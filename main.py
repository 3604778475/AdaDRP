import pandas as pd
import json
import argparse
import itertools
import time
from copy import deepcopy
import warnings

from utility import *
from dataload import get_multi_labeled_dataloader, get_drug_feat, get_unlabeled_dataloaders
from testing import testing
from finetune import fusion_taining
from pretraining import ResNetTissueClassifier, train_tissue_classifier, save_feat

def print_result(df, drugs, metrics, log_dir, brief_log_dir):
    write_to_logs(log_dir,"all results: ")
    write_to_logs(log_dir,df.astype('float64').round(4).to_string())
    write_to_logs(log_dir,"")

    print("5-fold average/std results: ")
    write_to_logs(log_dir,"5-fold average/std results: ")
    write_to_logs(brief_log_dir,"5-fold average/std results: ")
    mean_values = df.mean(axis=1)
    std_values = df.std(axis=1)
    max_drug_len = max(len(drug) for drug in drugs)
    for drug in drugs:
        drug_results = []
        for metric in metrics:
            drug_results.append(f"{metric}: {mean_values.loc[(drug, metric)]:.4f}/{std_values.loc[(drug, metric)]:.4f}")
        string = f"{drug.ljust(max_drug_len)}  {' | '.join(drug_results)}"
        print(string)
        write_to_logs(log_dir, string)
        write_to_logs(brief_log_dir, string)

    df_unstacked = df.unstack(level=1)
    five_metric_means = df_unstacked.mean(axis=1)
    five_metric_stds = df_unstacked.std(axis=1)
    formatted_results = " | ".join([f"{drug}: {mean:.4f}/{std:.4f}" for drug, mean, std in
                                    zip(five_metric_means.index, five_metric_means, five_metric_stds)])
    print("4 metrics average/std results: "+formatted_results)
    write_to_logs(log_dir,"4 metrics average/std results: "+formatted_results)
    write_to_logs(brief_log_dir,"4 metrics average/std results: "+formatted_results)

    #per-drug auc&aupr 5-fold average/std results:
    df_filtered = df.loc[df.index.get_level_values('metric').isin(['auc', 'aupr'])]
    two_metric_df_unstacked = df_filtered.unstack(level=1)
    two_metric_means = two_metric_df_unstacked.mean(axis=1)
    two_metric_drug_stds = two_metric_df_unstacked.std(axis=1)
    formatted_results = " | ".join([f"{drug}: {mean:.4f}/{std:.4f}" for drug, mean, std in
                                    zip(two_metric_means.index, two_metric_means, two_metric_drug_stds)])
    print("auc&aupr average/std results: "+formatted_results)
    write_to_logs(log_dir,"auc&aupr average/std results:"+formatted_results)
    write_to_logs(brief_log_dir,"auc&aupr average/std results:"+formatted_results)

    # per-metrics  average/std
    auc_data = df.xs('auc', level='metric')
    aupr_data = df.xs('aupr', level='metric')
    acc_data = df.xs('acc', level='metric')
    f1_data = df.xs('f1', level='metric')
    formatted_result = (f"auc：{auc_data.stack().mean():.4f}/{auc_data.stack().std():.4f}  "
                        f"aupr：{aupr_data.stack().mean():.4f}/{aupr_data.stack().std():.4f}  "
                        f"acc：{acc_data.stack().mean():.4f}/{acc_data.stack().std():.4f}  "
                        f"f1：{f1_data.stack().mean():.4f}/{f1_data.stack().std():.4f}")
    print(formatted_result)
    write_to_logs(log_dir,formatted_result)
    write_to_logs(brief_log_dir,formatted_result)

    # average/std
    df_single_row = np.array(df).flatten().tolist()
    formatted_result = f"all_metrics：{np.mean(df_single_row):.4f}/{np.std(df_single_row):.4f}"
    print(formatted_result)
    write_to_logs(log_dir,formatted_result)
    write_to_logs(brief_log_dir,formatted_result)

    # auc&aupr average/std
    df_single_row = np.array(df_filtered).flatten().tolist()
    formatted_result = f"auc_aupr_metrics：{np.mean(df_single_row):.4f}/{np.std(df_single_row):.4f}"
    print(formatted_result)
    write_to_logs(log_dir,formatted_result)
    write_to_logs(brief_log_dir,formatted_result)

def main(args, update_params_dict, now_time):
    seed = 2025
    set_seed(seed)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    with open(os.path.join('train_params.json'), 'r') as f:
        training_params = json.load(f)
    training_params['unlabeled'].update(update_params_dict)
    training_params['labeled'].update(update_params_dict)
    param_str = dict_to_str(update_params_dict)

    save_folder = os.path.join("logs", now_time)
    safe_make_dir(save_folder)
    log_dir = os.path.join(save_folder, 'logs.txt')
    brief_log_dir = os.path.join(save_folder, 'brief_logs.txt')

    print_logs(log_dir, "########################################################################################################", brief_log_dir)
    print_logs(log_dir, param_str, brief_log_dir)

    gex_features_df = pd.read_csv(config.gex_feature_file, index_col=0)
    drugs = ["5-Fluorouracil", "Cisplatin", "Cyclophosphamide", "Docetaxel", "Doxorubicin", "Etoposide", "Gemcitabine", "Paclitaxel", "Temozolomide"]

    training_params.update(
        {
            'device': device,
            'input_dim': gex_features_df.shape[-1]-1,
            'model_save_folder': os.path.join(save_folder, param_str),
            'retrain_flag': args.retrain_flag,
            'norm_flag': args.norm_flag,
            'alph': args.alph,
            'beta': args.beta,
        })
    safe_make_dir(training_params['model_save_folder'])

    print("======================= stage 1 =======================")

    # Data construction for the pre-training
    s_dataloaders, t_dataloaders = get_unlabeled_dataloaders(gex_features_df=gex_features_df, seed=seed, test_ratio=0.25,
                                                                    batch_size=training_params['unlabeled']['batch_size'])
    drug_feat = torch.from_numpy(get_drug_feat(drugs).astype('float32')).to(device)
    print_logs(log_dir, "s_dataloaders, t_dataloaders has been created")

    pretrain_model = ResNetTissueClassifier(
        input_dim=gex_features_df.shape[1] - 1,
        num_tissues=len(config.tissue_map),
        hidden_dims=[512, 256, 128, 64],
        drop=0.2
    ).to(device)

    # pre-training
    trained_model = train_tissue_classifier(
        model=pretrain_model,
        s_dataloaders=s_dataloaders,
        t_dataloaders=t_dataloaders,
        device=device,
        num_epochs=update_params_dict['pretrain_num_epochs'],
        lr=0.001,
        alpha=1.0,
        weight_decay=1e-4
    )

    save_feat(model=trained_model, s_dataloaders=s_dataloaders, t_dataloaders=t_dataloaders, device=device)


    print("======================= stage 2 & 3 =======================")
    # Data construction for feature fusion and drug response prediction
    labeled_dataloader = get_multi_labeled_dataloader(gex_features_df=gex_features_df, seed=seed, batch_size=training_params['labeled']['batch_size'],
                                                      drug=drugs, ccle_measurement=args.measurement, threshold_gdsc=args.thres_g, threshold_label=args.thres_s,
                                                      n_splits=args.n)

    fold = 0
    all_results = []
    metric_list = ['auc', 'aupr', 'f1', 'acc']
    result_df = pd.DataFrame(index=pd.MultiIndex.from_product([drugs, metric_list], names=['drug', 'metric']), columns=range(0, 5))  # drug metric fold
    for train_labeled_ccle, test_labeled_ccle, labeled_tcga in labeled_dataloader:
        print('Dataset Fold: {}'.format(fold))
        ft_encoder = deepcopy(trained_model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fusion_model = fusion_taining(encoder=ft_encoder, train_dataloader=train_labeled_ccle, val_dataloader=test_labeled_ccle, test_dataloader = labeled_tcga,
                                            drugs=drugs, gene_num = gex_features_df.shape[1] - 1, drug_feat = drug_feat, **wrap_params(training_params, type='labeled'))
            fusion_model.load_state_dict(torch.load(os.path.join(training_params['model_save_folder'], 'fusion_model.pt')))

            test_loss, results, y_true, y_pred, y_mask = testing(model=fusion_model, t_dataloader=labeled_tcga, drug_feat=drug_feat, device=training_params['device'])
            print('test-avg (TCGA) metrics (auc,aupr,f1,acc): {}'.format(np.around(results.mean(axis=1), 4)))
            write_to_logs(log_dir, 'test-avg (TCGA) metrics (auc,aupr,f1,acc): {}'.format(np.around(results.mean(axis=1), 4)))
            print('')

        results = pd.DataFrame(results)
        results.columns = drugs; results.index = ['auc','aupr','f1','acc']
        task_save_folder = os.path.join('results', 'fold-'+str(fold))
        safe_make_dir(task_save_folder)
        file_name = os.path.join(task_save_folder, param_str)
        with open(f'{file_name}.csv', 'w') as f:
            results.to_csv(f)
        all_results.append(results)

        for drug in drugs:
            for metric in metric_list:
                result_df.loc[(drug, metric), fold] = results.loc[metric, drug]
        fold = fold+1

    # Calculate the average result of 5-CV
    avg_result = np.mean(np.array(all_results), 0)
    avg_result = pd.DataFrame(avg_result)
    avg_result.columns = drugs
    avg_result.index = ['auc', 'aupr', 'f1', 'acc']
    file_name = os.path.join('results', param_str)
    with open(f'{file_name}.csv', 'w') as f:
        avg_result.to_csv(f)

    print_result(result_df, drugs, metric_list, log_dir, brief_log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pretraining and Fine_tuning')
    parser.add_argument('--metric', dest='metric', nargs='?', default='auroc', choices=['auroc', 'auprc'])
    parser.add_argument('--measurement', dest='measurement', nargs='?', default='Z_SCORE', choices=['Z_SCORE', 'LN_IC50', 'AUC'])
    parser.add_argument('--thres_gdsc', dest='thres_g', nargs='?', type=float, default=0.0)
    parser.add_argument('--thres_label', dest='thres_s', nargs='?', type=float, default=0.1)
    parser.add_argument('--n', dest='n', nargs='?', type=int, default=5)
    parser.add_argument('--alph', dest='alph', nargs='?', type=float, default=0.3, help='Coefficient of transfer loss')
    parser.add_argument('--beta', dest='beta', nargs='?', type=float, default=0.3, help='Coefficient of contrastive loss')

    train_group = parser.add_mutually_exclusive_group(required=False)
    train_group.add_argument('--train', dest='retrain_flag', action='store_true')
    train_group.add_argument('--no-train', dest='retrain_flag', action='store_false')
    parser.set_defaults(retrain_flag=True)

    norm_group = parser.add_mutually_exclusive_group(required=False)
    norm_group.add_argument('--norm', dest='norm_flag', action='store_true')
    norm_group.add_argument('--no-norm', dest='norm_flag', action='store_false')
    parser.set_defaults(norm_flag=True)

    args = parser.parse_args()
    print(f'current config is {args}')
    # params_grid = {
    #     "pretrain_num_epochs": [10],
    #     "train_num_epochs": [50, 100],
    #     "contrastive_num_epochs": [50, 100, 200],
    #     "K": [1,3,5,7,9]
    # }
    params_grid = {
        "pretrain_num_epochs": [100],
        "train_num_epochs": [100],
        "contrastive_num_epochs": [100],
        "K": [7]
    }

    warnings.filterwarnings("ignore")

    keys, values = zip(*params_grid.items())
    update_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    if args.retrain_flag == True:
        now_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    else:
        now_time = '2025-02-19_21-18-01'
    now_time = now_time + "bilinear+CDAN+gene_sim_K"


    for param_dict in update_params_dict_list:
        main(args=args, update_params_dict=param_dict, now_time=now_time)



#nohup python main.py >bilinear+CDAN+gene_sim_K.log 2>&1 &
#ps ux      kill -9 pid
#ps -f -p pid
