import os
import numpy as np
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
from mmpa_drp.stage2_program import fusion_training_program
from mmpa_drp.stage1_program import train_tissue_classifier_program, save_feat_program
from mmpa_drp.models import ProgramTissueClassifier


def print_result(df, drugs, metrics, log_dir, brief_log_dir):
    write_to_logs(log_dir, "all results: ")
    write_to_logs(log_dir, df.astype('float64').round(4).to_string())
    write_to_logs(log_dir, "")

    print("5-fold average/std results: ")
    write_to_logs(log_dir, "5-fold average/std results: ")
    write_to_logs(brief_log_dir, "5-fold average/std results: ")
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
    print("4 metrics average/std results: " + formatted_results)
    write_to_logs(log_dir, "4 metrics average/std results: " + formatted_results)
    write_to_logs(brief_log_dir, "4 metrics average/std results: " + formatted_results)

    df_filtered = df.loc[df.index.get_level_values('metric').isin(['auc', 'aupr'])]
    two_metric_df_unstacked = df_filtered.unstack(level=1)
    two_metric_means = two_metric_df_unstacked.mean(axis=1)
    two_metric_drug_stds = two_metric_df_unstacked.std(axis=1)
    formatted_results = " | ".join([f"{drug}: {mean:.4f}/{std:.4f}" for drug, mean, std in
                                    zip(two_metric_means.index, two_metric_means, two_metric_drug_stds)])
    print("auc&aupr average/std results: " + formatted_results)
    write_to_logs(log_dir, "auc&aupr average/std results:" + formatted_results)
    write_to_logs(brief_log_dir, "auc&aupr average/std results:" + formatted_results)

    auc_data = df.xs('auc', level='metric')
    aupr_data = df.xs('aupr', level='metric')
    acc_data = df.xs('acc', level='metric')
    f1_data = df.xs('f1', level='metric')
    formatted_result = (f"auc：{auc_data.stack().mean():.4f}/{auc_data.stack().std():.4f}  "
                        f"aupr：{aupr_data.stack().mean():.4f}/{aupr_data.stack().std():.4f}  "
                        f"acc：{acc_data.stack().mean():.4f}/{acc_data.stack().std():.4f}  "
                        f"f1：{f1_data.stack().mean():.4f}/{f1_data.stack().std():.4f}")
    print(formatted_result)
    write_to_logs(log_dir, formatted_result)
    write_to_logs(brief_log_dir, formatted_result)

    df_single_row = np.array(df).flatten().tolist()
    formatted_result = f"all_metrics：{np.mean(df_single_row):.4f}/{np.std(df_single_row):.4f}"
    print(formatted_result)
    write_to_logs(log_dir, formatted_result)
    write_to_logs(brief_log_dir, formatted_result)

    df_single_row = np.array(df_filtered).flatten().tolist()
    formatted_result = f"auc_aupr_metrics：{np.mean(df_single_row):.4f}/{np.std(df_single_row):.4f}"
    print(formatted_result)
    write_to_logs(log_dir, formatted_result)
    write_to_logs(brief_log_dir, formatted_result)


def save_stage2_features_for_visualization(model, dataloader, drug_feat, device, save_path):
    """
    保存 Stage2/3 特征用于后续 T-SNE 可视化。
    保存内容：gene_feat, fused_feature, labels, preds
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    all_gene_feat = []
    all_fused = []
    all_labels = []
    all_preds = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            tissue = batch[3].to(device)
            outputs = model(
                x, drug_feat=drug_feat, tissue_idx=tissue,
                return_dict=True, return_fused_feature=True
            )
            batch_size = outputs["gene_feat"].size(0)
            drug_num = outputs["fused_feature"].size(1)
            all_gene_feat.append(outputs["gene_feat"].cpu().numpy())
            all_fused.append(outputs["fused_feature"].cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_preds.append(outputs["pred"].view(batch_size, drug_num).cpu().numpy())

    np.savez(
        save_path,
        gene_feat=np.concatenate(all_gene_feat, axis=0),
        fused_feature=np.concatenate(all_fused, axis=0),
        labels=np.concatenate(all_labels, axis=0),
        preds=np.concatenate(all_preds, axis=0),
    )
    print(f"[Visualization] Saved Stage2/3 features to {save_path}")


def main(args, update_params_dict, now_time):
    seed = 2025
    set_seed(seed)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    with open(os.path.join('train_params.json'), 'r') as f:
        training_params = json.load(f)

    # Build param_str with abbreviated keys
    _key_abbr = {
        'pretrain_num_epochs': 'pt',
        'train_num_epochs': 'tr',
        'contrastive_num_epochs': 'ct',
        'domain_sep_hidden_dim': 'dsh',
        'domain_sep_output_dim': 'dso',
        'lambda_ortho': 'lo',
        'lambda_recon': 'lr',
        'lambda_private_domain': 'lpd',
        'lambda_migrate': 'lm',
    }
    param_dict_for_str = {}
    for k, v in update_params_dict.items():
        param_dict_for_str[_key_abbr.get(k, k)] = v

    # Keep folder names short to avoid Windows MAX_PATH (260) limits
    if args.use_fpi:
        param_dict_for_str['fpi'] = (
            f"gm{args.fpi_gamma_max}_gi{args.fpi_gamma_init}_"
            f"qp{args.fpi_q_prior_weight}_l{args.lambda_fpi_gamma_l1}_"
            f"d{args.fpi_detach_mode}_m{args.fpi_m_max}_fp{args.fpi_fp_dim}"
        )
    if args.use_domain_sep:
        param_dict_for_str['dom'] = (
            f"h{args.domain_sep_hidden_dim}_o{args.domain_sep_output_dim}_"
            f"o{args.lambda_ortho}_r{args.lambda_recon}_"
            f"p{args.lambda_private_domain}_m{args.lambda_migrate}"
        )
    if args.use_ot_alignment:
        param_dict_for_str['ot'] = (
            f"l{args.lambda_ot}_r{args.ot_reg}_{args.ot_cost_type[0]}{args.ot_marginal_type[0]}"
        )

    param_str = dict_to_str(param_dict_for_str)

    save_folder = os.path.join("logs", now_time)
    safe_make_dir(save_folder)
    log_dir = os.path.join(save_folder, 'logs.txt')
    brief_log_dir = os.path.join(save_folder, 'brief_logs.txt')

    print_logs(log_dir, "########################################################################################################", brief_log_dir)
    print_logs(log_dir, param_str, brief_log_dir)

    gex_features_df = pd.read_csv(config.gex_feature_file, index_col=0)
    drugs = ["5-Fluorouracil", "Cisplatin", "Cyclophosphamide", "Docetaxel",
             "Doxorubicin", "Etoposide", "Gemcitabine", "Paclitaxel", "Temozolomide"]

    training_params['unlabeled'].update(update_params_dict)
    training_params['labeled'].update(update_params_dict)

    training_params.update({
        'device': device,
        'input_dim': gex_features_df.shape[-1] - 1,
        'model_save_folder': os.path.join(save_folder, param_str),
        'retrain_flag': args.retrain_flag,
        'norm_flag': args.norm_flag,
        'alph': args.alph,
        'beta': args.beta,
        'num_programs': args.num_programs,
        'program_dim': args.program_dim,
        'lambda_div': args.lambda_div,
        'lambda_sparse': args.lambda_sparse,
        'use_ot_alignment': args.use_ot_alignment,
        'lambda_ot': args.lambda_ot,
        'ot_reg': args.ot_reg,
        'ot_max_iter': args.ot_max_iter,
        'ot_cost_type': args.ot_cost_type,
        'ot_marginal_type': args.ot_marginal_type,
        'use_cdan': args.use_cdan,
        'alpha': args.alph,
        'beta': args.beta,
        'topk_contrastive': args.K,
        # backbone
        'backbone_type': args.backbone_type,
        'num_gene_tokens': args.num_gene_tokens,
        'token_dim': args.token_dim,
        'convnext_depth': args.convnext_depth,
        'convnext_kernel_size': args.convnext_kernel_size,
        'convnext_expansion': args.convnext_expansion,
        # FPI
        'use_fpi': args.use_fpi,
        'fpi_config': {
            'program_dim': args.program_dim,
            'drug_dim': 64,
            'fragment_dim': args.program_dim,
            'num_programs': args.num_programs,
            'num_drugs': len(drugs),
            'max_fragments': args.fpi_m_max,
            'fp_dim': args.fpi_fp_dim,
            'q_prior_weight': args.fpi_q_prior_weight,
            'gamma_max': args.fpi_gamma_max,
            'gamma_init': args.fpi_gamma_init,
            'dropout': args.fpi_dropout,
            'detach_mode': args.fpi_detach_mode,
        },
        'lambda_fpi_gamma_l1': args.lambda_fpi_gamma_l1,
        'lambda_fpi_entropy': args.lambda_fpi_entropy,
        # Domain separation
        'use_domain_sep': args.use_domain_sep,
        'domain_sep_hidden_dim': args.domain_sep_hidden_dim,
        'domain_sep_output_dim': args.domain_sep_output_dim,
        'lambda_ortho': args.lambda_ortho,
        'lambda_recon': args.lambda_recon,
        'lambda_private_domain': args.lambda_private_domain,
        'lambda_migrate': args.lambda_migrate,
        'output_dir': args.output_dir if args.output_dir else os.path.dirname(log_dir),
    })
    safe_make_dir(training_params['model_save_folder'])

    print("======================= stage 1 (MMPA Program) =======================")

    s_dataloaders, t_dataloaders = get_unlabeled_dataloaders(
        gex_features_df=gex_features_df, seed=seed, test_ratio=0.25,
        batch_size=training_params['unlabeled']['batch_size'])

    drug_feat = torch.from_numpy(get_drug_feat(drugs).astype('float32')).to(device)
    print_logs(log_dir, "s_dataloaders, t_dataloaders has been created")

    pretrain_model = ProgramTissueClassifier(
        input_dim=gex_features_df.shape[1] - 1,
        num_tissues=len(config.tissue_map),
        num_programs=args.num_programs,
        program_dim=args.program_dim,
        hidden_dims=[512, 256, 128, 64],
        drop=0.2,
        backbone_type=args.backbone_type,
        num_gene_tokens=args.num_gene_tokens,
        token_dim=args.token_dim,
        convnext_depth=args.convnext_depth,
        convnext_kernel_size=args.convnext_kernel_size,
        convnext_expansion=args.convnext_expansion,
        use_domain_sep=args.use_domain_sep,
        domain_sep_hidden_dim=args.domain_sep_hidden_dim,
        domain_sep_output_dim=args.domain_sep_output_dim,
        program_encoder_type='latent',
    ).to(device)

    trained_model = train_tissue_classifier_program(
        model=pretrain_model,
        s_dataloaders=s_dataloaders,
        t_dataloaders=t_dataloaders,
        device=device,
        num_epochs=update_params_dict['pretrain_num_epochs'],
        lr=0.001,
        alpha=1.0,
        weight_decay=1e-4,
        lambda_div=args.lambda_div,
        lambda_sparse=args.lambda_sparse,
        lambda_ortho=args.lambda_ortho,
        lambda_recon=args.lambda_recon,
        lambda_private_domain=args.lambda_private_domain,
    )

    save_feat_program(model=trained_model, s_dataloaders=s_dataloaders, t_dataloaders=t_dataloaders, device=device)

    stage1_ckpt_dir = os.path.join("checkpoints", "stage1")
    safe_make_dir(stage1_ckpt_dir)
    stage1_ckpt_path = os.path.join(stage1_ckpt_dir, "stage1_model.pt")
    torch.save(trained_model.state_dict(), stage1_ckpt_path)
    print(f"Saved Stage1 model to {stage1_ckpt_path}")

    print("======================= stage 2 & 3 (MMPA Program) =======================")
    labeled_dataloader = get_multi_labeled_dataloader(
        gex_features_df=gex_features_df, seed=seed,
        batch_size=training_params['labeled']['batch_size'],
        drug=drugs, ccle_measurement=args.measurement,
        threshold_gdsc=args.thres_g, threshold_label=args.thres_s,
        n_splits=args.n)

    frag_fp = None
    frag_mask = None
    frag_smiles = None
    if args.use_fpi:
        from mmpa_drp.fragment_utils import load_drug_fragments
        frag_data = load_drug_fragments(args.fpi_fragment_npz, expected_drug_names=drugs)
        frag_fp = frag_data["frag_fp"].to(device)
        frag_mask = frag_data["frag_mask"].to(device)
        frag_smiles = frag_data["frag_smiles"]

    method_name = param_str
    fold = 0
    all_results = []
    metric_list = ['auc', 'aupr', 'f1', 'acc']
    result_df = pd.DataFrame(
        index=pd.MultiIndex.from_product([drugs, metric_list], names=['drug', 'metric']),
        columns=range(0, 5))

    for train_labeled_ccle, test_labeled_ccle, labeled_tcga in labeled_dataloader:
        print('Dataset Fold: {}'.format(fold))
        ft_encoder = deepcopy(trained_model.program_encoder)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fusion_model = fusion_training_program(
                encoder=ft_encoder,
                train_dataloader=train_labeled_ccle,
                val_dataloader=test_labeled_ccle,
                test_dataloader=labeled_tcga,
                drugs=drugs,
                gene_num=gex_features_df.shape[1] - 1,
                drug_feat=drug_feat,
                fold=fold,
                method_name=method_name,
                frag_fp=frag_fp,
                frag_mask=frag_mask,
                frag_smiles=frag_smiles,
                **wrap_params(training_params, type='labeled')
            )
            fusion_model.load_state_dict(torch.load(os.path.join(training_params['model_save_folder'], f'fusion_model_fold{fold}.pt')))

            vis_feat_path = os.path.join("outputs", "saved_features", f"stage2_fold{fold}_tcga.npz")
            save_stage2_features_for_visualization(
                model=fusion_model,
                dataloader=labeled_tcga,
                drug_feat=drug_feat,
                device=training_params['device'],
                save_path=vis_feat_path,
            )

            test_loss, results, y_true, y_pred, y_mask = testing(
                model=fusion_model, t_dataloader=labeled_tcga, drug_feat=drug_feat, device=training_params['device'])
            print('test-avg (TCGA) metrics (auc,aupr,f1,acc): {}'.format(np.around(results.mean(axis=1), 4)))
            write_to_logs(log_dir, 'test-avg (TCGA) metrics (auc,aupr,f1,acc): {}'.format(np.around(results.mean(axis=1), 4)))
            print('')

        results = pd.DataFrame(results)
        results.columns = drugs
        results.index = ['auc', 'aupr', 'f1', 'acc']
        task_save_folder = os.path.join('results', 'fold-' + str(fold))
        safe_make_dir(task_save_folder)
        file_name = os.path.join(task_save_folder, param_str + "_mmpa")
        with open(f'{file_name}.csv', 'w') as f:
            results.to_csv(f)
        all_results.append(results)

        for drug in drugs:
            for metric in metric_list:
                result_df.loc[(drug, metric), fold] = results.loc[metric, drug]
        fold = fold + 1

    avg_result = np.mean(np.array(all_results), 0)
    avg_result = pd.DataFrame(avg_result)
    avg_result.columns = drugs
    avg_result.index = ['auc', 'aupr', 'f1', 'acc']
    file_name = os.path.join('results', param_str + "_mmpa")
    with open(f'{file_name}.csv', 'w') as f:
        avg_result.to_csv(f)

    safe_make_dir("outputs")
    avg_result.to_csv("outputs/mmpa_step2_results.csv")

    print_result(result_df, drugs, metric_list, log_dir, brief_log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MMPA-DRP Training')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override output directory for diagnostics and logs')
    parser.add_argument('--metric', dest='metric', nargs='?', default='auroc',
                        choices=['auroc', 'auprc'])
    parser.add_argument('--measurement', dest='measurement', nargs='?', default='Z_SCORE',
                        choices=['Z_SCORE', 'LN_IC50', 'AUC'])
    parser.add_argument('--thres_gdsc', dest='thres_g', nargs='?', type=float, default=0.0)
    parser.add_argument('--thres_label', dest='thres_s', nargs='?', type=float, default=0.1)
    parser.add_argument('--n', dest='n', nargs='?', type=int, default=5)

    parser.add_argument('--alph', dest='alph', nargs='?', type=float, default=1.4,
                        help='Coefficient of transfer loss')
    parser.add_argument('--beta', dest='beta', nargs='?', type=float, default=0.066,
                        help='Coefficient of contrastive loss')

    parser.add_argument('--num_programs', type=int, default=8,
                        help='Number of molecular programs (K)')
    parser.add_argument('--program_dim', type=int, default=64,
                        help='Dimension of each program token')
    parser.add_argument('--lambda_div', type=float, default=0.01,
                        help='Diversity loss weight')
    parser.add_argument('--lambda_sparse', type=float, default=0.001,
                        help='Sparsity loss weight')

    parser.add_argument('--use_ot_alignment', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable OT alignment loss')
    parser.add_argument('--lambda_ot', type=float, default=0.005, help='OT loss weight')
    parser.add_argument('--ot_reg', type=float, default=0.1, help='Sinkhorn entropy regularization')
    parser.add_argument('--ot_max_iter', type=int, default=50, help='Sinkhorn max iterations')
    parser.add_argument('--ot_cost_type', type=str, default='cosine',
                        choices=['cosine', 'euclidean'], help='OT cost metric')
    parser.add_argument('--ot_marginal_type', type=str, default='uniform',
                        choices=['uniform', 'unbalanced', 'max_sim'], help='OT marginal type')

    parser.add_argument('--use_cdan', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable CDAN domain adaptation')

    parser.add_argument('--use_domain_sep', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable domain-level shared/private decomposition')
    parser.add_argument('--domain_sep_hidden_dim', type=int, default=256,
                        help='Hidden dim for domain separation MLPs')
    parser.add_argument('--domain_sep_output_dim', type=int, default=64,
                        help='Output dim for shared/private representations')
    parser.add_argument('--lambda_ortho', type=float, default=0.001,
                        help='Orthogonality loss weight between shared and private')
    parser.add_argument('--lambda_recon', type=float, default=0.005,
                        help='Reconstruction loss weight')
    parser.add_argument('--lambda_private_domain', type=float, default=0.005,
                        help='Private domain classification loss weight')
    parser.add_argument('--lambda_migrate', type=float, default=0.01,
                        help='Stage 2: migration loss weight')

    parser.add_argument('--use_fpi', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable Fragment-Program Interaction')
    parser.add_argument('--fpi_fragment_npz', type=str,
                        default='data/drug_fragments_brics_m16_fp1024.npz',
                        help='Path to precomputed fragment NPZ')
    parser.add_argument('--fpi_gamma_max', type=float, default=0.1, help='FPI gamma max')
    parser.add_argument('--fpi_gamma_init', type=float, default=0.05, help='FPI gamma init')
    parser.add_argument('--fpi_q_prior_weight', type=float, default=0.01,
                        help='FPI q prior weight')
    parser.add_argument('--fpi_dropout', type=float, default=0.1, help='FPI dropout')
    parser.add_argument('--fpi_m_max', type=int, default=16, help='Max fragments per drug')
    parser.add_argument('--fpi_fp_dim', type=int, default=1024,
                        help='Fragment fingerprint dimension')
    parser.add_argument('--lambda_fpi_gamma_l1', type=float, default=0.001,
                        help='L1 regularization on FPI gamma')
    parser.add_argument('--lambda_fpi_entropy', type=float, default=0.0,
                        help='FPI attention entropy loss weight')
    parser.add_argument('--fpi_detach_mode', type=str, default='none',
                        choices=['none', 'z', 'zq', 'zq_drug', 'all'],
                        help='Stop-gradient mode for FPI branch')

    parser.add_argument('--K', type=int, default=9, help='Top-K for contrastive domain isolation')
    parser.add_argument('--pretrain_num_epochs', type=int, default=100,
                        help='Pretraining epochs')
    parser.add_argument('--train_num_epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--contrastive_num_epochs', type=int, default=100,
                        help='Contrastive epochs')

    parser.add_argument('--backbone_type', type=str, default='convnext1d',
                        choices=['mlp', 'convnext1d', 'resnet'],
                        help='Backbone type for MolecularProgramEncoder')
    parser.add_argument('--num_gene_tokens', type=int, default=64,
                        help='Number of gene tokens for ConvNeXt1D tokenizer')
    parser.add_argument('--token_dim', type=int, default=64,
                        help='Token dimension for ConvNeXt1D')
    parser.add_argument('--convnext_depth', type=int, default=1,
                        help='Number of ConvNeXt1D blocks')
    parser.add_argument('--convnext_kernel_size', type=int, default=7,
                        help='Kernel size for ConvNeXt1D depthwise conv')
    parser.add_argument('--convnext_expansion', type=int, default=4,
                        help='Expansion factor for ConvNeXt1D MLP')

    parser.add_argument('--train', dest='retrain_flag', action=argparse.BooleanOptionalAction,
                        default=True, help='Run training')
    parser.add_argument('--norm', dest='norm_flag', action=argparse.BooleanOptionalAction,
                        default=True, help='Normalize input features')

    args = parser.parse_args()
    print(f'current config is {args}')

    params_grid = {
        "pretrain_num_epochs": [args.pretrain_num_epochs],
        "train_num_epochs": [args.train_num_epochs],
        "contrastive_num_epochs": [args.contrastive_num_epochs],
        "K": [args.K]
    }

    warnings.filterwarnings("ignore")

    keys, values = zip(*params_grid.items())
    update_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    if args.retrain_flag:
        now_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    else:
        now_time = '2025-02-19_21-18-01'
    now_time = now_time + "mmpa_program"

    for param_dict in update_params_dict_list:
        main(args=args, update_params_dict=param_dict, now_time=now_time)
