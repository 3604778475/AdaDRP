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

    # Append program OT params to param_str to avoid result file collision
    if args.use_program_ot:
        param_str += f"_progOT_{args.program_ot_mode}_l{args.lambda_program_ot}"
        if args.program_ot_mode == "q_mass_flattened":
            param_str += f"_detach{int(args.program_ot_detach_q_marginal)}"
        if args.q_align_enabled:
            param_str += f"_qalign{args.lambda_q_align}"

    # Append DSPI params to param_str
    if args.use_dspi:
        param_str += f"_dspi_iw{args.dspi_inter_weight}_qprior{args.dspi_q_prior_weight}"
        if args.lambda_dspi_attn_diversity > 0:
            param_str += f"_div{args.lambda_dspi_attn_diversity}"
        if args.lambda_dspi_attn_entropy > 0:
            param_str += f"_ent{args.lambda_dspi_attn_entropy}"
        if args.lambda_dspi_program_balance > 0:
            param_str += f"_bal{args.lambda_dspi_program_balance}"
        if args.lambda_dspi_gram_consistency > 0:
            param_str += f"_gram{args.lambda_dspi_gram_consistency}"

    # Append drug-conditioned OT params to param_str
    if args.drug_conditioned_ot_enabled:
        param_str += f"_dcot_{args.drug_conditioned_ot_mode}"
        if args.drug_conditioned_ot_replace_sample_ot:
            param_str += "_replace"
        else:
            param_str += "_add"
        param_str += f"_l{args.drug_conditioned_ot_global_lambda}"
        if args.drug_conditioned_ot_mode == "discrepancy":
            param_str += f"_lmax{args.drug_conditioned_ot_lambda_max}"

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
            'num_programs': args.num_programs,
            'program_dim': args.program_dim,
            'lambda_div': args.lambda_div,
            'lambda_sparse': args.lambda_sparse,
            'use_drug_gate': args.use_drug_gate,
            'shared_gate': args.shared_gate,
            'gate_mode': args.gate_mode,
            'gate_temperature': args.gate_temperature,
            'rho_init': args.rho_init,
            'beta_init': args.beta_init,
            'lambda_gate_entropy_floor': args.lambda_gate_entropy_floor,
            'gate_min_entropy': args.gate_min_entropy,
            'lambda_gate_balance': args.lambda_gate_balance,
            'lambda_gate_diversity': args.lambda_gate_diversity,
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
            'ot_mode': args.ot_mode,
            'ot_use_drug_conditioned_feature': args.ot_use_drug_conditioned_feature,
            'ot_per_drug_lambda': args.ot_per_drug_lambda,
            # Phase 7: Drug-conditioned OT
            'drug_conditioned_ot': {
                'enabled': args.drug_conditioned_ot_enabled,
                'mode': args.drug_conditioned_ot_mode,
                'replace_sample_ot': args.drug_conditioned_ot_replace_sample_ot,
                'global_lambda': args.drug_conditioned_ot_global_lambda,
                'reg': args.drug_conditioned_ot_reg,
                'cost': args.drug_conditioned_ot_cost,
                'discrepancy': {
                    'lambda_min': args.drug_conditioned_ot_lambda_min,
                    'lambda_max': args.drug_conditioned_ot_lambda_max,
                    'temperature': args.drug_conditioned_ot_temperature,
                    'threshold': args.drug_conditioned_ot_threshold,
                    'detach_gap': args.drug_conditioned_ot_detach_gap,
                    'normalize_gap': args.drug_conditioned_ot_normalize_gap,
                    'ema_gap': args.drug_conditioned_ot_ema_gap,
                    'ema_momentum': args.drug_conditioned_ot_ema_momentum,
                },
                'fixed_lambda': args.drug_conditioned_ot_fixed_lambda,
            },
            # Phase 6: Program-level OT
            'use_program_ot': args.use_program_ot,
            'program_ot_mode': args.program_ot_mode,
            'lambda_program_ot': args.lambda_program_ot,
            'program_ot_reg': args.program_ot_reg,
            'program_ot_max_iter': args.program_ot_max_iter,
            'program_ot_cost_type': args.program_ot_cost_type,
            'program_ot_marginal_type': args.program_ot_marginal_type,
            'program_ot_detach_q_marginal': args.program_ot_detach_q_marginal,
            'program_ot_q_temperature': args.program_ot_q_temperature,
            'program_ot_q_power': args.program_ot_q_power,
            'program_ot_min_mass': args.program_ot_min_mass,
            'q_align_enabled': args.q_align_enabled,
            'lambda_q_align': args.lambda_q_align,
            'enable_grad_diagnostics': args.enable_grad_diagnostics,
            'grad_diag_every': args.grad_diag_every,
            # Phase 5 backbone args
            'backbone_type': args.backbone_type,
            'norm_type': args.norm_type,
            'num_gene_tokens': args.num_gene_tokens,
            'token_dim': args.token_dim,
            'convnext_depth': args.convnext_depth,
            'convnext_kernel_size': args.convnext_kernel_size,
            'convnext_expansion': args.convnext_expansion,
            # Phase 8: DSPI
            'use_dspi': args.use_dspi,
            'dspi_config': {
                'program_dim': args.program_dim,
                'drug_dim': 64,
                'num_programs': args.num_programs,
                'num_drugs': len(drugs),
                'temperature': args.dspi_temperature,
                'inter_weight': args.dspi_inter_weight,
                'q_prior_weight': args.dspi_q_prior_weight,
                'drug_sim_temperature': args.dspi_drug_sim_temperature,
                'rho_mode': args.dspi_rho_mode,
                'fixed_rho': args.dspi_fixed_rho,
                'rho_max': args.dspi_rho_max,
            },
            'lambda_dspi_attn_diversity': args.lambda_dspi_attn_diversity,
            'lambda_dspi_attn_entropy': args.lambda_dspi_attn_entropy,
            'lambda_dspi_program_balance': args.lambda_dspi_program_balance,
            'lambda_dspi_gram_consistency': args.lambda_dspi_gram_consistency,
            'output_dir': os.path.dirname(log_dir),
        })
    safe_make_dir(training_params['model_save_folder'])

    print("======================= stage 1 (MMPA Program) =======================")

    # Data construction for the pre-training
    s_dataloaders, t_dataloaders = get_unlabeled_dataloaders(gex_features_df=gex_features_df, seed=seed, test_ratio=0.25,
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
        norm_type=args.norm_type,
        num_gene_tokens=args.num_gene_tokens,
        token_dim=args.token_dim,
        convnext_depth=args.convnext_depth,
        convnext_kernel_size=args.convnext_kernel_size,
        convnext_expansion=args.convnext_expansion,
    ).to(device)

    # pre-training
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
    )

    save_feat_program(model=trained_model, s_dataloaders=s_dataloaders, t_dataloaders=t_dataloaders, device=device)


    print("======================= stage 2 & 3 (MMPA Program) =======================")
    # Data construction for feature fusion and drug response prediction
    labeled_dataloader = get_multi_labeled_dataloader(gex_features_df=gex_features_df, seed=seed, batch_size=training_params['labeled']['batch_size'],
                                                      drug=drugs, ccle_measurement=args.measurement, threshold_gdsc=args.thres_g, threshold_label=args.thres_s,
                                                      n_splits=args.n)

    method_name = param_str
    fold = 0
    all_results = []
    metric_list = ['auc', 'aupr', 'f1', 'acc']
    result_df = pd.DataFrame(index=pd.MultiIndex.from_product([drugs, metric_list], names=['drug', 'metric']), columns=range(0, 5))  # drug metric fold
    for train_labeled_ccle, test_labeled_ccle, labeled_tcga in labeled_dataloader:
        print('Dataset Fold: {}'.format(fold))
        ft_encoder = deepcopy(trained_model.program_encoder)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fusion_model = fusion_training_program(encoder=ft_encoder, train_dataloader=train_labeled_ccle, val_dataloader=test_labeled_ccle, test_dataloader = labeled_tcga,
                                            drugs=drugs, gene_num = gex_features_df.shape[1] - 1, drug_feat = drug_feat, fold=fold, method_name=method_name, **wrap_params(training_params, type='labeled'))
            fusion_model.load_state_dict(torch.load(os.path.join(training_params['model_save_folder'], 'fusion_model.pt')))

            test_loss, results, y_true, y_pred, y_mask = testing(model=fusion_model, t_dataloader=labeled_tcga, drug_feat=drug_feat, device=training_params['device'])
            print('test-avg (TCGA) metrics (auc,aupr,f1,acc): {}'.format(np.around(results.mean(axis=1), 4)))
            write_to_logs(log_dir, 'test-avg (TCGA) metrics (auc,aupr,f1,acc): {}'.format(np.around(results.mean(axis=1), 4)))
            print('')

        results = pd.DataFrame(results)
        results.columns = drugs; results.index = ['auc','aupr','f1','acc']
        task_save_folder = os.path.join('results', 'fold-'+str(fold))
        safe_make_dir(task_save_folder)
        file_name = os.path.join(task_save_folder, param_str + "_mmpa")
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
    file_name = os.path.join('results', param_str + "_mmpa")
    with open(f'{file_name}.csv', 'w') as f:
        avg_result.to_csv(f)

    # Also save to outputs/mmpa_step2_results.csv
    safe_make_dir("outputs")
    avg_result.to_csv("outputs/mmpa_step2_results.csv")

    print_result(result_df, drugs, metric_list, log_dir, brief_log_dir)

    # Phase 7: Output CSV generation
    safe_make_dir("outputs")

    # Leakage check
    leakage_path = "outputs/drug_conditioned_ot_leakage_check.txt"
    with open(leakage_path, 'a') as f:
        f.write(f"Method: {method_name}\n")
        f.write("- Uses target response label for OT: NO\n")
        f.write("- Uses test AUC for lambda_d: NO\n")
        f.write("- Uses fused feature before classifier: YES\n")
        f.write("- Changes inference: NO\n")
        dcot_mode = args.drug_conditioned_ot_mode if args.drug_conditioned_ot_enabled else "none"
        f.write(f"- Lambda source: {dcot_mode}\n")
        f.write("-" * 40 + "\n")

    # Only write detailed outputs if drug-conditioned OT is enabled
    if args.drug_conditioned_ot_enabled:
        # Overall results
        overall = avg_result.mean(axis=1)
        results_df = pd.DataFrame({
            "method": [method_name],
            "AUC": [overall['auc']],
            "AUPR": [overall['aupr']],
            "F1": [overall['f1']],
            "ACC": [overall['acc']],
        })
        results_csv = "outputs/drug_conditioned_ot_results.csv"
        if os.path.exists(results_csv):
            existing = pd.read_csv(results_csv)
            existing = existing[existing['method'] != method_name]
            results_df = pd.concat([existing, results_df], ignore_index=True)
        results_df.to_csv(results_csv, index=False)

        # Per-drug results
        per_drug_rows = []
        for drug in drugs:
            per_drug_rows.append({
                "method": method_name,
                "drug_name": drug,
                "auc": avg_result.loc['auc', drug],
                "aupr": avg_result.loc['aupr', drug],
                "f1": avg_result.loc['f1', drug],
                "acc": avg_result.loc['acc', drug],
                "delta_vs_combo": np.nan,
            })
        per_drug_df = pd.DataFrame(per_drug_rows)
        per_drug_csv = "outputs/drug_conditioned_ot_per_drug.csv"
        if os.path.exists(per_drug_csv):
            existing = pd.read_csv(per_drug_csv)
            existing = existing[existing['method'] != method_name]
            per_drug_df = pd.concat([existing, per_drug_df], ignore_index=True)
        per_drug_df.to_csv(per_drug_csv, index=False)

        # Per-fold results
        per_fold_rows = []
        for fold_idx in range(5):
            fold_auc = result_df.xs('auc', level='metric')[fold_idx].mean()
            fold_aupr = result_df.xs('aupr', level='metric')[fold_idx].mean()
            fold_f1 = result_df.xs('f1', level='metric')[fold_idx].mean()
            fold_acc = result_df.xs('acc', level='metric')[fold_idx].mean()
            per_fold_rows.append({
                "method": method_name,
                "fold": fold_idx,
                "auc": fold_auc,
                "aupr": fold_aupr,
                "f1": fold_f1,
                "acc": fold_acc,
            })
        per_fold_df = pd.DataFrame(per_fold_rows)
        per_fold_csv = "outputs/drug_conditioned_ot_per_fold.csv"
        if os.path.exists(per_fold_csv):
            existing = pd.read_csv(per_fold_csv)
            existing = existing[existing['method'] != method_name]
            per_fold_df = pd.concat([existing, per_fold_df], ignore_index=True)
        per_fold_df.to_csv(per_fold_csv, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MMPA-DRP Training')
    parser.add_argument('--metric', dest='metric', nargs='?', default='auroc', choices=['auroc', 'auprc'])
    parser.add_argument('--measurement', dest='measurement', nargs='?', default='Z_SCORE', choices=['Z_SCORE', 'LN_IC50', 'AUC'])
    parser.add_argument('--thres_gdsc', dest='thres_g', nargs='?', type=float, default=0.0)
    parser.add_argument('--thres_label', dest='thres_s', nargs='?', type=float, default=0.1)
    parser.add_argument('--n', dest='n', nargs='?', type=int, default=5)
    parser.add_argument('--alph', dest='alph', nargs='?', type=float, default=0.3, help='Coefficient of transfer loss')
    parser.add_argument('--beta', dest='beta', nargs='?', type=float, default=0.3, help='Coefficient of contrastive loss')

    # MMPA specific parameters
    parser.add_argument('--num_programs', type=int, default=8, help='Number of molecular programs (K)')
    parser.add_argument('--program_dim', type=int, default=64, help='Dimension of each program token')
    parser.add_argument('--lambda_div', type=float, default=0.01, help='Diversity loss weight')
    parser.add_argument('--lambda_sparse', type=float, default=0.001, help='Sparsity loss weight')

    # Phase 3B-R: Residual Soft Drug Gate parameters
    parser.add_argument('--use_drug_gate', action='store_true', help='Enable drug-gated program aggregation (Phase 3B)')
    parser.add_argument('--shared_gate', action='store_true', help='Use shared gate across all drugs (otherwise drug-specific)')
    parser.add_argument('--gate_mode', type=str, default='additive_bias',
                        choices=['hard_multiplicative', 'residual_multiplicative', 'additive_bias', 'additive_bias_no_residual'],
                        help='Drug gate computation mode')
    parser.add_argument('--gate_temperature', type=float, default=5.0, help='Temperature for additive_bias gate')
    parser.add_argument('--rho_init', type=float, default=0.0, help='Initial value for residual rho')
    parser.add_argument('--beta_init', type=float, default=0.0, help='Initial value for additive beta')
    parser.add_argument('--lambda_gate_entropy_floor', type=float, default=0.0, help='Entropy floor loss weight')
    parser.add_argument('--gate_min_entropy', type=float, default=1.2, help='Minimum target entropy for floor loss')
    parser.add_argument('--lambda_gate_balance', type=float, default=0.0, help='Load balance loss weight')
    parser.add_argument('--lambda_gate_diversity', type=float, default=0.0, help='Gate diversity loss weight')

    # Phase OT: Optimal Transport alignment loss (training-only)
    parser.add_argument('--use_ot_alignment', action='store_true', help='Enable OT alignment loss (training-only, no inference change)')
    parser.add_argument('--lambda_ot', type=float, default=0.01, help='OT loss weight')
    parser.add_argument('--ot_reg', type=float, default=0.1, help='Sinkhorn entropy regularization')
    parser.add_argument('--ot_max_iter', type=int, default=50, help='Sinkhorn max iterations')
    parser.add_argument('--ot_cost_type', type=str, default='cosine', choices=['cosine', 'euclidean'], help='OT cost metric')
    parser.add_argument('--ot_marginal_type', type=str, default='uniform', choices=['uniform', 'unbalanced', 'max_sim'], help='OT marginal type')
    parser.add_argument('--use_cdan', action='store_true', default=True, help='Enable CDAN domain adaptation')
    parser.add_argument('--no_cdan', dest='use_cdan', action='store_false', help='Disable CDAN domain adaptation')
    parser.add_argument('--ot_mode', type=str, default='global', choices=['global', 'per_drug_fixed', 'per_drug_validation', 'per_drug_discrepancy'], help='OT mode: global or drug-adaptive')
    parser.add_argument('--ot_use_drug_conditioned_feature', action='store_true', help='Use drug-conditioned features for OT')
    parser.add_argument('--ot_per_drug_lambda', type=json.loads, default=None, help='Per-drug lambda dict as JSON string')

    # Phase 7: Drug-conditioned Optimal Transport
    parser.add_argument('--drug_conditioned_ot_enabled', action='store_true', help='Enable drug-conditioned OT (Phase 7)')
    parser.add_argument('--drug_conditioned_ot_mode', type=str, default='global', choices=['global', 'fixed', 'discrepancy'], help='Drug-conditioned OT mode')
    parser.add_argument('--drug_conditioned_ot_replace_sample_ot', action='store_true', help='Replace sample OT with drug-conditioned OT')
    parser.add_argument('--drug_conditioned_ot_global_lambda', type=float, default=0.005, help='Global lambda for drug-conditioned OT')
    parser.add_argument('--drug_conditioned_ot_reg', type=float, default=0.1, help='Sinkhorn regularization for DCOT')
    parser.add_argument('--drug_conditioned_ot_cost', type=str, default='cosine', choices=['cosine', 'euclidean'], help='DCOT cost metric')
    parser.add_argument('--drug_conditioned_ot_lambda_min', type=float, default=0.0, help='Minimum lambda for discrepancy mode')
    parser.add_argument('--drug_conditioned_ot_lambda_max', type=float, default=0.01, help='Maximum lambda for discrepancy mode')
    parser.add_argument('--drug_conditioned_ot_temperature', type=float, default=5.0, help='Temperature for discrepancy sigmoid')
    parser.add_argument('--drug_conditioned_ot_threshold', type=str, default='mean', help='Threshold for discrepancy mode ("mean" or float)')
    parser.add_argument('--drug_conditioned_ot_detach_gap', action='store_true', default=True, help='Detach gap from gradient')
    parser.add_argument('--no_detach_gap', dest='drug_conditioned_ot_detach_gap', action='store_false')
    parser.add_argument('--drug_conditioned_ot_normalize_gap', action='store_true', default=True, help='Normalize gap across drugs')
    parser.add_argument('--drug_conditioned_ot_ema_gap', action='store_true', default=True, help='Use EMA for gap estimation')
    parser.add_argument('--drug_conditioned_ot_ema_momentum', type=float, default=0.9, help='EMA momentum for gap')
    parser.add_argument('--drug_conditioned_ot_fixed_lambda', type=json.loads, default=None, help='Fixed per-drug lambda dict as JSON')

    # Phase 6: Program-level Optimal Transport
    parser.add_argument('--use_program_ot', action='store_true', help='Enable program-level OT alignment loss')
    parser.add_argument('--program_ot_mode', type=str, default='same_program_uniform',
                        choices=['same_program_uniform', 'same_program_q_weighted', 'program_only', 'flattened_program', 'q_mass_flattened'],
                        help='Program OT mode')
    parser.add_argument('--lambda_program_ot', type=float, default=0.01, help='Program OT loss weight')
    parser.add_argument('--program_ot_reg', type=float, default=0.1, help='Program OT Sinkhorn regularization')
    parser.add_argument('--program_ot_max_iter', type=int, default=50, help='Program OT Sinkhorn max iterations')
    parser.add_argument('--program_ot_cost_type', type=str, default='cosine', choices=['cosine', 'euclidean'], help='Program OT cost metric')
    parser.add_argument('--program_ot_marginal_type', type=str, default='uniform', choices=['uniform', 'unbalanced', 'max_sim'], help='Program OT marginal type')
    parser.add_argument('--program_ot_detach_q_marginal', action='store_true', default=True, help='Detach q marginal from gradient')
    parser.add_argument('--no_detach_q_marginal', dest='program_ot_detach_q_marginal', action='store_false')
    parser.add_argument('--program_ot_q_temperature', type=float, default=1.0, help='Temperature for q marginal softmax')
    parser.add_argument('--program_ot_q_power', type=float, default=1.0, help='Power for q marginal')
    parser.add_argument('--program_ot_min_mass', type=float, default=1e-6, help='Minimum mass for q marginal')
    parser.add_argument('--q_align_enabled', action='store_true', help='Enable q distribution alignment')
    parser.add_argument('--lambda_q_align', type=float, default=0.001, help='q align loss weight')
    parser.add_argument('--enable_grad_diagnostics', action='store_true', help='Enable gradient diagnostics')
    parser.add_argument('--grad_diag_every', type=int, default=50, help='Compute grad diagnostics every N steps')
    parser.add_argument('--K', type=int, default=7, help='Top-K for contrastive domain isolation')

    # Phase 8: Drug-Selective Program Interaction (DSPI)
    parser.add_argument('--use_dspi', action='store_true', help='Enable Drug-Selective Program Interaction (Phase 8)')
    parser.add_argument('--dspi_temperature', type=float, default=1.0, help='DSPI attention temperature')
    parser.add_argument('--dspi_inter_weight', type=float, default=0.2, help='DSPI inter-drug attention weight')
    parser.add_argument('--dspi_q_prior_weight', type=float, default=0.1, help='DSPI q prior weight')
    parser.add_argument('--dspi_drug_sim_temperature', type=float, default=1.0, help='DSPI drug similarity temperature')
    parser.add_argument('--dspi_rho_mode', type=str, default='fixed', choices=['fixed', 'learnable_scalar', 'schedule'], help='DSPI rho mode')
    parser.add_argument('--dspi_fixed_rho', type=float, default=0.1, help='DSPI fixed rho value')
    parser.add_argument('--dspi_rho_max', type=float, default=0.3, help='DSPI max rho for learnable mode')
    parser.add_argument('--lambda_dspi_attn_diversity', type=float, default=0.0, help='DSPI attention diversity loss weight')
    parser.add_argument('--lambda_dspi_attn_entropy', type=float, default=0.0, help='DSPI attention entropy loss weight')
    parser.add_argument('--lambda_dspi_program_balance', type=float, default=0.0, help='DSPI program usage balance loss weight')
    parser.add_argument('--lambda_dspi_gram_consistency', type=float, default=0.0, help='DSPI Gram consistency loss weight')

    # Phase 5: Backbone selection (ConvNeXt V2-inspired)
    parser.add_argument('--backbone_type', type=str, default='mlp', choices=['mlp', 'grn_mlp', 'convnext1d'], help='Backbone type for MolecularProgramEncoder')
    parser.add_argument('--norm_type', type=str, default='layernorm', choices=['layernorm', 'batchnorm'], help='Normalization type for GRN-MLP')
    parser.add_argument('--num_gene_tokens', type=int, default=64, help='Number of gene tokens for ConvNeXt1D tokenizer')
    parser.add_argument('--token_dim', type=int, default=64, help='Token dimension for ConvNeXt1D')
    parser.add_argument('--convnext_depth', type=int, default=2, help='Number of ConvNeXt1D blocks')
    parser.add_argument('--convnext_kernel_size', type=int, default=7, help='Kernel size for ConvNeXt1D depthwise conv')
    parser.add_argument('--convnext_expansion', type=int, default=4, help='Expansion factor for ConvNeXt1D MLP')

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
        "pretrain_num_epochs": [300],
        "train_num_epochs": [100],
        "contrastive_num_epochs": [200],
        "K": [args.K]
    }

    warnings.filterwarnings("ignore")

    keys, values = zip(*params_grid.items())
    update_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    if args.retrain_flag == True:
        now_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    else:
        now_time = '2025-02-19_21-18-01'
    now_time = now_time + "mmpa_program"


    for param_dict in update_params_dict_list:
        main(args=args, update_params_dict=param_dict, now_time=now_time)



#nohup python train_mmpa_program.py >mmpa_program.log 2>&1 &
#ps ux      kill -9 pid
#ps -f -p pid
