import os
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
    # Phase 11-C: Cross-Attention Fusion params
    update_params_dict['use_cross_attn'] = args.use_cross_attn
    update_params_dict['cross_attn_num_heads'] = args.cross_attn_num_heads

    training_params['unlabeled'].update(update_params_dict)
    training_params['labeled'].update(update_params_dict)
    # Build param_str with abbreviated keys to avoid overly long folder names (ENAMETOOLONG)
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
        if k in ('use_cross_attn', 'cross_attn_num_heads'):
            continue  # appended manually below with shorter suffix
        param_dict_for_str[_key_abbr.get(k, k)] = v
    param_str = dict_to_str(param_dict_for_str)

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

    # Append BPAF / LLRF params to param_str
    if args.use_bpaf:
        param_str += f"_bpaf_gmax{args.bpaf_gamma_max}_ginit{args.bpaf_gamma_init}"
        if args.lambda_fusion_gamma_l1 > 0:
            param_str += f"_gl1{args.lambda_fusion_gamma_l1}"
    elif args.use_llrf:
        param_str += f"_llrf_gmax{args.llrf_gamma_max}_ginit{args.llrf_gamma_init}"
        if args.lambda_fusion_gamma_l1 > 0:
            param_str += f"_gl1{args.lambda_fusion_gamma_l1}"

    # Append Cross-Attention Fusion params to param_str
    if args.use_cross_attn:
        param_str += f"_crossAttn_h{args.cross_attn_num_heads}"

    # Append FPI params to param_str
    if args.use_fpi:
        param_str += f"_fpi_gmax{args.fpi_gamma_max}_ginit{args.fpi_gamma_init}_qprior{args.fpi_q_prior_weight}"
        if args.lambda_fpi_gamma_l1 > 0:
            param_str += f"_fgl1{args.lambda_fpi_gamma_l1}"
        if args.lambda_fpi_entropy > 0:
            param_str += f"_fent{args.lambda_fpi_entropy}"
        param_str += f"_detach{args.fpi_detach_mode}"
        if args.use_fpi_sim:
            param_str += "_sim"
            param_str += f"_ua{args.fpi_sim_union_alpha}_ub{args.fpi_sim_union_beta}"
            param_str += f"_pt{args.fpi_sim_program_temperature}_ft{args.fpi_sim_fragment_temperature}_ct{args.fpi_sim_cross_temperature}"
        else:
            param_str += "_global"
        param_str += f"_m{args.fpi_m_max}_fp{args.fpi_fp_dim}"

    # Append FPI Router params to param_str
    if args.use_fpi_router:
        param_str += f"_router_ggmax{args.router_gamma_global_max}_gginit{args.router_gamma_global_init}"
        param_str += f"_gcmax{args.router_gamma_cross_max}_gcinit{args.router_gamma_cross_init}"
        if args.lambda_router_gamma_l1 > 0:
            param_str += f"_rgl1{args.lambda_router_gamma_l1}"
        if args.lambda_router_gamma_global_l1 is not None and args.lambda_router_gamma_global_l1 > 0:
            param_str += f"_rggl1{args.lambda_router_gamma_global_l1}"
        if args.lambda_router_gamma_cross_l1 is not None and args.lambda_router_gamma_cross_l1 > 0:
            param_str += f"_rgcl1{args.lambda_router_gamma_cross_l1}"
        if args.router_cross_start_epoch > 0 or args.router_cross_ramp_epochs > 0:
            param_str += f"_ramp{args.router_cross_start_epoch}+{args.router_cross_ramp_epochs}"
        if args.router_cross_detach_mode != "none":
            param_str += f"_crossdetach{args.router_cross_detach_mode}"
        if args.router_two_stage_cross_only:
            param_str += "_twostage"
        param_str += f"_rgs{args.router_residual_global_scale}_rcs{args.router_residual_cross_scale}"
        param_str += f"_m{args.fpi_m_max}_fp{args.fpi_fp_dim}"

    # Append PDG params to param_str
    if args.use_pdg:
        param_str += f"_pdgK{args.pdg_k}_pdgdim{args.pdg_dim}_pdgconcat"
        if args.pdg_no_normalize:
            param_str += "_pdgnonorm"

    # Append DCD-R params to param_str
    if args.use_dcd_residual:
        param_str += f"_dcdR_dgmax{args.dcd_gamma_max}_dginit{args.dcd_gamma_init}"
        if args.lambda_dcd_gamma_l1 > 0:
            param_str += f"_dgl1{args.lambda_dcd_gamma_l1}"
        if args.lambda_dcd_ortho > 0:
            param_str += f"_dortho{args.lambda_dcd_ortho}"
        if args.dcd_use_absdiff:
            param_str += "_dabs"

    # Append Phase 11 main fusion params to param_str
    if args.main_fusion_type != "bilinear":
        param_str += f"_mf{args.main_fusion_type}_r{args.main_fusion_rank_dim}_o{args.main_fusion_out_dim}"
        if args.main_fusion_type == "dcf_lrbf":
            param_str += f"_film{args.film_init_scale}"

    # Append Phase 12 VCB params to param_str
    if args.use_vcb:
        param_str += "_vcb"
        param_str += f"_vkl{args.lambda_vcb_kl}"
        param_str += f"_vwarm{args.vcb_warmup_ratio}"
        z_dim_str = args.vcb_z_dim if args.vcb_z_dim > 0 else "same"
        param_str += f"_vz{z_dim_str}"
        if args.vcb_no_normalize:
            param_str += "_vnonorm"

    # Append Phase 13 S3R params to param_str
    if args.use_s3r:
        param_str += "_s3r"
        sdneg = 0 if args.s3r_all_drug_negatives else 1
        fns = 0 if args.s3r_no_suppress_false_negatives else 1
        wpos = 1 if args.s3r_weighted_positives else 0
        param_str += f"_sdneg{sdneg}_fns{fns}_wpos{wpos}"
        if args.s3r_max_negatives_per_anchor > 0:
            param_str += f"_nmax{args.s3r_max_negatives_per_anchor}"

    # Append Phase 13-G: Gram volume alignment params to param_str
    if args.use_gram_alignment:
        param_str += f"_gram{args.lambda_gram}"

    # Append Phase 17: Counterfactual causal augmentation params to param_str
    if args.lambda_causal > 0.0:
        param_str += f"_cf{args.lambda_causal}_abr{args.causal_ablation_ratio}"

    # Append Phase 18: InfoNCE / hard negative / pos weighting params to param_str
    if args.use_infonce:
        param_str += "_infonce"
        if args.hard_negatives_only:
            param_str += f"_hn{args.num_hard_negs}"
        if args.use_pos_weights:
            param_str += "_pw"

    # Append Phase 19: Differentiated contrastive learning params to param_str
    if args.use_diff_contrastive:
        param_str += f"_diff{args.diff_contrastive_alpha}"

    # Append Phase 14-A Rank params to param_str
    if args.use_rank_loss:
        param_str += f"_rank_lrank{args.lambda_rank}_rmargin{args.rank_margin}"
        if args.rank_max_pairs_per_drug > 0:
            param_str += f"_rmaxp{args.rank_max_pairs_per_drug}"

    # Append DCDA params to param_str
    if args.use_dcda:
        param_str += f"_dcda_t{args.dcda_timesteps}_id{args.dcda_interaction_dim}"
        param_str += f"_ldiff{args.lambda_dcda_diff}_lalign{args.lambda_dcda_align}"

    # Append Phase 18 domain separation params to param_str
    if args.use_domain_sep:
        param_str += f"_domsep_h{args.domain_sep_hidden_dim}_o{args.domain_sep_output_dim}"
        param_str += f"_ortho{args.lambda_ortho}_recon{args.lambda_recon}_privdom{args.lambda_private_domain}"
        param_str += f"_migrate{args.lambda_migrate}"

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
            # Phase 8-F: BPAF / LLRF
            'use_bpaf': args.use_bpaf,
            'bpaf_config': {
                'program_dim': args.program_dim,
                'drug_dim': 64,
                'num_programs': args.num_programs,
                'num_drugs': len(drugs),
                'attention_temperature': args.bpaf_attention_temperature,
                'q_prior_weight': args.bpaf_q_prior_weight,
                'gamma_max': args.bpaf_gamma_max,
                'gamma_init': args.bpaf_gamma_init,
                'dropout': args.bpaf_dropout,
                'adapter_include_h_pool': args.bpaf_adapter_include_h_pool,
            },
            'use_llrf': args.use_llrf,
            'llrf_config': {
                'program_dim': args.program_dim,
                'drug_dim': 64,
                'num_programs': args.num_programs,
                'num_drugs': len(drugs),
                'attention_temperature': args.bpaf_attention_temperature,
                'q_prior_weight': args.bpaf_q_prior_weight,
                'gamma_max': args.llrf_gamma_max,
                'gamma_init': args.llrf_gamma_init,
                'dropout': args.bpaf_dropout,
            },
            'lambda_fusion_gamma_l1': args.lambda_fusion_gamma_l1,
            'lambda_selector_entropy': args.lambda_selector_entropy,
            'lambda_selector_div': args.lambda_selector_div,
            # Phase 9: FPI
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
                'use_fpi_sim': args.use_fpi_sim,
                'fpi_sim_config': {
                    'q_prior_weight': args.fpi_sim_q_prior_weight,
                    'program_temperature': args.fpi_sim_program_temperature,
                    'fragment_temperature': args.fpi_sim_fragment_temperature,
                    'cross_temperature': args.fpi_sim_cross_temperature,
                    'union_alpha': args.fpi_sim_union_alpha,
                    'union_beta': args.fpi_sim_union_beta,
                    'use_modal_interaction': args.fpi_sim_use_modal_interaction,
                    'selection_type': args.fpi_sim_selection_type,
                },
            },
            'lambda_fpi_gamma_l1': args.lambda_fpi_gamma_l1,
            'lambda_fpi_entropy': args.lambda_fpi_entropy,
            # Phase 9-R: FPI Router
            'lambda_router_gamma_global_l1': args.lambda_router_gamma_global_l1,
            'lambda_router_gamma_cross_l1': args.lambda_router_gamma_cross_l1,
            'use_fpi_router': args.use_fpi_router,
            'fpi_router_config': {
                'program_dim': args.program_dim,
                'drug_dim': 64,
                'fragment_dim': args.program_dim,
                'num_programs': args.num_programs,
                'num_drugs': len(drugs),
                'max_fragments': args.fpi_m_max,
                'q_prior_weight': args.fpi_q_prior_weight,
                'gamma_global_max': args.router_gamma_global_max,
                'gamma_global_init': args.router_gamma_global_init,
                'gamma_cross_max': args.router_gamma_cross_max,
                'gamma_cross_init': args.router_gamma_cross_init,
                'dropout': args.fpi_dropout,
                'router_cross_start_epoch': args.router_cross_start_epoch,
                'router_cross_ramp_epochs': args.router_cross_ramp_epochs,
                'router_cross_mode': args.router_cross_mode,
                'router_oracle_cross_drugs': args.router_oracle_cross_drugs,
                'router_residual_global_scale': args.router_residual_global_scale,
                'router_residual_cross_scale': args.router_residual_cross_scale,
                'router_branch_effect_threshold': args.router_branch_effect_threshold,
                'router_cross_detach_mode': args.router_cross_detach_mode,
                'debug_fixed_gamma_global': args.router_debug_fixed_gamma_global,
                'debug_fixed_gamma_cross': args.router_debug_fixed_gamma_cross,
            },
            'router_two_stage_cross_only': args.router_two_stage_cross_only,
            'router_freeze_main_for_cross': args.router_freeze_main_for_cross,
            'router_stage_a_checkpoint_dir': args.router_stage_a_checkpoint_dir,
            'lambda_router_gamma_l1': args.lambda_router_gamma_l1,
            # Phase 10: Pseudo-Domain Guide
            'use_pdg': args.use_pdg,
            'pdg_config': {
                'pdg_dim': args.pdg_dim,
                'assign_mode': args.pdg_assign_mode,
                'normalize': not args.pdg_no_normalize,
                'pdg_k': args.pdg_k,
            },
            'pdg_centroid_path': args.pdg_centroid_path,
            # Phase 10-D: DCD-R
            'use_dcd_residual': args.use_dcd_residual,
            'dcd_config': {
                'hidden_dim': args.program_dim,
                'drug_dim': 64,
                'num_drugs': len(drugs),
                'dcd_hidden_dim': args.dcd_hidden_dim,
                'gamma_max': args.dcd_gamma_max,
                'gamma_init': args.dcd_gamma_init,
                'dropout': args.dcd_dropout,
                'use_product': True,
                'use_absdiff': args.dcd_use_absdiff,
            },
            'lambda_dcd_gamma_l1': args.lambda_dcd_gamma_l1,
            'lambda_dcd_ortho': args.lambda_dcd_ortho,
            # Phase 11: Main Fusion Replacement
            'main_fusion_type': args.main_fusion_type,
            'main_fusion_config': {
                'rank_dim': args.main_fusion_rank_dim,
                'out_dim': args.main_fusion_out_dim,
                'dropout': args.dcd_dropout if hasattr(args, 'dcd_dropout') else 0.1,
                'use_layernorm': not args.main_fusion_no_layernorm,
                'film_init_scale': args.film_init_scale,
            },
            # Phase 12: VCB
            'use_vcb': args.use_vcb,
            'vcb_config': {
                'z_dim': None if args.vcb_z_dim <= 0 else args.vcb_z_dim,
                'hidden_dim': None if args.vcb_hidden_dim <= 0 else args.vcb_hidden_dim,
                'dropout': args.vcb_dropout,
                'logvar_min': args.vcb_logvar_min,
                'logvar_max': args.vcb_logvar_max,
                'normalize_z': not args.vcb_no_normalize,
            },
            'lambda_vcb_kl': args.lambda_vcb_kl,
            'vcb_warmup_ratio': args.vcb_warmup_ratio,
            # Phase 13: S3R
            'use_s3r': args.use_s3r,
            's3r_same_drug_negatives': not args.s3r_all_drug_negatives,
            's3r_suppress_false_negatives': not args.s3r_no_suppress_false_negatives,
            's3r_weighted_positives': args.s3r_weighted_positives,
            's3r_positive_temperature': args.s3r_positive_temperature,
            's3r_max_negatives_per_anchor': args.s3r_max_negatives_per_anchor,
            's3r_loss_temperature': args.s3r_loss_temperature,
            # Phase 13-G: Gram volume alignment
            'use_gram_alignment': args.use_gram_alignment,
            'lambda_gram': args.lambda_gram,
            # Phase 17: Counterfactual causal augmentation
            'lambda_causal': args.lambda_causal,
            'causal_ablation_ratio': args.causal_ablation_ratio,
            # Phase 18: InfoNCE / hard negative / pos weighting
            'use_infonce': args.use_infonce,
            'hard_negatives_only': args.hard_negatives_only,
            'num_hard_negs': args.num_hard_negs,
            'use_pos_weights': args.use_pos_weights,
            # Phase 19: Differentiated contrastive learning (DKC-inspired)
            'use_diff_contrastive': args.use_diff_contrastive,
            'diff_contrastive_alpha': args.diff_contrastive_alpha,
            # Phase 14-A / 15-B: Drug-wise Pairwise Ranking Loss
            'use_rank_loss': args.use_rank_loss,
            'lambda_rank': args.lambda_rank if args.use_rank_loss else 0.0,
            'rank_margin': args.rank_margin,
            'rank_max_pairs_per_drug': args.rank_max_pairs_per_drug,
            'rank_debug_sanity': args.rank_debug_sanity,
            # Phase 15-B: Effective Drug-wise Ranking Loss Sweep
            'ranking_warmup_epochs': args.ranking_warmup_epochs,
            'ranking_min_pos': args.ranking_min_pos,
            'ranking_min_neg': args.ranking_min_neg,
            'ranking_log_grad_ratio': args.ranking_log_grad_ratio,
            'ranking_log_drugwise_pair_acc': args.ranking_log_drugwise_pair_acc,
            'ranking_log_interval': args.ranking_log_interval,
            # Phase 16: FAMO
            'use_famo': args.use_famo,
            'famo_gamma': args.famo_gamma,
            'famo_w_lr': args.famo_w_lr,
            'famo_max_norm': args.famo_max_norm,
            # Phase DCDA
            'use_dcda': args.use_dcda,
            'dcda_config': {
                'timesteps': args.dcda_timesteps,
                'interaction_dim': args.dcda_interaction_dim,
            },
            'lambda_dcda_diff': args.lambda_dcda_diff,
            'lambda_dcda_align': args.lambda_dcda_align,
            # Phase 18: Domain Separation
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

    # Data construction for the pre-training
    s_dataloaders, t_dataloaders = get_unlabeled_dataloaders(gex_features_df=gex_features_df, seed=seed, test_ratio=0.25,
                                                                    batch_size=training_params['unlabeled']['batch_size'])
    # Load drug features: unified drug encoder interface
    drug_encoder_type = args.drug_encoder_type
    unimol_emb_path = args.unimol_emb_path
    unimol_dim = args.unimol_dim
    drug_emb_dim = args.drug_emb_dim
    drug_fusion_dropout = args.drug_fusion_dropout

    # Always load RDKit fingerprints as base drug_feat (used by Drug_MLP in mlp/fusion modes)
    drug_feat = torch.from_numpy(get_drug_feat(drugs).astype('float32')).to(device)

    # Pass unified drug encoder params to training pipeline
    training_params['drug_encoder_type'] = drug_encoder_type
    training_params['unimol_emb_path'] = unimol_emb_path
    training_params['unimol_dim'] = unimol_dim
    training_params['drug_emb_dim'] = drug_emb_dim
    training_params['drug_fusion_dropout'] = drug_fusion_dropout

    # Validate Uni-Mol path when needed
    if drug_encoder_type in ('unimol', 'fp_unimol_fusion'):
        if unimol_emb_path is None:
            raise ValueError(
                f"--drug_encoder_type={drug_encoder_type} requires --unimol_emb_path"
            )
        if not os.path.exists(unimol_emb_path):
            raise ValueError(
                f"Uni-Mol embedding file not found: {unimol_emb_path}\n"
                f"Please run: python extract_unimol_embeddings.py to generate it."
            )
        print_logs(log_dir, f"Drug encoder: {drug_encoder_type}, unimol_path={unimol_emb_path}, dim={unimol_dim}")
    else:
        print_logs(log_dir, f"Drug encoder: {drug_encoder_type} (original Drug_MLP)")

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
        use_domain_sep=args.use_domain_sep,
        domain_sep_hidden_dim=args.domain_sep_hidden_dim,
        domain_sep_output_dim=args.domain_sep_output_dim,
        # Hallmark gene-set encoder
        program_encoder_type=args.program_encoder_type,
        gene_set_mask_path=args.gene_set_mask_path,
        gene_set_names_path=args.gene_set_names_path,
        gene_set_token_dropout=args.gene_set_token_dropout,
        use_pathway_embedding=args.use_pathway_embedding,
        use_soft_gene_set_mask=args.use_soft_gene_set_mask,
        soft_mask_scale=args.soft_mask_scale,
        latent_num_programs=args.latent_num_programs,
        gene_set_num_final_programs=args.gene_set_num_final_programs,
        selector_num_heads=args.selector_num_heads,
        selector_dropout=args.selector_dropout,
        use_query_gate=args.use_query_gate,
        lambda_query_gate_l1=args.lambda_query_gate_l1,
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
        lambda_ortho=args.lambda_ortho,
        lambda_recon=args.lambda_recon,
        lambda_private_domain=args.lambda_private_domain,
    )

    save_feat_program(model=trained_model, s_dataloaders=s_dataloaders, t_dataloaders=t_dataloaders, device=device)

    # Save stage1 model for downstream use (e.g., pseudo-domain building)
    stage1_ckpt_dir = os.path.join("checkpoints", "stage1")
    safe_make_dir(stage1_ckpt_dir)
    stage1_ckpt_path = os.path.join(stage1_ckpt_dir, "stage1_model.pt")
    torch.save(trained_model.state_dict(), stage1_ckpt_path)
    print(f"Saved Stage1 model to {stage1_ckpt_path}")

    print("======================= stage 2 & 3 (MMPA Program) =======================")
    # Data construction for feature fusion and drug response prediction
    labeled_dataloader = get_multi_labeled_dataloader(gex_features_df=gex_features_df, seed=seed, batch_size=training_params['labeled']['batch_size'],
                                                      drug=drugs, ccle_measurement=args.measurement, threshold_gdsc=args.thres_g, threshold_label=args.thres_s,
                                                      n_splits=args.n)

    # Load drug fragments if FPI is enabled
    frag_fp = None
    frag_mask = None
    frag_smiles = None
    if args.use_fpi or args.use_fpi_router:
        from mmpa_drp.fragment_utils import load_drug_fragments
        frag_data = load_drug_fragments(args.fpi_fragment_npz, expected_drug_names=drugs)
        frag_fp = frag_data["frag_fp"].to(device)
        frag_mask = frag_data["frag_mask"].to(device)
        frag_smiles = frag_data["frag_smiles"]

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
    parser.add_argument('--output_dir', type=str, default=None, help='Override output directory for diagnostics and logs')
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
    parser.add_argument('--lambda_ot', type=float, default=0.005, help='OT loss weight')
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
    parser.add_argument('--K', type=int, default=3, help='Top-K for contrastive domain isolation')
    parser.add_argument('--pretrain_num_epochs', type=int, default=300, help='Pretraining epochs')
    parser.add_argument('--train_num_epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--contrastive_num_epochs', type=int, default=200, help='Contrastive epochs')

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

    # Phase 8-F: BasePreservingAdapterFusion (BPAF)
    parser.add_argument('--use_bpaf', action='store_true', help='Enable BasePreservingAdapterFusion (Phase 8-F)')
    parser.add_argument('--bpaf_gamma_max', type=float, default=0.05, help='BPAF gamma max')
    parser.add_argument('--bpaf_gamma_init', type=float, default=0.02, help='BPAF gamma init')
    parser.add_argument('--bpaf_q_prior_weight', type=float, default=0.01, help='BPAF selector q prior weight')
    parser.add_argument('--bpaf_attention_temperature', type=float, default=1.0, help='BPAF selector attention temperature')
    parser.add_argument('--bpaf_dropout', type=float, default=0.1, help='BPAF adapter dropout')
    parser.add_argument('--bpaf_adapter_include_h_pool', action='store_true', default=True, help='Include h_pool in adapter input')
    parser.add_argument('--no_bpaf_h_pool', dest='bpaf_adapter_include_h_pool', action='store_false')

    # Phase 8-F: LateLogitResidualFusion (LLRF)
    parser.add_argument('--use_llrf', action='store_true', help='Enable LateLogitResidualFusion (Phase 8-F)')
    parser.add_argument('--llrf_gamma_max', type=float, default=0.1, help='LLRF gamma max')
    parser.add_argument('--llrf_gamma_init', type=float, default=0.02, help='LLRF gamma init')

    # Phase 8-F: Fusion-Repair regularization
    parser.add_argument('--lambda_fusion_gamma_l1', type=float, default=0.001, help='L1 regularization on fusion gamma')
    parser.add_argument('--lambda_selector_entropy', type=float, default=0.0, help='Selector attention entropy loss weight')
    parser.add_argument('--lambda_selector_div', type=float, default=0.0, help='Selector attention diversity loss weight')

    # Phase 9: Fragment-Program Interaction (FPI)
    parser.add_argument('--use_fpi', action='store_true', help='Enable Fragment-Program Interaction (Phase 9)')
    parser.add_argument('--fpi_fragment_npz', type=str, default='data/drug_fragments_brics_m16_fp1024.npz', help='Path to precomputed fragment NPZ')
    parser.add_argument('--fpi_gamma_max', type=float, default=0.1, help='FPI gamma max')
    parser.add_argument('--fpi_gamma_init', type=float, default=0.05, help='FPI gamma init')
    parser.add_argument('--fpi_q_prior_weight', type=float, default=0.01, help='FPI q prior weight')
    parser.add_argument('--fpi_dropout', type=float, default=0.1, help='FPI dropout')
    parser.add_argument('--fpi_m_max', type=int, default=16, help='Max fragments per drug')
    parser.add_argument('--fpi_fp_dim', type=int, default=1024, help='Fragment fingerprint dimension')
    parser.add_argument('--lambda_fpi_gamma_l1', type=float, default=0.0, help='L1 regularization on FPI gamma')
    parser.add_argument('--lambda_fpi_entropy', type=float, default=0.0, help='FPI attention entropy loss weight')
    parser.add_argument('--fpi_detach_mode', type=str, default='none', choices=['none', 'z', 'zq', 'zq_drug', 'all'], help='Stop-gradient mode for FPI branch')
    parser.add_argument('--use_fpi_sim', action='store_true', help='Enable FPI-SIM (Signal-style selective interaction)')
    parser.add_argument('--fpi_sim_q_prior_weight', type=float, default=0.01, help='FPI-SIM q prior weight for program intra selection')
    parser.add_argument('--fpi_sim_program_temperature', type=float, default=1.0, help='FPI-SIM program intra selection temperature')
    parser.add_argument('--fpi_sim_fragment_temperature', type=float, default=1.0, help='FPI-SIM fragment intra selection temperature')
    parser.add_argument('--fpi_sim_cross_temperature', type=float, default=1.0, help='FPI-SIM cross-modal attention temperature')
    parser.add_argument('--fpi_sim_union_alpha', type=float, default=1.0, help='FPI-SIM union alpha weight')
    parser.add_argument('--fpi_sim_union_beta', type=float, default=1.0, help='FPI-SIM union beta weight')
    parser.add_argument('--fpi_sim_use_modal_interaction', action='store_true', help='Enable modal interaction in FPI-SIM (first version: disabled by default)')
    parser.add_argument('--fpi_sim_selection_type', type=str, default='soft', choices=['soft', 'hard_topk'], help='FPI-SIM selection type (first version: soft only)')

    # Phase 9-R: FPI Router (DualFPIResidualRouter)
    parser.add_argument('--use_fpi_router', action='store_true', help='Enable DualFPIResidualRouter (Phase 9-R)')
    parser.add_argument('--router_gamma_global_max', type=float, default=0.05, help='Router global gamma max')
    parser.add_argument('--router_gamma_global_init', type=float, default=0.0, help='Router global gamma init')
    parser.add_argument('--router_gamma_cross_max', type=float, default=0.05, help='Router cross gamma max')
    parser.add_argument('--router_gamma_cross_init', type=float, default=0.0, help='Router cross gamma init')
    parser.add_argument('--router_residual_global_scale', type=float, default=1.0, help='Router global residual scale')
    parser.add_argument('--router_residual_cross_scale', type=float, default=1.0, help='Router cross residual scale')
    parser.add_argument('--router_cross_start_epoch', type=int, default=0, help='Epoch to start cross branch activation')
    parser.add_argument('--router_cross_ramp_epochs', type=int, default=0, help='Number of epochs to ramp cross branch')
    parser.add_argument('--router_cross_mode', type=str, default='learnable', choices=['learnable', 'oracle_mask'], help='Cross branch mode')
    parser.add_argument('--router_oracle_cross_drugs', type=str, default='', help='Comma-separated drug names for oracle cross mask (diagnostic only)')
    parser.add_argument('--router_branch_effect_threshold', type=float, default=1e-4, help='Threshold for branch effect diagnostics')
    parser.add_argument('--router_cross_detach_mode', type=str, default='none', choices=['none', 'z', 'zq', 'zq_drug', 'all'], help='Detach mode for cross branch inputs (none/z/zq/zq_drug/all)')
    parser.add_argument('--router_two_stage_cross_only', action='store_true', help='Stage B: train only cross branch after loading Stage A checkpoint')
    parser.add_argument('--router_freeze_main_for_cross', action='store_true', help='Freeze main model (encoder/drug_mlp/bilinear/global_fpi/gamma_global) in Stage B')
    parser.add_argument('--router_stage_a_checkpoint_dir', type=str, default='', help='Directory containing Stage A checkpoint for two-stage training')
    parser.add_argument('--lambda_router_gamma_l1', type=float, default=0.0, help='L1 regularization on router gamma gates (deprecated, use separate global/cross L1)')
    parser.add_argument('--lambda_router_gamma_global_l1', type=float, default=None, help='L1 regularization on router global gamma gate')
    parser.add_argument('--lambda_router_gamma_cross_l1', type=float, default=None, help='L1 regularization on router cross gamma gate')
    parser.add_argument('--router_debug_fixed_gamma_global', type=float, default=None, help='DEBUG: fix global gamma to this value (overrides learnable gamma)')
    parser.add_argument('--router_debug_fixed_gamma_cross', type=float, default=None, help='DEBUG: fix cross gamma to this value (overrides learnable gamma_cross)')

    # Phase 11-C: Cross-Attention Fusion (Signal-style)
    parser.add_argument('--use_cross_attn', action='store_true', help='Enable Cross-Attention Fusion to replace Bilinear')
    parser.add_argument('--cross_attn_num_heads', type=int, default=4, help='Number of attention heads for Cross-Attention Fusion')

    # Phase 10: Pseudo-Domain Guide (PDG)
    parser.add_argument('--use_pdg', action='store_true', help='Enable Pseudo-Domain Guided Fusion (Phase 10)')
    parser.add_argument('--pdg_centroid_path', type=str, default=None, help='Path template to PDG centroids NPY (use {fold} for fold index)')
    parser.add_argument('--pdg_dim', type=int, default=32, help='PDG embedding dimension')
    parser.add_argument('--pdg_k', type=int, default=8, help='Number of pseudo-domain clusters')
    parser.add_argument('--pdg_assign_mode', type=str, default='nearest', choices=['nearest'], help='PDG assignment mode')
    parser.add_argument('--pdg_normalize', action='store_true', default=True, help='L2-normalize before cosine similarity')
    parser.add_argument('--pdg_no_normalize', action='store_true', help='Disable L2 normalization in PDG')

    # Phase 10-D: Residual Decoupled Fusion (DCD-R)
    parser.add_argument('--use_dcd_residual', action='store_true', help='Enable Residual Decoupled Fusion (Phase 10-D)')
    parser.add_argument('--dcd_gamma_max', type=float, default=0.03, help='DCD-R gamma max (default 0.03)')
    parser.add_argument('--dcd_gamma_init', type=float, default=0.005, help='DCD-R gamma init (default 0.005)')
    parser.add_argument('--dcd_hidden_dim', type=int, default=64, help='DCD-R residual head hidden dim')
    parser.add_argument('--dcd_use_absdiff', action='store_true', help='Use absdiff feature in DCD-R residual head')
    parser.add_argument('--dcd_dropout', type=float, default=0.1, help='DCD-R MLP dropout')
    parser.add_argument('--lambda_dcd_gamma_l1', type=float, default=0.0, help='L1 regularization on DCD gamma')
    parser.add_argument('--lambda_dcd_ortho', type=float, default=0.0, help='Orthogonality loss weight between shared and private')

    # Phase 11: Main Fusion Replacement
    parser.add_argument('--main_fusion_type', type=str, default='bilinear', choices=['bilinear', 'lrbf', 'dcf_lrbf'], help='Main gene-drug fusion type (Phase 11)')
    parser.add_argument('--main_fusion_rank_dim', type=int, default=64, help='Low-rank dimension for LRBF/DCF-LRBF')
    parser.add_argument('--main_fusion_out_dim', type=int, default=128, help='Output dimension for main fusion replacement')
    parser.add_argument('--film_init_scale', type=float, default=0.001, help='FiLM output layer init std (near-identity)')
    parser.add_argument('--main_fusion_layernorm', action='store_true', default=True, help='Use LayerNorm in main fusion MLPs')
    parser.add_argument('--main_fusion_no_layernorm', action='store_true', help='Disable LayerNorm in main fusion MLPs')

    # Phase 12: Variational Contrastive Bottleneck (VCB)
    parser.add_argument('--use_vcb', action='store_true', help='Enable Variational Contrastive Bottleneck (Phase 12)')
    parser.add_argument('--vcb_z_dim', type=int, default=0, help='VCB latent dim (0 = same as fused_feature_dim)')
    parser.add_argument('--vcb_hidden_dim', type=int, default=0, help='VCB backbone hidden dim (0 = same as fused_feature_dim)')
    parser.add_argument('--vcb_dropout', type=float, default=0.1, help='VCB dropout')
    parser.add_argument('--vcb_logvar_min', type=float, default=-6.0, help='VCB logvar clamp min')
    parser.add_argument('--vcb_logvar_max', type=float, default=2.0, help='VCB logvar clamp max')
    parser.add_argument('--vcb_no_normalize', action='store_true', help='Disable L2 normalization of VCB z')
    parser.add_argument('--lambda_vcb_kl', type=float, default=0.0, help='VCB KL loss weight')
    parser.add_argument('--vcb_warmup_ratio', type=float, default=0.1, help='VCB beta warm-up ratio')

    # Phase 13: Reliable Stage-3 Contrastive Anchoring (S3R)
    parser.add_argument('--use_s3r', action='store_true', help='Enable S3R reliable contrastive anchoring (Phase 13)')
    parser.add_argument('--s3r_same_drug_negatives', action='store_true', default=True, help='Use same-drug negatives only')
    parser.add_argument('--s3r_all_drug_negatives', action='store_true', help='Use all-drug negatives (override same-drug)')
    parser.add_argument('--s3r_suppress_false_negatives', action='store_true', default=True, help='Remove positives from negative pool')
    parser.add_argument('--s3r_no_suppress_false_negatives', action='store_true', help='Disable false-negative suppression')
    parser.add_argument('--s3r_weighted_positives', action='store_true', help='Weight positives by similarity/rank')
    parser.add_argument('--s3r_positive_temperature', type=float, default=0.1, help='Temperature for positive weighting')
    parser.add_argument('--s3r_max_negatives_per_anchor', type=int, default=0, help='Max negatives per anchor (0 = unlimited)')
    parser.add_argument('--s3r_loss_temperature', type=float, default=0.1, help='Temperature for contrastive loss')

    # Phase 13-G: Gram Volume Alignment (replaces contrastive)
    parser.add_argument('--use_gram_alignment', action='store_true', help='Enable Gram volume alignment in Stage 3 (replaces contrastive)')
    parser.add_argument('--lambda_gram', type=float, default=0.01, help='Weight for Gram volume loss')

    # Phase 17: Counterfactual Causal Augmentation
    parser.add_argument('--lambda_causal', type=float, default=0.0, help='Weight for counterfactual causal augmentation loss (Stage 3)')
    parser.add_argument('--causal_ablation_ratio', type=float, default=0.1, help='Ratio of genes to ablate for counterfactual augmentation')

    # Phase 18: InfoNCE / Hard Negative / Pos Weighting
    parser.add_argument('--use_infonce', action='store_true', help='Use standard InfoNCE instead of legacy multi-label BCE for Stage 3 contrastive')
    parser.add_argument('--hard_negatives_only', action='store_true', help='In InfoNCE, only use top-N hardest negatives')
    parser.add_argument('--num_hard_negs', type=int, default=50, help='Number of hard negatives to keep')
    parser.add_argument('--use_pos_weights', action='store_true', help='Weight positive samples by KNN similarity in InfoNCE')

    # Phase 19: Differentiated Contrastive Learning (DKC-inspired)
    parser.add_argument('--use_diff_contrastive', action='store_true', help='Enable differentiated contrastive learning: low-confidence samples get higher weights')
    parser.add_argument('--diff_contrastive_alpha', type=float, default=2.0, help='Max weight for low-confidence samples (weight = alpha - (alpha-1)*confidence)')

    # Phase 14-A: Drug-wise Pairwise Ranking Loss
    parser.add_argument('--use_rank_loss', action='store_true', help='Enable drug-wise pairwise ranking loss (Phase 14-A)')
    parser.add_argument('--lambda_rank', type=float, default=0.0, help='Weight for ranking loss')
    parser.add_argument('--rank_margin', type=float, default=0.0, help='Ranking margin')
    parser.add_argument('--rank_max_pairs_per_drug', type=int, default=0, help='Max pairs per drug (0 = unlimited)')
    parser.add_argument('--rank_debug_sanity', action='store_true', help='Enable rank loss sanity debug (allows lambda_rank=0)')

    # Phase 15-B: Effective Drug-wise Ranking Loss Sweep
    parser.add_argument('--ranking_warmup_epochs', type=int, default=0, help='Linear warm-up epochs for lambda_rank (0 = no warm-up)')
    parser.add_argument('--ranking_min_pos', type=int, default=1, help='Min positive samples per drug for ranking loss')
    parser.add_argument('--ranking_min_neg', type=int, default=1, help='Min negative samples per drug for ranking loss')
    parser.add_argument('--ranking_log_grad_ratio', action='store_true', help='Log ranking vs BCE gradient norm ratio')
    parser.add_argument('--ranking_log_drugwise_pair_acc', action='store_true', help='Log per-drug pairwise ranking accuracy')
    parser.add_argument('--ranking_log_interval', type=int, default=50, help='Interval (steps) for gradient ratio logging')

    # Phase 16: FAMO (Fast Adaptive Multitask Optimization)
    parser.add_argument('--use_famo', action='store_true', help='Enable FAMO for drug-wise multitask optimization')
    parser.add_argument('--famo_gamma', type=float, default=0.01, help='FAMO weight regularization coefficient')
    parser.add_argument('--famo_w_lr', type=float, default=0.025, help='FAMO task weight learning rate')
    parser.add_argument('--famo_max_norm', type=float, default=1.0, help='FAMO gradient clipping max norm')

    # Phase DCDA: Drug-Conditioned Domain Diffusion Alignment
    parser.add_argument('--use_dcda', action='store_true', help='Enable DCDA (HorizonRec-inspired domain alignment)')
    parser.add_argument('--dcda_timesteps', type=int, default=10, help='DCDA diffusion timesteps')
    parser.add_argument('--dcda_interaction_dim', type=int, default=128, help='DCDA interaction projection dim')
    parser.add_argument('--lambda_dcda_diff', type=float, default=0.1, help='DCDA diffusion loss weight')
    parser.add_argument('--lambda_dcda_align', type=float, default=0.05, help='DCDA domain alignment loss weight')

    # Phase 18: Domain Separation (DCDNet-style + BML Router)
    parser.add_argument('--use_domain_sep', action='store_true', help='Enable domain-level shared/private decomposition')
    parser.add_argument('--domain_sep_hidden_dim', type=int, default=256, help='Hidden dim for domain separation MLPs')
    parser.add_argument('--domain_sep_output_dim', type=int, default=64, help='Output dim for shared/private representations')
    parser.add_argument('--lambda_ortho', type=float, default=0.001, help='Orthogonality loss weight between shared and private')
    parser.add_argument('--lambda_recon', type=float, default=0.001, help='Reconstruction loss weight (hidden vs shared+private)')
    parser.add_argument('--lambda_private_domain', type=float, default=0.001, help='Private domain classification loss weight')
    parser.add_argument('--lambda_migrate', type=float, default=0.01, help='Stage 2: migration loss weight (private learns from reliable shared)')

    # Phase 5: Backbone selection (ConvNeXt V2-inspired)
    parser.add_argument('--backbone_type', type=str, default='convnext1d', choices=['mlp', 'grn_mlp', 'convnext1d', 'resnet'], help='Backbone type for MolecularProgramEncoder')
    parser.add_argument('--norm_type', type=str, default='layernorm', choices=['layernorm', 'batchnorm'], help='Normalization type for GRN-MLP')
    parser.add_argument('--num_gene_tokens', type=int, default=64, help='Number of gene tokens for ConvNeXt1D tokenizer')
    parser.add_argument('--token_dim', type=int, default=64, help='Token dimension for ConvNeXt1D')
    parser.add_argument('--convnext_depth', type=int, default=1, help='Number of ConvNeXt1D blocks')
    parser.add_argument('--convnext_kernel_size', type=int, default=7, help='Kernel size for ConvNeXt1D depthwise conv')
    parser.add_argument('--convnext_expansion', type=int, default=4, help='Expansion factor for ConvNeXt1D MLP')

    # Unified Drug Encoder parameters (Uni-Mol / MLP / Fusion)
    parser.add_argument('--drug_encoder_type', type=str, default='mlp',
                        choices=['mlp', 'unimol', 'fp_unimol_fusion'],
                        help='Drug encoder type: mlp (original), unimol (Uni-Mol only), fp_unimol_fusion (MLP+Uni-Mol)')
    parser.add_argument('--unimol_emb_path', type=str, default=None,
                        help='Path to offline Uni-Mol embedding .npy file (required for unimol/fp_unimol_fusion)')
    parser.add_argument('--unimol_dim', type=int, default=512,
                        help='Dimension of Uni-Mol embedding (e.g., 512 or 768)')
    parser.add_argument('--drug_emb_dim', type=int, default=64,
                        help='Output drug embedding dimension (must match model dim, default 64)')
    parser.add_argument('--drug_fusion_dropout', type=float, default=0.1,
                        help='Dropout for Uni-Mol projection and fusion MLPs')

    # Hallmark Gene-Set Program Encoder parameters
    parser.add_argument('--program_encoder_type', type=str, default='latent',
                        choices=['latent', 'gene_set_mask', 'hybrid_gene_set_latent', 'gene_set_selector'],
                        help='Program encoder type: latent (original), gene_set_mask (Hallmark), hybrid_gene_set_latent (Hallmark+latent), gene_set_selector (Hallmark->K via attention)')
    parser.add_argument('--gene_set_mask_path', type=str, default=None,
                        help='Path to gene_set_mask.pt or .npy (required for gene_set_mask/hybrid/gene_set_selector)')
    parser.add_argument('--gene_set_names_path', type=str, default=None,
                        help='Path to gene_set_names.txt for interpretability')
    parser.add_argument('--gene_set_token_dropout', type=float, default=0.1,
                        help='Dropout for gene-set token MLP')
    parser.add_argument('--use_pathway_embedding', action='store_true', default=True,
                        help='Enable learnable pathway bias in gene-set encoder')
    parser.add_argument('--no_pathway_embedding', dest='use_pathway_embedding', action='store_false')
    parser.add_argument('--use_soft_gene_set_mask', action='store_true', default=False,
                        help='Enable soft mask fine-tuning for gene sets')
    parser.add_argument('--soft_mask_scale', type=float, default=0.05,
                        help='Scale for soft mask contribution')
    parser.add_argument('--latent_num_programs', type=int, default=3,
                        help='Number of latent tokens in hybrid mode')
    parser.add_argument('--gene_set_num_final_programs', type=int, default=3,
                        help='Number of final program tokens for gene_set_selector (K)')
    parser.add_argument('--selector_num_heads', type=int, default=4,
                        help='Number of attention heads for gene-set selector')
    parser.add_argument('--selector_dropout', type=float, default=0.1,
                        help='Dropout for gene-set selector attention')
    parser.add_argument('--use_query_gate', action='store_true', default=False,
                        help='Enable learnable query gate for dynamic effective K')
    parser.add_argument('--lambda_query_gate_l1', type=float, default=0.01,
                        help='L1 sparsity regularization for query gate')

    train_group = parser.add_mutually_exclusive_group(required=False)
    train_group.add_argument('--train', dest='retrain_flag', action='store_true')
    train_group.add_argument('--no-train', dest='retrain_flag', action='store_false')
    parser.set_defaults(retrain_flag=True)

    norm_group = parser.add_mutually_exclusive_group(required=False)
    norm_group.add_argument('--norm', dest='norm_flag', action='store_true')
    norm_group.add_argument('--no-norm', dest='norm_flag', action='store_false')
    parser.set_defaults(norm_flag=True)

    args = parser.parse_args()
    if args.use_bpaf and args.use_llrf:
        parser.error("--use_bpaf and --use_llrf cannot be used together")
    if args.use_fpi and (args.use_bpaf or args.use_llrf or args.use_dspi or args.use_drug_gate):
        parser.error("--use_fpi is incompatible with --use_bpaf, --use_llrf, --use_dspi, --use_drug_gate")
    if args.use_rank_loss and args.lambda_rank < 0:
        parser.error("--use_rank_loss requires --lambda_rank >= 0")
    if args.use_fpi_router and (args.use_fpi or args.use_bpaf or args.use_llrf or args.use_dspi or args.use_drug_gate):
        parser.error("--use_fpi_router is incompatible with --use_fpi, --use_bpaf, --use_llrf, --use_dspi, --use_drug_gate")
    if args.use_pdg and args.pdg_centroid_path is None:
        parser.error("--use_pdg requires --pdg_centroid_path")
    print(f'current config is {args}')
    # params_grid = {
    #     "pretrain_num_epochs": [10],
    #     "train_num_epochs": [50, 100],
    #     "contrastive_num_epochs": [50, 100, 200],
    #     "K": [1,3,5,7,9]
    # }
    params_grid = {
        "pretrain_num_epochs": [args.pretrain_num_epochs],
        "train_num_epochs": [args.train_num_epochs],
        "contrastive_num_epochs": [args.contrastive_num_epochs],
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
