import random
import numpy as np
import torch
import os
import math
from sklearn.metrics import auc, precision_recall_curve

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

def wrap_params(training_params, type='unlabeled'):
    aux_dict = {k: v for k, v in training_params.items() if k not in ['unlabeled', 'labeled']}
    aux_dict.update(**training_params[type])
    return aux_dict

def safe_make_dir(new_folder_name):
    if not os.path.exists(new_folder_name):
        os.makedirs(new_folder_name)

def dict_to_str(d):
    return "_".join(["_".join([k, str(v)]) for k, v in d.items()])

def write_to_logs(dir, str):
    with open(dir,'a') as f:
         f.write(str+'\n')

def print_logs(log_dir, str, brief_log_dir = None):
    print(str)
    write_to_logs(log_dir, str)
    if brief_log_dir is not None:
        write_to_logs(brief_log_dir, str)

def auprc(y_true, y_score):
    lr_precision, lr_recall, _ = precision_recall_curve(y_true=y_true, y_score=y_score)
    return auc(lr_recall, lr_precision)

def uniform(size, tensor):
    if tensor is not None:
        bound = 1.0 / math.sqrt(size)
        tensor.data.uniform_(-bound, bound)



from dataload import get_drug_feat
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import config
def print_tissue_img(train_labeled_ccle, test_labeled_ccle, labeled_tcga, drugs, network, with_fusion, fold, dir):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    drug_feat = torch.from_numpy(get_drug_feat(drugs).astype('float32')).to(device)
    network.eval()

    if with_fusion == True:
        with_fusion_name = "drug_response_" + str(fold)
    else:
        with_fusion_name = "gene_" + str(fold)

    s_all, t_all = [], []
    s_types, t_types = [], []
    with torch.no_grad():
        for i, batch in enumerate(train_labeled_ccle):
            s_ty = batch[3].to(device)
            s_x = batch[0].to(device)
            if with_fusion == True:
                outputs = network(s_x, drug_feat, s_ty)
                s_feat = outputs[2]
                s_ty = outputs[3]
            else:
                _, _, s_feat = network(s_x)
            s_all.append(s_feat)
            s_types.append(s_ty)
        for i, batch in enumerate(test_labeled_ccle):
            s_ty = batch[3].to(device)
            s_x = batch[0].to(device)
            if with_fusion == True:
                outputs = network(s_x, drug_feat, s_ty)
                s_feat = outputs[2]
                s_ty = outputs[3]
            else:
                _, _, s_feat = network(s_x)
            s_all.append(s_feat)
            s_types.append(s_ty)
        for i, batch in enumerate(labeled_tcga):
            t_ty = batch[3].to(device)
            t_x = batch[0].to(device)
            if with_fusion == True:
                outputs = network(t_x, drug_feat, t_ty)
                t_feat = outputs[2]
                t_ty = outputs[3]
            else:
                _, _, t_feat = network(t_x)
            t_all.append(t_feat)
            t_types.append(t_ty)
    s_all = torch.cat(s_all, dim=0)
    s_types = torch.cat(s_types, dim=0)
    t_all = torch.cat(t_all, dim=0)
    t_types = torch.cat(t_types, dim=0)

    all_features = torch.cat([s_all, t_all], dim=0)
    all_types = torch.cat([s_types, t_types], dim=0)
    s_num = len(s_all)
    t_num = len(t_all)
    drug_num = len(drugs)

    sub_drug = True
    if sub_drug:
        sample_num = int(all_features.shape[0]/drug_num)
        s_num = int(s_num/drug_num)
        t_num = int(t_num/drug_num)
        all_features = all_features.reshape(sample_num, drug_num, -1)
        all_types = all_types.reshape(sample_num, drug_num)

        drugs = ["5-Fluorouracil", "Cisplatin", "Cyclophosphamide", "Docetaxel"]
        drug_num = len(drugs)
        all_features = all_features[:, 0:drug_num, :].reshape(sample_num*drug_num, -1)
        all_types = all_types[:, 0:drug_num].reshape(sample_num*drug_num)
        s_num = s_num*drug_num
        t_num = t_num*drug_num

    all_features_cpu = all_features.cpu().detach().numpy()
    all_types_cpu = all_types.cpu().detach().numpy()

    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(all_features_cpu)
    plt.figure(figsize=(15, 10))

    source_types = np.unique(all_types_cpu[:s_num])
    target_types = np.unique(all_types_cpu[t_num:])

    common_types = np.intersect1d(source_types, target_types)
    source_only_types = np.setdiff1d(source_types, target_types)
    target_only_types = np.setdiff1d(target_types, source_types)

    colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
    colors2 = plt.cm.tab20b(np.linspace(0, 1, 20))
    colors3 = plt.cm.tab20c(np.linspace(0, 1, 20))
    colors = np.vstack([colors1, colors2, colors3])

    for i, tissue_type in enumerate(common_types):
        tissue_key = [k for k, v in config.tissue_map.items() if v == tissue_type][0]

        source_mask = (all_types_cpu[:s_num] == tissue_type)
        plt.scatter(features_2d[:s_num][source_mask, 0],
                    features_2d[:s_num][source_mask, 1],
                    color=colors[i], marker='o',
                    label=f'{tissue_key}', alpha=0.7, s=50, edgecolors='none')

        target_mask = (all_types_cpu[s_num:] == tissue_type)
        plt.scatter(features_2d[s_num:][target_mask, 0],
                    features_2d[s_num:][target_mask, 1],
                    color=colors[i], marker='^',
                    label='_nolegend_', alpha=0.7, s=50, edgecolors='none')

    for i, tissue_type in enumerate(source_only_types):
        tissue_key = [k for k, v in config.tissue_map.items() if v == tissue_type][0]
        source_mask = (all_types_cpu[:s_num] == tissue_type)
        plt.scatter(features_2d[:s_num][source_mask, 0],
                    features_2d[:s_num][source_mask, 1],
                    color=colors[i + len(common_types)], marker='o',
                    label=f'{tissue_key} (CCLE only)', alpha=0.7, s=50, edgecolors='none')

    for i, tissue_type in enumerate(target_only_types):
        tissue_key = [k for k, v in config.tissue_map.items() if v == tissue_type][0]
        target_mask = (all_types_cpu[s_num:] == tissue_type)
        plt.scatter(features_2d[s_num:][target_mask, 0],
                    features_2d[s_num:][target_mask, 1],
                    color=colors[i + len(common_types) + len(source_only_types)], marker='^',
                    label=f'{tissue_key} (TCGA only)', alpha=0.7, s=50, edgecolors='none')

    plt.title('t-SNE Visualization of Tissue Features', fontsize=14, pad=20)
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
               borderaxespad=0., fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(dir, with_fusion_name + "_tissue.png"), dpi=300, bbox_inches='tight')
    plt.close()



    # drugtype
    if with_fusion == True:
        num_samples = all_features_cpu.shape[0]
        num_drugs = len(drugs)
        assert num_samples % num_drugs == 0
        samples_per_drug = num_samples // num_drugs

        drug_labels = np.tile(np.arange(num_drugs), samples_per_drug)

        plt.figure(figsize=(15, 10))
        s_d_color = ["#d2b381","#393b79", "#92bd2d", "#CE182E"]
        t_d_color = ["#d16200","#9edae5", "#237323", "#AC64B0"]

        for i, drug_name in enumerate(drugs):
            source_mask = (drug_labels[:s_num] == i)
            plt.scatter(features_2d[:s_num][source_mask, 0], features_2d[:s_num][source_mask, 1],
                        color=s_d_color[i], marker='o', label=f'{drug_name} (CCLE)', alpha=0.5, s=100, edgecolors='none')

            target_mask = (drug_labels[s_num:] == i)
            plt.scatter(features_2d[s_num:][target_mask, 0], features_2d[s_num:][target_mask, 1],
                        color=t_d_color[i], marker='^', label=f'{drug_name} (TCGA)', alpha=0.5, s=100, edgecolors='none')

        plt.title('t-SNE Visualization by Drug Type', fontsize=14, pad=20)
        plt.xlabel('t-SNE 1', fontsize=12)
        plt.ylabel('t-SNE 2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(dir, with_fusion_name + "_drugtype.png"), dpi=300, bbox_inches='tight')
        plt.close()

