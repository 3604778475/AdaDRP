# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import torch.nn as nn
from myloss import contrastive_loss
from itertools import cycle
from models import Fusion_MLP
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from utility import auprc
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

def fusion_model_train_step(model, batch, drug_feat, device, optimizer, loss_fn):

    model.train()
    optimizer.zero_grad()

    s_x = batch[0][0].to(device)   #B,F
    s_y = batch[0][1].to(device)   #B,drug_num
    s_mask = batch[0][2].to(device)   #B,drug_num
    s_mask = (s_mask > 0)
    s_tissue = batch[0][3].to(device)  #B

    s_outputs = model(s_x, drug_feat, s_tissue)
    s_yp = s_outputs[0]
    s_cdan_domain_output = s_outputs[4]

    s_yp = s_yp.reshape(s_y.shape)
    s_loss_mat = loss_fn(s_yp, s_y.double())
    s_loss_mat = torch.where(
        s_mask, s_loss_mat,
        torch.zeros(s_loss_mat.shape).to(device).to(s_loss_mat.dtype))
    s_loss = torch.sum(s_loss_mat) / torch.sum(s_mask)


    t_x = batch[1][0].to(device)
    t_tissue = batch[1][3].to(device)

    t_outputs = model(t_x, drug_feat, t_tissue)
    t_cdan_domain_output = t_outputs[4]

    # CDAN
    if s_cdan_domain_output is not None and t_cdan_domain_output is not None:
        s_domain_labels = torch.zeros(s_cdan_domain_output.size(0), 1).to(device)
        t_domain_labels = torch.ones(t_cdan_domain_output.size(0), 1).to(device)

        domain_outputs = torch.cat([s_cdan_domain_output, t_cdan_domain_output], dim=0)
        domain_labels = torch.cat([s_domain_labels, t_domain_labels], dim=0)

        domain_criterion = nn.BCEWithLogitsLoss()
        cdan_domain_loss = domain_criterion(domain_outputs, domain_labels)
    else:
        cdan_domain_loss = torch.tensor(0.0).to(device)

    loss = s_loss + cdan_domain_loss
    loss.backward()
    optimizer.step()
    return loss

def fusion_model_train_step_with_contrastive(model, batch, drug_feat, device, optimizer, all_target_fusion_feat, loss_fn):

    model.train()
    optimizer.zero_grad()

    s_x = batch[0][0].to(device)   #B,F
    s_y = batch[0][1].to(device)   #B,drug_num
    s_mask = batch[0][2].to(device)   #B,drug_num
    s_mask = (s_mask > 0)
    s_tissue = batch[0][3].to(device)  #B
    s_topk_idx = batch[0][5].to(device)   #B, drug_num, K

    s_outputs = model(s_x, drug_feat, s_tissue)
    s_yp = s_outputs[0]
    s_fusion_feat = s_outputs[2]  # [B*drug_num, D]

    s_yp = s_yp.reshape(s_y.shape)
    s_loss_mat = loss_fn(s_yp, s_y.double())
    s_loss_mat = torch.where(
        s_mask, s_loss_mat,
        torch.zeros(s_loss_mat.shape).to(device).to(s_loss_mat.dtype))
    cls_loss = torch.sum(s_loss_mat) / torch.sum(s_mask)

    # 构建正样本特征
    B = s_y.size(0)
    drug_num = s_y.size(1)
    D = s_fusion_feat.size(-1)

    fusion_feat = s_fusion_feat.view(B, drug_num, D)  # [B, drug_num, D]
    anchor = fusion_feat.reshape(-1, D)  # [B*drug_num, D]

    B, drug_num, K = s_topk_idx.shape
    M, _, D = all_target_fusion_feat.shape

    topk_idx_flat = s_topk_idx.reshape(-1, K)  # [B*drug_num, K]
    drug_ids = torch.arange(drug_num, device=s_topk_idx.device).unsqueeze(0).expand(B, drug_num)
    drug_ids_flat = drug_ids.reshape(-1, 1).expand(-1, K)  # [B*drug_num, K]

    pos_samples = all_target_fusion_feat[topk_idx_flat, drug_ids_flat, :]  # [B*drug_num, K, D]
    neg_samples = all_target_fusion_feat.reshape(-1, D)  # [M*drug_num, D]

    contrastive_loss_val = contrastive_loss(anchor, pos_samples, neg_samples)

    contrastive_weight = 0.1
    loss = cls_loss + contrastive_weight * contrastive_loss_val
    loss.backward()

    optimizer.step()
    return loss


def fusion_eval_epoch(model, loader, drug_feat, loss_fn, device):
    y_true, y_pred, y_mask = [], [], []
    auc_list, aupr_list, f1_list, acc_list = [], [], [], []
    avg_loss = 0
    for x, y, mask, tissue in loader:
        x = x.to(device)
        y = y.to(device)
        mask_tmp = (mask > 0).to(device)
        with torch.no_grad():
            outputs = model(x, drug_feat, tissue)
            yp = outputs[0].reshape(y.shape)
            loss_mat = loss_fn(yp, y.double())
            loss_mat = torch.where(mask_tmp, loss_mat,
                                   torch.zeros(loss_mat.shape).to(device))
            loss = torch.sum(loss_mat) / torch.sum(mask_tmp)
            y_true += y.cpu().detach().numpy().tolist()
            y_pred += yp.cpu().detach().numpy().tolist()
            y_mask += mask.cpu().detach().numpy().tolist()
            avg_loss += loss / x.size(0)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_mask = np.array(y_mask)
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1.0) > 0 and np.sum(y_true[:, i] == 0.0) > 0:
            is_valid = (y_mask[:, i] > 0)
            auc_list.append(roc_auc_score(y_true[is_valid, i], y_pred[is_valid, i]))
            aupr_list.append(auprc(y_true[is_valid, i], y_pred[is_valid, i]))
            f1_list.append(f1_score(y_true[is_valid, i], (y_pred[is_valid, i] >= 0.5).astype('int')))
            acc_list.append(accuracy_score(y_true[is_valid, i], (y_pred[is_valid, i] >= 0.5).astype('int')))
        else:
            print('{} is invalid'.format(i))
    all_results = [auc_list, aupr_list, acc_list, f1_list]

    return np.array(all_results), avg_loss.cpu().detach().item()

def get_featuers(fusion_model, train_dataloader, test_dataloader, drugs, drug_feat, device):
    all_source_gene_feat = []
    all_source_fusion_feat = []
    fusion_model.eval()
    with torch.no_grad():
        for batch in train_dataloader:
            x = batch[0].to(device)
            tissue = batch[3].to(device)
            outputs = fusion_model(x, drug_feat, tissue)
            gene_feat = outputs[1]
            fusion_feat = outputs[2]
            batch_size = x.size(0)
            drug_num = len(drugs)
            fusion_feat = fusion_feat.view(batch_size, drug_num, -1)   #[B, drug_num, D]
            all_source_gene_feat.append(gene_feat)
            all_source_fusion_feat.append(fusion_feat)
    all_source_gene_feat = torch.cat(all_source_gene_feat, dim=0)  # [N, F]
    all_source_fusion_feat = torch.cat(all_source_fusion_feat, dim=0)  # [N, drug_num, D]

    all_target_gene_feat = []
    all_target_fusion_feat = []
    with torch.no_grad():
        for batch in test_dataloader:
            x = batch[0].to(device)
            tissue = batch[3].to(device)
            outputs = fusion_model(x, drug_feat, tissue)
            gene_feat = outputs[1]
            fusion_feat = outputs[2]
            batch_size = x.size(0)
            drug_num = len(drugs)
            fusion_feat = fusion_feat.view(batch_size, drug_num, -1)
            all_target_gene_feat.append(gene_feat)
            all_target_fusion_feat.append(fusion_feat)
    all_target_gene_feat = torch.cat(all_target_gene_feat, dim=0)  # [M, F]
    all_target_fusion_feat = torch.cat(all_target_fusion_feat, dim=0)  # [N, drug_num, D]
    return all_source_gene_feat, all_source_fusion_feat, all_target_gene_feat, all_target_fusion_feat


def get_gene_feat_sim(all_source_gene_feat, all_target_gene_feat, drug_num, K):
    sim = F.cosine_similarity(all_source_gene_feat.unsqueeze(1), all_target_gene_feat.unsqueeze(0), dim=-1)  # [N, M]
    sim_matrix = sim.unsqueeze(1).expand(-1, drug_num, -1)  # [N, drug_num, M]
    topk_sim, topk_idx = torch.topk(sim_matrix, K, dim=2)  # [N, drug_num, K]
    return topk_sim, topk_idx

def fusion_taining(encoder, train_dataloader, val_dataloader, test_dataloader, drugs, gene_num, drug_feat, **kwargs):
    drug_feat_out_dim = 64
    bilinear_dim = 128
    fusion_model = Fusion_MLP(encoder = encoder, input_dim = kwargs['encoder_hidden_dims'][-1]+drug_feat_out_dim+bilinear_dim, output_dim = 1,
                              hidden_dims=[256, 128, 64], gene_num = gene_num, bilinear_dim = bilinear_dim, gene_feat_dim = kwargs['encoder_hidden_dims'][-1],
                              drug_feat_dim = drug_feat_out_dim, drug_num=len(drugs), drop=0.1).to(kwargs['device'])
    fusion_optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=kwargs['lr'])
    classifier_loss = nn.BCEWithLogitsLoss(reduction='none')
    print("fusion_pretaining")
    batch_size = kwargs['batch_size']
    cls_best_auc = 0

    for cls_epoch in range(kwargs['train_num_epochs']):
        #Training
        train_loss = 0
        fusion_model.train()
        for step, batch in enumerate(zip(train_dataloader, cycle(test_dataloader))):
            loss = fusion_model_train_step(model=fusion_model, batch=batch, drug_feat=drug_feat, device=kwargs['device'],
                                    optimizer=fusion_optimizer, loss_fn=classifier_loss)
            train_loss += loss.cpu().detach().item()
        # if (epoch + 1) % 100 == 0:
        #     print('classification training epoch = {}'.format(epoch + 1))
        #     print('training loss : {:.4f}'.format(train_loss))

        #Evaluating
        fusion_model.eval()
        metrics, valid_loss = fusion_eval_epoch(model=fusion_model, loader=val_dataloader, drug_feat=drug_feat,
                                                loss_fn=classifier_loss, device=kwargs['device'])
        # if epoch % 50 == 0:
        #     print('validating loss : {:.4f}'.format(valid_loss))
        #     print('validation avg (GDSC) metrics (auc,aupr,f1,acc): {}'.format(np.around(metrics.mean(axis=1), 4)))

        if metrics[0].mean() > cls_best_auc:
            torch.save(fusion_model.state_dict(), os.path.join(kwargs['model_save_folder'], 'fusion_model.pt'))
            cls_best_auc = metrics[0].mean()

    con_best_auc = 0
    K = kwargs['K']  # Top-K
    for con_epoch in range(kwargs['contrastive_num_epochs']):
        all_source_gene_feat, all_source_fusion_feat, all_target_gene_feat, all_target_fusion_feat = get_featuers(fusion_model, train_dataloader, test_dataloader, drugs, drug_feat, kwargs['device'])
        topk_sim, topk_idx = get_gene_feat_sim(all_source_gene_feat, all_target_gene_feat, len(drugs), K)
        datasets = train_dataloader.dataset.tensors
        inputs, label, mask, tissue = datasets[0], datasets[1], datasets[2], datasets[3]
        sim_data = TensorDataset(inputs, label, mask, tissue, topk_sim, topk_idx)
        train_dataloader = DataLoader(sim_data, batch_size=batch_size, shuffle=False)

        #Training
        train_loss = 0
        fusion_model.train()
        for step, batch in enumerate(zip(train_dataloader, cycle(test_dataloader))):
            loss = fusion_model_train_step_with_contrastive(model=fusion_model, batch=batch, drug_feat=drug_feat, device=kwargs['device'],
                                    optimizer=fusion_optimizer, loss_fn=classifier_loss, all_target_fusion_feat=all_target_fusion_feat)
            train_loss += loss.cpu().detach().item()
        # if (epoch + 1) % 100 == 0:
        #     print('classification training epoch = {}'.format(epoch + 1))
        #     print('training loss : {:.4f}'.format(train_loss))

        #Evaluating
        fusion_model.eval()
        metrics, valid_loss = fusion_eval_epoch(model=fusion_model, loader=val_dataloader, drug_feat=drug_feat,
                                                loss_fn=classifier_loss, device=kwargs['device'])
        # if epoch % 50 == 0:
        #     print('validating loss : {:.4f}'.format(valid_loss))
        #     print('validation avg (GDSC) metrics (auc,aupr,f1,acc): {}'.format(np.around(metrics.mean(axis=1), 4)))

        if metrics[0].mean() > con_best_auc:
            torch.save(fusion_model.state_dict(), os.path.join(kwargs['model_save_folder'], 'fusion_model.pt'))
            con_best_auc = metrics[0].mean()

    return fusion_model

