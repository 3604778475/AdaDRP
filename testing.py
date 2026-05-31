# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, auc, precision_recall_curve
from utility import auprc
import config

def multi_eval_epoch(model, loader, drug_feat, device, domain_idx=None):
    model.eval()
    total_loss = 0
    y_true, y_pred, y_mask = [], [], []
    auc_list, aupr_list, acc_list, f1_list = [], [], [], []
    for x, y, mask, tissue in loader:
        x = x.to(device)
        y = y.to(device)
        mask_tmp = (mask > 0).to(device)
        tissue = tissue.to(device)
        with torch.no_grad():
            if getattr(model.encoder, 'use_domain_sep', False):
                batch_domain_idx = torch.full((x.size(0),), domain_idx, dtype=torch.long, device=device) if domain_idx is not None else None
            else:
                batch_domain_idx = None
            outputs = model(x, drug_feat, tissue, domain_idx=batch_domain_idx)
            yp = outputs[0].reshape(y.shape)
            yp = yp.reshape(y.shape)
            loss_mat = nn.BCEWithLogitsLoss(reduction='none')(yp, y.double())
            loss_mat = torch.where(mask_tmp, loss_mat,
                                   torch.zeros(loss_mat.shape).to(device))
            loss = torch.sum(loss_mat) / torch.sum(mask_tmp)
            total_loss += loss
            y_true += y.cpu().detach().numpy().tolist()
            y_pred += yp.cpu().detach().numpy().tolist()
            y_mask += mask.cpu().detach().numpy().tolist()

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
    return total_loss / len(loader), np.array(all_results), y_true, y_pred, y_mask

def testing(model, t_dataloader, drug_feat, device):
    model.train(False)
    drug_feat = drug_feat.to(device)
    # TCGA is target domain (domain_idx=1) for domain separation
    domain_idx = 1 if getattr(model.encoder, 'use_domain_sep', False) else None
    test_loss, results, y_true, y_pred, y_mask = multi_eval_epoch(model=model, loader=t_dataloader, drug_feat=drug_feat, device=device, domain_idx=domain_idx)
    return test_loss, results, y_true, y_pred, y_mask
