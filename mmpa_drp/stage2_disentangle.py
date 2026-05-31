"""
MMPA-DRP Stage 2 & 3 Training (Disentangled Program-aware Fusion)
=================================================================
Phase 3A: Uses DisentangledProgramFusionMLP.
"""

import os
import torch
import numpy as np
from itertools import cycle
from torch.utils.data import DataLoader, TensorDataset

from mmpa_drp.models import DisentangledProgramFusionMLP
from finetune import (
    fusion_model_train_step,
    fusion_model_train_step_with_contrastive,
    fusion_eval_epoch,
    get_featuers,
    get_gene_feat_sim,
)


def fusion_training_disentangle(
    encoder,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    drugs,
    gene_num,
    drug_feat,
    **kwargs,
):
    """
    Disentangled Stage 2 & 3 training.
    Mirrors finetune.fusion_taining but instantiates DisentangledProgramFusionMLP.
    """
    drug_feat_out_dim = 64
    bilinear_dim = 128
    fusion_model = DisentangledProgramFusionMLP(
        encoder=encoder,
        input_dim=kwargs["encoder_hidden_dims"][-1] + drug_feat_out_dim + bilinear_dim,
        output_dim=1,
        hidden_dims=[256, 128, 64],
        gene_num=gene_num,
        bilinear_dim=bilinear_dim,
        gene_feat_dim=kwargs["encoder_hidden_dims"][-1],
        drug_feat_dim=drug_feat_out_dim,
        drug_num=len(drugs),
        drop=0.1,
    ).to(kwargs["device"])

    fusion_optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=kwargs["lr"])
    classifier_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
    print("fusion_training_disentangle (MMPA Phase 3A Stage 2)")
    batch_size = kwargs["batch_size"]
    cls_best_auc = 0

    # Stage 2: Classification + CDAN
    for cls_epoch in range(kwargs["train_num_epochs"]):
        train_loss = 0
        fusion_model.train()
        for step, batch in enumerate(zip(train_dataloader, cycle(test_dataloader))):
            loss = fusion_model_train_step(
                model=fusion_model,
                batch=batch,
                drug_feat=drug_feat,
                device=kwargs["device"],
                optimizer=fusion_optimizer,
                loss_fn=classifier_loss,
            )
            train_loss += loss.cpu().detach().item()

        fusion_model.eval()
        metrics, valid_loss = fusion_eval_epoch(
            model=fusion_model,
            loader=val_dataloader,
            drug_feat=drug_feat,
            loss_fn=classifier_loss,
            device=kwargs["device"],
        )

        if metrics[0].mean() > cls_best_auc:
            torch.save(
                fusion_model.state_dict(),
                os.path.join(kwargs["model_save_folder"], "fusion_model.pt"),
            )
            cls_best_auc = metrics[0].mean()

    # Stage 3: Contrastive domain isolation (unchanged logic)
    con_best_auc = 0
    K = kwargs["K"]
    for con_epoch in range(kwargs["contrastive_num_epochs"]):
        (
            all_source_gene_feat,
            all_source_fusion_feat,
            all_target_gene_feat,
            all_target_fusion_feat,
        ) = get_featuers(
            fusion_model, train_dataloader, test_dataloader, drugs, drug_feat, kwargs["device"]
        )
        topk_sim, topk_idx = get_gene_feat_sim(
            all_source_fusion_feat, all_target_fusion_feat, len(drugs), K
        )
        datasets = train_dataloader.dataset.tensors
        inputs, label, mask, tissue = datasets[0], datasets[1], datasets[2], datasets[3]
        sim_data = TensorDataset(inputs, label, mask, tissue, topk_sim, topk_idx)
        train_dataloader = DataLoader(sim_data, batch_size=batch_size, shuffle=False)

        train_loss = 0
        fusion_model.train()
        for step, batch in enumerate(zip(train_dataloader, cycle(test_dataloader))):
            loss = fusion_model_train_step_with_contrastive(
                model=fusion_model,
                batch=batch,
                drug_feat=drug_feat,
                device=kwargs["device"],
                optimizer=fusion_optimizer,
                loss_fn=classifier_loss,
                all_target_fusion_feat=all_target_fusion_feat,
            )
            train_loss += loss.cpu().detach().item()

        fusion_model.eval()
        metrics, valid_loss = fusion_eval_epoch(
            model=fusion_model,
            loader=val_dataloader,
            drug_feat=drug_feat,
            loss_fn=classifier_loss,
            device=kwargs["device"],
        )

        if metrics[0].mean() > con_best_auc:
            torch.save(
                fusion_model.state_dict(),
                os.path.join(kwargs["model_save_folder"], "fusion_model.pt"),
            )
            con_best_auc = metrics[0].mean()

    return fusion_model
