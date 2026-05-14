"""
MMPA-DRP Stage 1 Training (Disentangled Program Classifier)
===========================================================
Phase 3A: Causal-inspired Response-Domain Disentanglement.
Uses DisentangledProgramTissueClassifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import cycle
import config
from mmpa_drp.models import DisentangledProgramTissueClassifier
from mmpa_drp.losses import compute_program_aux_losses, orthogonality_loss
from pretraining import calculate_class_weights


def train_tissue_classifier_disentangle(
    model,
    s_dataloaders,
    t_dataloaders,
    device,
    num_epochs=100,
    lr=0.001,
    alpha=1.0,
    weight_decay=1e-4,
    lambda_rec=0.0,
    lambda_div=0.01,
    lambda_sparse=0.001,
    lambda_adv_response=0.005,
    lambda_domain_nuisance=0.005,
    lambda_ortho=0.001,
    grl_lambda=1.0,
    teacher=None,
    lambda_feat_distill=0.0,
    freeze_warmup_epochs=0,
):
    """
    Train DisentangledProgramTissueClassifier.
    """
    s_train = s_dataloaders[0]
    s_test = s_dataloaders[1]
    t_train = t_dataloaders[0]
    t_test = t_dataloaders[1]

    source_weights = calculate_class_weights(s_train).to(device)
    target_weights = calculate_class_weights(t_train).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    tissue_criterion = nn.CrossEntropyLoss(weight=source_weights)
    target_tissue_criterion = nn.CrossEntropyLoss(weight=target_weights)
    domain_criterion = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        # Freeze warm-up handling
        freeze_mode = (freeze_warmup_epochs > 0 and epoch < freeze_warmup_epochs)
        if freeze_mode:
            for param in model.program_encoder.base_encoder.parameters():
                param.requires_grad = False
        else:
            for param in model.program_encoder.base_encoder.parameters():
                param.requires_grad = True

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for (source_batch, target_batch) in zip(s_train, cycle(t_train)):
            source_inputs = source_batch[0].to(device)
            source_labels = source_batch[1].to(device)

            target_inputs = target_batch[0].to(device)
            target_labels = target_batch[1].to(device)

            bs_s = source_inputs.size(0)
            bs_t = target_inputs.size(0)

            source_domain_labels = torch.zeros(bs_s, 1).to(device)
            target_domain_labels = torch.ones(bs_t, 1).to(device)

            # Combine for disentangle forward
            combined_inputs = torch.cat([source_inputs, target_inputs], dim=0)
            combined_domain_labels = torch.cat(
                [source_domain_labels, target_domain_labels], dim=0
            )

            optimizer.zero_grad()

            out = model(combined_inputs, alpha=alpha)

            # Tissue classification (source + target)
            tissue_pred = out["tissue_pred"]
            source_tissue_loss = tissue_criterion(tissue_pred[:bs_s], source_labels)
            target_tissue_loss = target_tissue_criterion(tissue_pred[bs_s:], target_labels)

            # Original domain discriminator (on h_pred)
            domain_pred = out["domain_pred"]
            source_domain_loss = domain_criterion(
                domain_pred[:bs_s], source_domain_labels
            )
            target_domain_loss = domain_criterion(
                domain_pred[bs_s:], target_domain_labels
            )

            # Response branch adversarial (GRL already applied in model forward)
            adv_response_loss = domain_criterion(
                out["response_domain_pred"], combined_domain_labels
            )

            # Domain-nuisance branch
            domain_nuisance_loss = domain_criterion(
                out["domain_nuisance_pred"], combined_domain_labels
            )

            # Orthogonality loss
            ortho_loss = orthogonality_loss(out["z_response"], out["z_domain"])

            # Program auxiliary losses on source
            program_out = {
                "z": out["z_base"][:bs_s],
                "q": out["q"][:bs_s],
            }
            aux_loss, _, _, _ = compute_program_aux_losses(
                program_out,
                lambda_rec=lambda_rec,
                lambda_div=lambda_div,
                lambda_sparse=lambda_sparse,
            )

            total_loss = (
                source_tissue_loss
                + target_tissue_loss
                + alpha * (source_domain_loss + target_domain_loss)
                + lambda_adv_response * adv_response_loss
                + lambda_domain_nuisance * domain_nuisance_loss
                + lambda_ortho * ortho_loss
                + aux_loss
            )

            # Teacher distillation
            if teacher is not None and lambda_feat_distill > 0:
                with torch.no_grad():
                    teacher_out = teacher(combined_inputs)
                    h_teacher = teacher_out["h"]
                h_pred = out["gene_feat"]
                feat_distill_loss = F.mse_loss(h_pred, h_teacher)
                total_loss = total_loss + lambda_feat_distill * feat_distill_loss

            total_loss.backward()
            optimizer.step()

            _, predicted = tissue_pred[:bs_s].max(1)
            train_total += source_labels.size(0)
            train_correct += predicted.eq(source_labels).sum().item()
            train_loss += source_tissue_loss.item()

        train_acc = train_correct / train_total

        model.eval()

        # Evaluate source domain training set
        s_train_correct = 0
        s_train_total = 0
        with torch.no_grad():
            for batch in s_train:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                out = model(inputs)
                _, predicted = out["tissue_pred"].max(1)
                s_train_total += labels.size(0)
                s_train_correct += predicted.eq(labels).sum().item()
        s_train_acc = s_train_correct / s_train_total

        # Evaluate source domain test set
        s_test_correct = 0
        s_test_total = 0
        with torch.no_grad():
            for batch in s_test:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                out = model(inputs)
                _, predicted = out["tissue_pred"].max(1)
                s_test_total += labels.size(0)
                s_test_correct += predicted.eq(labels).sum().item()
        s_test_acc = s_test_correct / s_test_total

        # Evaluate target domain training set
        t_train_correct = 0
        t_train_total = 0
        with torch.no_grad():
            for batch in t_train:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                out = model(inputs)
                _, predicted = out["tissue_pred"].max(1)
                t_train_total += labels.size(0)
                t_train_correct += predicted.eq(labels).sum().item()
        t_train_acc = t_train_correct / t_train_total

        # Evaluate target domain test set
        t_test_correct = 0
        t_test_total = 0
        with torch.no_grad():
            for batch in t_test:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                out = model(inputs)
                _, predicted = out["tissue_pred"].max(1)
                t_test_total += labels.size(0)
                t_test_correct += predicted.eq(labels).sum().item()
        t_test_acc = t_test_correct / t_test_total

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            print(f"Train Loss: {train_loss / len(s_train):.4f}, Train Acc: {train_acc:.4f}")
            print(f"Source Domain - Train Acc: {s_train_acc:.4f}, Test Acc: {s_test_acc:.4f}")
            print(f"Target Domain - Train Acc: {t_train_acc:.4f}, Test Acc: {t_test_acc:.4f}")

    return model


def save_feat_disentangle(model, s_dataloaders, t_dataloaders, device):
    """
    Save h_pred features for t-SNE / analysis.
    """
    model.eval()
    s_train = s_dataloaders[0]
    t_train = t_dataloaders[0]

    source_features = []
    source_labels = []
    target_features = []
    target_labels = []

    with torch.no_grad():
        for batch in s_train:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            out = model.program_encoder(inputs)
            features = out["h_pred"]
            source_features.append(features)
            source_labels.append(labels)

    with torch.no_grad():
        for batch in t_train:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            out = model.program_encoder(inputs)
            features = out["h_pred"]
            target_features.append(features)
            target_labels.append(labels)

    source_features = torch.cat(source_features, dim=0)
    target_features = torch.cat(target_features, dim=0)

    np.save("source_features.npy", source_features.cpu().detach().numpy())
    np.save("target_features.npy", target_features.cpu().detach().numpy())
