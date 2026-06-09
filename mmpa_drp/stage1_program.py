"""
MMPA-DRP Stage 1 Training (Program-aware Tissue Classifier)
===========================================================
Drop-in replacement for pretraining.py::train_tissue_classifier.
Uses ProgramTissueClassifier instead of ResNetTissueClassifier.
Adds optional program auxiliary losses (diversity, sparsity).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import cycle
import config
from mmpa_drp.models import ProgramTissueClassifier
from mmpa_drp.losses import compute_program_aux_losses
from pretraining import calculate_class_weights


def train_tissue_classifier_program(
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
    # Phase 18: domain separation losses
    lambda_ortho=0.0,
    lambda_recon=0.0,
    lambda_private_domain=0.0,
    # Gene-set selector query gate
    lambda_query_gate_l1=0.0,
):
    """
    Train ProgramTissueClassifier with optional program auxiliary losses.
    Phase 18: supports domain-separated shared/private training.
    Interface mirrors pretraining.train_tissue_classifier.
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
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_ortho_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_priv_domain_loss = 0.0
        epoch_query_gate_loss = 0.0

        for (source_batch, target_batch) in zip(s_train, cycle(t_train)):
            source_inputs = source_batch[0].to(device)
            source_labels = source_batch[1].to(device)

            target_inputs = target_batch[0].to(device)
            target_labels = target_batch[1].to(device)

            source_domain_labels = torch.zeros(source_inputs.size(0), 1).to(device)
            target_domain_labels = torch.ones(target_inputs.size(0), 1).to(device)
            source_domain_idx = torch.zeros(source_inputs.size(0), dtype=torch.long, device=device)
            target_domain_idx = torch.ones(target_inputs.size(0), dtype=torch.long, device=device)

            optimizer.zero_grad()

            # Source forward
            source_tissue_pred, source_domain_pred, _ = model(source_inputs, domain_idx=source_domain_idx)
            source_tissue_loss = tissue_criterion(source_tissue_pred, source_labels)
            source_domain_loss = domain_criterion(source_domain_pred, source_domain_labels)

            # Program auxiliary losses on source
            program_out = model.program_encoder(source_inputs, domain_idx=source_domain_idx)
            aux_loss, _, _, _ = compute_program_aux_losses(
                program_out,
                lambda_rec=lambda_rec,
                lambda_div=lambda_div,
                lambda_sparse=lambda_sparse,
            )

            # Gene-set selector query gate loss
            query_gate_loss_val = 0.0
            if "query_gate_loss" in program_out and lambda_query_gate_l1 > 0:
                query_gate_loss_val = lambda_query_gate_l1 * program_out["query_gate_loss"]
                aux_loss = aux_loss + query_gate_loss_val
                epoch_query_gate_loss += query_gate_loss_val.item()

            # Target forward
            target_tissue_pred, target_domain_pred, _ = model(target_inputs, domain_idx=target_domain_idx)
            target_tissue_loss = target_tissue_criterion(target_tissue_pred, target_labels)
            target_domain_loss = domain_criterion(target_domain_pred, target_domain_labels)

            total_loss = (
                source_tissue_loss
                + target_tissue_loss
                + alpha * (source_domain_loss + target_domain_loss)
                + aux_loss
            )

            # Phase 18: domain separation losses
            ortho_loss_val = 0.0
            recon_loss_val = 0.0
            priv_domain_loss_val = 0.0
            if getattr(model, 'use_domain_sep', False) and getattr(model.program_encoder, 'use_domain_sep', False):
                # Get source domain sep outputs directly (avoid last_domain_sep_out overwrite)
                source_enc_out = model.program_encoder(source_inputs, domain_idx=source_domain_idx)
                source_h_shared = source_enc_out["h_shared"]
                source_h_private = source_enc_out["h_private"]
                source_hidden = source_enc_out["hidden"]

                # Get target domain sep outputs directly
                target_enc_out = model.program_encoder(target_inputs, domain_idx=target_domain_idx)
                target_h_shared = target_enc_out["h_shared"]
                target_h_private = target_enc_out["h_private"]
                target_hidden = target_enc_out["hidden"]

                # Orthogonality: encourage h_shared ⟂ h_private (both domains)
                if lambda_ortho > 0:
                    h_shared_cat = torch.cat([source_h_shared, target_h_shared], dim=0)
                    h_private_cat = torch.cat([source_h_private, target_h_private], dim=0)
                    ortho = (h_shared_cat * h_private_cat).sum(dim=1).pow(2).mean()
                    ortho_loss_val = lambda_ortho * ortho
                    total_loss = total_loss + ortho_loss_val
                    epoch_ortho_loss += ortho_loss_val.item()

                # Reconstruction: hidden ≈ h_shared + h_private (both domains)
                if lambda_recon > 0:
                    hidden_cat = torch.cat([source_hidden, target_hidden], dim=0)
                    h_shared_cat = torch.cat([source_h_shared, target_h_shared], dim=0)
                    h_private_cat = torch.cat([source_h_private, target_h_private], dim=0)
                    recon = F.mse_loss(h_shared_cat + h_private_cat, hidden_cat)
                    recon_loss_val = lambda_recon * recon
                    total_loss = total_loss + recon_loss_val
                    epoch_recon_loss += recon_loss_val.item()

                # Private domain discriminator: private should predict domain
                if lambda_private_domain > 0 and model.private_domain_discriminator is not None:
                    source_priv_domain_pred = model.private_domain_discriminator(source_h_private)
                    target_priv_domain_pred = model.private_domain_discriminator(target_h_private)
                    source_priv_domain_loss = domain_criterion(source_priv_domain_pred, source_domain_labels)
                    target_priv_domain_loss = domain_criterion(target_priv_domain_pred, target_domain_labels)
                    priv_domain_loss_val = lambda_private_domain * (source_priv_domain_loss + target_priv_domain_loss)
                    total_loss = total_loss + priv_domain_loss_val
                    epoch_priv_domain_loss += priv_domain_loss_val.item()

            total_loss.backward()
            optimizer.step()

            _, predicted = source_tissue_pred.max(1)
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
                outputs, _, _ = model(inputs)
                _, predicted = outputs.max(1)
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
                outputs, _, _ = model(inputs)
                _, predicted = outputs.max(1)
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
                outputs, _, _ = model(inputs)
                _, predicted = outputs.max(1)
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
                outputs, _, _ = model(inputs)
                _, predicted = outputs.max(1)
                t_test_total += labels.size(0)
                t_test_correct += predicted.eq(labels).sum().item()
        t_test_acc = t_test_correct / t_test_total

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            print(f"Train Loss: {train_loss / len(s_train):.4f}, Train Acc: {train_acc:.4f}")
            if lambda_ortho > 0:
                print(f"  Ortho Loss: {epoch_ortho_loss / len(s_train):.4f}")
            if lambda_recon > 0:
                print(f"  Recon Loss: {epoch_recon_loss / len(s_train):.4f}")
            if lambda_private_domain > 0:
                print(f"  PrivDomain Loss: {epoch_priv_domain_loss / len(s_train):.4f}")
            if lambda_query_gate_l1 > 0:
                print(f"  QueryGate Loss: {epoch_query_gate_loss / len(s_train):.4f}")
            print(f"Source Domain - Train Acc: {s_train_acc:.4f}, Test Acc: {s_test_acc:.4f}")
            print(f"Target Domain - Train Acc: {t_train_acc:.4f}, Test Acc: {t_test_acc:.4f}")

    return model


def save_feat_program(model, s_dataloaders, t_dataloaders, device):
    """
    Save features for t-SNE / analysis.
    Compatible with ProgramTissueClassifier (uses program_encoder to get h).
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
            features = out["h"]
            source_features.append(features)
            source_labels.append(labels)

    with torch.no_grad():
        for batch in t_train:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            out = model.program_encoder(inputs)
            features = out["h"]
            target_features.append(features)
            target_labels.append(labels)

    source_features = torch.cat(source_features, dim=0)
    target_features = torch.cat(target_features, dim=0)

    np.save("source_features.npy", source_features.cpu().detach().numpy())
    np.save("target_features.npy", target_features.cpu().detach().numpy())
