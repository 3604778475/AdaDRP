#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract intermediate features from a trained FMPA-DRP / MMPA-DRP checkpoint for
t-SNE visualization.

The script loads the best fusion-model checkpoint (ProgramFusionMLP or the
legacy Fusion_MLP), runs a no-gradient forward pass on source-domain GDSC
(CCLE) and target-domain TCGA data, and saves the requested intermediate
representations to a compressed NPZ file.

Example
-------
    source activate macb_drp
    python tools/extract_fmpa_tsne_features.py \
        --checkpoint logs/2026-07-02_07-57-33mmpa_program/.../fusion_model_fold0.pt \
        --fold 0 \
        --output-dir artifacts/tsne/fold_0 \
        --device cuda:0 \
        --seed 2025
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as project_config
from utility import set_seed
from dataload import get_drug_feat, get_multi_labeled_dataloader, get_unlabeled_dataloaders
from mmpa_drp.models import MolecularProgramEncoder, ProgramFusionMLP

# Default drug panel used throughout the project.
DEFAULT_DRUGS = [
    "5-Fluorouracil",
    "Cisplatin",
    "Cyclophosphamide",
    "Docetaxel",
    "Doxorubicin",
    "Etoposide",
    "Gemcitabine",
    "Paclitaxel",
    "Temozolomide",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract FMPA-DRP intermediate features for t-SNE visualization."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the trained fusion_model_fold*.pt checkpoint."
    )
    parser.add_argument(
        "--fold", type=int, default=0,
        help="Which CV fold to use for source/train split (0-4)."
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory where the NPZ file and logs will be saved."
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device to use for inference."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for feature extraction (default: 64)."
    )
    parser.add_argument(
        "--seed", type=int, default=2025,
        help="Random seed for data splitting."
    )
    parser.add_argument(
        "--measurement", type=str, default="Z_SCORE",
        choices=["Z_SCORE", "LN_IC50", "AUC"],
        help="GDSC measurement column used during training."
    )
    parser.add_argument(
        "--thres-gdsc", type=float, default=0.0,
        help="GDSC response threshold used during training."
    )
    parser.add_argument(
        "--thres-label", type=float, default=0.1,
        help="Label-graph threshold used during training."
    )
    parser.add_argument(
        "--n-splits", type=int, default=5,
        help="Number of CV folds in get_multi_labeled_dataloader."
    )
    parser.add_argument(
        "--drug-names", type=str, default=",".join(DEFAULT_DRUGS),
        help="Comma-separated drug names in the exact training order."
    )
    parser.add_argument(
        "--include-val-source", action="store_true",
        help="Include the validation GDSC split as additional source samples."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Optional JSON config overriding auto-inferred architecture."
    )
    parser.add_argument(
        "--strict", action="store_true", default=True,
        help="Use strict=True when loading the state_dict."
    )
    parser.add_argument(
        "--no-strict", dest="strict", action="store_false",
        help="Allow missing/unexpected keys during state_dict loading."
    )
    return parser.parse_args(argv)


def load_checkpoint(path: str, device: str) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    """
    Load a checkpoint and return (raw_checkpoint, state_dict, was_data_parallel).

    Supports:
      - checkpoint is itself a state dict
      - checkpoint["model_state_dict"]
      - checkpoint["state_dict"]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    raw_ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(raw_ckpt, dict):
        if "model_state_dict" in raw_ckpt:
            state_dict = raw_ckpt["model_state_dict"]
        elif "state_dict" in raw_ckpt:
            state_dict = raw_ckpt["state_dict"]
        else:
            # assume the dict is the state dict itself
            state_dict = raw_ckpt
    else:
        state_dict = raw_ckpt

    was_data_parallel = any(k.startswith("module.") for k in state_dict.keys())
    if was_data_parallel:
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    return raw_ckpt, state_dict, was_data_parallel


def _shape_of(sd: Dict[str, Any], key: str) -> Optional[Tuple[int, ...]]:
    tensor = sd.get(key)
    if tensor is None:
        return None
    return tuple(tensor.shape)


def _detect_encoder_backbone(sd: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Infer MolecularProgramEncoder backbone type and its constructor kwargs from
    the state dict.
    """
    if "encoder.backbone.tokenizer.weight" in sd:
        # ConvNeXt1D
        w = sd["encoder.backbone.tokenizer.weight"]
        b = sd["encoder.backbone.tokenizer.bias"]
        g = w.shape[1]                      # gene_num
        num_tokens_times_token_dim = w.shape[0]

        token_dim = sd["encoder.backbone.final_norm.weight"].shape[0]
        if num_tokens_times_token_dim % token_dim != 0:
            raise ValueError(
                f"Cannot infer num_gene_tokens: {num_tokens_times_token_dim} not divisible by {token_dim}"
            )
        num_gene_tokens = num_tokens_times_token_dim // token_dim

        # Count blocks
        depth = 0
        while f"encoder.backbone.blocks.{depth}.dwconv.weight" in sd:
            depth += 1
        if depth == 0:
            raise ValueError("ConvNeXt backbone found but no blocks in state dict")

        dw = sd["encoder.backbone.blocks.0.dwconv.weight"]
        kernel_size = dw.shape[2]
        pw1 = sd["encoder.backbone.blocks.0.pwconv1.weight"]
        expansion = pw1.shape[0] // token_dim

        # Output dim of backbone (input dim of program_proj)
        program_proj_w = sd["encoder.program_proj.weight"]
        backbone_output_dim = program_proj_w.shape[1]

        return "convnext1d", {
            "num_gene_tokens": num_gene_tokens,
            "token_dim": token_dim,
            "convnext_depth": depth,
            "convnext_kernel_size": kernel_size,
            "convnext_expansion": expansion,
            "hidden_dims": [backbone_output_dim],
        }

    if "encoder.backbone.fc1.weight" in sd:
        # GRN-MLP backbone
        hidden_dim = sd["encoder.backbone.fc1.weight"].shape[0]
        return "grn_mlp", {"hidden_dims": [hidden_dim]}

    if "encoder.backbone.0.0.weight" in sd:
        # Standard MLP backbone: Sequential modules are [Linear, act, BN, Drop]
        dims = []
        i = 0
        while f"encoder.backbone.{i}.0.weight" in sd:
            dims.append(sd[f"encoder.backbone.{i}.0.weight"].shape[0])
            i += 1
        return "mlp", {"hidden_dims": dims}

    if any(k.startswith("encoder.backbone.residual_blocks") for k in sd):
        # ResNetBackbone (legacy MACB)
        dims = []
        i = 0
        while f"encoder.backbone.residual_blocks.{i}.block.0.weight" in sd:
            dims.append(sd[f"encoder.backbone.residual_blocks.{i}.block.0.weight"].shape[0])
            i += 1
        return "resnet", {"hidden_dims": dims}

    raise ValueError("Could not infer encoder backbone type from checkpoint keys.")


def infer_model_config(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Infer a model-configuration dictionary from a ProgramFusionMLP state dict.

    The returned dict contains all arguments required to reconstruct the
    MolecularProgramEncoder and ProgramFusionMLP used during training.
    """
    sd = state_dict
    cfg: Dict[str, Any] = {}

    # Basic shapes
    attention_masks = sd.get("attention_masks")
    if attention_masks is None:
        raise ValueError("Checkpoint missing 'attention_masks'; cannot infer tissue/gene dimensions.")
    num_tissues, gene_num = attention_masks.shape
    cfg["gene_num"] = int(gene_num)
    cfg["num_tissues"] = int(num_tissues)

    # Program encoder
    activation_bias = sd["encoder.activation_proj.bias"]
    num_programs = int(activation_bias.shape[0])
    program_proj_bias = sd["encoder.program_proj.bias"]
    if program_proj_bias.shape[0] % num_programs != 0:
        raise ValueError("Cannot infer program_dim from program_proj.bias")
    program_dim = program_proj_bias.shape[0] // num_programs
    cfg["num_programs"] = num_programs
    cfg["program_dim"] = program_dim

    backbone_type, backbone_kwargs = _detect_encoder_backbone(sd)
    cfg["backbone_type"] = backbone_type
    cfg["backbone_kwargs"] = backbone_kwargs
    backbone_output_dim = backbone_kwargs["hidden_dims"][-1]

    # Domain separation
    use_domain_sep = "encoder.domain_sep.shared_mlp.0.weight" in sd
    cfg["use_domain_sep"] = use_domain_sep
    if use_domain_sep:
        domain_sep_hidden_dim = int(sd["encoder.domain_sep.shared_mlp.0.weight"].shape[0])
        domain_sep_output_dim = int(sd["encoder.domain_sep.shared_mlp.4.weight"].shape[0])
        cfg["domain_sep_hidden_dim"] = domain_sep_hidden_dim
        cfg["domain_sep_output_dim"] = domain_sep_output_dim
        gene_feat_dim = domain_sep_output_dim
    else:
        cfg["domain_sep_hidden_dim"] = 256
        cfg["domain_sep_output_dim"] = program_dim
        gene_feat_dim = program_dim

    # Bilinear / fusion dims
    bilinear_weight = sd["bilinear.weight"]
    bilinear_dim = int(bilinear_weight.shape[0])
    cfg["bilinear_dim"] = bilinear_dim
    cfg["gene_feat_dim"] = int(bilinear_weight.shape[1])
    cfg["drug_feat_dim"] = int(bilinear_weight.shape[2])

    # Consistency check
    if cfg["gene_feat_dim"] != gene_feat_dim:
        warnings.warn(
            f"Inferred gene_feat_dim from bilinear ({cfg['gene_feat_dim']}) does not match "
            f"encoder output ({gene_feat_dim}); using bilinear value."
        )
        gene_feat_dim = cfg["gene_feat_dim"]

    # Fusion MLP hidden dims and input dim
    fusion_first = sd["fusion_module.module.0.0.weight"]
    input_dim = int(fusion_first.shape[1])
    cfg["input_dim"] = input_dim
    hidden_dims = []
    i = 0
    while f"fusion_module.module.{i}.0.weight" in sd:
        hidden_dims.append(int(sd[f"fusion_module.module.{i}.0.weight"].shape[0]))
        i += 1
    cfg["hidden_dims"] = hidden_dims
    cfg["fused_feature_dim"] = hidden_dims[-1]

    # Drug count
    drug_num = int(sd["drug_classifier.4.weight"].shape[0])
    cfg["drug_num"] = drug_num

    # Optional modules (detected by key prefixes)
    cfg["use_fpi"] = any(k.startswith("fpi.") for k in sd)
    cfg["use_fpi_router"] = any(k.startswith("fpi_router.") for k in sd)
    cfg["use_bpaf"] = any(k.startswith("bpaf.") for k in sd)
    cfg["use_llrf"] = any(k.startswith("llrf.") for k in sd)
    cfg["use_dspi"] = any(k.startswith("dspi.") for k in sd)
    cfg["use_dcda"] = any(k.startswith("dcda_") or k.startswith("domain_aligner") for k in sd)
    cfg["use_dcd_residual"] = "dcd_residual.residual_head.0.weight" in sd
    cfg["use_pdg"] = "pdg.proj.0.weight" in sd
    cfg["use_vcb"] = any(k.startswith("vcb.") for k in sd)
    cfg["use_cross_attn"] = "cross_attn_fusion.gene_to_drug.weight" in sd
    cfg["use_drug_relation_refiner"] = "drug_relation_refiner.layers.0.weight" in sd

    # Drug encoder type
    if "drug_unimol_emb" in sd or "unimol_proj.projector.0.weight" in sd:
        if "drug_fusion.fusion.0.weight" in sd:
            cfg["drug_encoder_type"] = "fp_unimol_fusion"
        else:
            cfg["drug_encoder_type"] = "unimol"
        if "unimol_proj.projector.0.weight" in sd:
            cfg["unimol_dim"] = int(sd["unimol_proj.projector.0.weight"].shape[1])
    else:
        cfg["drug_encoder_type"] = "mlp"
        cfg["unimol_dim"] = 512

    # Main fusion type
    if "main_fusion_replacement" in sd or any(k.startswith("main_fusion_replacement.") for k in sd):
        cfg["main_fusion_type"] = "lrbf"  # default; dcf_lrbf would look identical
    else:
        cfg["main_fusion_type"] = "bilinear"

    # FPI fragment configuration
    if cfg["use_fpi"] or cfg["use_fpi_router"]:
        frag_fp = sd.get("_frag_fp")
        if frag_fp is None:
            raise ValueError("FPI/FPI-Router enabled but checkpoint missing '_frag_fp' buffer.")
        _, max_fragments, fp_dim = frag_fp.shape
        cfg["fpi_config"] = {
            "fp_dim": int(fp_dim),
            "max_fragments": int(max_fragments),
            "hidden_dim": gene_feat_dim,
            "program_dim": program_dim,
            "drug_dim": cfg["drug_feat_dim"],
            "num_programs": num_programs,
            "num_drugs": drug_num,
            "q_prior_weight": 0.01,
            "gamma_max": 0.1,
            "gamma_init": 0.05,
            "dropout": 0.1,
        }
    else:
        cfg["fpi_config"] = None

    # Fragment tensors (used as buffers by ProgramFusionMLP)
    cfg["frag_fp"] = sd.get("_frag_fp")
    cfg["frag_mask"] = sd.get("_frag_mask")

    return cfg


def merge_with_user_config(cfg: Dict[str, Any], config_path: Optional[str]) -> Dict[str, Any]:
    """Override inferred config with a user-supplied JSON config."""
    if config_path is None:
        return cfg
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"User config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        user_cfg = json.load(f)
    cfg = dict(cfg)
    cfg.update(user_cfg)
    return cfg


def build_encoder(cfg: Dict[str, Any]) -> MolecularProgramEncoder:
    """Build a MolecularProgramEncoder matching the inferred config."""
    kwargs = dict(cfg["backbone_kwargs"])
    return MolecularProgramEncoder(
        input_dim=cfg["gene_num"],
        num_programs=cfg["num_programs"],
        program_dim=cfg["program_dim"],
        drop=0.1,
        act_fn=nn.SELU,
        backbone_type=cfg["backbone_type"],
        use_domain_sep=cfg.get("use_domain_sep", False),
        domain_sep_hidden_dim=cfg.get("domain_sep_hidden_dim", 256),
        domain_sep_output_dim=cfg.get("domain_sep_output_dim", cfg["program_dim"]),
        **kwargs,
    )


def build_fusion_model(
    encoder: MolecularProgramEncoder,
    cfg: Dict[str, Any],
    state_dict: Dict[str, Any],
    device: str,
    strict: bool = True,
) -> ProgramFusionMLP:
    """
    Build and load a ProgramFusionMLP from the inferred config and state dict.
    """
    model = ProgramFusionMLP(
        encoder=encoder,
        input_dim=cfg["input_dim"],
        output_dim=1,
        hidden_dims=cfg["hidden_dims"],
        gene_num=cfg["gene_num"],
        bilinear_dim=cfg["bilinear_dim"],
        gene_feat_dim=cfg["gene_feat_dim"],
        drug_feat_dim=cfg["drug_feat_dim"],
        drug_num=cfg["drug_num"],
        drop=0.1,
        act_fn=nn.SELU,
        use_fpi=cfg.get("use_fpi", False),
        fpi_config=cfg.get("fpi_config"),
        frag_fp=cfg.get("frag_fp"),
        frag_mask=cfg.get("frag_mask"),
        use_domain_sep=cfg.get("use_domain_sep", False),  # also read from encoder attr
        use_cross_attn=cfg.get("use_cross_attn", False),
        drug_encoder_type=cfg.get("drug_encoder_type", "mlp"),
        unimol_dim=cfg.get("unimol_dim", 512),
        use_drug_relation_refiner=cfg.get("use_drug_relation_refiner", False),
        main_fusion_type=cfg.get("main_fusion_type", "bilinear"),
        use_bpaf=cfg.get("use_bpaf", False),
        use_llrf=cfg.get("use_llrf", False),
        use_dspi=cfg.get("use_dspi", False),
        use_dcda=cfg.get("use_dcda", False),
        use_dcd_residual=cfg.get("use_dcd_residual", False),
        use_pdg=cfg.get("use_pdg", False),
        use_vcb=cfg.get("use_vcb", False),
        use_fpi_router=cfg.get("use_fpi_router", False),
    ).to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        print(f"[Warning] {len(missing)} missing keys:")
        for k in missing[:10]:
            print(f"  - {k}")
    if unexpected:
        print(f"[Warning] {len(unexpected)} unexpected keys:")
        for k in unexpected[:10]:
            print(f"  - {k}")

    model.eval()
    return model


def _model_accepts_domain_idx(model: nn.Module) -> bool:
    """Check whether the model forward accepts a domain_idx argument."""
    sig = inspect.signature(model.forward)
    return "domain_idx" in sig.parameters


def _model_supports_dict_output(model: nn.Module) -> bool:
    """Check whether the model forward accepts return_dict / return_fused_feature."""
    sig = inspect.signature(model.forward)
    return "return_dict" in sig.parameters and "return_fused_feature" in sig.parameters


def reshape_fused_feature(fused: torch.Tensor, batch_size: int, drug_num: int) -> torch.Tensor:
    """
    Ensure fused_feature has shape [B, D, H].

    If the model returns [B*D, H], validate the first dimension and reshape.
    """
    if fused.dim() == 3:
        if fused.shape[0] != batch_size or fused.shape[1] != drug_num:
            raise ValueError(
                f"fused_feature 3D shape {tuple(fused.shape)} does not match "
                f"expected [B={batch_size}, D={drug_num}, H]."
            )
        return fused
    if fused.dim() == 2:
        if fused.shape[0] != batch_size * drug_num:
            raise ValueError(
                f"fused_feature 2D shape {tuple(fused.shape)} cannot be reshaped to "
                f"[B={batch_size}, D={drug_num}, H] because dim0 != {batch_size * drug_num}."
            )
        return fused.view(batch_size, drug_num, -1)
    raise ValueError(f"Unexpected fused_feature dimension: {fused.dim()}")


def extract_domain_features(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    drug_feat: torch.Tensor,
    device: str,
    domain_idx: int,
    drug_num: int,
) -> Dict[str, np.ndarray]:
    """
    Run no-gradient inference on one domain and collect intermediate features.

    Returns a dict with numpy arrays:
      gene_feat, fused_feature, hidden, h_pool, h_shared, h_private,
      tissue, sample_id (if available).
    """
    feats: Dict[str, List[torch.Tensor]] = {
        "gene_feat": [],
        "fused_feature": [],
        "hidden": [],
        "h_pool": [],
        "h_shared": [],
        "h_private": [],
        "tissue": [],
        "sample_id": [],
    }

    supports_dict = _model_supports_dict_output(model)
    accepts_domain_idx = _model_accepts_domain_idx(model)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            tissue = batch[3].to(device)

            kwargs: Dict[str, Any] = {}
            if accepts_domain_idx:
                kwargs["domain_idx"] = torch.full(
                    (x.size(0),), domain_idx, dtype=torch.long, device=device
                )

            if supports_dict:
                outputs = model(
                    x,
                    drug_feat=drug_feat,
                    tissue_idx=tissue,
                    return_dict=True,
                    return_fused_feature=True,
                    return_encoder_out=True,
                    **kwargs,
                )
                gene_feat = outputs["gene_feat"]
                fused_feature = outputs["fused_feature"]
                encoder_out = outputs.get("encoder_out", {})
            else:
                # Legacy Fusion_MLP tuple path
                outputs = model(x, drug_feat, tissue, **kwargs)
                gene_feat = outputs[1]
                feat = outputs[2]
                fused_feature = feat.view(x.size(0), drug_num, -1)
                encoder_out = outputs[6] if len(outputs) > 6 else {}

            batch_size = x.size(0)
            fused_feature = reshape_fused_feature(fused_feature, batch_size, drug_num)

            feats["gene_feat"].append(gene_feat.cpu())
            feats["fused_feature"].append(fused_feature.cpu())
            feats["hidden"].append(encoder_out.get("hidden", gene_feat).cpu())
            feats["h_pool"].append(encoder_out.get("h", gene_feat).cpu())
            if "h_shared" in encoder_out:
                feats["h_shared"].append(encoder_out["h_shared"].cpu())
            if "h_private" in encoder_out:
                feats["h_private"].append(encoder_out["h_private"].cpu())
            feats["tissue"].append(tissue.cpu())

            if len(batch) > 4 and batch[4] is not None:
                feats["sample_id"].append(batch[4])

    result: Dict[str, np.ndarray] = {}
    for key, tensors in feats.items():
        if not tensors:
            continue
        if key == "sample_id":
            result[key] = np.array([str(s) for t in tensors for s in t])
        else:
            result[key] = torch.cat(tensors, dim=0).numpy()
    return result


def validate_feature_arrays(
    source: Dict[str, np.ndarray],
    target: Dict[str, np.ndarray],
    drug_names: List[str],
) -> None:
    """
    Validate extracted feature arrays before saving.

    Checks shapes, NaN/Inf, all-zero features, tissue counts, and sample-ID
    uniqueness.
    """
    print("\n=== Feature validation ===")
    n_src = source["gene_feat"].shape[0]
    n_tgt = target["gene_feat"].shape[0]
    d = len(drug_names)

    for name, feats in [("source", source), ("target", target)]:
        print(f"\n{name} samples: {feats['gene_feat'].shape[0]}")
        for key, arr in feats.items():
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
            if arr.dtype.kind == "f":
                if np.isnan(arr).any():
                    raise ValueError(f"{name}.{key} contains NaN.")
                if np.isinf(arr).any():
                    raise ValueError(f"{name}.{key} contains Inf.")
                if np.allclose(arr, 0.0):
                    print(f"  [Warning] {name}.{key} is all zeros.")
        if feats["tissue"].shape[0] != feats["gene_feat"].shape[0]:
            raise ValueError(
                f"{name}: tissue count ({feats['tissue'].shape[0]}) != "
                f"gene_feat count ({feats['gene_feat'].shape[0]})."
            )
        if feats["fused_feature"].shape[0] != feats["gene_feat"].shape[0]:
            raise ValueError(
                f"{name}: fused_feature batch dim ({feats['fused_feature'].shape[0]}) != "
                f"gene_feat batch dim ({feats['gene_feat'].shape[0]})."
            )
        if feats["fused_feature"].shape[1] != d:
            raise ValueError(
                f"{name}: fused_feature drug dim ({feats['fused_feature'].shape[1]}) != "
                f"len(drug_names)={d}."
            )
        if "sample_id" in feats and len(feats["sample_id"]) != feats["gene_feat"].shape[0]:
            raise ValueError(f"{name}: sample_id length mismatch.")

    print(f"\nDrug dimension: {d}")
    print(f"Source tissues: {np.unique(source['tissue'])}")
    print(f"Target tissues: {np.unique(target['tissue'])}")
    src_counts = pd.Series(source["tissue"]).value_counts().sort_index()
    tgt_counts = pd.Series(target["tissue"]).value_counts().sort_index()
    print("Source tissue counts:\n", src_counts)
    print("Target tissue counts:\n", tgt_counts)

    if "sample_id" in source:
        if len(set(source["sample_id"])) != len(source["sample_id"]):
            raise ValueError("Duplicate sample IDs detected in source domain.")
    if "sample_id" in target:
        if len(set(target["sample_id"])) != len(target["sample_id"]):
            raise ValueError("Duplicate sample IDs detected in target domain.")

    print("\nValidation passed.\n")


def build_dataloaders(
    args: argparse.Namespace,
    gex_features_df: Optional[pd.DataFrame] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, List[str]]:
    """
    Build source (train CCLE), optional validation CCLE, and target TCGA loaders.
    """
    set_seed(args.seed)
    if gex_features_df is None:
        gex_features_df = pd.read_csv(project_config.gex_feature_file, index_col=0)
    drugs = [d.strip() for d in args.drug_names.split(",") if d.strip()]

    labeled_iter = get_multi_labeled_dataloader(
        gex_features_df=gex_features_df,
        drug=drugs,
        seed=args.seed,
        batch_size=args.batch_size,
        ccle_measurement=args.measurement,
        threshold_gdsc=args.thres_gdsc,
        threshold_label=args.thres_label,
        n_splits=args.n_splits,
    )

    train_loader = None
    val_loader = None
    tcga_loader = None
    for fold_idx, (train_ccle, val_ccle, tcga) in enumerate(labeled_iter):
        if fold_idx == args.fold:
            train_loader = train_ccle
            val_loader = val_ccle
            tcga_loader = tcga
            break

    if train_loader is None:
        raise ValueError(f"Fold {args.fold} not found in {args.n_splits}-fold splitter.")

    return train_loader, val_loader, tcga_loader, drugs


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    raw_ckpt, state_dict, was_dp = load_checkpoint(args.checkpoint, str(device))
    print(f"Checkpoint type: {type(raw_ckpt).__name__}")
    if was_dp:
        print("Removed 'module.' prefix from DataParallel checkpoint.")

    cfg = infer_model_config(state_dict)
    cfg = merge_with_user_config(cfg, args.config)

    print("\nInferred model config:")
    for k, v in sorted(cfg.items()):
        if k in ("frag_fp", "frag_mask"):
            v = None if v is None else tuple(v.shape)
        print(f"  {k}: {v}")

    # Ensure config.tissue_map is populated before ProgramFusionMLP instantiation.
    gex_features_df = pd.read_csv(project_config.gex_feature_file, index_col=0)
    get_unlabeled_dataloaders(
        gex_features_df=gex_features_df,
        seed=args.seed,
        test_ratio=0.25,
        batch_size=args.batch_size,
    )

    encoder = build_encoder(cfg).to(device)
    model = build_fusion_model(encoder, cfg, state_dict, str(device), strict=args.strict)
    print("Model loaded successfully.")

    # Build dataloaders
    train_loader, val_loader, tcga_loader, drugs = build_dataloaders(args, gex_features_df=gex_features_df)
    drug_feat = torch.from_numpy(get_drug_feat(drugs).astype("float32")).to(device)
    print(f"Drug features shape: {tuple(drug_feat.shape)}")

    # Extract source features (domain_idx=0)
    source_loaders = [train_loader]
    if args.include_val_source and val_loader is not None:
        source_loaders.append(val_loader)

    source_feats_list = [
        extract_domain_features(model, loader, drug_feat, str(device), domain_idx=0, drug_num=len(drugs))
        for loader in source_loaders
    ]
    source: Dict[str, np.ndarray] = {}
    for key in source_feats_list[0]:
        arrs = [s[key] for s in source_feats_list]
        if key == "sample_id":
            source[key] = np.concatenate(arrs)
        else:
            source[key] = np.concatenate(arrs, axis=0)

    # Extract target features (domain_idx=1)
    target = extract_domain_features(
        model, tcga_loader, drug_feat, str(device), domain_idx=1, drug_num=len(drugs)
    )

    # Tissue map -> names
    tissue_map = project_config.tissue_map
    max_label = max(int(v) for v in tissue_map.values())
    tissue_names = [""] * (max_label + 1)
    for name, idx in tissue_map.items():
        tissue_names[int(idx)] = str(name)

    validate_feature_arrays(source, target, drugs)

    # Save NPZ
    output_path = os.path.join(args.output_dir, "fmpa_tsne_features.npz")
    save_dict = {
        "source_gene_feat": source["gene_feat"],
        "target_gene_feat": target["gene_feat"],
        "source_fused_feat": source["fused_feature"],
        "target_fused_feat": target["fused_feature"],
        "source_tissue": source["tissue"],
        "target_tissue": target["tissue"],
        "drug_names": np.asarray(drugs),
        "tissue_names": np.asarray(tissue_names),
        "fold": np.array(args.fold),
        "checkpoint": np.array(args.checkpoint),
    }
    if "hidden" in source:
        save_dict.update(
            source_hidden=source["hidden"],
            target_hidden=target["hidden"],
        )
    if "h_pool" in source:
        save_dict.update(
            source_h_pool=source["h_pool"],
            target_h_pool=target["h_pool"],
        )
    if "h_shared" in source:
        save_dict.update(
            source_h_shared=source["h_shared"],
            target_h_shared=target["h_shared"],
        )
    if "h_private" in source:
        save_dict.update(
            source_h_private=source["h_private"],
            target_h_private=target["h_private"],
        )
    if "sample_id" in source:
        save_dict.update(
            source_sample_id=source["sample_id"],
            target_sample_id=target["sample_id"],
        )

    np.savez_compressed(output_path, **save_dict)
    print(f"Saved features to: {output_path}")

    # Save a small human-readable summary
    summary_path = os.path.join(args.output_dir, "extraction_summary.json")
    summary = {
        "checkpoint": args.checkpoint,
        "fold": args.fold,
        "seed": args.seed,
        "device": str(device),
        "source_samples": int(source["gene_feat"].shape[0]),
        "target_samples": int(target["gene_feat"].shape[0]),
        "gene_feat_dim": int(source["gene_feat"].shape[1]),
        "fused_feature_shape": list(source["fused_feature"].shape),
        "drug_names": drugs,
        "tissue_count": len(tissue_names),
        "model_config": {k: (tuple(v.shape) if torch.is_tensor(v) else v)
                         for k, v in cfg.items() if k not in ("frag_fp", "frag_mask")},
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
