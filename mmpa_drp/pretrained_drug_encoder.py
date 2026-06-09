"""
Pretrained Drug Encoders
========================
Lightweight wrappers around pretrained molecular representation models.
All base models are frozen; only a small projection MLP is trainable.

Supported models:
- ChemBERTa: RoBERTa pretrained on ZINC-250M SMILES (seyonec/ChemBERTa-zinc-base-v1)
- Mol2Vec: Word2Vec on Morgan fingerprints (requires external model file)
- Uni-Mol: 3D molecular Transformer (requires unimol/unicore, handled externally)

Recommended: ChemBERTa (best balance of quality and ease-of-use).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_transformers():
    try:
        from transformers import AutoModel, AutoTokenizer
        return AutoModel, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "'transformers' is required for pretrained drug encoders. "
            "Install with: pip install transformers"
        ) from e


class ChemBERTaDrugEncoder(nn.Module):
    """
    ChemBERTa-based drug encoder.

    Uses 'seyonec/ChemBERTa-zinc-base-v1' (RoBERTa trained on ZINC-250M)
    to encode SMILES strings. The BERT weights are frozen; only a small
    projection MLP (~131K params) is trainable.

    Args:
        smiles_list: list of SMILES strings (one per drug)
        output_dim: final drug embedding dimension (default 64)
        hidden_dim: projection MLP hidden dimension (default 256)
        dropout: dropout rate for projection MLP
        model_name: HuggingFace model identifier
    """

    def __init__(
        self,
        smiles_list,
        output_dim=64,
        hidden_dim=256,
        dropout=0.1,
        model_name="seyonec/ChemBERTa-zinc-base-v1",
    ):
        super(ChemBERTaDrugEncoder, self).__init__()
        self.smiles_list = smiles_list
        self.output_dim = output_dim

        AutoModel, AutoTokenizer = _check_transformers()

        # Load pretrained tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        # Freeze BERT permanently (we only fine-tune the projection head)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.bert.eval()

        bert_dim = self.bert.config.hidden_size  # 768 for ChemBERTa-base

        # Projection MLP: bert_dim -> hidden_dim -> output_dim
        self.projector = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self.reset_parameters()

        # Precompute BERT embeddings at init time (drugs are fixed)
        # This avoids running BERT on every forward pass during training.
        self._precompute_embeddings()

    def reset_parameters(self):
        for m in self.projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _precompute_embeddings(self):
        """Encode all SMILES once and store as buffer."""
        self.bert.eval()
        with torch.no_grad():
            embs = []
            for smi in self.smiles_list:
                inputs = self.tokenizer(
                    smi,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                # Move to same device as BERT (CPU is fine for one-time compute)
                outputs = self.bert(**inputs)
                # Use [CLS] token embedding (index 0)
                cls_emb = outputs.last_hidden_state[:, 0, :]  # [1, 768]
                embs.append(cls_emb)
            precomputed = torch.cat(embs, dim=0)  # [D, 768]
        self.register_buffer("_precomputed_emb", precomputed)

    def forward(self, dummy=None):
        """
        dummy: placeholder for interface compatibility with Drug_MLP.
        Returns: (output, embed) both [D, output_dim]
        """
        out = self.projector(self._precomputed_emb)  # [D, output_dim]
        return out, out


class ECFPMLPEncoder(nn.Module):
    """
    ECFP (Morgan fingerprint) + MLP encoder.
    Not a pretrained neural model, but ECFP is a strong classical baseline.
    Uses RDKit to generate 2048-bit Morgan fingerprints and projects them.

    Args:
        smiles_list: list of SMILES strings
        fp_dim: Morgan fingerprint dimension (default 2048)
        radius: Morgan fingerprint radius (default 2)
        output_dim: final embedding dimension (default 64)
        hidden_dim: MLP hidden dimension
        dropout: dropout rate
    """

    def __init__(
        self,
        smiles_list,
        fp_dim=2048,
        radius=2,
        output_dim=64,
        hidden_dim=512,
        dropout=0.1,
    ):
        super(ECFPMLPEncoder, self).__init__()
        self.smiles_list = smiles_list
        self.output_dim = output_dim

        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError as e:
            raise ImportError("RDKit is required for ECFP encoder.") from e

        # Precompute ECFP fingerprints
        fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smi}")
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=fp_dim)
            arr = torch.zeros(fp_dim, dtype=torch.float32)
            arr[list(fp.GetOnBits())] = 1.0
            fps.append(arr)
        precomputed = torch.stack(fps, dim=0)  # [D, fp_dim]
        self.register_buffer("_fp", precomputed)

        self.projector = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, dummy=None):
        out = self.projector(self._fp)  # [D, output_dim]
        return out, out


def build_pretrained_drug_encoder(
    smiles_list,
    encoder_type="chemberta",
    output_dim=64,
    **kwargs,
):
    """
    Factory function to build a pretrained drug encoder.

    Args:
        smiles_list: list of SMILES strings
        encoder_type: 'chemberta' | 'ecfp'
        output_dim: output embedding dimension
        **kwargs: passed to the specific encoder class
    """
    if encoder_type == "chemberta":
        return ChemBERTaDrugEncoder(
            smiles_list=smiles_list,
            output_dim=output_dim,
            **kwargs,
        )
    elif encoder_type == "ecfp":
        return ECFPMLPEncoder(
            smiles_list=smiles_list,
            output_dim=output_dim,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}. Choose 'chemberta' or 'ecfp'.")
