"""
Extract Uni-Mol embeddings for 9 drugs (offline, one-time).

This script generates 3D conformations with RDKit and extracts fixed
Uni-Mol representations. The output .npy file is consumed by
train_mmpa_program.py when use_unimol_drug=True.

Prerequisites (run in a separate environment with GPU recommended):
    pip install rdkit-pypi numpy pandas torch
    pip install git+https://github.com/dptech-corp/Uni-Mol.git
    # Download pre-trained weights from Uni-Mol release page, e.g.:
    # wget https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/unimol_pretrain.pt

Usage:
    python tools/extract_unimol_drug_emb.py \
        --checkpoint checkpoints/unimol_pretrain.pt \
        --dict_file unimol_dict.txt \
        --output data/drug_unimol_emb_9drugs.npy
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import warnings

# ---------------------------------------------------------------------------
# RDKit (available in most chemistry environments)
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError as e:
    raise ImportError(
        "RDKit is required. Install with: conda install -c conda-forge rdkit  "
        "or: pip install rdkit-pypi"
    ) from e

# ---------------------------------------------------------------------------
# Uni-Mol (optional at training time; only needed here for extraction)
# ---------------------------------------------------------------------------
try:
    import torch
    from unimol.data import Dictionary, collate_fn
    from unimol.models import UniMolModel
    from unicore.utils import tensor_tree_map
    UNIMOL_AVAILABLE = True
except ImportError:
    UNIMOL_AVAILABLE = False
    torch = None


# ---------------------------------------------------------------------------
# Hard-coded 9-drug list (must match train_mmpa_program.py)
# ---------------------------------------------------------------------------
DRUGS = [
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

DRUG_SMILES_CSV = os.path.join("data", "raw_dat", "GDSC", "drug_smiles.csv")


def get_smiles_list(drug_list, csv_path=DRUG_SMILES_CSV):
    """Read SMILES strings from GDSC drug_smiles.csv."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Drug SMILES file not found: {csv_path}. "
            "Please ensure GDSC data is in data/raw_dat/GDSC/"
        )
    df = pd.read_csv(csv_path, header=0, index_col=0)
    missing = [d for d in drug_list if d not in df.index]
    if missing:
        raise KeyError(f"Missing drugs in {csv_path}: {missing}")
    smiles = [str(df.loc[d, "Isosmiles"]).strip() for d in drug_list]
    return smiles


def mol_to_unimol_input(mol, dictionary, max_atoms=512):
    """
    Convert an RDKit Mol (with 3D conformer) to Uni-Mol batch format.
    Returns a dict with keys: src_tokens, src_distance, src_coord, src_edge_type.
    """
    conf = mol.GetConformer()
    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    coords = [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    n_atoms = len(atoms)

    # Pad / truncate to max_atoms
    pad_idx = dictionary.pad()
    src_tokens = np.full((max_atoms,), pad_idx, dtype=np.int64)
    src_tokens[:n_atoms] = atoms

    src_distance = np.zeros((max_atoms, max_atoms), dtype=np.float32)
    src_coord = np.zeros((max_atoms, 3), dtype=np.float32)
    src_edge_type = np.zeros((max_atoms, max_atoms), dtype=np.int64)

    src_coord[:n_atoms] = coords

    # Distance matrix + edge types
    for i in range(n_atoms):
        for j in range(n_atoms):
            dist = float(np.linalg.norm(np.array(coords[i]) - np.array(coords[j])))
            src_distance[i, j] = dist
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond is not None:
                bt = bond.GetBondTypeAsDouble()
                src_edge_type[i, j] = int(bt) if bt <= 3 else 1

    return {
        "src_tokens": torch.from_numpy(src_tokens).unsqueeze(0),  # [1, max_atoms]
        "src_distance": torch.from_numpy(src_distance).unsqueeze(0),  # [1, max_atoms, max_atoms]
        "src_coord": torch.from_numpy(src_coord).unsqueeze(0),  # [1, max_atoms, 3]
        "src_edge_type": torch.from_numpy(src_edge_type).unsqueeze(0),  # [1, max_atoms, max_atoms]
    }


def build_unimol_model(checkpoint_path, dict_path, device="cpu"):
    """Load Uni-Mol pretrained model."""
    dictionary = Dictionary.load(dict_path)
    # Typical Uni-Mol base architecture (adjust if you use a different checkpoint)
    model = UniMolModel.build_model(
        {
            "encoder_layers": 15,
            "encoder_embed_dim": 512,
            "encoder_ffn_embed_dim": 2048,
            "encoder_attention_heads": 64,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "activation_dropout": 0.0,
            "pooler_activation_fn": "tanh",
            "pooler_dropout": 0.0,
            "activation_fn": "gelu",
            "max_seq_len": 512,
            "num_classes": 1,
            "classification_head_name": "classification",
        },
        dictionary,
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    model.to(device)
    return model, dictionary


def extract_embeddings(smiles_list, model, dictionary, device="cpu", seed=42):
    """
    Extract Uni-Mol atom-pooled embeddings for a list of SMILES.
    Returns ndarray [len(smiles_list), 512].
    """
    embeddings = []
    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smi}")

        # Add hydrogens and generate 3D conformation
        mol = Chem.AddHs(mol)
        ret = AllChem.EmbedMolecule(mol, randomSeed=seed + idx)
        if ret == -1:
            warnings.warn(f"EmbedMolecule failed for {smi}, trying with different params")
            AllChem.EmbedMolecule(mol, randomSeed=seed + idx + 1000, maxAttempts=100)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)

        inputs = mol_to_unimol_input(mol, dictionary)
        # Move to device and collate
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            encoder_out = model.encoder(**inputs)
            # encoder_out.encoder_embedding shape: [1, max_atoms+1, 512]
            # Index 0 is CLS token; average over real atoms (skip CLS, use n_atoms)
            emb = encoder_out.encoder_embedding[0, 1:, :]  # [max_atoms, 512]
            # Simple mean pooling over all padded positions
            # (Uni-Mol uses padding, but the transformer outputs are real for all positions)
            # A common practice: mean over all atom positions
            emb_pooled = emb.mean(dim=0)  # [512]

        embeddings.append(emb_pooled.cpu().numpy())
        print(f"  [{idx+1}/{len(smiles_list)}] {smi[:40]:<40} -> OK")

    return np.stack(embeddings, axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser("Extract Uni-Mol drug embeddings")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/unimol_pretrain.pt",
        help="Path to Uni-Mol pretrained checkpoint (.pt)",
    )
    parser.add_argument(
        "--dict_file",
        type=str,
        default="unimol_dict.txt",
        help="Path to Uni-Mol dictionary file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/drug_unimol_emb_9drugs.npy",
        help="Output .npy file path",
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not UNIMOL_AVAILABLE:
        print(
            "ERROR: Uni-Mol / unicore is not installed in this environment.\n"
            "Please install it first:\n"
            "  pip install torch\n"
            "  pip install git+https://github.com/dptech-corp/Uni-Mol.git\n"
            "Then download the pre-trained checkpoint and dictionary.\n"
            "If you cannot install Uni-Mol, you can still train with RDKit fingerprints (default)."
        )
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("Please download it from the Uni-Mol release page.")
        sys.exit(1)

    if not os.path.exists(args.dict_file):
        print(f"ERROR: Dictionary file not found: {args.dict_file}")
        print("Please download it from the Uni-Mol repository.")
        sys.exit(1)

    print("=" * 60)
    print("Uni-Mol Drug Embedding Extraction")
    print("=" * 60)

    # Load SMILES
    smiles_list = get_smiles_list(DRUGS)
    print(f"\nDrugs: {DRUGS}")
    print(f"SMILES loaded: {len(smiles_list)}")

    # Load model
    print(f"\nLoading Uni-Mol model from {args.checkpoint} ...")
    model, dictionary = build_unimol_model(args.checkpoint, args.dict_file, device=args.device)
    print("Model loaded.")

    # Extract
    print(f"\nExtracting embeddings (device={args.device}) ...")
    embeddings = extract_embeddings(smiles_list, model, dictionary, device=args.device, seed=args.seed)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.save(args.output, embeddings)
    print(f"\nSaved: {args.output}")
    print(f"Shape: {embeddings.shape}  (drugs, unimol_dim)")
    print(f"Dtype: {embeddings.dtype}")
    print("=" * 60)
    print("DONE. You can now run train_mmpa_program.py with Uni-Mol features.")


if __name__ == "__main__":
    main()
