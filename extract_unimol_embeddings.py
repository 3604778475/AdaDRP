"""
extract_unimol_embeddings.py
============================
Offline script to extract Uni-Mol embeddings for drugs used in AdaDRP.

Usage:
    python extract_unimol_embeddings.py --output data/drug_unimol_emb.npy

Requirements:
    pip install unimol-tools rdkit pandas numpy

Notes:
    - Uni-Mol encoder is NOT trained; we only use the official pretrained model.
    - The output .npy shape is [num_drugs, unimol_dim].
    - A drug_order.txt is saved alongside to verify drug ordering.
    - For metal-coordination drugs (e.g., Cisplatin), Uni-Mol featurization may fail.
      In that case, the script raises an error so you can provide a fallback SMILES.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Check unimol_tools
# ---------------------------------------------------------------------------
try:
    from unimol_tools import UniMolRepr
except ImportError as e:
    print(
        "[ERROR] unimol_tools is not installed.\n"
        "Please install it first:\n"
        "    pip install unimol-tools\n"
        "If you encounter dependency conflicts, try:\n"
        "    pip install unimol-tools --no-deps\n"
        "    pip install torch torchvision rdkit-pypi pandas numpy scikit-learn\n"
    )
    raise

# ---------------------------------------------------------------------------
# Drug list (must match training code)
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

# ---------------------------------------------------------------------------
# Fallback SMILES for problematic drugs (metal-coordination etc.)
# If Uni-Mol fails on a drug, you can add a sanitized SMILES here.
# ---------------------------------------------------------------------------
FALLBACK_SMILES = {
    # Example: "Cisplatin": "N.N.Cl[Pt]Cl",  # simplified representation
}


def load_smiles(csv_path, drug_list):
    """Load SMILES strings for the given drug list from drug_smiles.csv."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Drug SMILES file not found: {csv_path}")

    df = pd.read_csv(csv_path, header=0)
    # Expected columns: Drug_name, PubChem_ID, Isosmiles
    if "Drug_name" not in df.columns or "Isosmiles" not in df.columns:
        raise ValueError(
            f"CSV must contain 'Drug_name' and 'Isosmiles' columns. Got: {list(df.columns)}"
        )

    df = df.set_index("Drug_name")
    smiles_map = {}
    for drug in drug_list:
        if drug in FALLBACK_SMILES:
            smiles_map[drug] = FALLBACK_SMILES[drug]
            print(f"[Fallback] Using fallback SMILES for {drug}")
            continue
        if drug not in df.index:
            raise ValueError(f"Drug '{drug}' not found in {csv_path}")
        smiles = str(df.loc[drug, "Isosmiles"]).strip()
        if pd.isna(smiles) or smiles == "":
            raise ValueError(f"Empty SMILES for drug '{drug}'")
        smiles_map[drug] = smiles

    ordered_smiles = [smiles_map[d] for d in drug_list]
    return ordered_smiles


def extract_embeddings(smiles_list, model_name="unimolv1", remove_hs=False):
    """Extract Uni-Mol CLS token representations."""
    print(f"[Uni-Mol] Loading pretrained model: {model_name}")
    repr_model = UniMolRepr(
        data_type="molecule",
        remove_hs=remove_hs,
        model_name=model_name,
        use_gpu=True,
    )

    print(f"[Uni-Mol] Encoding {len(smiles_list)} molecules...")
    result = repr_model.get_repr(smiles_list, return_atomic_reprs=False)

    # cls_repr is a list of numpy arrays or a single array
    cls_repr = result.get("cls_repr")
    if cls_repr is None:
        raise RuntimeError("Uni-Mol returned no 'cls_repr'. Output keys: {}".format(list(result.keys())))

    # Ensure numpy array of shape [N, D]
    emb = np.array(cls_repr)
    if emb.ndim == 1 and len(smiles_list) == 1:
        emb = emb.reshape(1, -1)
    if emb.shape[0] != len(smiles_list):
        raise RuntimeError(
            f"Embedding count mismatch: expected {len(smiles_list)}, got {emb.shape[0]}"
        )

    print(f"[Uni-Mol] Embedding shape: {emb.shape}, dtype: {emb.dtype}")
    return emb


def main():
    parser = argparse.ArgumentParser(description="Extract Uni-Mol embeddings for AdaDRP drugs")
    parser.add_argument(
        "--drug_smiles_csv",
        type=str,
        default="data/raw_dat/GDSC/drug_smiles.csv",
        help="Path to drug_smiles.csv with Drug_name and Isosmiles columns",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/drug_unimol_emb.npy",
        help="Output path for .npy embedding file",
    )
    parser.add_argument(
        "--order_txt",
        type=str,
        default="data/drug_unimol_order.txt",
        help="Output path for drug order text file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="unimolv1",
        choices=["unimolv1", "unimolv2"],
        help="Uni-Mol model version",
    )
    parser.add_argument(
        "--remove_hs",
        action="store_true",
        help="Remove hydrogens before encoding (default: keep H)",
    )
    args = parser.parse_args()

    # 1. Load SMILES
    print("[Step 1/4] Loading drug SMILES...")
    smiles_list = load_smiles(args.drug_smiles_csv, DRUGS)
    for d, s in zip(DRUGS, smiles_list):
        print(f"  {d:20s} -> {s}")

    # 2. Extract embeddings
    print("\n[Step 2/4] Extracting Uni-Mol embeddings...")
    embeddings = extract_embeddings(smiles_list, model_name=args.model_name, remove_hs=args.remove_hs)

    # 3. Validate
    print("\n[Step 3/4] Validating outputs...")
    assert embeddings.ndim == 2, f"Expected 2D array, got shape {embeddings.shape}"
    assert embeddings.shape[0] == len(DRUGS), f"Drug count mismatch: {embeddings.shape[0]} vs {len(DRUGS)}"
    print(f"  Drugs : {embeddings.shape[0]}")
    print(f"  Dim   : {embeddings.shape[1]}")
    print(f"  Dtype : {embeddings.dtype}")

    # 4. Save
    print("\n[Step 4/4] Saving embeddings...")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.save(args.output, embeddings.astype(np.float32))
    print(f"  Saved: {args.output}")

    with open(args.order_txt, "w") as f:
        for i, drug in enumerate(DRUGS):
            f.write(f"{i}\t{drug}\t{smiles_list[i]}\n")
    print(f"  Saved: {args.order_txt}")

    print("\n[Done] Please verify the drug order matches your training pipeline.")
    print("       Expected order:")
    for i, d in enumerate(DRUGS):
        print(f"       [{i}] {d}")


if __name__ == "__main__":
    main()
