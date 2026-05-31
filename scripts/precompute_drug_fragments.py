"""
Precompute drug fragment fingerprints using RDKit BRICS decomposition.

Output:
    data/drug_fragments_brics_m16_fp1024.npz
    outputs/drug_fragments_brics.csv
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS, AllChem

GDSC_DRUGS_PATH = "data/raw_dat/GDSC/drug_smiles.csv"


def precompute_drug_fragments(
    drug_names,
    smiles_dict,
    m_max=16,
    fp_dim=1024,
    radius=2,
    output_npz="data/drug_fragments_brics_m16_fp1024.npz",
    output_csv="outputs/drug_fragments_brics.csv",
):
    """
    Decompose drugs into BRICS fragments and compute Morgan fingerprints.

    Args:
        drug_names: list of drug names in desired order
        smiles_dict: dict mapping drug_name -> SMILES string
        m_max: max fragments per drug
        fp_dim: fingerprint dimension
        radius: Morgan fingerprint radius
        output_npz: path to save .npz file
        output_csv: path to save .csv file
    """
    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    D = len(drug_names)
    frag_fp = np.zeros((D, m_max, fp_dim), dtype=np.float32)
    frag_mask = np.zeros((D, m_max), dtype=np.float32)
    frag_smiles_arr = np.full((D, m_max), "", dtype=object)

    csv_rows = []

    for d_idx, drug_name in enumerate(drug_names):
        smiles = smiles_dict.get(drug_name)
        if smiles is None or pd.isna(smiles):
            warnings.warn(f"No SMILES found for {drug_name}, using fallback.")
            # fallback: empty entry will trigger whole-mol fallback below
            mol = None
        else:
            mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            warnings.warn(f"Failed to parse SMILES for {drug_name}: {smiles}")
            fragments = []
        else:
            try:
                frag_set = BRICS.BRICSDecompose(mol)
                fragments = list(frag_set)
            except Exception as e:
                warnings.warn(f"BRICS decomposition failed for {drug_name}: {e}")
                fragments = []

        # Fallback: use whole molecule as one fragment
        if len(fragments) == 0:
            if mol is not None:
                fragments = [smiles]
                warnings.warn(f"Drug {drug_name}: BRICS returned 0 fragments, using whole molecule as fallback.")
            else:
                warnings.warn(f"Drug {drug_name}: no valid molecule, all-zero fragments.")
                fragments = []

        # Compute heavy atom counts for sorting
        frag_info = []
        for f_smiles in fragments:
            f_mol = Chem.MolFromSmiles(f_smiles)
            if f_mol is not None:
                heavy_atoms = f_mol.GetNumHeavyAtoms()
                frag_info.append((f_smiles, f_mol, heavy_atoms))
            else:
                warnings.warn(f"Drug {drug_name}: failed to parse fragment {f_smiles}")

        # Sort by heavy atom count descending, keep top m_max
        frag_info.sort(key=lambda x: x[2], reverse=True)
        frag_info = frag_info[:m_max]

        for m_idx, (f_smiles, f_mol, heavy_atoms) in enumerate(frag_info):
            fp = AllChem.GetMorganFingerprintAsBitVect(f_mol, radius=radius, nBits=fp_dim)
            fp_vec = np.zeros((fp_dim,), dtype=np.float32)
            Chem.DataStructs.ConvertToNumpyArray(fp, fp_vec)
            frag_fp[d_idx, m_idx, :] = fp_vec
            frag_mask[d_idx, m_idx] = 1.0
            frag_smiles_arr[d_idx, m_idx] = f_smiles

            csv_rows.append({
                "drug_name": drug_name,
                "fragment_id": m_idx,
                "fragment_smiles": f_smiles,
                "is_valid": 1,
                "heavy_atom_count": heavy_atoms,
            })

        # Pad entries in CSV
        for m_idx in range(len(frag_info), m_max):
            csv_rows.append({
                "drug_name": drug_name,
                "fragment_id": m_idx,
                "fragment_smiles": "",
                "is_valid": 0,
                "heavy_atom_count": 0,
            })

    np.savez(
        output_npz,
        frag_fp=frag_fp,
        frag_mask=frag_mask,
        drug_names=np.array(drug_names, dtype=object),
        frag_smiles=frag_smiles_arr,
    )

    pd.DataFrame(csv_rows).to_csv(output_csv, index=False)
    print(f"Saved NPZ: {output_npz}")
    print(f"Saved CSV: {output_csv}")
    print(f"Drugs: {D}, M_max: {m_max}, FP_dim: {fp_dim}")


def main():
    parser = argparse.ArgumentParser(description="Precompute drug fragment fingerprints")
    parser.add_argument("--m_max", type=int, default=16, help="Max fragments per drug")
    parser.add_argument("--fp_dim", type=int, default=1024, help="Fingerprint dimension")
    parser.add_argument("--radius", type=int, default=2, help="Morgan fingerprint radius")
    parser.add_argument("--output_npz", type=str, default="data/drug_fragments_brics_m16_fp1024.npz")
    parser.add_argument("--output_csv", type=str, default="outputs/drug_fragments_brics.csv")
    args = parser.parse_args()

    # Read drug SMILES from the same source as dataload.py
    smiles_df = pd.read_csv(GDSC_DRUGS_PATH, header=0, index_col=0)
    smiles_df.index = smiles_df.index.map(str.lower)

    # Default 9 drugs used in the project
    drugs = [
        "5-Fluorouracil", "Cisplatin", "Cyclophosphamide",
        "Docetaxel", "Doxorubicin", "Etoposide",
        "Gemcitabine", "Paclitaxel", "Temozolomide",
    ]
    drug_names_lower = [d.lower() for d in drugs]

    smiles_dict = {}
    for d, d_lower in zip(drugs, drug_names_lower):
        if d_lower in smiles_df.index:
            smiles_dict[d] = smiles_df.loc[d_lower, "Isosmiles"]
        else:
            smiles_dict[d] = None
            warnings.warn(f"Drug {d} not found in {config.gdsc_drugs}")

    precompute_drug_fragments(
        drug_names=drugs,
        smiles_dict=smiles_dict,
        m_max=args.m_max,
        fp_dim=args.fp_dim,
        radius=args.radius,
        output_npz=args.output_npz,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
