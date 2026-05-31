"""
Utility functions for loading and managing drug fragment data.
"""

import numpy as np
import torch


def load_drug_fragments(npz_path, expected_drug_names=None):
    """
    Load precomputed drug fragment data from .npz file.

    Args:
        npz_path: path to the .npz file
        expected_drug_names: optional list of drug names to validate/reorder against

    Returns:
        dict with keys:
            frag_fp: torch.FloatTensor [D, M, F]
            frag_mask: torch.FloatTensor [D, M]
            frag_smiles: list[list[str]]
            drug_names: list[str]
    """
    data = np.load(npz_path, allow_pickle=True)

    frag_fp = torch.from_numpy(data["frag_fp"].astype(np.float32))
    frag_mask = torch.from_numpy(data["frag_mask"].astype(np.float32))
    drug_names = list(data["drug_names"])
    frag_smiles_arr = data["frag_smiles"]

    # Convert frag_smiles to list of lists
    frag_smiles = []
    for d_idx in range(len(drug_names)):
        smiles_row = []
        for m_idx in range(frag_smiles_arr.shape[1]):
            s = frag_smiles_arr[d_idx, m_idx]
            if s is not None and str(s) != "":
                smiles_row.append(str(s))
            else:
                smiles_row.append("")
        frag_smiles.append(smiles_row)

    # Validate and reorder if expected_drug_names provided
    if expected_drug_names is not None:
        if set(expected_drug_names) != set(drug_names):
            missing = set(expected_drug_names) - set(drug_names)
            extra = set(drug_names) - set(expected_drug_names)
            raise ValueError(
                f"Drug name mismatch. Missing in npz: {missing}, Extra in npz: {extra}"
            )
        if expected_drug_names != drug_names:
            # Reorder to match expected
            name_to_idx = {name: i for i, name in enumerate(drug_names)}
            reorder_idx = [name_to_idx[name] for name in expected_drug_names]
            frag_fp = frag_fp[reorder_idx]
            frag_mask = frag_mask[reorder_idx]
            frag_smiles = [frag_smiles[i] for i in reorder_idx]
            drug_names = [drug_names[i] for i in reorder_idx]

    return {
        "frag_fp": frag_fp,
        "frag_mask": frag_mask,
        "frag_smiles": frag_smiles,
        "drug_names": drug_names,
    }
