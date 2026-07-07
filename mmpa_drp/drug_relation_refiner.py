"""
Drug Relation Refiner
=====================
Refine per-drug embeddings by propagating information along a drug-drug
relation graph (e.g., chemical similarity, co-occurrence, shared targets).

Compatible with ProgramFusionMLP: the refiner is applied after the base
drug encoder and before bilinear/FPI branches.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DrugRelationRefiner(nn.Module):
    """
    Simple GCN-style refiner over a fixed drug relation graph.

    Args:
        emb_dim: drug embedding dimension.
        num_layers: number of propagation layers.
        dropout: dropout on hidden representations.
        residual: whether to add residual connection.
        activation: 'elu' or 'relu'.
    """

    def __init__(
        self,
        emb_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        residual: bool = True,
        activation: str = "elu",
    ):
        super(DrugRelationRefiner, self).__init__()
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.residual = residual

        self.layers = nn.ModuleList(
            [nn.Linear(emb_dim, emb_dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if activation == "elu":
            self.activation = F.elu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        Args:
            x:   [D, emb_dim] drug embeddings.
            adj: [D, D] pre-normalized relation adjacency matrix.
        Returns:
            [D, emb_dim] refined embeddings.
        """
        for lin in self.layers:
            out = self.activation(lin(x))
            out = self.dropout(out)
            propagated = torch.matmul(adj, out)
            x = x + propagated if self.residual else propagated
        return x


def build_drug_relation_graph(
    smiles_list,
    graph_type: str = "chemical",
    k: int = 3,
    threshold: float = 0.3,
    fp_radius: int = 2,
    fp_bits: int = 1024,
    label_graph=None,
):
    """
    Build a normalized drug-drug relation adjacency matrix.

    Args:
        smiles_list: list of SMILES strings, length D.
        graph_type:
            - 'chemical': Tanimoto similarity on ECFP/Morgan fingerprints.
            - 'label_graph': use the provided (D, D) binary/weighted adjacency.
        k: for chemical graph, keep top-k neighbors per drug (excluding self).
        threshold: for chemical graph, also drop edges below this similarity.
        fp_radius, fp_bits: Morgan fingerprint parameters.
        label_graph: optional [D, D] numpy array for 'label_graph' mode.

    Returns:
        [D, D] float32 normalized adjacency matrix (with self-loops).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    D = len(smiles_list)
    if graph_type == "chemical":
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        for i, m in enumerate(mols):
            if m is None:
                raise ValueError(f"Invalid SMILES at index {i}: {smiles_list[i]}")
        fps = [
            AllChem.GetMorganFingerprintAsBitVect(m, radius=fp_radius, nBits=fp_bits)
            for m in mols
        ]

        sim = np.zeros((D, D), dtype=np.float64)
        for i in range(D):
            for j in range(i, D):
                s = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                sim[i, j] = s
                sim[j, i] = s

        # Build weighted adjacency: self-loop + top-k neighbors above threshold
        adj = np.eye(D, dtype=np.float64)
        for i in range(D):
            order = np.argsort(sim[i])[::-1]
            kept = 0
            for j in order:
                if j == i:
                    continue
                if sim[i, j] < threshold:
                    break
                adj[i, j] = sim[i, j]
                kept += 1
                if kept >= k:
                    break

        # Symmetrize
        adj = np.maximum(adj, adj.T)

    elif graph_type == "label_graph":
        if label_graph is None:
            raise ValueError("graph_type='label_graph' requires label_graph")
        adj = np.asarray(label_graph, dtype=np.float64).copy()
        np.fill_diagonal(adj, 1.0)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    degree = adj.sum(axis=1, keepdims=True)
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
    adj_norm = degree_inv_sqrt * adj * degree_inv_sqrt
    return adj_norm.astype(np.float32)
