"""
Pure PyTorch Graph Transformer Drug Encoder
=============================================
Aligned with GraTransDRP's drug encoder but without PyTorch Geometric dependency.
Processes molecular graphs with Transformer + GAT + GCN.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Lazy import RDKit to avoid mandatory dependency at module load time
try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"Input {x} not in allowable set {allowable_set}")
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom):
    """
    78-dim atom feature vector (same as GraTransDRP).
    Includes: atom type (44), degree (11), total Hs (11), implicit valence (11), aromatic (1).
    """
    return np.array(
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                "C",
                "N",
                "O",
                "S",
                "F",
                "Si",
                "P",
                "Cl",
                "Br",
                "Mg",
                "Na",
                "Ca",
                "Fe",
                "As",
                "Al",
                "I",
                "B",
                "V",
                "K",
                "Tl",
                "Yb",
                "Sb",
                "Sn",
                "Ag",
                "Pd",
                "Co",
                "Se",
                "Ti",
                "Zn",
                "H",
                "Li",
                "Ge",
                "Cu",
                "Au",
                "Ni",
                "Cd",
                "In",
                "Mn",
                "Zr",
                "Cr",
                "Pt",
                "Hg",
                "Pb",
                "Unknown",
            ],
        )
        + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + [atom.GetIsAromatic()]
    )


def smiles_to_graph(smiles):
    """
    Convert SMILES to molecular graph with normalized adjacency matrix.
    Returns dict with 'x' [N, 78] and 'adj' [N, N].
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for graph construction. Install with: pip install rdkit-pypi")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Node features
    features = [atom_features(atom) for atom in mol.GetAtoms()]
    x = np.array(features, dtype=np.float32)  # [N, 78]

    # Adjacency matrix (undirected)
    N = x.shape[0]
    adj = np.zeros((N, N), dtype=np.float32)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    # Add self-loops
    adj += np.eye(N, dtype=np.float32)

    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    degree = adj.sum(axis=1)
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(degree_inv_sqrt)
    adj_normalized = D_inv_sqrt.dot(adj).dot(D_inv_sqrt)

    return {
        "x": torch.from_numpy(x),  # [N, 78]
        "adj": torch.from_numpy(adj_normalized).float(),  # [N, N]
    }


class GCNLayer(nn.Module):
    """Pure PyTorch GCN layer: X' = D^{-1/2} A D^{-1/2} X W."""

    def __init__(self, in_dim, out_dim):
        super(GCNLayer, self).__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, x, adj):
        # x: [N, in_dim], adj: [N, N] (pre-normalized)
        support = self.W(x)  # [N, out_dim]
        output = torch.matmul(adj, support)  # [N, out_dim]
        return F.relu(output)


class GATLayer(nn.Module):
    """
    Pure PyTorch GAT layer (multi-head).
    Input: [N, in_dim], adj: [N, N]
    Output: [N, out_dim * num_heads]  (concatenated across heads)

    Args:
        out_dim: output dimension PER HEAD (same semantics as PyG GATConv)
    """

    def __init__(self, in_dim, out_dim, num_heads=10, dropout=0.0):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim  # per-head dimension

        # Total output dimension = out_dim * num_heads (same as PyG GATConv)
        self.W = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.a = nn.Parameter(torch.zeros(num_heads, 2 * self.head_dim))
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x, adj):
        # x: [N, in_dim], adj: [N, N]
        N = x.size(0)
        Wh = self.W(x)  # [N, out_dim * num_heads]
        Wh = Wh.view(N, self.num_heads, self.head_dim)  # [N, H, d]

        # Compute attention coefficients e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        Wh_i = Wh.unsqueeze(1).expand(N, N, self.num_heads, self.head_dim)  # [N, N, H, d]
        Wh_j = Wh.unsqueeze(0).expand(N, N, self.num_heads, self.head_dim)  # [N, N, H, d]
        a_input = torch.cat([Wh_i, Wh_j], dim=-1)  # [N, N, H, 2d]

        e = self.leakyrelu(torch.einsum("ijhd,hd->ijh", a_input, self.a))  # [N, N, H]

        # Mask non-edges
        mask = adj.unsqueeze(-1).expand(-1, -1, self.num_heads)  # [N, N, H]
        e = e.masked_fill(mask == 0, float("-inf"))

        # Softmax normalization over neighbors (dim=1)
        alpha = F.softmax(e, dim=1)  # [N, N, H]
        alpha = self.dropout(alpha)

        # Aggregate neighbors
        out = torch.einsum("ijh,jhd->ihd", alpha, Wh)  # [N, H, d]
        out = out.reshape(N, -1)  # [N, H*d]

        return F.elu(out)


class GraphTransformerDrugEncoder(nn.Module):
    """
    Pure PyTorch Graph Transformer drug encoder aligned with GraTransDRP.

    Architecture:
        UGformer-1 (Transformer) -> GAT (10 heads) ->
        UGformer-2 (Transformer) -> GCN ->
        Global Mean/Max Pool -> MLP -> [output_dim]

    Processes each drug's molecular graph independently (suitable for small fixed drug sets).
    """

    def __init__(
        self,
        smiles_list,
        num_features_xd=78,
        n_heads_gat=10,
        output_dim=64,
        dropout=0.2,
        freeze_gnn=True,
    ):
        super(GraphTransformerDrugEncoder, self).__init__()
        self.smiles_list = smiles_list
        self.output_dim = output_dim
        self.freeze_gnn = freeze_gnn

        # Preprocess molecular graphs (fixed set of drugs)
        self.graphs = [smiles_to_graph(s) for s in smiles_list]

        # UGformer Layer 1
        encoder_layer_1 = nn.TransformerEncoderLayer(
            d_model=num_features_xd,
            nhead=1,
            dropout=0.5,
            dim_feedforward=num_features_xd * 4,
            batch_first=True,
        )
        self.ugformer_1 = nn.TransformerEncoder(encoder_layer_1, num_layers=1)

        # GAT Layer
        self.gat = GATLayer(
            num_features_xd,
            num_features_xd,
            num_heads=n_heads_gat,
            dropout=dropout,
        )

        # UGformer Layer 2
        gat_out_dim = num_features_xd * n_heads_gat
        encoder_layer_2 = nn.TransformerEncoderLayer(
            d_model=gat_out_dim,
            nhead=1,
            dropout=0.5,
            dim_feedforward=gat_out_dim * 4,
            batch_first=True,
        )
        self.ugformer_2 = nn.TransformerEncoder(encoder_layer_2, num_layers=1)

        # GCN Layer
        self.gcn = GCNLayer(gat_out_dim, gat_out_dim)

        # Pooling + MLP
        pool_dim = gat_out_dim * 2  # mean + max concatenation
        self.fc_g1 = nn.Linear(pool_dim, 1500)
        self.fc_g2 = nn.Linear(1500, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

        # Freeze GNN layers if requested (recommended for small drug sets)
        if freeze_gnn:
            self._freeze_gnn_layers()
            print(f"[GraphTransformerDrugEncoder] GNN layers frozen. "
                  f"Only fc_g1/fc_g2 projection layers are trainable. "
                  f"Trainable params: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        else:
            total = sum(p.numel() for p in self.parameters())
            print(f"[GraphTransformerDrugEncoder] GNN layers unfrozen. "
                  f"Total trainable params: {total:,}")

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _freeze_gnn_layers(self):
        """Freeze Transformer, GAT, and GCN layers; only keep MLP projection trainable."""
        gnn_modules = [self.ugformer_1, self.gat, self.ugformer_2, self.gcn]
        for module in gnn_modules:
            for param in module.parameters():
                param.requires_grad = False

    def encode_single(self, x, adj):
        """
        Encode a single molecular graph.
        x: [N, 78], adj: [N, N]
        Returns: [1, output_dim]
        """
        # UGformer 1: [N, 78] -> [1, N, 78] -> [N, 78]
        x = x.unsqueeze(0)
        x = self.ugformer_1(x)
        x = x.squeeze(0)

        # GAT: [N, 78] -> [N, 780]
        x = self.gat(x, adj)

        # UGformer 2: [N, 780] -> [1, N, 780] -> [N, 780]
        x = x.unsqueeze(0)
        x = self.ugformer_2(x)
        x = x.squeeze(0)

        # GCN: [N, 780] -> [N, 780]
        x = self.gcn(x, adj)

        # Global pooling: mean + max
        x_mean = x.mean(dim=0, keepdim=True)  # [1, 780]
        x_max = x.max(dim=0, keepdim=True)[0]  # [1, 780]
        x = torch.cat([x_mean, x_max], dim=-1)  # [1, 1560]

        # MLP
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)  # [1, output_dim]

        return x

    def forward(self, dummy=None):
        """
        Encode all preprocessed drug graphs.
        dummy: placeholder for interface compatibility with Drug_MLP.
        Returns: (output, embed) both [D, output_dim]
        """
        device = next(self.parameters()).device
        embs = []
        for g in self.graphs:
            x = g["x"].to(device)
            adj = g["adj"].to(device)
            emb = self.encode_single(x, adj)
            embs.append(emb)
        out = torch.cat(embs, dim=0)  # [D, output_dim]
        return out, out
