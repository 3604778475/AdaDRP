import torch.nn as nn
from typing import TypeVar
from torch.nn import functional as F
from copy import deepcopy
from utility import *
from torch.autograd import Function
import config

Tensor = TypeVar('torch.tensor')

class FeatMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None, drop=0.1, act_fn=nn.SELU, **kwargs):
        super(FeatMLP, self).__init__()
        self.output_dim = output_dim
        self.drop = drop
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        hidden_dims = deepcopy(hidden_dims)
        hidden_dims.insert(0, input_dim)
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                    act_fn(),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.Dropout(self.drop))
            )
        self.module = nn.Sequential(*modules)
        self.output_layer = nn.Sequential(nn.Linear(hidden_dims[-1], output_dim, bias=True))
        self.reset_para()

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return

    def forward(self, inputs):
        embed = self.module(inputs)
        output = self.output_layer(embed)
        return output, embed

class Drug_MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None, drop=0.1, act_fn=nn.SELU, **kwargs):
        super(Drug_MLP, self).__init__()
        self.output_dim = output_dim
        self.drop = drop

        if hidden_dims is None:
            hidden_dims = [128, 64]
        hidden_dims = deepcopy(hidden_dims)
        hidden_dims.insert(0, input_dim)

        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                    act_fn(),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.Dropout(self.drop)
                )
            )

        self.module = nn.Sequential(*modules)
        self.output_layer = nn.Sequential(nn.Linear(hidden_dims[-1], output_dim, bias=True))

        self.reset_para()

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return

    def forward(self, inputs):
        embed = self.module(inputs)
        output = self.output_layer(embed)
        return output, embed

class Fusion_MLP(nn.Module):
    def __init__(self, encoder, input_dim, output_dim, hidden_dims = [128,64], gene_num = 1306, bilinear_dim = 128, gene_feat_dim = 64, drug_feat_dim = 64, drug_num = 9, drop=0.1, act_fn=nn.SELU, **kwargs):
        super(Fusion_MLP, self).__init__()
        self.encoder = encoder
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.drug_num = drug_num
        self.drop = drop

        self.encoder_input_dim = gene_num
        self.attention_masks = nn.Parameter(torch.ones(len(config.tissue_map), self.encoder_input_dim))
        self.attention_activation = nn.Sigmoid()

        self.drug_mlp = Drug_MLP(
            input_dim=128,
            output_dim=64,
            hidden_dims=[128, 64],
            drop=0.1
        )

        self.bilinear = nn.Bilinear(gene_feat_dim, drug_feat_dim, bilinear_dim)

        self.fusion_module = FeatMLP(input_dim, output_dim, hidden_dims, drop=drop, act_fn=act_fn)

        cdan_input_dim = hidden_dims[-1] + drug_num
        self.cdan_discriminator = nn.Sequential(
            nn.Linear(cdan_input_dim, cdan_input_dim // 2),
            nn.BatchNorm1d(cdan_input_dim // 2),
            nn.SELU(),
            nn.Dropout(drop),
            nn.Linear(cdan_input_dim // 2, cdan_input_dim // 4),
            nn.BatchNorm1d(cdan_input_dim // 4),
            nn.SELU(),
            nn.Dropout(drop),
            nn.Linear(cdan_input_dim // 4, 1)
        )

        self.drug_classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.BatchNorm1d(hidden_dims[-1] // 2),
            nn.SELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dims[-1] // 2, drug_num)
        )

    def forward(self, inputs, drug_feat, tissue_idx, alpha=1.0):
        if tissue_idx is None:
            attention_weights = self.attention_activation(self.attention_masks.mean(dim=0))
        else:
            attention_weights = self.attention_activation(self.attention_masks[tissue_idx])


        inputs = inputs * attention_weights
        _, _, gene_feat = self.encoder(inputs)
        gene_feat = F.normalize(gene_feat, p=2, dim=1)     #[B,D]

        drug_feat, _ = self.drug_mlp(drug_feat)      #[drug_num, D]
        drug_feat = F.normalize(drug_feat, p=2, dim=1)


        gene_feat_expand = gene_feat.unsqueeze(1).expand(-1, self.drug_num, -1)  # (B, 1, D)  [B, drug_num, 64]
        drug_feat_expand = drug_feat.unsqueeze(0).expand(gene_feat.size(0), -1, -1)  # (1, drug_num, D)  [B, drug_num, 64]

        z = self.bilinear(
            gene_feat_expand.reshape(-1, gene_feat_expand.size(-1)),   # [B*drug_num, 64]
            drug_feat_expand.reshape(-1, drug_feat_expand.size(-1))    # [B*drug_num, 64]
        )  # [B*drug_num, bilinear_dim]

        fusion_input = torch.cat([
            z,
            gene_feat_expand.reshape(-1, gene_feat_expand.size(-1)),
            drug_feat_expand.reshape(-1, drug_feat_expand.size(-1))
        ], dim=1)  # [B*drug_num, bilinear_dim+64+64]

        probs, feat = self.fusion_module(fusion_input)
        outputs = torch.sigmoid(probs)


        # CDAN
        if self.training:
            drug_pred = self.drug_classifier(feat)  # [B*drug_num, drug_num]
            drug_prob = F.softmax(drug_pred, dim=1)  # [B*drug_num, drug_num]

            conditional_feat = torch.cat([feat, drug_prob], dim=1)  # [B*drug_num, feat_dim + drug_num]

            reversed_conditional_feat = ReverseLayerF.apply(conditional_feat, alpha)
            cdan_domain_output = self.cdan_discriminator(reversed_conditional_feat)
        else:
            cdan_domain_output = None
            drug_pred = None

        if tissue_idx is not None:
            # tissue_idx: [batch]，expand to [batch * drug_num]
            expanded_tissue_idx = tissue_idx.repeat_interleave(self.drug_num)
        else:
            expanded_tissue_idx = None
        return outputs, gene_feat, feat, expanded_tissue_idx, cdan_domain_output, drug_pred

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
