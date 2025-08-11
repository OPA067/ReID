from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

class MHAModel_v2(nn.Module):
    def __init__(self, embed_dim=768, n_head=8, hidden_dim=768, pos_index=0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_head)
        self.out_layer = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_dim),
            )
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(embed_dim, embed_dim * 4)),
                    ("gelu", nn.GELU()),
                    ("c_proj", nn.Linear(embed_dim * 4, embed_dim)),
                ]
            )
        )
        self.pos_index = pos_index
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.ln_3 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """x: patch feats \in [B, N_P=192+1, D]"""
        cls = x[:, 0, :]
        cls = self.linear(cls)
        cls = self.ln_1(cls)
        cls = cls.unsqueeze(1)

        cls = cls.permute(1, 0, 2)
        patch = self.ln_2(x).permute(1, 0, 2)
        x = cls + self.attn(cls, patch, patch)[0]
        x = x + self.mlp(self.ln_3(x))
        x = x.permute(1, 0, 2)
        x = self.out_layer(x).squeeze(1)
        return x