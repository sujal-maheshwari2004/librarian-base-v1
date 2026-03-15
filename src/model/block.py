import torch.nn as nn

from .attention import SelfAttention
from .mlp import MLP
from .rmsnorm import RMSNorm


class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.attn_norm = RMSNorm(config.dim)
        self.attn = SelfAttention(config)

        self.mlp_norm = RMSNorm(config.dim)
        self.mlp = MLP(config)

    def forward(self, x):

        x = x + self.attn(self.attn_norm(x))

        x = x + self.mlp(self.mlp_norm(x))

        return x