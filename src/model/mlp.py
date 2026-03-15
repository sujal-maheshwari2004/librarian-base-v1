import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.gate_proj = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, config.hidden_dim, bias=False)

        self.down_proj = nn.Linear(config.hidden_dim, config.dim, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):

        gate = self.gate_proj(x)
        up = self.up_proj(x)

        x = F.silu(gate) * up

        x = self.down_proj(x)

        return self.dropout(x)