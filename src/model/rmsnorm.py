import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    Used in LLaMA / modern GPT architectures
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):

        # x : (batch, seq, dim)

        norm = x.pow(2).mean(-1, keepdim=True)
        norm = torch.rsqrt(norm + self.eps)

        return x * norm * self.weight