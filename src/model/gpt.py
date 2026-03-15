import torch
import torch.nn as nn

from .block import TransformerBlock
from .rmsnorm import RMSNorm


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.token_emb = nn.Embedding(
            config.vocab_size,
            config.dim
        )

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.norm = RMSNorm(config.dim)

        self.lm_head = nn.Linear(
            config.dim,
            config.vocab_size,
            bias=False
        )

        if config.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):

        x = self.token_emb(idx)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        logits = self.lm_head(x)

        return logits