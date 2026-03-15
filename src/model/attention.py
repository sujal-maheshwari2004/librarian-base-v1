import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import precompute_rope_freqs, apply_rope


class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.dim, bias=False)

        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)

        cos, sin = precompute_rope_freqs(
            self.head_dim,
            config.max_seq_len,
            config.rope_theta
        )

        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):

        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, self.cos, self.sin)
        k = apply_rope(k, self.cos, self.sin)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )

        out = attn_out.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.out_proj(out))