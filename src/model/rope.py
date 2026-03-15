import torch


def precompute_rope_freqs(dim: int, seq_len: int, theta: float = 10000.0):

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    t = torch.arange(seq_len)

    freqs = torch.outer(t, freqs)

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    return cos, sin


def apply_rope(x, cos, sin):

    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    cos = cos[: x.size(2)].unsqueeze(0).unsqueeze(0)
    sin = sin[: x.size(2)].unsqueeze(0).unsqueeze(0)

    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    out = torch.stack((out1, out2), dim=-1)
    out = out.flatten(-2)

    return out