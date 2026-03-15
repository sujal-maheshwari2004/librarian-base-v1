import torch

from .sampler import sample_next_token


def generate(
    model,
    idx,
    max_new_tokens,
    temperature=1.0,
    top_k=50
):

    device = idx.device

    for _ in range(max_new_tokens):

        idx_cond = idx[:, -model.config.max_seq_len:]

        logits = model(idx_cond)

        logits = logits[:, -1, :]

        next_token = sample_next_token(
            logits,
            temperature,
            top_k
        )

        idx = torch.cat((idx, next_token), dim=1)

    return idx
