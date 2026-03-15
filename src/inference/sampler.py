import torch
import torch.nn.functional as F


def sample_next_token(logits, temperature=1.0, top_k=None):

    logits = logits / temperature

    if top_k is not None:

        values, _ = torch.topk(logits, top_k)

        min_val = values[:, -1].unsqueeze(-1)

        logits = torch.where(
            logits < min_val,
            torch.full_like(logits, -float("inf")),
            logits
        )

    probs = F.softmax(logits, dim=-1)

    return torch.multinomial(probs, num_samples=1)