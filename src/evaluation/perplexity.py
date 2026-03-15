import torch
import torch.nn.functional as F


def compute_perplexity(model, dataloader, device):

    model.eval()

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():

        for x, y in dataloader:

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="sum"
            )

            total_loss += loss.item()
            total_tokens += y.numel()

    ppl = torch.exp(torch.tensor(total_loss / total_tokens))

    return ppl.item()