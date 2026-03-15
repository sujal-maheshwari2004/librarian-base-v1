import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    LoRA injected Linear layer.

    W_out = Wx + BAx
    """

    def __init__(
        self,
        linear_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.linear = linear_layer

        in_features = linear_layer.in_features
        out_features = linear_layer.out_features

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(
            torch.zeros(rank, in_features)
        )

        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank)
        )

        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

        # freeze base weights
        for param in self.linear.parameters():
            param.requires_grad = False

    def reset_parameters(self):

        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):

        result = self.linear(x)

        lora = self.dropout(x) @ self.lora_A.T
        lora = lora @ self.lora_B.T

        return result + lora * self.scaling
    
def inject_lora(model, rank=8, alpha=16, dropout=0.0):

    for name, module in model.named_modules():

        if isinstance(module, nn.Linear):

            parent = model
            *path, last = name.split(".")

            for p in path:
                parent = getattr(parent, p)

            setattr(
                parent,
                last,
                LoRALinear(module, rank, alpha, dropout)
            )

    return model

def enable_bitfit(model):

    for name, param in model.named_parameters():

        if "bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model


def print_trainable_parameters(model):

    trainable = 0
    total = 0

    for p in model.parameters():
        total += p.numel()

        if p.requires_grad:
            trainable += p.numel()

    percent = 100 * trainable / total

    print(
        f"Trainable params: {trainable:,} / {total:,} ({percent:.2f}%)"
    )

def get_lora_state_dict(model):

    lora_state = {}

    for name, param in model.named_parameters():
        if "lora_" in name:
            lora_state[name] = param.detach().cpu()

    return lora_state