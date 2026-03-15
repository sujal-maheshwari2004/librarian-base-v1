import torch

from .generate import generate


def chat(model, tokenizer):

    device = next(model.parameters()).device

    while True:

        prompt = input(">> ")

        tokens = tokenizer.encode(prompt).ids

        idx = torch.tensor(tokens).unsqueeze(0).to(device)

        out = generate(model, idx, 100)

        decoded = tokenizer.decode(out[0].tolist())

        print(decoded)