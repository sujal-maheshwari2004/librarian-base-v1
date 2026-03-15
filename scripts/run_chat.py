import torch
from tokenizers import Tokenizer

from configs.load_configs import load_model_config
from src.inference.load_model import load_model
from src.inference.generate import generate


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_config = load_model_config("configs/model.json")

    tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

    model = load_model(
        model_config,
        "checkpoints/best/best_800.pt",
        device
    )

    print("\nChat started. Type 'exit' to quit.\n")

    while True:

        prompt = input(">> ")

        if prompt == "exit":
            break

        tokens = tokenizer.encode(prompt).ids

        idx = torch.tensor(tokens).unsqueeze(0).to(device)

        out = generate(model, idx, 100)

        text = tokenizer.decode(out[0].tolist())

        print(text)
        print()


if __name__ == "__main__":
    main()