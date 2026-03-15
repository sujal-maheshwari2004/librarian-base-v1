import argparse
import torch

from configs.load_configs import load_model_config
from src.inference.load_model import load_model
from src.inference.generate import generate
from tokenizers import Tokenizer


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best/best_800.pt"
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_config = load_model_config("configs/model.json")

    tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

    model = load_model(
        model_config,
        args.checkpoint,
        device
    )

    tokens = tokenizer.encode(args.prompt).ids

    idx = torch.tensor(tokens).unsqueeze(0).to(device)

    out = generate(model, idx, max_new_tokens=100)

    text = tokenizer.decode(out[0].tolist())

    print("\n=== Output ===\n")
    print(text)


if __name__ == "__main__":
    main()