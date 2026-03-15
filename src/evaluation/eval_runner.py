import argparse
import torch
from torch.utils.data import DataLoader

from configs.load_configs import load_model_config
from src.model.gpt import GPT
from src.training.checkpoint import load_checkpoint
from src.data.dataset import PackedDataset
from src.evaluation.perplexity import compute_perplexity


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate trained Librarian model"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    device = args.device

    print("=================================")
    print(" Librarian Evaluation")
    print("=================================")

    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Device: {device}")

    # ------------------------------------------------
    # Load model config
    # ------------------------------------------------

    model_config = load_model_config("configs/model_dummy.json")

    # ------------------------------------------------
    # Build model
    # ------------------------------------------------

    model = GPT(model_config).to(device)

    optimizer = None

    load_checkpoint(
        model,
        optimizer,
        args.checkpoint
    )

    model.eval()

    print("\nModel loaded.")

    # ------------------------------------------------
    # Load validation dataset
    # ------------------------------------------------

    val_dataset = PackedDataset(
        "data/tokenized/validation_packed.bin",
        model_config.max_seq_len
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True
    )

    print("Validation dataset loaded.")
    print(f"Validation samples: {len(val_dataset)}")

    # ------------------------------------------------
    # Compute perplexity
    # ------------------------------------------------

    print("\nRunning evaluation...")

    perplexity = compute_perplexity(
        model,
        val_loader,
        device
    )

    print("\n=================================")
    print(" Evaluation Results")
    print("=================================")

    print(f"Perplexity : {perplexity:.4f}")
    print("=================================")


if __name__ == "__main__":
    main()