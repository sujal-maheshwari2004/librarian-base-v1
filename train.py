import argparse
import os
import torch
from torch.utils.data import DataLoader

from configs.load_configs import load_model_config, load_train_config
from src.model.gpt import GPT
from src.training.trainer import Trainer
from src.data.dataset import PackedDataset
from src.training.checkpoint import load_checkpoint


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", required=True)
    parser.add_argument("--train_config", required=True)
    parser.add_argument("--resume", default=None)

    args = parser.parse_args()

    # ─────────────────────────────────────
    # Load configs
    # ─────────────────────────────────────
    model_config = load_model_config(args.model_config)
    train_config = load_train_config(args.train_config)

    device = train_config.device if torch.cuda.is_available() else "cpu"

    # ─────────────────────────────────────
    # Model
    # ─────────────────────────────────────
    model = GPT(model_config).to(device)

    # ─────────────────────────────────────
    # Resume
    # ─────────────────────────────────────
    start_step = 0
    optimizer = None

    if args.resume:
        print(f"Resuming from {args.resume}")
        start_step = load_checkpoint(model, optimizer, args.resume)

    # ─────────────────────────────────────
    # Dataset
    # ─────────────────────────────────────
    train_dataset = PackedDataset(
        "data/tokenized/train_packed.bin",
        model_config.max_seq_len
    )

    val_dataset = PackedDataset(
        "data/tokenized/validation_packed.bin",
        model_config.max_seq_len
    )

    if len(train_dataset) <= 0:
        raise ValueError("Train dataset is empty")

    if len(val_dataset) <= 0:
        raise ValueError("Validation dataset is empty")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ─────────────────────────────────────
    # Trainer
    # ─────────────────────────────────────
    run_id = int(os.environ.get("RUN_ID", 0)) or None

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        train_config,
        device,
        run_id=run_id
    )

    trainer.step = start_step

    # ─────────────────────────────────────
    # Train
    # ─────────────────────────────────────
    trainer.train()


if __name__ == "__main__":
    main()
