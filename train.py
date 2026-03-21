import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader

from configs.load_configs import load_model_config, load_train_config
from src.model.gpt import GPT
from src.data.dataset import PackedDataset
from src.training.trainer import Trainer
from src.training.checkpoint import load_checkpoint
from src.utils.logging import StageLogger

_RUN_ID = int(os.environ.get("RUN_ID", 0)) or None


def parse_args():
    parser = argparse.ArgumentParser(description="Train Librarian Base v1")
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model_130M.json",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="configs/train_130M.json",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_config = load_model_config(args.model_config)
    train_config = load_train_config(args.train_config)

    device = train_config.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"

    print("=================================")
    print(" Librarian Base v1 Training")
    print("=================================")
    print(f"Model config : {args.model_config}")
    print(f"Train config : {args.train_config}")
    print(f"Device       : {device}")
    print(f"Resume       : {args.resume or 'none'}")

    stage_log = StageLogger(run_id=_RUN_ID)
    stage_log.start("train")
    stage_log.progress("train", {
        "model_config": args.model_config,
        "train_config": args.train_config,
        "device":       device,
        "resume":       args.resume or "none",
    })

    # ── model ──────────────────────────────────────────────
    model = GPT(model_config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters   : {n_params:,}  ({n_params/1e6:.1f}M)")

    # ── datasets ───────────────────────────────────────────
    train_dataset = PackedDataset(
        "data/tokenized/train_packed.bin",
        model_config.max_seq_len,
    )
    val_dataset = PackedDataset(
        "data/tokenized/validation_packed.bin",
        model_config.max_seq_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples  : {len(val_dataset):,}")

    stage_log.progress("train", {
        "train_samples": len(train_dataset),
        "val_samples":   len(val_dataset),
        "n_params_M":    round(n_params / 1e6, 1),
    })

    # ── trainer ────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_config=train_config,
        device=device,
        run_id=_RUN_ID,
    )

    # ── resume ─────────────────────────────────────────────
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        step = load_checkpoint(model, trainer.optimizer, args.resume)
        trainer.step = step
        trainer.progress.update(step)
        stage_log.progress("train", {"resumed_from_step": step})

    # ── train ──────────────────────────────────────────────
    try:
        trainer.train()
        stage_log.end("train", {
            "final_step":      trainer.step,
            "best_val_loss":   round(trainer.best_val_loss, 4),
        })
    except Exception as e:
        stage_log.error("train", str(e))
        raise


if __name__ == "__main__":
    main()
