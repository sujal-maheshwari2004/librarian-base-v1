import os
import time
import torch
from torch.utils.data import DataLoader

from configs.load_configs import load_model_config, load_train_config
from src.model.gpt import GPT
from src.data.dataset import PackedDataset
from src.training.trainer import Trainer
from src.utils.logging import StageLogger

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    # Read the shared RUN_ID set by train.ps1 so all pipeline stages
    # appear under the same run. Falls back to a new timestamp when
    # running train.py standalone (outside the full pipeline).
    run_id = int(os.environ.get("RUN_ID", 0)) or int(time.time())

    stage_log = StageLogger(run_id=run_id)

    # ── load configs ────────────────────────────────────────
    model_config = load_model_config("configs/model_130M.json")
    train_config = load_train_config("configs/train_130M.json")

    # ── build model ─────────────────────────────────────────
    model = GPT(model_config)

    # ── datasets ────────────────────────────────────────────
    train_dataset = PackedDataset(
        "data/tokenized/train_packed.bin",
        model_config.max_seq_len
    )
    val_dataset = PackedDataset(
        "data/tokenized/validation_packed.bin",
        model_config.max_seq_len
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        num_workers=4,
        pin_memory=True,
    )

    # ── training stage ──────────────────────────────────────
    stage_log.start("train")
    stage_log.progress("train", {
        "model_params":  sum(p.numel() for p in model.parameters()),
        "train_samples": len(train_dataset),
        "val_samples":   len(val_dataset),
        "total_steps":   train_config.total_steps,
        "batch_size":    train_config.batch_size,
    })

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        train_config,
        device=train_config.device,
        run_id=run_id,
    )

    try:
        trainer.train()
        stage_log.end("train", {
            "steps_completed": trainer.step,
            "best_val_loss":   round(trainer.best_val_loss, 4),
        })
    except Exception as e:
        stage_log.error("train", str(e))
        raise


if __name__ == "__main__":
    main()