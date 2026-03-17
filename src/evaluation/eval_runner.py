import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
import argparse
import torch
from torch.utils.data import DataLoader

from configs.load_configs import load_model_config
from src.model.gpt import GPT
from src.training.checkpoint import load_checkpoint
from src.data.dataset import PackedDataset
from src.evaluation.perplexity import compute_perplexity
from src.utils.logging import StageLogger


def main():

    parser = argparse.ArgumentParser(description="Evaluate trained Librarian model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Share the same run_id as the rest of the pipeline
    run_id = int(os.environ.get("RUN_ID", 0)) or None
    stage_log = StageLogger(run_id=run_id)

    stage_log.start("evaluate")
    stage_log.progress("evaluate", {
        "checkpoint": Path(args.checkpoint).name,
        "device":     args.device,
    })

    print("=================================")
    print(" Librarian Evaluation")
    print("=================================")
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Device:     {args.device}")

    try:
        # ── model ──────────────────────────────────────────
        model_config = load_model_config("configs/model_dummy.json")
        model        = GPT(model_config).to(args.device)
        load_checkpoint(model, None, args.checkpoint)
        model.eval()
        print("\nModel loaded.")

        # ── dataset ────────────────────────────────────────
        val_dataset = PackedDataset(
            "data/tokenized/validation_packed.bin",
            model_config.max_seq_len
        )
        val_loader = DataLoader(
            val_dataset, batch_size=32,
            num_workers=4, pin_memory=True
        )
        print(f"Validation samples: {len(val_dataset)}")

        stage_log.progress("evaluate", {
            "val_samples": len(val_dataset),
            "status":      "computing perplexity",
        })

        # ── perplexity ─────────────────────────────────────
        print("\nRunning evaluation...")
        perplexity = compute_perplexity(model, val_loader, args.device)

        print("\n=================================")
        print(" Evaluation Results")
        print("=================================")
        print(f"Perplexity : {perplexity:.4f}")
        print("=================================")

        stage_log.end("evaluate", {
            "perplexity":  round(float(perplexity), 4),
            "checkpoint":  Path(args.checkpoint).name,
        })

    except Exception as e:
        stage_log.error("evaluate", str(e))
        raise


if __name__ == "__main__":
    main()