"""
pipeline.py — Orchestrates the full preprocessing pipeline.

Stages run in order:
  download → clean → tokenize → pack

Each stage:
  1. Checks its input manifest is complete before starting
  2. Writes progress to its own manifest (shard-level)
  3. Validates outputs before deleting previous stage
  4. Is safe to restart mid-stage

Usage
-----
    # Full pipeline
    python -m src.pipeline.pipeline

    # Start from a specific stage (useful after partial failure)
    python -m src.pipeline.pipeline --start-from tokenize

    # Dry-run cleanup check
    python -m src.pipeline.pipeline --dry-run-cleanup
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pipeline.manifest import StageManifest
from src.utils.logging import StageLogger

MANIFEST_DIR = Path("data/manifests")
_RUN_ID      = int(os.environ.get("RUN_ID", 0)) or None

STAGE_ORDER = ["download", "clean", "tokenize", "pack"]


def stage_is_complete(stage: str) -> bool:
    manifest_path = MANIFEST_DIR / f"{stage}.json"
    if not manifest_path.exists():
        return False
    return StageManifest(manifest_path).is_complete()


def run_pipeline(
    start_from: str = "download",
    dry_run_cleanup: bool = False,
):
    stage_log = StageLogger(run_id=_RUN_ID)

    start_idx = STAGE_ORDER.index(start_from)
    stages    = STAGE_ORDER[start_idx:]

    print(f"\n{'='*60}")
    print(f"  Librarian Preprocessing Pipeline")
    print(f"{'='*60}")
    for stage in STAGE_ORDER:
        done = stage_is_complete(stage)
        print(f"  {'✓' if done else '○'} {stage}")
    print(f"{'='*60}")
    print(f"  Starting from: {start_from}")

    t_start = time.time()

    for stage in stages:
        if stage_is_complete(stage):
            print(f"\n[pipeline] Stage '{stage}' already complete — skipping")
            continue

        print(f"\n[pipeline] Starting stage: {stage}")
        stage_log.start(stage)
        t0 = time.time()

        try:
            if stage == "download":
                from src.data.download import download_datasets
                summary = download_datasets(stage_log=stage_log)

            elif stage == "clean":
                from src.data.clean import run_clean
                summary = run_clean(stage_log=stage_log)

            elif stage == "tokenize":
                from src.data.tokenizer import run_tokenize
                summary = run_tokenize(stage_log=stage_log)

            elif stage == "pack":
                from src.data.pack import run_pack
                summary = run_pack(stage_log=stage_log)

            else:
                raise ValueError(f"Unknown stage: {stage}")

        except Exception as e:
            stage_log.error(stage, str(e))
            print(f"\n[pipeline] FATAL: stage '{stage}' failed: {e}")
            print(f"[pipeline] Fix the issue and re-run with --start-from {stage}")
            raise

        elapsed = time.time() - t0
        stage_log.end(stage, {**summary, "elapsed_s": round(elapsed, 1)})
        print(f"[pipeline] Stage '{stage}' complete in {elapsed:.1f}s")

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {total:.1f}s")
    print(f"{'='*60}")

    # Print storage summary
    _print_storage_summary()


def _print_storage_summary():
    dirs = {
        "data/raw":             Path("data/raw"),
        "data/cleaned":         Path("data/cleaned"),
        "data/tokenized/shards":Path("data/tokenized/shards"),
        "data/tokenized (packed)": Path("data/tokenized"),
        "tokenizer":            Path("tokenizer"),
        "checkpoints":          Path("checkpoints"),
        "logs":                 Path("logs"),
        "data/manifests":       Path("data/manifests"),
    }

    print("\n  Storage summary:")
    for label, path in dirs.items():
        if not path.exists():
            continue
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        print(f"    {label:35s}: {size / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Librarian preprocessing pipeline")
    parser.add_argument(
        "--start-from",
        choices=STAGE_ORDER,
        default="download",
        help="Start (or resume) from this stage",
    )
    parser.add_argument(
        "--dry-run-cleanup",
        action="store_true",
        help="Show what cleanup would delete without deleting",
    )
    args = parser.parse_args()

    run_pipeline(
        start_from      = args.start_from,
        dry_run_cleanup = args.dry_run_cleanup,
    )


if __name__ == "__main__":
    main()