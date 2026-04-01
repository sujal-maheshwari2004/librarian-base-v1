"""
download.py — Streaming shard downloader with manifest-based resume.

Changes from original:
  - Shard-level manifest replaces in-memory state
  - Downloads are sorted deterministically (reproducibility)
  - Writes to data/raw/shards/<source>/<shard_id>.txt
  - Does NOT use datasets.map() or Arrow caching
  - HuggingFace streaming API used directly (no full dataset download)
  - After all shards verified, downstream stages can safely delete this dir
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pipeline.manifest import StageManifest, file_checksum
from src.pipeline.atomic_writer import AtomicTextWriter
from src.pipeline.cleanup import safe_delete_stage
from src.utils.logging import StageLogger

# ── Configuration ────────────────────────────────────────────────────

RAW_DIR      = Path("data/raw/shards")
MANIFEST_DIR = Path("data/manifests")
_RUN_ID      = int(os.environ.get("RUN_ID", 0)) or None

DATASET_CONFIGS = [
    {
        "name":   "wikitext",
        "hf_id":  "wikitext",
        "config": "wikitext-103-raw-v1",
        "splits": ["train", "validation", "test"],
        "text_col": "text",
        "weight": 0.30,
    },
    {
        "name":   "bookcorpus",
        "hf_id":  "bookcorpus",
        "config": None,
        "splits": ["train"],
        "text_col": "text",
        "weight": 0.50,
    },
    {
        "name":   "openwebtext",
        "hf_id":  "openwebtext",
        "config": None,
        "splits": ["train"],
        "text_col": "text",
        "weight": 0.20,
    },
]

# How many documents to write per shard file.
# Smaller = faster resume after crash; larger = fewer files.
DOCS_PER_SHARD = 50_000


# ── Shard ID helpers ─────────────────────────────────────────────────

def shard_id(source: str, split: str, shard_idx: int) -> str:
    return f"{source}__{split}__{shard_idx:06d}"


def shard_path(source: str, split: str, shard_idx: int) -> Path:
    return RAW_DIR / source / split / f"shard_{shard_idx:06d}.txt"


# ── Per-dataset streaming download ──────────────────────────────────

def _stream_dataset(hf_id: str, config: str | None, split: str,
                    text_col: str):
    """
    Yield individual text strings using HuggingFace streaming.
    No Arrow caching, no full download.
    """
    from datasets import load_dataset  # type: ignore

    kwargs: dict = {"streaming": True, "trust_remote_code": False}
    if config:
        kwargs["name"] = config

    ds = load_dataset(hf_id, split=split, **kwargs)
    for example in ds:
        text = example.get(text_col, "")
        if text and text.strip():
            yield text.strip()


def download_source(
    cfg: dict,
    manifest: StageManifest,
    stage_log: StageLogger | None = None,
) -> int:
    """
    Download all splits for one data source into sharded .txt files.
    Idempotent: skips shards already in DONE state.
    Returns total documents written.
    """
    source = cfg["name"]
    total  = 0

    for split in cfg["splits"]:
        print(f"\n[download] {source}/{split} — streaming…")
        out_dir = RAW_DIR / source / split
        out_dir.mkdir(parents=True, exist_ok=True)

        # Buffer docs into shards
        shard_idx  = 0
        buffer: list[str] = []
        doc_count  = 0

        def flush_shard():
            nonlocal shard_idx, buffer, doc_count
            sid  = shard_id(source, split, shard_idx)
            path = shard_path(source, split, shard_idx)

            # Skip if already done
            if sid in manifest._entries:
                from src.pipeline.manifest import ShardState
                if manifest._entries[sid].state == ShardState.DONE:
                    shard_idx += 1
                    buffer = []
                    return

            manifest.mark_processing(sid)

            with AtomicTextWriter(path) as w:
                for doc in buffer:
                    w.write(doc + "\n")

            checksum = file_checksum(path)
            manifest.mark_verified(sid, str(path), checksum, len(buffer))
            manifest.mark_done(sid)

            if stage_log:
                stage_log.progress("download", {
                    "source": source,
                    "split":  split,
                    "shard":  shard_idx,
                    "docs":   len(buffer),
                })

            shard_idx += 1
            buffer = []

        for text in _stream_dataset(cfg["hf_id"], cfg["config"],
                                     split, cfg["text_col"]):
            buffer.append(text)
            doc_count += 1
            if len(buffer) >= DOCS_PER_SHARD:
                flush_shard()

        if buffer:
            flush_shard()

        total += doc_count
        print(f"[download] {source}/{split}: {doc_count:,} docs in {shard_idx} shards")

    return total


# ── Main entrypoint ──────────────────────────────────────────────────

def download_datasets(stage_log: StageLogger | None = None) -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    manifest = StageManifest(MANIFEST_DIR / "download.json")
    manifest.reset_stale()

    print("\n=== DOWNLOAD STAGE ===")

    total_docs = 0
    for cfg in DATASET_CONFIGS:
        docs = download_source(cfg, manifest, stage_log)
        total_docs += docs

    print(f"\n[download] Total documents: {total_docs:,}")
    print(f"[download] Manifest: {manifest.summary()}")

    return {
        "total_docs": total_docs,
        "manifest":   manifest.summary(),
    }


if __name__ == "__main__":
    log = StageLogger(run_id=_RUN_ID)
    log.start("download")
    try:
        summary = download_datasets(stage_log=log)
        log.end("download", summary)
    except Exception as e:
        log.error("download", str(e))
        raise
