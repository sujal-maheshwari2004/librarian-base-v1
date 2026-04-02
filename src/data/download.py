"""
download.py — Streaming shard downloader with manifest-based resume.

Changes from original:
  - Shard-level manifest replaces in-memory state
  - Downloads are sorted deterministically (reproducibility)
  - Writes to data/raw/shards/<source>/<shard_id>.txt
  - Wikitext fetched directly via HF parquet (bypasses broken S3 URL)
  - bookcorpus / openwebtext use HF datasets streaming with fallback
  - After all shards verified, downstream stages can safely delete this dir
"""

from __future__ import annotations

import io
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
        "name":     "wikitext",
        "hf_id":    "wikitext",
        "config":   "wikitext-103-raw-v1",
        "splits":   ["train", "validation", "test"],
        "text_col": "text",
        "weight":   0.30,
    },
    {
        "name":     "bookcorpus",
        "hf_id":    "bookcorpus",
        "config":   None,
        "splits":   ["train"],
        "text_col": "text",
        "weight":   0.50,
    },
    {
        "name":     "openwebtext",
        "hf_id":    "openwebtext",
        "config":   None,
        "splits":   ["train"],
        "text_col": "text",
        "weight":   0.20,
    },
]

# Wikitext-103 parquet files hosted directly on HuggingFace
WIKITEXT_PARQUET = {
    "train": [
        "https://huggingface.co/datasets/Salesforce/wikitext/resolve/refs%2Fconvert%2Fparquet/wikitext-103-raw-v1/train/0000.parquet",
        "https://huggingface.co/datasets/Salesforce/wikitext/resolve/refs%2Fconvert%2Fparquet/wikitext-103-raw-v1/train/0001.parquet",
    ],
    "validation": [
        "https://huggingface.co/datasets/Salesforce/wikitext/resolve/refs%2Fconvert%2Fparquet/wikitext-103-raw-v1/validation/0000.parquet",
    ],
    "test": [
        "https://huggingface.co/datasets/Salesforce/wikitext/resolve/refs%2Fconvert%2Fparquet/wikitext-103-raw-v1/test/0000.parquet",
    ],
}

DOCS_PER_SHARD = 50_000


# ── Shard ID helpers ─────────────────────────────────────────────────

def shard_id(source: str, split: str, shard_idx: int) -> str:
    return f"{source}__{split}__{shard_idx:06d}"


def shard_path(source: str, split: str, shard_idx: int) -> Path:
    return RAW_DIR / source / split / f"shard_{shard_idx:06d}.txt"


# ── Wikitext direct parquet download ─────────────────────────────────

def _stream_wikitext(split: str):
    """
    Download wikitext-103 directly from HF parquet files.
    Bypasses the broken S3 ZIP URL that datasets 2.2.x tries to use.
    Train has 2 shards; validation and test have 1 each.
    """
    import requests
    import pyarrow.parquet as pq

    token   = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    urls    = WIKITEXT_PARQUET[split]
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    for url in urls:
        fname = url.split("/")[-1]
        print(f"[download] fetching wikitext/{split}/{fname} from HF…")
        r = requests.get(url, headers=headers, timeout=120)
        r.raise_for_status()

        table = pq.read_table(io.BytesIO(r.content))
        for batch in table.to_batches():
            for val in batch.column("text"):
                text = val.as_py()
                if text and text.strip():
                    yield text.strip()


# ── Per-dataset streaming download ───────────────────────────────────

def _stream_dataset(hf_id: str, config: str | None, split: str,
                    text_col: str):
    """
    Yield individual text strings.
    - wikitext: fetched directly via parquet (S3 URL is dead)
    - others: HF datasets streaming with non-streaming fallback
    """
    if hf_id == "wikitext":
        yield from _stream_wikitext(split)
        return

    from datasets import load_dataset

    kwargs: dict = {}
    if config:
        kwargs["name"] = config

    try:
        ds = load_dataset(hf_id, split=split, streaming=True, **kwargs)
        for example in ds:
            text = example.get(text_col, "")
            if text and text.strip():
                yield text.strip()
    except Exception as e:
        print(f"[download] streaming failed for {hf_id}/{split} ({e}), "
              f"falling back to non-streaming…")
        ds = load_dataset(hf_id, split=split, streaming=False, **kwargs)
        for example in ds:
            text = example.get(text_col, "")
            if text and text.strip():
                yield text.strip()


# ── Per-source download ───────────────────────────────────────────────

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

        shard_idx         = 0
        buffer: list[str] = []
        doc_count         = 0

        def flush_shard():
            nonlocal shard_idx, buffer, doc_count
            sid  = shard_id(source, split, shard_idx)
            path = shard_path(source, split, shard_idx)

            if sid in manifest._entries:
                from src.pipeline.manifest import ShardState
                if manifest._entries[sid].state == ShardState.DONE:
                    shard_idx += 1
                    buffer = []
                    return

            # Register shard if not already in manifest
            manifest.register_shards([sid])
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


# ── Compatibility shims (expected by sanity_check.py) ────────────────

def discover(source: str, split: str) -> list[str]:
    manifest = StageManifest(MANIFEST_DIR / "download.json")
    prefix   = f"{source}__{split}__"
    return sorted(
        sid for sid in manifest._entries if sid.startswith(prefix)
    )


def download_shard_to_disk(
    shard_id_str: str,
    manifest: StageManifest,
    stage_log=None,
) -> Path | None:
    from src.pipeline.manifest import ShardState
    if shard_id_str in manifest._entries and \
       manifest._entries[shard_id_str].state == ShardState.DONE:
        return Path(manifest._entries[shard_id_str].output_path)
    source, split, _ = shard_id_str.split("__")
    cfg = next((c for c in DATASET_CONFIGS if c["name"] == source), None)
    if cfg is None:
        raise ValueError(f"Unknown source in shard_id: {source}")
    download_source(cfg, manifest, stage_log)
    entry = manifest._entries.get(shard_id_str)
    return Path(entry.output_path) if entry else None


def parquet_to_txt(parquet_path: str, out_path: str, text_col: str = "text") -> int:
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path, columns=[text_col])
    rows  = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for batch in table.to_batches():
            for val in batch.column(text_col):
                text = val.as_py()
                if text and text.strip():
                    f.write(text.strip() + "\n")
                    rows += 1
    return rows
