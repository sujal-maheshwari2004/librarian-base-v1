"""
download.py — Streaming shard downloader with manifest-based resume.

Changes from original:
  - Shard-level manifest replaces in-memory state
  - Downloads are sorted deterministically (reproducibility)
  - Writes to data/raw/shards/<source>/<shard_id>.txt
  - Wikitext fetched directly via HF parquet (bypasses broken S3 URL)
  - bookcorpus / openwebtext use HF datasets streaming with fallback
  - After all shards verified, downstream stages can safely delete this dir
  - Each source has a max_shards cap (None = unlimited) to allow a fast
    tokenizer-only run before committing to the full dataset download
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

DOCS_PER_SHARD = 50_000

# max_shards caps how many shards to download per source/split.
# None = no limit (download everything) — use this for full training runs.
# Set to a small number (e.g. bookcorpus=50, openwebtext=20) for a fast
# tokenizer-only run. The manifest is additive, so a subsequent full run
# with max_shards=None will resume from where the capped run left off
# without re-downloading already-completed shards.
DATASET_CONFIGS = [
    {
        "name":       "wikitext",
        "hf_id":      "wikitext",
        "config":     "wikitext-103-raw-v1",
        "splits":     ["train", "validation", "test"],
        "text_col":   "text",
        "weight":     0.30,
        "max_shards": None,   # wikitext is small (~4 shards total), always full
    },
    {
        "name":       "bookcorpus",
        "hf_id":      "bookcorpus",
        "config":     None,
        "splits":     ["train"],
        "text_col":   "text",
        "weight":     0.50,
        "max_shards": None,   # set to e.g. 50 for a tokenizer-only run
    },
    {
        "name":       "openwebtext",
        "hf_id":      "openwebtext",
        "config":     None,
        "splits":     ["train"],
        "text_col":   "text",
        "weight":     0.20,
        "max_shards": None,   # set to e.g. 20 for a tokenizer-only run
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


# ── Fineweb-edu direct parquet download ──────────────────────────────

def _stream_fineweb_edu(split: str):
    import requests
    import pyarrow.parquet as pq

    token   = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    api_url = ("https://datasets-server.huggingface.co/parquet"
               "?dataset=HuggingFaceFW/fineweb-edu&config=sample-10BT&split=train")
    r = requests.get(api_url, headers=headers, timeout=30)
    r.raise_for_status()
    files = [f["url"] for f in r.json()["parquet_files"]]

    print(f"[download] fineweb-edu: fetching {len(files)} of available parquet shards")

    for i, url in enumerate(files):
        print(f"[download] fineweb-edu shard {i+1}/{len(files)}…")
        r = requests.get(url, headers=headers, timeout=300)
        r.raise_for_status()
        table = pq.read_table(io.BytesIO(r.content), columns=["text"])
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
    - fineweb-edu: fetched directly via HF datasets-server API
    - others: HF datasets streaming with non-streaming fallback
    """
    if hf_id == "wikitext":
        yield from _stream_wikitext(split)
        return

    if hf_id == "HuggingFaceFW/fineweb-edu":
        yield from _stream_fineweb_edu(split)
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
    Stops early if max_shards is set and reached.
    Returns total documents written.
    """
    source     = cfg["name"]
    max_shards = cfg.get("max_shards")   # None = unlimited
    total      = 0

    for split in cfg["splits"]:
        print(f"\n[download] {source}/{split} — streaming…")
        if max_shards is not None:
            print(f"[download] {source}/{split} — capped at {max_shards} shards "
                  f"({max_shards * DOCS_PER_SHARD:,} docs max)")

        out_dir = RAW_DIR / source / split
        out_dir.mkdir(parents=True, exist_ok=True)

        shard_idx         = 0
        buffer: list[str] = []
        doc_count         = 0
        stop_early        = False

        def flush_shard():
            nonlocal shard_idx, buffer, doc_count, stop_early
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

            # Check cap AFTER incrementing shard_idx
            if max_shards is not None and shard_idx >= max_shards:
                stop_early = True

        for text in _stream_dataset(cfg["hf_id"], cfg["config"],
                                    split, cfg["text_col"]):
            if stop_early:
                break
            buffer.append(text)
            doc_count += 1
            if len(buffer) >= DOCS_PER_SHARD:
                flush_shard()
                if stop_early:
                    break

        # Flush remaining partial shard (unless we hit the cap exactly)
        if buffer and not stop_early:
            flush_shard()

        total += doc_count
        print(f"[download] {source}/{split}: {doc_count:,} docs in {shard_idx} shards"
              + (" (capped)" if stop_early else ""))

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
