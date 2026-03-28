"""
Robust HF dataset downloader (datasets==2.2.1 safe)
Features:
- Direct HTTP streaming download (bypasses hf_hub_download timeout issues)
- Chunk-level progress bar per shard
- Dynamic shard discovery via HF API
- Mirror fallback support
- Manifest-based shard-level resume
- Exponential backoff retry on every failure mode
- No datasets.load_dataset()
"""
import os
import time
import random
import json
import hashlib
from pathlib import Path

import requests
import pyarrow.parquet as pq
from huggingface_hub import list_repo_files

from src.utils.logging import StageLogger


MAX_RETRIES    = 10
WAIT           = 3
CHUNK_SIZE     = 8 * 1024 * 1024   # 8 MB chunks
CONNECT_TIMEOUT = 30               # seconds to establish connection
READ_TIMEOUT    = 120              # seconds between chunks (not total transfer)

_RUN_ID = int(os.environ.get("RUN_ID", 0)) or None

HF_TOKEN = os.environ.get("HF_TOKEN", "")

DATASET_CONFIGS = {
    "wikitext": [
        ("Salesforce/wikitext", "wikitext-103-raw-v1/train"),
    ],
    # BookCorpus replaced with FineWeb-Edu
    "bookcorpus": [
        ("HuggingFaceFW/fineweb-edu", "train"),
    ],
    "openwebtext": [
        ("Skylion007/openwebtext", "train"),
    ],
}


# ── shard discovery ───────────────────────────────────────────────────

def discover(repo_id: str, pattern: str) -> list[str]:
    files = list_repo_files(repo_id, repo_type="dataset")
    return sorted(
        f for f in files
        if f.endswith(".parquet") and pattern in f
    )


# ── direct HTTP streaming download ───────────────────────────────────

def shard_url(repo_id: str, shard_path: str) -> str:
    return (
        f"https://huggingface.co/datasets/{repo_id}"
        f"/resolve/main/{shard_path}"
    )


def download_shard_to_disk(repo_id: str, shard: str, dest: Path) -> Path:
    """
    Stream-download a single parquet shard to dest via direct HTTP.
    Uses chunk-level READ_TIMEOUT so a stalled connection is detected
    quickly even on a 500 MB file.
    Resumes partial downloads if dest.partial already exists.
    Returns dest on success.
    """
    url      = shard_url(repo_id, shard)
    partial  = dest.with_suffix(".partial")
    headers  = {}

    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    # Resume support: if a .partial exists, ask for the remaining bytes
    start_byte = 0
    if partial.exists():
        start_byte = partial.stat().st_size
        if start_byte > 0:
            headers["Range"] = f"bytes={start_byte}-"
            print(f"    resuming from byte {start_byte:,}")

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            )

            # 416 = range not satisfiable → server thinks we have it all
            if resp.status_code == 416:
                print("    server returned 416 — treating partial as complete")
                partial.rename(dest)
                return dest

            resp.raise_for_status()

            total    = int(resp.headers.get("Content-Length", 0)) + start_byte
            received = start_byte
            mode     = "ab" if start_byte > 0 else "wb"

            with partial.open(mode) as fh:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)
                        received += len(chunk)

                        if total:
                            pct = 100 * received / total
                            mb  = received / 1e6
                            print(
                                f"\r    {mb:.0f} MB / {total/1e6:.0f} MB  "
                                f"({pct:.1f}%)",
                                end="",
                                flush=True,
                            )

            print()  # newline after progress
            partial.rename(dest)
            return dest

        except Exception as exc:
            wait = WAIT * (2 ** attempt)
            print(f"\n    attempt {attempt+1}/{MAX_RETRIES} failed: {exc}")

            if attempt < MAX_RETRIES - 1:
                print(f"    retrying in {wait}s ...")
                time.sleep(wait)

                # Update resume header for next attempt
                if partial.exists():
                    start_byte         = partial.stat().st_size
                    headers["Range"]   = f"bytes={start_byte}-"
            else:
                raise RuntimeError(
                    f"Failed to download {shard} after {MAX_RETRIES} attempts"
                ) from exc


# ── parquet → txt ─────────────────────────────────────────────────────

def parquet_to_txt(src: Path, dst: Path) -> int:
    pf   = pq.ParquetFile(str(src))
    rows = 0
    with dst.open("a", encoding="utf-8") as f:
        for rg in range(pf.num_row_groups):
            batch = pf.read_row_group(rg)
            for row in batch.to_pylist():
                text = (row.get("text") or "").strip()
                if text:
                    f.write(text + "\n\n")
                    rows += 1
    return rows


# ── manifest helpers ──────────────────────────────────────────────────

def manifest_path(output_dir: Path, name: str) -> Path:
    return output_dir / f".{name}_manifest.json"

def load_manifest(output_dir: Path, name: str) -> dict:
    p = manifest_path(output_dir, name)
    if p.exists():
        return json.loads(p.read_text())
    return {"completed_shards": [], "total_docs": 0}

def save_manifest(output_dir: Path, name: str, manifest: dict) -> None:
    manifest_path(output_dir, name).write_text(
        json.dumps(manifest, indent=2)
    )


# ── per-dataset downloader ────────────────────────────────────────────

def download_dataset(name: str, output_dir: Path, stage_logger=None):
    mirrors     = DATASET_CONFIGS[name]
    output_file = output_dir / f"{name}_train.txt"
    cache_dir   = output_dir / ".shard_cache" / name
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest    = load_manifest(output_dir, name)
    done_shards = set(manifest["completed_shards"])

    for repo, pattern in mirrors:
        print(f"\n{name}: discovering shards in {repo} ...")

        try:
            all_shards = discover(repo, pattern)
        except Exception as e:
            print(f"  discover failed → {e}")
            continue

        if not all_shards:
            print(f"  no shards matched pattern '{pattern}'")
            continue

        remaining = [s for s in all_shards if s not in done_shards]
        n_total   = len(all_shards)
        n_done    = len(done_shards)

        print(f"  {n_total} shards total  |  {n_done} done  |  "
              f"{len(remaining)} remaining")

        if not remaining:
            print(f"{name}: already complete ({manifest['total_docs']:,} docs)")
            return

        total_docs = manifest["total_docs"]

        for i, shard in enumerate(remaining, start=1):
            position  = n_done + i
            shard_name = shard.replace("/", "_")
            dest       = cache_dir / shard_name

            print(f"\n  [{position}/{n_total}] {shard}")

            # Download parquet to local cache
            download_shard_to_disk(repo, shard, dest)

            # Extract text
            rows        = parquet_to_txt(dest, output_file)
            total_docs += rows

            # Update manifest immediately
            manifest["completed_shards"].append(shard)
            manifest["total_docs"] = total_docs
            save_manifest(output_dir, name, manifest)

            print(f"    +{rows:,} docs  (running total: {total_docs:,})")

            if stage_logger:
                stage_logger.progress("download", {
                    "dataset":      name,
                    "shards_done":  position,
                    "shards_total": n_total,
                    "total_docs":   total_docs,
                })

            # Remove cached parquet after successful extraction to save disk
            dest.unlink(missing_ok=True)

        print(f"\n{name}: complete — {total_docs:,} docs across {n_total} shards")
        return

    raise RuntimeError(f"{name}: all mirrors failed")


# ── top-level entry point ─────────────────────────────────────────────

def download_datasets(seed=42, output_dir="data/raw", stage_logger=None):
    random.seed(seed)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\nDirect HTTP streaming download mode\n")

    for name in DATASET_CONFIGS:
        download_dataset(name, out, stage_logger=stage_logger)

    print("\nDownload complete.\n")


if __name__ == "__main__":
    log = StageLogger(run_id=_RUN_ID)
    log.start("download")
    try:
        download_datasets(stage_logger=log)
        log.end("download", {"datasets": list(DATASET_CONFIGS.keys())})
    except Exception as e:
        log.error("download", str(e))
        raise
