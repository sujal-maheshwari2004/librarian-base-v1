"""
src/data/download.py  (hardened replacement)
─────────────────────────────────────────────
Changes vs original:
  • load_with_retry()       — exponential back-off, configurable attempts
  • skip_if_exists()        — skips splits already on disk (safe to re-run)
  • streaming-with-fallback — tries streaming first; falls back to non-streaming
                              for datasets<3.0 where Parquet streaming was flaky
  • writes incrementally    — no giant list accumulation before writing
  • per-dataset fallback    — bookcorpus & OWT have tested fallback sources
  • clear error messages    — prints which dataset/split failed and why

datasets version compatibility:
  • >=4.x  : streaming=True, trust_remote_code ignored (script-based datasets dead)
  • 2.x-3.x: streaming=True attempted; falls back to streaming=False on failure
             trust_remote_code=True still works for script-based datasets
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import random
from pathlib import Path

import datasets as _datasets_module
from datasets import load_dataset
from src.utils.logging import StageLogger

# Detect datasets version once at import time
_DATASETS_VERSION = tuple(int(x) for x in _datasets_module.__version__.split(".")[:2])
_STREAMING_RELIABLE = _DATASETS_VERSION >= (3, 0)   # streaming Parquet stable in 3.x+

_RUN_ID = int(os.environ.get("RUN_ID", 0)) or None

# ── Retry config ─────────────────────────────────────────────────────────────
MAX_RETRIES     = 4        # total attempts before giving up on a dataset
RETRY_BASE_WAIT = 5        # seconds — doubles each attempt (5, 10, 20, 40)

# ── Dataset configs ──────────────────────────────────────────────────────────
#   Each entry: (hf_path, hf_config_name, trust_remote_code)
#   Primary is tried first; fallbacks are tried in order on failure.
DATASET_CONFIGS = {
    "wikitext": {
        "primary":   ("Salesforce/wikitext", "wikitext-103-raw-v1", False),
        "fallbacks": [],   # WikiText is very stable; no fallback needed
        "splits":    None, # None = use whatever HF returns
    },
    "bookcorpus": {
        # bookcorpus and bookcorpus/bookcorpus use loading scripts which
        # datasets>=4.x dropped entirely. rojagtap/bookcorpus is the same
        # 74M row dataset re-hosted as native Parquet, updated Aug 2025.
        "primary":   ("rojagtap/bookcorpus", None, False),
        "fallbacks": [
            # Cleaned + near-deduplicated + benchmark-decontaminated
            ("Geralt-Targaryen/bookcorpus", None, False),
            # Another clean Parquet mirror of the same 74M rows
            ("SamuelYang/bookcorpus", None, False),
        ],
        "splits":    None,
    },
    "openwebtext": {
        "primary":   ("openwebtext", None, False),   # trust_remote_code no longer supported
        "fallbacks": [
            ("Skylion007/openwebtext", None, False),
        ],
        "splits":    None,
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_with_retry(hf_path, hf_name=None, trust_remote_code=False,
                    max_retries=MAX_RETRIES, base_wait=RETRY_BASE_WAIT):
    """
    Attempt load_dataset() up to max_retries times with exponential back-off.

    Streaming strategy:
      - datasets >= 3.0 : streaming=True  (Parquet streaming is stable)
      - datasets <  3.0 : tries streaming=True first; if it fails or stalls,
                          falls back to streaming=False (loads to disk cache)
                          trust_remote_code is passed through for script-based
                          datasets which still work on older versions.

    Raises the last exception if all attempts fail.
    """
    last_exc = None

    # On old datasets, non-streaming is the safer default for Parquet sources
    streaming_modes = [True] if _STREAMING_RELIABLE else [True, False]

    for streaming in streaming_modes:
        for attempt in range(max_retries):
            try:
                kwargs = dict(streaming=streaming, trust_remote_code=trust_remote_code)
                if hf_name:
                    kwargs["name"] = hf_name
                if not streaming:
                    print(f"  [non-streaming mode — datasets {_datasets_module.__version__}]")
                result = load_dataset(hf_path, **kwargs)
                if not streaming:
                    print(f"  Non-streaming load OK — will iterate from disk cache")
                return result
            except Exception as e:
                last_exc = e
                if attempt < max_retries - 1:
                    wait = base_wait * (2 ** attempt)
                    print(f"  [retry {attempt+1}/{max_retries-1}] {type(e).__name__}: {e}")
                    print(f"  Waiting {wait}s before retry...")
                    time.sleep(wait)
        if not _STREAMING_RELIABLE and streaming:
            print(f"  Streaming failed on datasets {_datasets_module.__version__}, trying non-streaming...")

    raise last_exc


def output_path_for(output_dir: Path, prefix: str, split: str) -> Path:
    return output_dir / f"{prefix}_{split}.txt"


def split_already_done(output_dir: Path, prefix: str, split: str) -> bool:
    p = output_path_for(output_dir, prefix, split)
    if p.exists() and p.stat().st_size > 1024:   # >1 KB = probably not empty
        return True
    return False


def _save_split_streaming(dataset_split, file_path: Path,
                           log, tag: str) -> int:
    """
    Write a HF dataset split (streaming or non-streaming) to a flat .txt file.
    Writes incrementally — never accumulates rows in RAM.
    Returns doc count.
    """
    rows = 0
    with file_path.open("w", encoding="utf-8") as f:
        for example in dataset_split:
            text = (example.get("text") or "").strip()
            if text:
                f.write(text + "\n\n")
                rows += 1

            # Progress heartbeat every 50k rows
            if rows > 0 and rows % 50_000 == 0:
                print(f"    … {rows:,} docs written", flush=True)
                if log:
                    log.progress("download", {"tag": tag, "rows_so_far": rows})

    if log:
        log.progress("download", {"file": file_path.name, "rows": rows, "tag": tag})
    print(f"  Saved {file_path.name}  ({rows:,} docs)")
    return rows


# ── Per-dataset downloader ────────────────────────────────────────────────────

def download_one(
    key: str,
    prefix: str,
    output_dir: Path,
    log,
    summary: dict,
):
    """
    Download one logical dataset (e.g. 'wikitext').
    Tries primary source then fallbacks.
    Skips splits whose output files already exist.
    """
    cfg      = DATASET_CONFIGS[key]
    sources  = [cfg["primary"]] + cfg.get("fallbacks", [])
    want_splits = cfg.get("splits")   # None = accept whatever

    print(f"\n── {key} {'─'*(50 - len(key))}")
    if log:
        log.progress("download", {"dataset": key, "status": "starting"})

    dataset = None
    used_source = None

    for hf_path, hf_name, trust in sources:
        try:
            print(f"  Loading: {hf_path}" + (f"  [{hf_name}]" if hf_name else ""))
            dataset = load_with_retry(hf_path, hf_name, trust)
            used_source = hf_path
            print(f"  OK — splits: {list(dataset.keys())}")
            break
        except Exception as e:
            print(f"  FAILED ({hf_path}): {type(e).__name__}: {e}")
            if hf_path == sources[-1][0]:
                raise RuntimeError(
                    f"All sources for '{key}' failed. Last error: {e}"
                ) from e
            print(f"  Trying next source...")

    splits = want_splits or list(dataset.keys())
    total_rows = 0

    for split in splits:
        if split not in dataset:
            print(f"  Split '{split}' not in dataset, skipping.")
            continue

        out_path = output_path_for(output_dir, prefix, split)

        if split_already_done(output_dir, prefix, split):
            size_mb = out_path.stat().st_size / (1024**2)
            print(f"  SKIP {out_path.name}  (already exists, {size_mb:.1f} MB)")
            summary[f"{prefix}_{split}_skipped"] = True
            continue

        rows = _save_split_streaming(
            dataset[split],
            out_path,
            log,
            tag=f"{prefix}/{split}",
        )
        total_rows += rows

    summary[f"{prefix}_rows"]   = total_rows
    summary[f"{prefix}_source"] = used_source


# ── Public entry point ────────────────────────────────────────────────────────

def download_datasets(
    seed: int = 42,
    output_dir: str = "data/raw",
    stage_logger: StageLogger | None = None,
):
    log = stage_logger
    random.seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary: dict = {}
    failed: list[str] = []

    for key, prefix in [
        ("wikitext",    "wikitext"),
        ("bookcorpus",  "bookcorpus"),
        ("openwebtext", "openwebtext"),
    ]:
        try:
            download_one(key, prefix, output_path, log, summary)
        except Exception as e:
            print(f"\n  ERROR downloading {key}: {e}")
            failed.append(key)
            summary[f"{key}_error"] = str(e)
            # Continue to next dataset — don't abort the whole pipeline
            # (wikitext alone is sufficient to start training)

    if failed:
        print(f"\n  WARNING: {len(failed)} dataset(s) failed: {failed}")
        print(f"  Pipeline can continue if wikitext succeeded.")
        if "wikitext" in failed:
            raise RuntimeError("WikiText download failed — cannot proceed.")

    print("\nDownload complete.")
    return summary


if __name__ == "__main__":
    log = StageLogger(run_id=_RUN_ID)
    log.start("download")
    try:
        s = download_datasets(stage_logger=log)
        log.end("download", s)
    except Exception as e:
        log.error("download", str(e))
        raise
