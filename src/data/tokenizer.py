"""
tokenizer.py — Streaming per-shard tokenization with manifest tracking.

Changes from original:
  - Tokenizes one cleaned shard at a time (no 150GB RAM requirement)
  - Each tokenized shard is a uint16 .bin written atomically
  - Deletes data/cleaned/ after all tokenized shards verified
  - No intermediate text files held in memory

Output per shard: data/tokenized/shards/<source>/<split>/shard_XXXXXX.bin
Final merged splits happen in pack.py via streaming concat.
"""

from __future__ import annotations

import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pipeline.manifest import StageManifest, file_checksum, validate_bin_file
from src.pipeline.atomic_writer import AtomicBinaryWriter
from src.pipeline.cleanup import safe_delete_stage
from src.utils.logging import StageLogger

# ── Paths ────────────────────────────────────────────────────────────
TOKENIZER_PATH  = Path("tokenizer/tokenizer.json")
CLEANED_DIR     = Path("data/cleaned/shards")
TOKENIZED_DIR   = Path("data/tokenized/shards")
MANIFEST_DIR    = Path("data/manifests")
DTYPE           = np.uint16
_RUN_ID         = int(os.environ.get("RUN_ID", 0)) or None


# ── Per-shard tokenization ───────────────────────────────────────────

def tokenize_shard(
    cleaned_path: Path,
    out_path: Path,
    tokenizer,
) -> int:
    """
    Stream-tokenize one cleaned text shard into a uint16 .bin file.
    Returns number of tokens written.
    """
    FLUSH_EVERY = 100_000    # tokens — flush to writer in batches
    DTYPE_SIZE  = np.dtype(DTYPE).itemsize

    buffer: list[int] = []
    total_tokens = 0

    with AtomicBinaryWriter(out_path) as w:
        with cleaned_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                ids = tokenizer.encode(line).ids
                buffer.extend(ids)

                if len(buffer) >= FLUSH_EVERY:
                    arr = np.array(buffer, dtype=DTYPE)
                    w.write(arr.tobytes())
                    total_tokens += len(buffer)
                    buffer = []

        if buffer:
            arr = np.array(buffer, dtype=DTYPE)
            w.write(arr.tobytes())
            total_tokens += len(buffer)

    return total_tokens


# ── Main tokenize stage ──────────────────────────────────────────────

def run_tokenize(stage_log: StageLogger | None = None) -> dict:
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {TOKENIZER_PATH}. Run train_tokenizer first."
        )

    from tokenizers import Tokenizer  # type: ignore
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    clean_manifest    = StageManifest(MANIFEST_DIR / "clean.json")
    tokenize_manifest = StageManifest(MANIFEST_DIR / "tokenize.json")
    tokenize_manifest.reset_stale()

    if not clean_manifest.is_complete():
        raise RuntimeError(
            f"Clean stage not complete: {clean_manifest.summary()}"
        )

    clean_entries = clean_manifest.verified_entries()
    tokenize_manifest.register_shards(
        [e.shard_id for e in clean_entries],
        meta={"stage": "tokenize"},
    )

    print(f"\n=== TOKENIZE STAGE ===")
    print(f"Cleaned shards: {len(clean_entries)}")
    print(f"Tokenize manifest: {tokenize_manifest.summary()}")

    total_tokens = 0

    for entry in clean_entries:
        sid = entry.shard_id

        from src.pipeline.manifest import ShardState
        if tokenize_manifest._entries.get(sid) and \
           tokenize_manifest._entries[sid].state == ShardState.DONE:
            continue

        cleaned_path = Path(entry.output_path)
        # Mirror path structure under tokenized/shards/
        rel       = cleaned_path.relative_to(CLEANED_DIR)
        tok_path  = TOKENIZED_DIR / rel.with_suffix(".bin")
        tok_path.parent.mkdir(parents=True, exist_ok=True)

        tokenize_manifest.mark_processing(sid)

        try:
            n_tokens = tokenize_shard(cleaned_path, tok_path, tokenizer)
        except Exception as e:
            tokenize_manifest.mark_failed(sid, str(e))
            print(f"[tokenize] FAILED shard {sid}: {e}")
            continue

        checksum = file_checksum(tok_path)
        tokenize_manifest.mark_verified(sid, str(tok_path), checksum, n_tokens)
        tokenize_manifest.mark_done(sid)
        total_tokens += n_tokens

        if stage_log:
            stage_log.progress("tokenize", {
                "shard":    sid,
                "tokens":   n_tokens,
                "total_M":  total_tokens / 1e6,
            })

        print(f"[tokenize] {sid}: {n_tokens:,} tokens")

    print(f"\n[tokenize] Total tokens: {total_tokens:,}")
    print(f"[tokenize] Manifest: {tokenize_manifest.summary()}")

    # ── Delete cleaned shards after tokenize completes ─────────────
    if tokenize_manifest.is_complete():
        print("\n[tokenize] All shards tokenized — deleting cleaned data…")
        result = safe_delete_stage(
            stage_name               = "clean",
            artifact_dir             = CLEANED_DIR,
            downstream_manifest_path = MANIFEST_DIR / "tokenize.json",
            dry_run                  = False,
        )
        print(f"[tokenize] Freed {result['deleted_bytes'] / 1e9:.2f} GB")

    return {
        "total_tokens": total_tokens,
        "manifest":     tokenize_manifest.summary(),
    }


if __name__ == "__main__":
    log = StageLogger(run_id=_RUN_ID)
    log.start("tokenize")
    try:
        summary = run_tokenize(stage_log=log)
        log.end("tokenize", summary)
    except Exception as e:
        log.error("tokenize", str(e))
        raise
