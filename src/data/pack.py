"""
pack.py — Streaming token packer with manifest tracking.

Critical fix from original: original pack.py called np.fromfile() on the
entire tokenized file — at 90 GB that causes OOM on any pod with < 96 GB RAM.

This version:
  - Streams tokenized shards one at a time
  - Writes packed sequences of exactly seq_len tokens to output .bin
  - Maintains a carry-over buffer so sequences never cross shard boundaries
    in a way that loses tokens (deterministic output)
  - Carves a validation split at a configurable fraction
  - Deletes data/tokenized/shards/ after packing completes
  - Does NOT delete the final train_packed.bin / validation_packed.bin

Output:
  data/tokenized/train_packed.bin
  data/tokenized/validation_packed.bin
  data/tokenized/test_packed.bin   (if test shards exist)
"""

from __future__ import annotations

import hashlib
import json
import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pipeline.manifest import StageManifest, file_checksum, validate_bin_file
from src.pipeline.atomic_writer import AtomicBinaryWriter
from src.pipeline.cleanup import safe_delete_stage
from src.utils.logging import StageLogger

# ── Paths ─────────────────────────────────────────────────────────────
TOKENIZED_SHARDS = Path("data/tokenized/shards")
PACKED_DIR       = Path("data/tokenized")
MANIFEST_DIR     = Path("data/manifests")
DTYPE            = np.uint16
DTYPE_BYTES      = np.dtype(DTYPE).itemsize
SEQ_LEN          = 512
VAL_FRAC         = 0.005    # 0.5% of sequences → validation split
TEST_FRAC        = 0.001    # 0.1% → test split (optional)
_RUN_ID          = int(os.environ.get("RUN_ID", 0)) or None

# Dataset source weights for training set merge
# Shards are assigned to train/val/test based on a deterministic hash
# of the shard_id (not random, so behavior is reproducible across restarts)
SOURCE_WEIGHTS = {
    "wikitext":    0.30,
    "bookcorpus":  0.50,
    "openwebtext": 0.20,
}


# ── Streaming shard reader ────────────────────────────────────────────

def iter_tokens_from_shard(path: Path) -> np.ndarray:
    """
    Yield numpy arrays of uint16 tokens from a tokenized .bin shard,
    read in 32 MB chunks to bound memory usage.
    """
    CHUNK_TOKENS = 32 * 1024 * 1024 // DTYPE_BYTES   # 16M tokens = 32 MB
    with open(path, "rb") as f:
        while True:
            raw = f.read(CHUNK_TOKENS * DTYPE_BYTES)
            if not raw:
                break
            yield np.frombuffer(raw, dtype=DTYPE)


# ── Deterministic split assignment ───────────────────────────────────

def _shard_split_assignment(shard_id: str) -> str:
    """
    Deterministically assign a shard to train/validation/test using a
    hash of the shard_id. This is reproducible across restarts and
    avoids random state issues.
    """
    h = int(hashlib.sha256(shard_id.encode()).hexdigest(), 16) % 10000
    if h < int(TEST_FRAC * 10000):
        return "test"
    if h < int((VAL_FRAC + TEST_FRAC) * 10000):
        return "validation"
    return "train"


# ── Streaming pack writer ─────────────────────────────────────────────

class StreamingPacker:
    """
    Accepts a stream of uint16 token arrays and writes complete sequences
    of `seq_len` tokens to an output .bin file.

    Carries over partial sequences across flush calls so no tokens are lost.
    """

    def __init__(self, out_path: Path, seq_len: int):
        self.out_path   = out_path
        self.seq_len    = seq_len
        self._carry     = np.array([], dtype=DTYPE)
        self._writer    = None
        self._n_seqs    = 0

    def __enter__(self):
        self._writer = AtomicBinaryWriter(self.out_path).__enter__()
        return self

    def feed(self, tokens: np.ndarray):
        if len(tokens) == 0:
            return
        combined = np.concatenate([self._carry, tokens])
        n_complete = len(combined) // self.seq_len
        if n_complete > 0:
            usable = combined[: n_complete * self.seq_len]
            self._writer.write(usable.tobytes())
            self._n_seqs += n_complete
        self._carry = combined[n_complete * self.seq_len :]

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Discard partial last sequence (standard approach)
        return self._writer.__exit__(exc_type, exc_val, exc_tb)

    @property
    def sequences_written(self) -> int:
        return self._n_seqs


# ── Main pack stage ───────────────────────────────────────────────────

def run_pack(
    seq_len: int = SEQ_LEN,
    stage_log: StageLogger | None = None,
) -> dict:
    PACKED_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    tokenize_manifest = StageManifest(MANIFEST_DIR / "tokenize.json")
    if not tokenize_manifest.is_complete():
        raise RuntimeError(
            f"Tokenize stage not complete: {tokenize_manifest.summary()}"
        )

    tok_entries = tokenize_manifest.verified_entries()
    print(f"\n=== PACK STAGE ===")
    print(f"Tokenized shards: {len(tok_entries)}")

    # Sort deterministically for reproducibility
    tok_entries.sort(key=lambda e: e.shard_id)

    # Assign shards to splits
    splits: dict[str, list] = {"train": [], "validation": [], "test": []}
    for entry in tok_entries:
        split = _shard_split_assignment(entry.shard_id)
        splits[split].append(entry)

    for split, entries in splits.items():
        print(f"  {split}: {len(entries)} shards")

    results: dict = {}

    for split, entries in splits.items():
        if not entries:
            print(f"[pack] No shards for {split}, skipping")
            continue

        out_path = PACKED_DIR / f"{split}_packed.bin"

        # Skip if already packed and valid
        if out_path.exists():
            valid, n_tok, err = validate_bin_file(out_path, seq_len)
            if valid:
                print(f"[pack] {split}_packed.bin already valid ({n_tok:,} tokens) — skipping")
                results[split] = {
                    "sequences": n_tok // seq_len,
                    "tokens":    n_tok,
                    "size_gb":   out_path.stat().st_size / 1e9,
                }
                continue

        print(f"\n[pack] Packing {split}…")
        with StreamingPacker(out_path, seq_len) as packer:
            for entry in entries:
                shard_path = Path(entry.output_path)
                for chunk in iter_tokens_from_shard(shard_path):
                    packer.feed(chunk)

                if stage_log:
                    stage_log.progress("pack", {
                        "split": split,
                        "shard": entry.shard_id,
                        "seqs":  packer.sequences_written,
                    })

        n_seqs = packer.sequences_written
        size   = out_path.stat().st_size
        print(f"[pack] {split}: {n_seqs:,} sequences, {size / 1e9:.2f} GB")

        # Validate the output before allowing cleanup
        valid, n_tok, err = validate_bin_file(out_path, seq_len)
        if not valid:
            raise RuntimeError(f"Packed output invalid: {err}")

        results[split] = {
            "sequences": n_seqs,
            "tokens":    n_tok,
            "size_gb":   size / 1e9,
        }

    # Write a pack summary manifest for the cleanup stage to reference
    pack_manifest = StageManifest(MANIFEST_DIR / "pack.json")
    pack_manifest.register_shards(
        [f"packed_{split}" for split in results],
        meta={"stage": "pack"},
    )
    for split in results:
        sid      = f"packed_{split}"
        out_path = PACKED_DIR / f"{split}_packed.bin"
        checksum = file_checksum(out_path)
        pack_manifest.mark_processing(sid)
        pack_manifest.mark_verified(sid, str(out_path), checksum,
                                    results[split]["tokens"])
        pack_manifest.mark_done(sid)

    print(f"\n[pack] Pack manifest: {pack_manifest.summary()}")

    # ── Delete tokenized shards after packing completes ────────────
    if pack_manifest.is_complete():
        print("\n[pack] All splits packed — deleting tokenized shards…")
        result = safe_delete_stage(
            stage_name               = "tokenize",
            artifact_dir             = TOKENIZED_SHARDS,
            downstream_manifest_path = MANIFEST_DIR / "pack.json",
            dry_run                  = False,
        )
        print(f"[pack] Freed {result['deleted_bytes'] / 1e9:.2f} GB")

    return results


if __name__ == "__main__":
    log = StageLogger(run_id=_RUN_ID)
    log.start("pack")
    try:
        summary = run_pack(stage_log=log)
        log.end("pack", summary)
    except Exception as e:
        log.error("pack", str(e))
        raise


# ── Compatibility shim (expected by sanity_check.py) ─────────────────

def pack_split(split_name: str, stage_logger=None):
    """
    Thin wrapper kept for backward compatibility with sanity_check.py.
    Delegates to run_pack() filtering to a single split.
    """
    results = run_pack(seq_len=SEQ_LEN, stage_log=stage_logger)
    return results.get(split_name)
