"""
manifest.py — Shard-level progress manifest for restart-safe pipeline execution.

Each stage writes a JSON manifest tracking per-shard state:
  pending → processing → verified → done

A stage is only considered complete when ALL shards reach 'done'.
Manifests survive pod eviction and enable fine-grained resume.
"""

from __future__ import annotations

import json
import os
import time
import hashlib
import tempfile
from enum import Enum
from pathlib import Path
from typing import Iterator


class ShardState(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    VERIFIED   = "verified"
    DONE       = "done"
    FAILED     = "failed"


class ManifestEntry:
    __slots__ = ("shard_id", "state", "output_path", "checksum",
                 "token_count", "updated_at", "error")

    def __init__(
        self,
        shard_id: str,
        state: ShardState = ShardState.PENDING,
        output_path: str = "",
        checksum: str = "",
        token_count: int = 0,
        updated_at: float = 0.0,
        error: str = "",
    ):
        self.shard_id    = shard_id
        self.state       = ShardState(state)
        self.output_path = output_path
        self.checksum    = checksum
        self.token_count = token_count
        self.updated_at  = updated_at or time.time()
        self.error       = error

    def to_dict(self) -> dict:
        return {
            "shard_id":    self.shard_id,
            "state":       self.state.value,
            "output_path": self.output_path,
            "checksum":    self.checksum,
            "token_count": self.token_count,
            "updated_at":  self.updated_at,
            "error":       self.error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ManifestEntry":
        return cls(**d)


class StageManifest:
    """
    Atomic, restart-safe manifest for a single pipeline stage.

    Writes are atomic: always write to a .tmp file then os.replace() to
    ensure the manifest is never partially-written on disk.

    Usage
    -----
        manifest = StageManifest("data/manifests/clean.json")
        manifest.register_shards(["shard_0", "shard_1", ...])

        for shard_id in manifest.pending_shards():
            manifest.mark_processing(shard_id)
            # ... do work ...
            manifest.mark_done(shard_id, output_path, checksum, token_count)

        if manifest.is_complete():
            # safe to delete previous stage
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, ManifestEntry] = {}
        self._meta: dict = {}
        self._load()

    # ── persistence ───────────────────────────────────────────────────

    def _load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self._meta    = data.get("meta", {})
                self._entries = {
                    e["shard_id"]: ManifestEntry.from_dict(e)
                    for e in data.get("shards", [])
                }
            except (json.JSONDecodeError, KeyError):
                # Corrupted manifest — start fresh but preserve the file
                # for debugging; rename to .corrupt
                corrupt = self.path.with_suffix(".corrupt")
                self.path.rename(corrupt)
                print(f"[manifest] Corrupted manifest renamed to {corrupt}")

    def _save(self):
        payload = json.dumps(
            {
                "meta":   self._meta,
                "shards": [e.to_dict() for e in self._entries.values()],
            },
            indent=2,
        )
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(payload)
        os.replace(tmp, self.path)   # atomic on POSIX + most K8s CSI drivers

    # ── registration ─────────────────────────────────────────────────

    def register_shards(self, shard_ids: list[str], meta: dict | None = None):
        """
        Idempotent: skips shards already in the manifest so resume works.
        """
        if meta:
            self._meta.update(meta)
        changed = False
        for sid in shard_ids:
            if sid not in self._entries:
                self._entries[sid] = ManifestEntry(shard_id=sid)
                changed = True
        if changed:
            self._save()

    # ── state transitions ────────────────────────────────────────────

    def mark_processing(self, shard_id: str):
        self._entries[shard_id].state      = ShardState.PROCESSING
        self._entries[shard_id].updated_at = time.time()
        self._save()

    def mark_verified(self, shard_id: str, output_path: str,
                      checksum: str, token_count: int = 0):
        e = self._entries[shard_id]
        e.state        = ShardState.VERIFIED
        e.output_path  = output_path
        e.checksum     = checksum
        e.token_count  = token_count
        e.updated_at   = time.time()
        self._save()

    def mark_done(self, shard_id: str):
        self._entries[shard_id].state      = ShardState.DONE
        self._entries[shard_id].updated_at = time.time()
        self._save()

    def mark_failed(self, shard_id: str, error: str):
        self._entries[shard_id].state      = ShardState.FAILED
        self._entries[shard_id].error      = error
        self._entries[shard_id].updated_at = time.time()
        self._save()

    # ── queries ───────────────────────────────────────────────────────

    def pending_shards(self) -> list[str]:
        """Return shards not yet done (includes failed for retry)."""
        return [
            sid for sid, e in self._entries.items()
            if e.state not in (ShardState.DONE,)
        ]

    def done_shards(self) -> list[str]:
        return [
            sid for sid, e in self._entries.items()
            if e.state == ShardState.DONE
        ]

    def verified_entries(self) -> list[ManifestEntry]:
        return [e for e in self._entries.values()
                if e.state in (ShardState.VERIFIED, ShardState.DONE)]

    def is_complete(self) -> bool:
        if not self._entries:
            return False
        return all(e.state == ShardState.DONE for e in self._entries.values())

    def summary(self) -> dict:
        counts = {s.value: 0 for s in ShardState}
        for e in self._entries.values():
            counts[e.state.value] += 1
        counts["total"] = len(self._entries)
        return counts

    def total_tokens(self) -> int:
        return sum(e.token_count for e in self._entries.values()
                   if e.state == ShardState.DONE)

    # ── reset stale processing entries ───────────────────────────────

    def reset_stale(self, max_age_s: float = 3600.0):
        """
        On restart, any shard left in 'processing' state was interrupted
        mid-flight. Reset it to 'pending' so it gets reprocessed.
        """
        now = time.time()
        for e in self._entries.values():
            if e.state == ShardState.PROCESSING:
                if now - e.updated_at > max_age_s:
                    e.state      = ShardState.PENDING
                    e.updated_at = now
        self._save()


# ── Checksum helpers ─────────────────────────────────────────────────

def file_checksum(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """SHA-256 of a file, read in chunks to avoid OOM on large files."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def validate_bin_file(path: str | Path, seq_len: int,
                      dtype_bytes: int = 2) -> tuple[bool, int, str]:
    """
    Validate a packed .bin file:
      - Must exist and be non-empty
      - Size must be a multiple of seq_len * dtype_bytes (no partial sequences)

    Returns (is_valid, num_tokens, error_message)
    """
    p = Path(path)
    if not p.exists():
        return False, 0, "file does not exist"
    size = p.stat().st_size
    if size == 0:
        return False, 0, "file is empty"
    token_size = dtype_bytes
    if size % token_size != 0:
        return False, 0, f"size {size} not multiple of dtype_bytes {token_size}"
    num_tokens = size // token_size
    if num_tokens < seq_len + 1:
        return False, 0, f"only {num_tokens} tokens, need at least {seq_len + 1}"
    return True, num_tokens, ""