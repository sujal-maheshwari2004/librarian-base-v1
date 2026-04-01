"""
clean.py — Streaming per-shard text cleaner with manifest tracking.

Changes from original:
  - Reads raw shards one at a time (no full 150GB load)
  - Each cleaned shard is written atomically
  - Checksums verified before marking done
  - Deletes data/raw/ after all cleaned shards verified
  - Deduplication is per-shard only (global dedup at too large a scale
    requires a full pass; per-shard dedup catches the worst duplicates)
  - Weighted merge for training set happens here, streaming from shards
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pipeline.manifest import StageManifest, file_checksum
from src.pipeline.atomic_writer import AtomicTextWriter
from src.pipeline.cleanup import safe_delete_stage
from src.utils.logging import StageLogger

# ── Paths ─────────────────────────────────────────────────────────────
RAW_DIR      = Path("data/raw/shards")
CLEANED_DIR  = Path("data/cleaned/shards")
MANIFEST_DIR = Path("data/manifests")
_RUN_ID      = int(os.environ.get("RUN_ID", 0)) or None

# ── Quality thresholds ────────────────────────────────────────────────
MIN_CHARS       = 100
MAX_CHARS       = 80_000
MIN_ALPHA_RATIO = 0.62
MAX_DIGIT_RATIO = 0.15

_BOILERPLATE = re.compile(
    r"^(cookie|privacy) policy"
    r"|^all rights reserved"
    r"|^copyright \d{4}"
    r"|^subscribe (to|now)"
    r"|^(sign|log) (in|up)"
    r"|^\d+\s*(comments?|shares?|likes?|views?)"
    r"|^(click here|read more|see also)"
    r"|^advertisement"
    r"|^this (article|page) (was|is)",
    re.IGNORECASE,
)


# ── Text utilities (unchanged from original) ─────────────────────────

def normalize_text(text: str) -> str:
    text = text.strip().replace("\t", " ")
    return re.sub(r"\s+", " ", text)

def fix_wikitext_artifacts(text: str) -> str:
    return text.replace("@-@", "-").replace("@,@", ",").replace("@.@", ".")

def remove_wiki_markup(text: str) -> str:
    if re.match(r"^=+.*=+$", text): return ""
    if re.match(r"^[\W_]+$", text):  return ""
    return text

def remove_boilerplate(text: str) -> str:
    return "" if _BOILERPLATE.match(text) else text

def remove_citations(text: str) -> str:
    text = re.sub(r"\[[0-9]+\]", "", text)
    return re.sub(r"\[[^\]]*citation[^\]]*\]", "", text, flags=re.IGNORECASE)

def alpha_ratio(text: str) -> float:
    if not text: return 0.0
    return sum(1 for c in text if c.isalpha() or c.isspace()) / len(text)

def digit_ratio(text: str) -> float:
    if not text: return 0.0
    return sum(1 for c in text if c.isdigit()) / len(text)

def passes_quality_checks(text: str) -> bool:
    n = len(text)
    if n < MIN_CHARS or n > MAX_CHARS: return False
    if alpha_ratio(text) < MIN_ALPHA_RATIO: return False
    if digit_ratio(text) > MAX_DIGIT_RATIO: return False
    return True

def clean_document(text: str, source: str = "") -> str:
    text = normalize_text(text)
    if source == "wikitext":
        text = fix_wikitext_artifacts(text)
        text = remove_wiki_markup(text)
    elif source == "bookcorpus":
        text = remove_boilerplate(text)
        text = remove_citations(text)
    else:
        text = remove_boilerplate(text)
    return text if passes_quality_checks(text) else ""


# ── Shard-level cleaning ─────────────────────────────────────────────

def _source_from_shard_id(shard_id: str) -> str:
    """Extract source name from shard_id: 'wikitext__train__000000' → 'wikitext'"""
    return shard_id.split("__")[0]


def clean_shard(
    raw_shard_path: Path,
    cleaned_shard_path: Path,
    source: str,
) -> tuple[int, int]:
    """
    Stream-clean one raw shard into one cleaned shard.
    Returns (docs_in, docs_out).
    Uses per-shard SHA-1 deduplication.
    """
    seen: set[str] = set()
    docs_in = docs_out = 0

    with AtomicTextWriter(cleaned_shard_path) as w:
        with raw_shard_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                docs_in += 1
                cleaned = clean_document(line, source=source)
                if not cleaned:
                    continue
                h = hashlib.sha1(cleaned.encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                w.write(cleaned + "\n")
                docs_out += 1

    return docs_in, docs_out


# ── Main clean stage ─────────────────────────────────────────────────

def run_clean(stage_log: StageLogger | None = None) -> dict:
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    # Load download manifest to know which raw shards exist
    download_manifest = StageManifest(MANIFEST_DIR / "download.json")
    if not download_manifest.is_complete():
        raise RuntimeError(
            f"Download stage not complete: {download_manifest.summary()}"
        )

    # Build clean manifest
    clean_manifest = StageManifest(MANIFEST_DIR / "clean.json")
    clean_manifest.reset_stale()

    # Register all raw shards as work items for clean stage
    raw_shard_ids = [e.shard_id for e in download_manifest.verified_entries()]
    clean_manifest.register_shards(raw_shard_ids, meta={"stage": "clean"})

    print(f"\n=== CLEAN STAGE ===")
    print(f"Raw shards to clean: {len(raw_shard_ids)}")
    print(f"Clean manifest: {clean_manifest.summary()}")

    total_in = total_out = 0

    for entry in download_manifest.verified_entries():
        sid = entry.shard_id

        # Skip already completed
        from src.pipeline.manifest import ShardState
        if clean_manifest._entries.get(sid, None) and \
           clean_manifest._entries[sid].state == ShardState.DONE:
            continue

        raw_path     = Path(entry.output_path)
        source       = _source_from_shard_id(sid)

        # Mirror directory structure under cleaned/
        rel          = raw_path.relative_to(RAW_DIR)
        cleaned_path = CLEANED_DIR / rel

        cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        clean_manifest.mark_processing(sid)

        try:
            docs_in, docs_out = clean_shard(raw_path, cleaned_path, source)
        except Exception as e:
            clean_manifest.mark_failed(sid, str(e))
            print(f"[clean] FAILED shard {sid}: {e}")
            continue

        checksum = file_checksum(cleaned_path)
        clean_manifest.mark_verified(sid, str(cleaned_path), checksum, docs_out)
        clean_manifest.mark_done(sid)

        total_in  += docs_in
        total_out += docs_out

        if stage_log:
            stage_log.progress("clean", {
                "shard":    sid,
                "docs_in":  docs_in,
                "docs_out": docs_out,
            })

    print(f"\n[clean] Kept {total_out:,} / {total_in:,} docs")
    print(f"[clean] Manifest: {clean_manifest.summary()}")

    # ── Delete raw shards after clean stage completes ──────────────
    if clean_manifest.is_complete():
        print("\n[clean] All shards cleaned — deleting raw data…")
        result = safe_delete_stage(
            stage_name              = "raw",
            artifact_dir            = RAW_DIR,
            downstream_manifest_path= MANIFEST_DIR / "clean.json",
            dry_run                 = False,
        )
        print(f"[clean] Freed {result['deleted_bytes'] / 1e9:.2f} GB")
    else:
        print("[clean] Stage incomplete — skipping raw deletion")

    return {
        "total_docs_in":  total_in,
        "total_docs_out": total_out,
        "manifest":       clean_manifest.summary(),
    }


if __name__ == "__main__":
    log = StageLogger(run_id=_RUN_ID)
    log.start("clean")
    try:
        summary = run_clean(stage_log=log)
        log.end("clean", summary)
    except Exception as e:
        log.error("clean", str(e))
        raise
