import os
import re
import random
import hashlib
from pathlib import Path
from typing import Generator

from src.utils.logging import StageLogger


RAW_DIR     = Path("data/raw")
CLEANED_DIR = Path("data/cleaned")

MIN_CHARS       = 100
MAX_CHARS       = 80_000
MIN_ALPHA_RATIO = 0.62
MAX_DIGIT_RATIO = 0.15

WEIGHTS = {
    "wikitext":    0.30,
    "bookcorpus":  0.50,
    "openwebtext": 0.20,
}

SEED    = 42
_RUN_ID = int(os.environ.get("RUN_ID", 0)) or None

_READ_CHUNK = 64 * 1024 * 1024  # 64 MB


# ── Text normalization ────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    text = text.strip()
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text

def fix_wikitext_artifacts(text: str) -> str:
    text = text.replace("@-@", "-")
    text = text.replace("@,@", ",")
    text = text.replace("@.@", ".")
    return text

def remove_wiki_markup(text: str) -> str:
    if re.match(r"^=+.*=+$", text):
        return ""
    if re.match(r"^[\W_]+$", text):
        return ""
    return text

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

def remove_boilerplate(text: str) -> str:
    return "" if _BOILERPLATE.match(text) else text

def remove_citations(text: str) -> str:
    text = re.sub(r"\[[0-9]+\]", "", text)
    text = re.sub(r"\[[^\]]*citation[^\]]*\]", "", text, flags=re.IGNORECASE)
    return text

def alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if c.isalpha() or c.isspace()) / len(text)

def digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if c.isdigit()) / len(text)

def passes_quality_checks(text: str) -> bool:
    n = len(text)
    if n < MIN_CHARS or n > MAX_CHARS:
        return False
    if alpha_ratio(text) < MIN_ALPHA_RATIO:
        return False
    if digit_ratio(text) > MAX_DIGIT_RATIO:
        return False
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


# ── Streaming document iterator ───────────────────────────────────────

def iter_documents(file_path: Path) -> Generator[str, None, None]:
    """
    Yield double-newline-separated documents without loading the full
    file into memory. Carries a leftover buffer across 64 MB chunks.
    """
    buffer = ""
    with file_path.open("r", encoding="utf-8", errors="replace") as fh:
        while True:
            chunk = fh.read(_READ_CHUNK)
            if not chunk:
                break
            buffer += chunk
            parts = re.split(r"\n\s*\n", buffer)
            for part in parts[:-1]:
                part = part.strip()
                if part:
                    yield part
            buffer = parts[-1]
    if buffer.strip():
        yield buffer.strip()


# ── File cleaning ─────────────────────────────────────────────────────

def process_file(file_path: Path) -> int:
    print(f"\nCleaning: {file_path.name}")

    name = file_path.name
    if name.startswith("wikitext"):
        source = "wikitext"
    elif name.startswith("bookcorpus"):
        source = "bookcorpus"
    elif name.startswith("openwebtext"):
        source = "openwebtext"
    else:
        source = ""

    output_path  = CLEANED_DIR / file_path.name
    seen_hashes: set[str] = set()
    total = 0
    kept  = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for doc in iter_documents(file_path):
            total += 1
            cleaned = clean_document(doc, source=source)
            if not cleaned:
                continue
            doc_hash = hashlib.sha1(cleaned.encode()).hexdigest()
            if doc_hash in seen_hashes:
                continue
            seen_hashes.add(doc_hash)
            out_f.write(cleaned + "\n")
            kept += 1

    print(f"  Kept {kept:,} / {total:,} documents  "
          f"({len(seen_hashes):,} unique)")
    return kept


# ── Streaming helpers ─────────────────────────────────────────────────

def iter_lines(path: Path) -> Generator[str, None, None]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.strip():
                yield line

def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


# ── Weighted merge (streaming reservoir sampling) ─────────────────────

def weighted_merge_train(stage_logger=None) -> int:
    random.seed(SEED)

    paths  = {src: CLEANED_DIR / f"{src}_train.txt" for src in WEIGHTS}
    counts = {src: count_lines(p) for src, p in paths.items()}

    total_available = sum(counts.values())
    if total_available == 0:
        print("  No training data found.")
        return 0

    print("\n  Available lines per source:")
    for src, n in counts.items():
        print(f"    {src:15s}: {n:>12,}")

    # Compute targets; redistribute shortfall proportionally
    raw_targets = {src: int(total_available * w) for src, w in WEIGHTS.items()}
    take        = {}
    shortfall   = 0
    for src, target in raw_targets.items():
        actual       = min(target, counts[src])
        take[src]    = actual
        shortfall   += target - actual

    if shortfall > 0:
        surplus_sources = {
            src: counts[src] - take[src]
            for src in take if counts[src] > take[src]
        }
        total_surplus = sum(surplus_sources.values())
        if total_surplus > 0:
            for src, surplus in surplus_sources.items():
                extra     = int(shortfall * surplus / total_surplus)
                take[src] = min(take[src] + extra, counts[src])

    all_sampled: list[str] = []

    for src, n_take in take.items():
        if n_take == 0:
            print(f"  WARNING: {src} has no data, skipping.")
            continue

        reservoir: list[str] = []
        for i, line in enumerate(iter_lines(paths[src])):
            if len(reservoir) < n_take:
                reservoir.append(line)
            else:
                j = random.randint(0, i)
                if j < n_take:
                    reservoir[j] = line

        all_sampled.extend(reservoir)
        print(f"  Sampled {len(reservoir):,} from {src}")

        if stage_logger:
            stage_logger.progress("clean", {
                "merge_source": src,
                "sampled":      len(reservoir),
            })

    random.shuffle(all_sampled)

    output_path = CLEANED_DIR / "merged_train.txt"
    with output_path.open("w", encoding="utf-8") as f:
        for line in all_sampled:
            f.write(line + "\n")

    total = len(all_sampled)
    print(f"\n  Total merged train lines: {total:,}")
    return total


# ── Validation / test merge ───────────────────────────────────────────

def merge_split_other(split_name: str, stage_logger=None) -> int:
    print(f"\n── Merging split: {split_name} ──────────────────────")

    src_path    = CLEANED_DIR / f"wikitext_{split_name}.txt"
    output_path = CLEANED_DIR / f"merged_{split_name}.txt"

    lines = list(iter_lines(src_path))
    if not lines:
        print(f"  {split_name}: no data found, skipping.")
        return 0

    random.seed(SEED)
    random.shuffle(lines)

    with output_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"  {split_name}: {len(lines):,} lines (WikiText only)")

    if stage_logger:
        stage_logger.progress("clean", {f"merged_{split_name}": len(lines)})

    return len(lines)


# ── Main entrypoint ───────────────────────────────────────────────────

def run_clean(stage_logger=None) -> dict:
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== CLEANING STAGE ===")

    total_docs      = 0
    files_processed = 0

    for file_path in sorted(RAW_DIR.glob("*.txt")):
        kept = process_file(file_path)
        total_docs      += kept
        files_processed += 1

        if stage_logger:
            stage_logger.progress("clean", {
                "file":       file_path.name,
                "docs_kept":  kept,
                "files_done": files_processed,
            })

    print("\n=== MERGING STAGE ===")

    merged_totals                  = {}
    merged_totals["merged_train"]  = weighted_merge_train(stage_logger=stage_logger)

    for split in ["validation", "test"]:
        merged_totals[f"merged_{split}"] = merge_split_other(
            split, stage_logger=stage_logger
        )

    print("\nData cleaning complete.")
    return {
        "files_processed": files_processed,
        "total_docs_kept": total_docs,
        **merged_totals,
    }


if __name__ == "__main__":
    log = StageLogger(run_id=_RUN_ID)
    log.start("clean")
    try:
        summary = run_clean(stage_logger=log)
        log.end("clean", summary)
    except Exception as e:
        log.error("clean", str(e))
        raise
