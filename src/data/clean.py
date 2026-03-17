import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import re
import random
import os
from pathlib import Path

from src.utils.logging import StageLogger

RAW_DIR     = Path("data/raw")
CLEANED_DIR = Path("data/cleaned")

MIN_CHARS   = 20
MAX_CHARS   = 20000
TINY_WEIGHT = 0.6
WIKI_WEIGHT = 0.4
SEED        = 42

_RUN_ID = int(os.environ.get("RUN_ID", 0)) or None


# ── cleaning helpers ────────────────────────────────────────
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
    if re.match(r"^=+.*=+$", text): return ""
    if re.match(r"^[\W_]+$", text):  return ""
    return text

def passes_quality_checks(text: str) -> bool:
    if not text:             return False
    if len(text) < MIN_CHARS: return False
    if len(text) > MAX_CHARS: return False
    return True

def clean_document(text: str) -> str:
    text = normalize_text(text)
    text = fix_wikitext_artifacts(text)
    text = remove_wiki_markup(text)
    if not passes_quality_checks(text): return ""
    return text


# ── file cleaning ───────────────────────────────────────────
def process_file(file_path: Path):
    print(f"\nCleaning: {file_path.name}")
    raw_text   = file_path.read_text(encoding="utf-8")
    documents  = re.split(r"\n\s*\n", raw_text)
    cleaned    = [clean_document(d) for d in documents]
    cleaned    = [d for d in cleaned if d]
    output_path = CLEANED_DIR / file_path.name
    with output_path.open("w", encoding="utf-8") as f:
        for doc in cleaned:
            f.write(doc + "\n")
    print(f"Kept {len(cleaned)} documents.")
    return len(cleaned)


# ── merge logic ─────────────────────────────────────────────
def read_lines(path: Path):
    return path.read_text(encoding="utf-8").splitlines()

def weighted_merge(tiny_lines, wiki_lines):
    random.seed(SEED)
    total        = len(tiny_lines) + len(wiki_lines)
    tiny_sample  = random.sample(tiny_lines, min(int(total * TINY_WEIGHT), len(tiny_lines)))
    wiki_sample  = random.sample(wiki_lines, min(int(total * WIKI_WEIGHT), len(wiki_lines)))
    merged       = tiny_sample + wiki_sample
    random.shuffle(merged)
    return merged

def merge_split(split_name: str):
    tiny_file = CLEANED_DIR / f"tinystories_{split_name}.txt"
    wiki_file = CLEANED_DIR / f"wikitext_{split_name}.txt"
    if not tiny_file.exists() or not wiki_file.exists():
        print(f"Skipping merge for {split_name}")
        return 0
    merged = weighted_merge(read_lines(tiny_file), read_lines(wiki_file))
    output_path = CLEANED_DIR / f"merged_{split_name}.txt"
    with output_path.open("w", encoding="utf-8") as f:
        for line in merged:
            f.write(line + "\n")
    print(f"Merged {split_name}: {len(merged)} samples")
    return len(merged)


# ── main ────────────────────────────────────────────────────
def run_clean(stage_logger: StageLogger | None = None):
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    log = stage_logger

    print("\n=== CLEANING STAGE ===")
    total_docs = 0
    files_processed = 0

    for file_path in RAW_DIR.glob("*.txt"):
        kept = process_file(file_path)
        total_docs     += kept
        files_processed += 1
        if log:
            log.progress("clean", {
                "file":           file_path.name,
                "docs_kept":      kept,
                "files_done":     files_processed,
            })

    print("\n=== MERGING STAGE ===")
    merged_totals = {}
    for split in ["train", "validation", "test"]:
        n = merge_split(split)
        merged_totals[f"merged_{split}"] = n
        if log:
            log.progress("clean", {f"merged_{split}": n})

    print("\nData stage complete.")

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