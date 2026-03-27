import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import re
import random
from pathlib import Path

from src.utils.logging import StageLogger


RAW_DIR     = Path("data/raw")
CLEANED_DIR = Path("data/cleaned")


# ── Quality thresholds (tuned for FineWeb-Edu style corpus) ─────────────

MIN_CHARS       = 100
MAX_CHARS       = 80_000
MIN_ALPHA_RATIO = 0.62
MAX_DIGIT_RATIO = 0.15


# ── Merge weights (decoder pretraining tuned for 390M model) ───────────

WEIGHTS = {
    "wikitext":    0.30,
    "bookcorpus":  0.50,   # FineWeb-Edu replacement signal
    "openwebtext": 0.20,
}


SEED    = 42
_RUN_ID = int(os.environ.get("RUN_ID", 0)) or None


# ── Text normalization helpers ─────────────────────────────────────────


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


def remove_boilerplate(text: str) -> str:

    lower = text.lower()

    patterns = [
        r"^(cookie|privacy) policy",
        r"^all rights reserved",
        r"^copyright \d{4}",
        r"^subscribe (to|now)",
        r"^(sign|log) (in|up)",
        r"^\d+\s*(comments?|shares?|likes?|views?)",
        r"^(click here|read more|see also)",
        r"^advertisement",
        r"^this (article|page) (was|is)",
    ]

    for p in patterns:
        if re.match(p, lower):
            return ""

    return text


def remove_citations(text: str) -> str:

    text = re.sub(r"\[[0-9]+\]", "", text)
    text = re.sub(r"\[[^\]]*citation[^\]]*\]", "", text, flags=re.IGNORECASE)

    return text


def alpha_ratio(text: str) -> float:

    if not text:
        return 0.0

    return sum(
        1 for c in text if c.isalpha() or c.isspace()
    ) / len(text)


def digit_ratio(text: str) -> float:

    if not text:
        return 0.0

    return sum(
        1 for c in text if c.isdigit()
    ) / len(text)


def passes_quality_checks(text: str) -> bool:

    if not text:
        return False

    if len(text) < MIN_CHARS:
        return False

    if len(text) > MAX_CHARS:
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

    if not passes_quality_checks(text):
        return ""

    return text


# ── File cleaning ─────────────────────────────────────────────────────


def process_file(file_path: Path) -> int:

    print(f"\nCleaning: {file_path.name}")

    if file_path.name.startswith("wikitext"):
        source = "wikitext"

    elif file_path.name.startswith("bookcorpus"):
        source = "bookcorpus"

    elif file_path.name.startswith("openwebtext"):
        source = "openwebtext"

    else:
        source = ""

    raw_text  = file_path.read_text(encoding="utf-8")

    documents = re.split(r"\n\s*\n", raw_text)

    cleaned = [
        clean_document(doc, source=source)
        for doc in documents
    ]

    cleaned = [doc for doc in cleaned if doc]

    output_path = CLEANED_DIR / file_path.name

    with output_path.open("w", encoding="utf-8") as f:

        for doc in cleaned:
            f.write(doc + "\n")

    print(f"  Kept {len(cleaned):,} / {len(documents):,} documents")

    return len(cleaned)


# ── Merge helpers ─────────────────────────────────────────────────────


def read_lines(path: Path):

    if not path.exists():
        return []

    return [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def weighted_merge_train():

    random.seed(SEED)

    sources = {

        "wikitext": read_lines(
            CLEANED_DIR / "wikitext_train.txt"
        ),

        "bookcorpus": read_lines(
            CLEANED_DIR / "bookcorpus_train.txt"
        ),

        "openwebtext": read_lines(
            CLEANED_DIR / "openwebtext_train.txt"
        ),
    }

    total_available = sum(
        len(v) for v in sources.values()
    )

    if total_available == 0:

        print("  No training data found.")
        return []

    print("\n  Available lines per source:")

    for src, lines in sources.items():

        print(f"    {src:15s}: {len(lines):>10,}")

    sampled = []

    for src, weight in WEIGHTS.items():

        lines = sources[src]

        if not lines:

            print(f"  WARNING: {src} has no data, skipping.")
            continue

        target = int(total_available * weight)

        take = min(target, len(lines))

        sampled.extend(random.sample(lines, take))

        print(f"  Sampled {take:,} from {src}")

    random.shuffle(sampled)

    print(f"\n  Total merged train lines: {len(sampled):,}")

    return sampled


def merge_split(split_name: str, stage_logger=None):

    print(f"\n── Merging split: {split_name} ──────────────────────")

    if split_name == "train":

        merged = weighted_merge_train()

    else:

        merged = read_lines(
            CLEANED_DIR / f"wikitext_{split_name}.txt"
        )

        random.seed(SEED)
        random.shuffle(merged)

        print(
            f"  {split_name}: {len(merged):,} lines (WikiText only)"
        )

    if not merged:
        return 0

    output_path = CLEANED_DIR / f"merged_{split_name}.txt"

    with output_path.open("w", encoding="utf-8") as f:

        for line in merged:
            f.write(line + "\n")

    if stage_logger:

        stage_logger.progress(
            "clean",
            {f"merged_{split_name}": len(merged)}
        )

    return len(merged)


# ── Main entrypoint ───────────────────────────────────────────────────


def run_clean(stage_logger=None):

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    log = stage_logger

    print("\n=== CLEANING STAGE ===")

    total_docs = 0
    files_processed = 0

    for file_path in sorted(RAW_DIR.glob("*.txt")):

        kept = process_file(file_path)

        total_docs += kept
        files_processed += 1

        if log:

            log.progress(
                "clean",
                {
                    "file": file_path.name,
                    "docs_kept": kept,
                    "files_done": files_processed,
                },
            )

    print("\n=== MERGING STAGE ===")

    merged_totals = {}

    for split in ["train", "validation", "test"]:

        n = merge_split(split, stage_logger=log)

        merged_totals[f"merged_{split}"] = n

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
