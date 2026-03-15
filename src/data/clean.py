# src/data/clean.py

import re
import random
from pathlib import Path


RAW_DIR = Path("data/raw")
CLEANED_DIR = Path("data/cleaned")

MIN_CHARS = 20
MAX_CHARS = 20000

TINY_WEIGHT = 0.6
WIKI_WEIGHT = 0.4
SEED = 42


# ------------------------------------------------
# Cleaning functions
# ------------------------------------------------
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


def passes_quality_checks(text: str) -> bool:
    if not text:
        return False
    if len(text) < MIN_CHARS:
        return False
    if len(text) > MAX_CHARS:
        return False
    return True


def clean_document(text: str) -> str:
    text = normalize_text(text)
    text = fix_wikitext_artifacts(text)
    text = remove_wiki_markup(text)

    if not passes_quality_checks(text):
        return ""

    return text


# ------------------------------------------------
# File cleaning
# ------------------------------------------------
def process_file(file_path: Path):
    print(f"\nCleaning: {file_path.name}")

    raw_text = file_path.read_text(encoding="utf-8")

    # Split on blank lines (documents preserved from download stage)
    documents = re.split(r"\n\s*\n", raw_text)

    cleaned_docs = []
    for doc in documents:
        cleaned = clean_document(doc)
        if cleaned:
            cleaned_docs.append(cleaned)

    output_path = CLEANED_DIR / file_path.name

    with output_path.open("w", encoding="utf-8") as f:
        for doc in cleaned_docs:
            f.write(doc + "\n")

    print(f"Kept {len(cleaned_docs)} documents.")
    return output_path


# ------------------------------------------------
# Merge logic
# ------------------------------------------------
def read_lines(path: Path):
    return path.read_text(encoding="utf-8").splitlines()


def weighted_merge(tiny_lines, wiki_lines):
    random.seed(SEED)

    total = len(tiny_lines) + len(wiki_lines)
    tiny_target = int(total * TINY_WEIGHT)
    wiki_target = int(total * WIKI_WEIGHT)

    tiny_sample = random.sample(tiny_lines, min(tiny_target, len(tiny_lines)))
    wiki_sample = random.sample(wiki_lines, min(wiki_target, len(wiki_lines)))

    merged = tiny_sample + wiki_sample
    random.shuffle(merged)

    return merged


def merge_split(split_name: str):
    tiny_file = CLEANED_DIR / f"tinystories_{split_name}.txt"
    wiki_file = CLEANED_DIR / f"wikitext_{split_name}.txt"

    if not tiny_file.exists() or not wiki_file.exists():
        print(f"Skipping merge for {split_name}")
        return

    tiny_lines = read_lines(tiny_file)
    wiki_lines = read_lines(wiki_file)

    merged = weighted_merge(tiny_lines, wiki_lines)

    output_path = CLEANED_DIR / f"merged_{split_name}.txt"

    with output_path.open("w", encoding="utf-8") as f:
        for line in merged:
            f.write(line + "\n")

    print(f"Merged {split_name}: {len(merged)} samples")


# ------------------------------------------------
# MAIN
# ------------------------------------------------
if __name__ == "__main__":

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== CLEANING STAGE ===")

    for file_path in RAW_DIR.glob("*.txt"):
        process_file(file_path)

    print("\n=== MERGING STAGE ===")

    for split in ["train", "validation", "test"]:
        merge_split(split)

    print("\nData stage complete.")