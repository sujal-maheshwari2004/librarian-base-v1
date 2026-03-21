import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datasets import load_dataset
from pathlib import Path
import random
import os

from src.utils.logging import StageLogger

_RUN_ID = int(os.environ.get("RUN_ID", 0)) or None


def _save_split(dataset, file_path: Path, log, tag: str) -> int:
    """Write a HF dataset split to a flat .txt file, return doc count."""
    rows = 0
    with file_path.open("w", encoding="utf-8") as f:
        for example in dataset:
            text = (example.get("text") or "").strip()
            if text:
                f.write(text + "\n\n")
                rows += 1
    if log:
        log.progress("download", {
            "file": file_path.name,
            "rows": rows,
            "tag":  tag,
        })
    print(f"  Saved {file_path.name}  ({rows:,} docs)")
    return rows


def download_datasets(
    seed: int = 42,
    output_dir: str = "data/raw",
    stage_logger: StageLogger | None = None,
):
    log = stage_logger
    random.seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = {}

    # ────────────────────────────────────────────────────────
    # 1.  WikiText-103
    # ────────────────────────────────────────────────────────
    print("\n── WikiText-103 ──────────────────────────────────────")
    if log:
        log.progress("download", {"dataset": "WikiText-103", "status": "downloading"})

    wiki      = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    wiki_rows = 0

    for split in wiki.keys():
        rows = _save_split(
            wiki[split],
            output_path / f"wikitext_{split}.txt",
            log,
            f"wiki/{split}",
        )
        wiki_rows += rows

    summary["wikitext_splits"] = len(wiki)
    summary["wikitext_rows"]   = wiki_rows

    # ────────────────────────────────────────────────────────
    # 2.  BookCorpus
    # ────────────────────────────────────────────────────────
    print("\n── BookCorpus ────────────────────────────────────────")
    if log:
        log.progress("download", {"dataset": "BookCorpus", "status": "downloading"})

    books     = load_dataset("bookcorpus", trust_remote_code=True)
    book_rows = 0

    for split in books.keys():
        rows = _save_split(
            books[split],
            output_path / f"bookcorpus_{split}.txt",
            log,
            f"books/{split}",
        )
        book_rows += rows

    summary["bookcorpus_splits"] = len(books)
    summary["bookcorpus_rows"]   = book_rows

    # ────────────────────────────────────────────────────────
    # 3.  OpenWebText
    # ────────────────────────────────────────────────────────
    print("\n── OpenWebText ───────────────────────────────────────")
    if log:
        log.progress("download", {"dataset": "OpenWebText", "status": "downloading"})

    owt      = load_dataset("openwebtext", trust_remote_code=True)
    owt_rows = 0

    for split in owt.keys():
        rows = _save_split(
            owt[split],
            output_path / f"openwebtext_{split}.txt",
            log,
            f"owt/{split}",
        )
        owt_rows += rows

    summary["openwebtext_splits"] = len(owt)
    summary["openwebtext_rows"]   = owt_rows

    summary["total_files"] = (
        summary["wikitext_splits"] +
        summary["bookcorpus_splits"] +
        summary["openwebtext_splits"]
    )

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
