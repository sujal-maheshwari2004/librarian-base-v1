import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datasets import load_dataset
from pathlib import Path
import random
import os

from src.utils.logging import StageLogger

_RUN_ID = int(os.environ.get("RUN_ID", 0)) or None


def download_datasets(
    use_fraction: float = 1.0,
    seed: int = 42,
    output_dir: str = "data/raw",
    stage_logger: StageLogger | None = None,
):
    assert 0 < use_fraction <= 1.0, "use_fraction must be between 0 and 1"

    log = stage_logger

    random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------
    # TinyStories
    # ----------------------------------------
    if log: log.progress("download", {"dataset": "TinyStories", "status": "downloading"})
    print("Downloading TinyStories...")
    tiny = load_dataset("roneneldan/TinyStories")

    tiny_files = 0
    tiny_rows  = 0

    for split in tiny.keys():
        dataset = tiny[split]

        if use_fraction < 1.0:
            dataset = dataset.shuffle(seed=seed)
            keep_size = int(len(dataset) * use_fraction)
            dataset = dataset.select(range(keep_size))

        file_path = output_path / f"tinystories_{split}.txt"

        with file_path.open("w", encoding="utf-8") as f:
            for example in dataset:
                text = example.get("text", "").strip()
                if text:
                    f.write(text + "\n\n")

        tiny_files += 1
        tiny_rows  += len(dataset)
        print(f"Saved {file_path}")

        if log:
            log.progress("download", {
                "tinystories_split": split,
                "rows": len(dataset),
                "file": file_path.name,
            })

    # ----------------------------------------
    # WikiText
    # ----------------------------------------
    if log: log.progress("download", {"dataset": "WikiText-103", "status": "downloading"})
    print("Downloading WikiText...")
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1")

    wiki_files = 0
    wiki_rows  = 0

    for split in wiki.keys():
        dataset = wiki[split]

        if use_fraction < 1.0:
            dataset = dataset.shuffle(seed=seed)
            keep_size = int(len(dataset) * use_fraction)
            dataset = dataset.select(range(keep_size))

        file_path = output_path / f"wikitext_{split}.txt"

        with file_path.open("w", encoding="utf-8") as f:
            for example in dataset:
                text = example.get("text", "").strip()
                if text:
                    f.write(text + "\n\n")

        wiki_files += 1
        wiki_rows  += len(dataset)
        print(f"Saved {file_path}")

        if log:
            log.progress("download", {
                "wikitext_split": split,
                "rows": len(dataset),
                "file": file_path.name,
            })

    print("\nDownload complete.")

    return {
        "tinystories_splits": tiny_files,
        "tinystories_rows":   tiny_rows,
        "wikitext_splits":    wiki_files,
        "wikitext_rows":      wiki_rows,
        "total_files":        tiny_files + wiki_files,
    }


if __name__ == "__main__":
    log = StageLogger(run_id=_RUN_ID)
    log.start("download")
    try:
        summary = download_datasets(use_fraction=0.33, stage_logger=log)
        log.end("download", summary)
    except Exception as e:
        log.error("download", str(e))
        raise