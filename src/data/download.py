from datasets import load_dataset
from pathlib import Path
import random


def download_datasets(
    use_fraction: float = 1.0,
    seed: int = 42,
    output_dir: str = "data/raw"
):
    assert 0 < use_fraction <= 1.0, "use_fraction must be between 0 and 1"

    random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # TinyStories
    # -----------------------------
    print("Downloading TinyStories...")
    tiny = load_dataset("roneneldan/TinyStories")

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
                    # IMPORTANT: double newline preserves document boundary
                    f.write(text + "\n\n")

        print(f"Saved {file_path}")

    # -----------------------------
    # WikiText
    # -----------------------------
    print("Downloading WikiText...")
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1")

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

        print(f"Saved {file_path}")

    print("\nDownload complete.")


if __name__ == "__main__":
    download_datasets(use_fraction=0.001)