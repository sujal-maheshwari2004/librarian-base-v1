# src/data/tokenize.py

from tokenizers import Tokenizer
from pathlib import Path
import numpy as np
from tqdm import tqdm


# ------------------------------------------------
# Config
# ------------------------------------------------
TOKENIZER_PATH = Path("tokenizer/tokenizer.json")

CLEANED_DIR = Path("data/cleaned")
TOKENIZED_DIR = Path("data/tokenized")

DTYPE = np.uint16  # since vocab < 65535


# ------------------------------------------------
# Encode one split
# ------------------------------------------------
def encode_split(split_name: str):

    input_file = CLEANED_DIR / f"merged_{split_name}.txt"
    output_file = TOKENIZED_DIR / f"{split_name}.bin"

    if not input_file.exists():
        print(f"Skipping {split_name} (file not found)")
        return

    print(f"\nEncoding {split_name}...")

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    lines = input_file.read_text(encoding="utf-8").splitlines()

    all_ids = []

    for line in tqdm(lines, desc=f"Tokenizing {split_name}"):
        encoded = tokenizer.encode(line)
        all_ids.extend(encoded.ids)

    arr = np.array(all_ids, dtype=DTYPE)
    arr.tofile(output_file)

    print(f"Saved {output_file}")
    print(f"Total tokens: {len(arr)}")
    print(f"Memory size: {arr.nbytes / (1024**2):.2f} MB")


# ------------------------------------------------
# Main
# ------------------------------------------------
if __name__ == "__main__":

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError("Tokenizer not found. Train it first.")

    TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== TOKENIZATION STAGE ===")

    for split in ["train", "validation", "test"]:
        encode_split(split)

    print("\nTokenization complete.")