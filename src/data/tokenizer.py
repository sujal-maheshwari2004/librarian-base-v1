import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tokenizers import Tokenizer
from pathlib import Path
import numpy as np
import os
from tqdm import tqdm

from src.utils.logging import StageLogger

TOKENIZER_PATH = Path("tokenizer/tokenizer.json")
CLEANED_DIR    = Path("data/cleaned")
TOKENIZED_DIR  = Path("data/tokenized")
DTYPE          = np.uint16

_RUN_ID = int(os.environ.get("RUN_ID", 0)) or None


def encode_split(split_name: str, stage_logger: StageLogger | None = None):
    input_file  = CLEANED_DIR / f"merged_{split_name}.txt"
    output_file = TOKENIZED_DIR / f"{split_name}.bin"

    if not input_file.exists():
        print(f"Skipping {split_name} (file not found)")
        return None

    print(f"\nEncoding {split_name}...")

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    lines     = input_file.read_text(encoding="utf-8").splitlines()
    all_ids   = []

    for i, line in enumerate(tqdm(lines, desc=f"Tokenizing {split_name}")):
        encoded = tokenizer.encode(line)
        all_ids.extend(encoded.ids)

        # emit progress every 10 %
        if stage_logger and len(lines) > 0 and (i+1) % max(1, len(lines)//10) == 0:
            pct = round(100 * (i+1) / len(lines))
            stage_logger.progress("tokenize", {
                f"{split_name}_progress_pct": pct,
                f"{split_name}_tokens_so_far": len(all_ids),
            })

    arr = np.array(all_ids, dtype=DTYPE)
    arr.tofile(output_file)

    print(f"Saved {output_file}")
    print(f"Total tokens: {len(arr)}")
    print(f"Memory size: {arr.nbytes / (1024**2):.2f} MB")

    return {
        "split":       split_name,
        "total_tokens": int(len(arr)),
        "size_mb":      round(arr.nbytes / (1024**2), 2),
    }


def run_tokenize(stage_logger: StageLogger | None = None):
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError("Tokenizer not found. Train it first.")

    TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== TOKENIZATION STAGE ===")

    results = {}
    for split in ["train", "validation", "test"]:
        info = encode_split(split, stage_logger=stage_logger)
        if info:
            results[f"{split}_tokens"] = info["total_tokens"]
            results[f"{split}_mb"]     = info["size_mb"]

    print("\nTokenization complete.")
    return results


if __name__ == "__main__":
    log = StageLogger(run_id=_RUN_ID)
    log.start("tokenize")
    try:
        summary = run_tokenize(stage_logger=log)
        log.end("tokenize", summary)
    except Exception as e:
        log.error("tokenize", str(e))
        raise