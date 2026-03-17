import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import os
from pathlib import Path

from src.utils.logging import StageLogger

TOKENIZED_DIR   = Path("data/tokenized")
PACKED_DIR      = Path("data/tokenized")
CONTEXT_LENGTH  = 512
DTYPE           = np.uint16

_RUN_ID = int(os.environ.get("RUN_ID", 0)) or None


def pack_split(split_name: str, stage_logger: StageLogger | None = None):
    input_file  = TOKENIZED_DIR / f"{split_name}.bin"
    output_file = PACKED_DIR    / f"{split_name}_packed.bin"

    if not input_file.exists():
        print(f"Skipping {split_name} (file not found)")
        return None

    print(f"\nPacking {split_name}...")

    data           = np.fromfile(input_file, dtype=DTYPE)
    total_tokens   = len(data)
    num_sequences  = total_tokens // CONTEXT_LENGTH
    trimmed_tokens = num_sequences * CONTEXT_LENGTH
    packed         = data[:trimmed_tokens]
    packed.tofile(output_file)

    size_mb = packed.nbytes / (1024**2)

    print(f"Original tokens: {total_tokens}")
    print(f"Packed tokens  : {trimmed_tokens}")
    print(f"Sequences      : {num_sequences}")
    print(f"Saved to       : {output_file}")
    print(f"Final size     : {size_mb:.2f} MB")

    result = {
        f"{split_name}_original_tokens":  int(total_tokens),
        f"{split_name}_packed_tokens":    int(trimmed_tokens),
        f"{split_name}_sequences":        int(num_sequences),
        f"{split_name}_size_mb":          round(size_mb, 2),
    }

    if stage_logger:
        stage_logger.progress("pack", result)

    return result


def run_pack(stage_logger: StageLogger | None = None):
    print("\n=== PACKING STAGE ===")

    combined = {}
    for split in ["train", "validation", "test"]:
        info = pack_split(split, stage_logger=stage_logger)
        if info:
            combined.update(info)

    print("\nPacking complete.")
    return combined


if __name__ == "__main__":
    log = StageLogger(run_id=_RUN_ID)
    log.start("pack")
    try:
        summary = run_pack(stage_logger=log)
        log.end("pack", summary)
    except Exception as e:
        log.error("pack", str(e))
        raise