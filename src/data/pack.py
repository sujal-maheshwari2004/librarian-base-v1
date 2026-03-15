# src/data/pack.py

import numpy as np
from pathlib import Path


# ------------------------------------------------
# Config
# ------------------------------------------------
TOKENIZED_DIR = Path("data/tokenized")
PACKED_DIR = Path("data/tokenized")  # can keep same folder

CONTEXT_LENGTH = 512
DTYPE = np.uint16


# ------------------------------------------------
# Pack one split
# ------------------------------------------------
def pack_split(split_name: str):

    input_file = TOKENIZED_DIR / f"{split_name}.bin"
    output_file = PACKED_DIR / f"{split_name}_packed.bin"

    if not input_file.exists():
        print(f"Skipping {split_name} (file not found)")
        return

    print(f"\nPacking {split_name}...")

    data = np.fromfile(input_file, dtype=DTYPE)

    total_tokens = len(data)
    num_sequences = total_tokens // CONTEXT_LENGTH

    trimmed_tokens = num_sequences * CONTEXT_LENGTH

    packed = data[:trimmed_tokens]

    packed.tofile(output_file)

    print(f"Original tokens: {total_tokens}")
    print(f"Packed tokens  : {trimmed_tokens}")
    print(f"Sequences      : {num_sequences}")
    print(f"Saved to       : {output_file}")
    print(f"Final size     : {packed.nbytes / (1024**2):.2f} MB")


# ------------------------------------------------
# Main
# ------------------------------------------------
if __name__ == "__main__":

    print("\n=== PACKING STAGE ===")

    for split in ["train", "validation", "test"]:
        pack_split(split)

    print("\nPacking complete.")