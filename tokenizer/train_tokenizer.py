import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFKC
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from pathlib import Path
import json
import os
import time

from src.utils.logging import StageLogger

DATA_FILE      = Path("data/cleaned/merged_train.txt")
TOKENIZER_DIR  = Path("tokenizer")
VOCAB_SIZE     = 32000
MIN_FREQUENCY  = 2
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

_RUN_ID = int(os.environ.get("RUN_ID", 0)) or None


def train_tokenizer(stage_logger: StageLogger | None = None):
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Merged training file not found at data/cleaned/merged_train.txt"
        )

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Initializing Tokenizer ===")

    if stage_logger:
        stage_logger.progress("train_tokenizer", {
            "vocab_size":    VOCAB_SIZE,
            "min_frequency": MIN_FREQUENCY,
            "data_file":     DATA_FILE.name,
        })

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer    = NFKC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
    )

    print("Training tokenizer...")
    t0 = time.time()
    tokenizer.train([str(DATA_FILE)], trainer)
    train_elapsed = time.time() - t0

    print("Adding BOS/EOS post-processing...")
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )
    tokenizer.decoder = ByteLevelDecoder()

    print("Saving tokenizer files...")
    tokenizer.save(str(TOKENIZER_DIR / "tokenizer.json"))

    config = {
        "vocab_size":     tokenizer.get_vocab_size(),
        "model":          "BPE",
        "pre_tokenizer":  "ByteLevel",
        "normalizer":     "NFKC",
        "min_frequency":  MIN_FREQUENCY,
        "special_tokens": SPECIAL_TOKENS,
    }
    with open(TOKENIZER_DIR / "tokenizer_config.json", "w") as f:
        json.dump(config, f, indent=4)

    print("\nTokenizer training complete.")
    print(f"Final vocab size: {tokenizer.get_vocab_size()}")

    return {
        "final_vocab_size": tokenizer.get_vocab_size(),
        "train_elapsed_s":  round(train_elapsed, 2),
        "special_tokens":   len(SPECIAL_TOKENS),
    }


if __name__ == "__main__":
    log = StageLogger(run_id=_RUN_ID)
    log.start("train_tokenizer")
    try:
        summary = train_tokenizer(stage_logger=log)
        log.end("train_tokenizer", summary)
    except Exception as e:
        log.error("train_tokenizer", str(e))
        raise
