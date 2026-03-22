from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFKC
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from pathlib import Path
import json


# ------------------------------------------------
# Configuration
# ------------------------------------------------
DATA_FILE = Path("../src/data/data/cleaned/merged_train.txt")
TOKENIZER_DIR = Path("tokenizer")

VOCAB_SIZE = 16000
MIN_FREQUENCY = 2

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


# ------------------------------------------------
# Training Function
# ------------------------------------------------
def train_tokenizer():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Merged training file not found at data/cleaned/merged_train.txt"
        )

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Initializing Tokenizer ===")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    # Unicode normalization
    tokenizer.normalizer = NFKC()

    # GPT-style byte-level pre-tokenization
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS
    )

    print("Training tokenizer...")
    tokenizer.train([str(DATA_FILE)], trainer)

    print("Adding BOS/EOS post-processing...")

    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )

    # Proper ByteLevel decoder
    tokenizer.decoder = ByteLevelDecoder()

    print("Saving tokenizer files...")

    tokenizer.save(str(TOKENIZER_DIR / "tokenizer.json"))

    # Save readable config
    config = {
        "vocab_size": tokenizer.get_vocab_size(),
        "model": "BPE",
        "pre_tokenizer": "ByteLevel",
        "normalizer": "NFKC",
        "min_frequency": MIN_FREQUENCY,
        "special_tokens": SPECIAL_TOKENS
    }

    with open(TOKENIZER_DIR / "tokenizer_config.json", "w") as f:
        json.dump(config, f, indent=4)

    print("\nTokenizer training complete.")
    print(f"Final vocab size: {tokenizer.get_vocab_size()}")


# ------------------------------------------------
# Main
# ------------------------------------------------
if __name__ == "__main__":
    train_tokenizer()
