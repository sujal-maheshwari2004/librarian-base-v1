# tokenizer/evaluate_tokenizer.py

from tokenizers import Tokenizer
from pathlib import Path
import statistics


TOKENIZER_PATH = Path("tokenizer/tokenizer.json")
DATA_PATH = Path("data/cleaned/merged_train.txt")

MAX_SAMPLES = 50000  # limit evaluation size for speed


def evaluate():

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError("Tokenizer not found.")

    if not DATA_PATH.exists():
        raise FileNotFoundError("Merged training file not found.")

    print("\n=== Loading Tokenizer ===")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    print("\n=== Loading Data Sample ===")
    lines = DATA_PATH.read_text(encoding="utf-8").splitlines()

    if len(lines) > MAX_SAMPLES:
        lines = lines[:MAX_SAMPLES]

    print(f"Evaluating on {len(lines)} documents...")

    token_counts = []
    total_chars = 0
    total_tokens = 0
    unk_count = 0

    for line in lines:
        total_chars += len(line)

        encoded = tokenizer.encode(line)
        tokens = encoded.ids

        token_counts.append(len(tokens))
        total_tokens += len(tokens)

        unk_id = tokenizer.token_to_id("<unk>")
        if unk_id is not None:
            unk_count += tokens.count(unk_id)

    avg_tokens = statistics.mean(token_counts)
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)

    compression_ratio = total_chars / total_tokens if total_tokens > 0 else 0

    print("\n=== Tokenization Statistics ===")
    print(f"Average tokens per document : {avg_tokens:.2f}")
    print(f"Min tokens per document     : {min_tokens}")
    print(f"Max tokens per document     : {max_tokens}")
    print(f"Total tokens evaluated      : {total_tokens}")
    print(f"Unknown token count         : {unk_count}")
    print(f"Char-to-token ratio         : {compression_ratio:.2f}")

    print("\n=== Example Tokenization ===")

    sample_text = lines[0]
    encoded = tokenizer.encode(sample_text)

    print("\nSample Text:")
    print(sample_text[:200])

    print("\nTokens:")
    print(encoded.tokens[:30])

    print("\nToken IDs:")
    print(encoded.ids[:30])


if __name__ == "__main__":
    evaluate()