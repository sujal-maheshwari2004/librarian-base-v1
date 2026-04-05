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

import json
import os
import tempfile
import time
from pathlib import Path

from src.utils.logging import StageLogger
from src.data.clean import iter_documents_from_shards, CLEANED_DIR
from src.pipeline.atomic_writer import recover_stranded_tmps

TOKENIZER_DIR  = Path("tokenizer")
VOCAB_SIZE     = 32000
MIN_FREQUENCY  = 2
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

_RUN_ID = int(os.environ.get("RUN_ID", 0)) or None


def _write_temp_corpus(tmp_path: Path, stage_logger=None) -> int:
    """
    Stream all cleaned shards into a single temporary text file that
    the HuggingFace BpeTrainer can read (it requires a file path, not
    an iterator).  The temp file is deleted after training completes.

    Returns the number of lines written.
    """
    if not CLEANED_DIR.exists():
        raise FileNotFoundError(
            f"Cleaned shards directory not found: {CLEANED_DIR}\n"
            "Run the clean stage before training the tokenizer."
        )

    # ── Recover any stranded .tmp shards left by an interrupted clean
    # run before scanning for .txt files. ────────────────────────────
    recovered = recover_stranded_tmps(CLEANED_DIR, src_ext=".tmp", dst_ext=".txt")
    if recovered:
        print(f"[train_tokenizer] Recovered {recovered} stranded .tmp shards.")

    shard_files = sorted(CLEANED_DIR.rglob("*.txt"))
    if not shard_files:
        raise FileNotFoundError(
            f"No cleaned shard files found under {CLEANED_DIR}\n"
            "Run the clean stage before training the tokenizer."
        )

    print(f"[train_tokenizer] Streaming {len(shard_files)} shard(s) into temp corpus…")

    lines_written = 0
    with tmp_path.open("w", encoding="utf-8") as out:
        for doc in iter_documents_from_shards(CLEANED_DIR):
            out.write(doc + "\n")
            lines_written += 1
            if lines_written % 500_000 == 0:
                print(f"[train_tokenizer]   {lines_written:,} lines streamed…")
                if stage_logger:
                    stage_logger.progress("train_tokenizer", {
                        "lines_streamed": lines_written,
                    })

    print(f"[train_tokenizer] Corpus ready: {lines_written:,} lines")
    return lines_written


def train_tokenizer(stage_logger=None):
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Initializing Tokenizer ===")

    if stage_logger:
        stage_logger.progress("train_tokenizer", {
            "vocab_size":    VOCAB_SIZE,
            "min_frequency": MIN_FREQUENCY,
            "shards_dir":    str(CLEANED_DIR),
        })

    # Stream cleaned shards into a temp file (BpeTrainer needs a path)
    with tempfile.TemporaryDirectory(prefix="librarian_tok_") as tmp_dir:
        corpus_path = Path(tmp_dir) / "corpus.txt"
        lines = _write_temp_corpus(corpus_path, stage_logger)

        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.normalizer    = NFKC()
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)

        trainer = BpeTrainer(
            vocab_size=VOCAB_SIZE,
            min_frequency=MIN_FREQUENCY,
            special_tokens=SPECIAL_TOKENS,
        )

        print("Training tokenizer…")
        t0 = time.time()
        tokenizer.train([str(corpus_path)], trainer)
        train_elapsed = time.time() - t0
        # corpus_path is deleted automatically when the with-block exits

    print("Adding BOS/EOS post-processing…")
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )
    tokenizer.decoder = ByteLevelDecoder()

    print("Saving tokenizer files…")
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
    print(f"Final vocab size : {tokenizer.get_vocab_size()}")
    print(f"Lines trained on : {lines:,}")
    print(f"Elapsed          : {train_elapsed:.1f}s")

    return {
        "final_vocab_size": tokenizer.get_vocab_size(),
        "lines_trained_on": lines,
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
