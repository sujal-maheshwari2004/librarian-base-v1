"""
Robust HF dataset downloader (datasets==2.2.1 safe)

Features:
• Dynamic shard discovery
• Mirror fallback support
• Retry logic
• Streaming parquet reading
• No datasets.load_dataset()
"""

import os
import time
import random
from pathlib import Path
from typing import List

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

from src.utils.logging import StageLogger


MAX_RETRIES = 5
WAIT = 5


DATASET_CONFIGS = {

    "wikitext": [
        ("Salesforce/wikitext",
         "wikitext-103-raw-v1/train"),
    ],

    # replacing BookCorpus with FineWeb-Edu
    "bookcorpus": [
        ("HuggingFaceFW/fineweb-edu",
         "train"),
    ],

    "openwebtext": [
        ("Skylion007/openwebtext",
         "train"),
    ],
}


def discover(repo_id, pattern):

    files = list_repo_files(repo_id, repo_type="dataset")

    shards = [
        f for f in files
        if f.endswith(".parquet") and pattern in f
    ]

    return shards


def download(repo, shard):

    for attempt in range(MAX_RETRIES):

        try:

            return hf_hub_download(
                repo_id=repo,
                filename=shard,
                repo_type="dataset",
                token=os.environ.get("HF_TOKEN"),
            )

        except Exception:

            if attempt == MAX_RETRIES - 1:
                raise

            time.sleep(WAIT * (2 ** attempt))


def parquet_to_txt(src, dst):

    pf = pq.ParquetFile(src)

    rows = 0

    with open(dst, "a", encoding="utf-8") as f:

        for rg in range(pf.num_row_groups):

            batch = pf.read_row_group(rg)

            for row in batch.to_pylist():

                text = (row.get("text") or "").strip()

                if text:

                    f.write(text + "\n\n")
                    rows += 1

    return rows


def download_dataset(name, output_dir):

    mirrors = DATASET_CONFIGS[name]

    output_file = output_dir / f"{name}_train.txt"

    if output_file.exists():

        print(f"{name}: already exists")
        return

    for repo, pattern in mirrors:

        print(f"\n{name}: trying {repo}")

        try:

            shards = discover(repo, pattern)

            if not shards:
                continue

            total = 0

            for shard in shards:

                print(f" downloading {shard}")

                local = download(repo, shard)

                total += parquet_to_txt(local, output_file)

            print(f"{name}: success ({total:,} docs)")

            return

        except Exception as e:

            print(f"{repo} failed → {e}")

    raise RuntimeError(f"{name} failed on all mirrors")


def download_datasets(seed=42,
                      output_dir="data/raw",
                      stage_logger=None):

    random.seed(seed)

    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True,
                     exist_ok=True)

    print("\nDirect parquet download mode\n")

    for dataset in DATASET_CONFIGS:

        download_dataset(dataset,
                         output_dir)

    print("\nDownload complete.\n")
