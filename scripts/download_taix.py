"""Download the TAIX-Ray `default` config (all 215,381 images, ~58 GB of parquet) locally.

Usage:
    python scripts/download_taix.py [--data-dir data/TAIX-Ray] [--revision <sha>]

Re-running is safe: files already present are skipped.
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

DATASET_ID = "TLAIM/TAIX-Ray"
# Revision used for the team's first full-data run; pin it so everyone trains on the same files.
DEFAULT_REVISION = "7f43720e2c7ab6400608880c0cb8b950ac0e5afb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("TAIX_DATA_DIR", "data/TAIX-Ray"))
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.data_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET_ID}@{args.revision[:8]} (config 'default') -> {out}")
    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=args.revision,
        local_dir=out,
        allow_patterns=["data/*.parquet", "README.md"],
        max_workers=args.workers,
        token=os.environ.get("HF_TOKEN"),
    )
    files = sorted((out / "data").glob("*.parquet"))
    total = sum(f.stat().st_size for f in files)
    print(f"Done: {len(files)} parquet files, {total / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
