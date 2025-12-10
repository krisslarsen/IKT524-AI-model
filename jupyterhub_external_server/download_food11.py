#!/usr/bin/env python3
"""
download_food11.py
Download the Food-11 dataset from Kaggle via kagglehub and copy it into a chosen folder.

Usage:
  python download_food11.py --out /datasets/dataset-03-Food-11
  # or
  python download_food11.py --out ~/datasets/dataset-03-Food-11

Options:
  --force   Overwrite the output folder if it already exists.

Prereqs:
  pip install kagglehub
"""

import argparse
import shutil
from pathlib import Path
import sys

DATASET_REF = "trolukovich/food11-image-dataset"

def main():
    parser = argparse.ArgumentParser(description="Download Food-11 dataset to a target folder.")
    parser.add_argument("--out", type=Path, default=Path("/datasets/dataset-03-Food-11"),
                        help="Target folder to place the dataset (default: /datasets/dataset-03-Food-11).")
    parser.add_argument("--force", action="store_true",
                        help="If set, remove the target folder first if it exists.")
    args = parser.parse_args()

    try:
        import kagglehub
    except Exception as e:
        print("ERROR: 'kagglehub' is not installed. Run: pip install kagglehub", file=sys.stderr)
        sys.exit(1)

    print(f"[1/3] Downloading via kagglehub: {DATASET_REF}")
    cache_path = kagglehub.dataset_download(DATASET_REF)
    if not cache_path:
        print("ERROR: kagglehub did not return a path. Check dataset ref or authentication.", file=sys.stderr)
        sys.exit(2)

    cache_dir = Path(cache_path).resolve()
    print(f"[2/3] KaggleHub cache at: {cache_dir}")

    out_dir = args.out.expanduser().resolve()
    if out_dir.exists():
        if args.force:
            print(f"[force] Removing existing folder: {out_dir}")
            shutil.rmtree(out_dir)
        else:
            print(f"[skip] Output folder already exists: {out_dir}")
            print("       Use --force to overwrite, or choose a different --out path.")
            sys.exit(0)

    print(f"[3/3] Copying to: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cache_dir, out_dir, dirs_exist_ok=True)
    print("✅ Done.")
    print(f"Path to dataset files: {out_dir}")

if __name__ == "__main__":
    main()
