#!/usr/bin/env python3
"""
download_nutrition5k.py
Download the Nutrition5k dataset from Kaggle via kagglehub and copy it into a chosen folder.

By default this fetches: siddhantrout/nutrition5k-dataset
(An alternative with JPEGs is zygmuntyt/nutrition5k-dataset-side-angle-images.)

Usage:
  python download_nutrition5k.py --out /home/kristoffel/datasets/dataset-01-Nutrition5k
  python download_nutrition5k.py --out ~/datasets/dataset-01-Nutrition5k --force

Options:
  --ref    Kaggle dataset ref (default: siddhantrout/nutrition5k-dataset)
  --force  Overwrite the output folder if it already exists.

Prereqs:
  pip install kagglehub
"""

import argparse
import shutil
from pathlib import Path
import sys

DEFAULT_REF = "siddhantrout/nutrition5k-dataset"

def main():
    parser = argparse.ArgumentParser(description="Download Nutrition5k dataset to a target folder.")
    parser.add_argument("--out", type=Path, default=Path("/datasets/dataset-01-Nutrition5k"),
                        help="Target folder to place the dataset (default: /datasets/dataset-01-Nutrition5k).")
    parser.add_argument("--ref", type=str, default=DEFAULT_REF,
                        help=f"Kaggle dataset reference. Default: {DEFAULT_REF}")
    parser.add_argument("--force", action="store_true",
                        help="If set, remove the target folder first if it exists.")
    args = parser.parse_args()

    try:
        import kagglehub
    except Exception:
        print("ERROR: 'kagglehub' is not installed. Run: pip install kagglehub", file=sys.stderr)
        sys.exit(1)

    print(f"[1/3] Downloading via kagglehub: {args.ref}")
    cache_path = kagglehub.dataset_download(args.ref)
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
    print("Note: If this ref stores images as bytes inside tables, use the training notebook to materialize JPEGs.")

if __name__ == "__main__":
    main()
