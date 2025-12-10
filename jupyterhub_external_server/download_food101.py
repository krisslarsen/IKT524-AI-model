#!/usr/bin/env python3
"""
download_food101.py  (v3, robust nested layout support)
Downloads Food-101 via kagglehub and ensures final layout:
  <out>/images/
  <out>/meta/

Handles:
- cache/food-101/images + cache/food-101/meta
- cache/food-101/food-101/images + meta  (double-nested)
- monolithic food-101.* archive
- separate images.* and meta.* archives
- already extracted at version root

Usage:
  python download_food101.py --out /home/kristoffel/datasets/dataset-02-Food-101 --force
"""

import argparse
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple, List

DATASET_REF = "dansbecker/food-101"

def extract_any(archive: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    if name.endswith(('.tar.gz', '.tgz', '.tar')):
        with tarfile.open(archive, 'r:*') as tf:
            tf.extractall(dest)
    elif name.endswith('.zip'):
        with zipfile.ZipFile(archive, 'r') as zf:
            zf.extractall(dest)
    else:
        raise ValueError(f"Unsupported archive format: {archive}")

def iter_dirs_upto_depth(root: Path, max_depth: int = 3):
    # Breadth-first traversal yielding directories up to max_depth relative to root
    queue = [(root, 0)]
    seen = set()
    while queue:
        cur, depth = queue.pop(0)
        if cur in seen: 
            continue
        seen.add(cur)
        yield cur, depth
        if depth < max_depth:
            try:
                for p in cur.iterdir():
                    if p.is_dir():
                        queue.append((p, depth+1))
            except Exception:
                pass

def find_images_meta(roots: List[Path], max_depth: int = 3) -> Tuple[Optional[Path], Optional[Path]]:
    for r in roots:
        for p, d in iter_dirs_upto_depth(r, max_depth=max_depth):
            img = p / "images"
            met = p / "meta"
            if img.is_dir() and met.is_dir():
                return img, met
    return None, None

def main():
    parser = argparse.ArgumentParser(description="Download Food-101 to a target folder (robust extractor v3).")
    parser.add_argument("--out", type=Path, default=Path("/datasets/dataset-02-Food-101"),
                        help="Target folder to place the dataset.")
    parser.add_argument("--force", action="store_true", help="Remove the target folder first if it exists.")
    args = parser.parse_args()

    try:
        import kagglehub
    except Exception:
        print("ERROR: 'kagglehub' is not installed. Run: pip install kagglehub", file=sys.stderr)
        sys.exit(1)

    print(f"[1/6] Downloading via kagglehub: {DATASET_REF}")
    cache_path = kagglehub.dataset_download(DATASET_REF)
    if not cache_path:
        print("ERROR: kagglehub did not return a path. Check dataset ref or authentication.", file=sys.stderr)
        sys.exit(2)
    cache_dir = Path(cache_path).resolve()
    print(f"[2/6] KaggleHub cache at: {cache_dir}")

    out_dir = args.out.expanduser().resolve()
    if out_dir.exists():
        if args.force:
            print(f"[force] Removing existing folder: {out_dir}")
            shutil.rmtree(out_dir)
        else:
            print(f"[skip] Output folder already exists: {out_dir}")
            print("       Use --force to overwrite, or choose a different --out path.")
            sys.exit(0)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Candidate roots to search
    candidates = [cache_dir]
    if (cache_dir / "food-101").exists():
        candidates.append(cache_dir / "food-101")
    # Some mirrors nest twice (food-101/food-101)
    if (cache_dir / "food-101" / "food-101").exists():
        candidates.append(cache_dir / "food-101" / "food-101")

    # 1) Try to locate ready-made directories (nested search up to depth=3)
    images_dir, meta_dir = find_images_meta(candidates, max_depth=3)
    if images_dir and meta_dir:
        print(f"[3/6] Found images/meta under: {images_dir.parent}")
    else:
        print("[3/6] No ready 'images/' and 'meta/' directories found. Searching for archives...")
        images_archive = None
        meta_archive   = None
        food101_archive = None
        exts = ('.tar.gz', '.tgz', '.tar', '.zip')
        for root in candidates:
            for p in root.glob("*"):
                if not p.is_file():
                    continue
                low = p.name.lower()
                if low.startswith('images') and low.endswith(exts):
                    images_archive = p
                elif low.startswith('meta') and low.endswith(exts):
                    meta_archive = p
                elif low.startswith('food-101') and low.endswith(exts):
                    food101_archive = p

        if food101_archive:
            print(f"  • Extracting monolithic archive: {food101_archive.name}")
            extract_any(food101_archive, out_dir)
            # search again in out_dir
            images_dir, meta_dir = find_images_meta([out_dir], max_depth=2)

        else:
            if images_archive:
                print(f"  • Extracting images archive: {images_archive.name}")
                extract_any(images_archive, out_dir)
            if meta_archive:
                print(f"  • Extracting meta archive: {meta_archive.name}")
                extract_any(meta_archive, out_dir)
            images_dir, meta_dir = find_images_meta([out_dir], max_depth=2)

        if not (images_dir and meta_dir):
            print("ERROR: Could not locate 'images/' and 'meta/' after extraction.", file=sys.stderr)
            sys.exit(3)

    # 2) Finalize into canonical layout
    final_images = out_dir / "images"
    final_meta   = out_dir / "meta"

    if images_dir.resolve() != final_images.resolve():
        print(f"[4/6] Copying images -> {final_images}")
        shutil.copytree(images_dir, final_images)
    else:
        print(f"[4/6] images already at {final_images}")

    if meta_dir.resolve() != final_meta.resolve():
        print(f"[5/6] Copying meta -> {final_meta}")
        shutil.copytree(meta_dir, final_meta)
    else:
        print(f"[5/6] meta already at {final_meta}")

    # 3) Verify
    num_classes = len([d for d in final_images.iterdir() if d.is_dir()]) if final_images.exists() else 0
    print("[6/6] Verify:")
    print(f"  images: {final_images} ({num_classes} class folders)")
    print(f"  meta  : {final_meta}")
    print("✅ Done.")
    print(f"Path to dataset files: {out_dir}")

if __name__ == "__main__":
    main()
