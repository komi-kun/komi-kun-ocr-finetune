#!/usr/bin/env python3
"""Download the gated Manga109 / Manga109-s release from Hugging Face.

Requires an HF account that has accepted the dataset terms:
  https://huggingface.co/datasets/hal-utokyo/Manga109-s   (auto-gated)
  https://huggingface.co/datasets/hal-utokyo/Manga109      (manual review)

Then either `hf auth login` (or `huggingface-cli login`) once, or set HF_TOKEN.

Usage:
  python scripts/fetch_manga109.py --repo hal-utokyo/Manga109-s
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="hal-utokyo/Manga109-s", choices=["hal-utokyo/Manga109-s", "hal-utokyo/Manga109"])
    ap.add_argument("--dest", type=Path, default=RAW)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    dest = args.dest / args.repo.split("/")[-1]
    dest.mkdir(parents=True, exist_ok=True)
    print(f"downloading {args.repo} -> {dest}")
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=["*.zip", "*.txt", "*.md"],
    )

    for z in sorted(dest.glob("*.zip")):
        out = dest / z.stem
        if out.exists() and any(out.iterdir()):
            print(f"already extracted: {out}")
            continue
        print(f"extracting {z.name} ...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        # normalise: if the zip contained a single folder, note it
        print("done")

    # Report the layout we expect by build_real_dataset.py
    for p in sorted(dest.rglob("*")):
        if p.is_dir() and (p / "images").is_dir():
            print(f"images dir: {p / 'images'}")
        if p.is_dir() and (p / "annotations").is_dir():
            print(f"annotations dir: {p / 'annotations'}")


if __name__ == "__main__":
    main()
