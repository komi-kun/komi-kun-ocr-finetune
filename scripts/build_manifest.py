#!/usr/bin/env python3
"""Mix real (Manga109/COO) and synthetic samples into PaddleOCR label files."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_labels(path: Path) -> list[tuple[str, str, str, str]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p, text, source, split = line.split("\t")
        rows.append((str(ROOT / p), text, source, split))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=Path, default=ROOT / "data/real/labels.tsv")
    ap.add_argument("--synth", type=Path, default=ROOT / "data/synth/labels.tsv")
    ap.add_argument("--synth-eval", type=Path, default=ROOT / "data/synth_eval/labels.tsv")
    ap.add_argument("--synth-ratio", type=float, default=0.5, help="synthetic samples per real sample (train only)")
    ap.add_argument("--out", type=Path, default=ROOT / "data/manifests")
    args = ap.parse_args()

    real = read_labels(args.real)
    synth = read_labels(args.synth)
    synth_eval = read_labels(args.synth_eval)
    args.out.mkdir(parents=True, exist_ok=True)

    def dump(name: str, rows):
        with open(args.out / name, "w", encoding="utf-8") as f:
            for p, text, _, _ in rows:
                f.write(f"{p}\t{text}\n")
        print(f"{name}: {len(rows)}")

    real_train = [r for r in real if r[3] == "train"]
    real_val = [r for r in real if r[3] == "val"]
    real_test = [r for r in real if r[3] == "test"]
    synth_train = [r for r in synth if r[3] == "train"]
    synth_val = [r for r in synth if r[3] == "val"]

    if real_train:
        k = min(len(synth_train), int(len(real_train) * args.synth_ratio))
        import random

        random.Random(0).shuffle(synth_train)
        train = real_train + synth_train[:k]
    else:
        # synthetic-only bootstrap
        train = synth_train

    dump("train.txt", train)
    # held-out-font synthetic rows are a leakage-free validation set
    dump("val.txt", real_val + synth_eval + synth_val[: max(1, min(len(synth_val), len(real_val) or 2000))])
    dump("test_real.txt", real_test)
    with open(args.out / "test_tsv.txt", "w", encoding="utf-8") as f:
        for p, text, source, _ in real_test:
            f.write(f"{p}\t{text}\t{source}\n")

    print("train sources:", Counter(r[2] for r in train))
    for name, split in (("real_train", real_train), ("real_val", real_val), ("test_real", real_test)):
        print(f"{name}: {len(split)}", Counter(r[2] for r in split))


if __name__ == "__main__":
    main()
