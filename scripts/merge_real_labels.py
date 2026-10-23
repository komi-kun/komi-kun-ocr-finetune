#!/usr/bin/env python3
"""Merge and sanitise the real-data label files.

* dialogue labels from label_lines.py  (data/real/labels_dialogue.tsv)
* COO SFX labels from extract_coo_sfx.py (data/real/labels_sfx.tsv)

Drops labels longer than the recogniser's max_text_length or containing
characters outside the shipped dictionary (PaddleOCR silently skips those, which
would corrupt supervision).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import load_dict  # noqa: E402

MAX_LEN = 25


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogue", type=Path, default=ROOT / "data/real/labels_dialogue.tsv")
    ap.add_argument("--sfx", type=Path, default=ROOT / "data/real/labels_sfx.tsv")
    ap.add_argument("--out", type=Path, default=ROOT / "data/real/labels.tsv")
    args = ap.parse_args()

    vocab = set(load_dict())
    rows, dropped = [], Counter()
    for path in (args.dialogue, args.sfx):
        if not path.exists():
            print(f"missing {path} (skipped)")
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            p, text, source, split = line.split("\t")
            text = "".join(text.split())
            if not text:
                dropped["empty"] += 1
                continue
            if len(text) > MAX_LEN:
                dropped["too_long"] += 1
                continue
            bad = [c for c in text if c != " " and c not in vocab]
            if bad:
                dropped["oov"] += 1
                continue
            rows.append((p, text, source, split))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write("\t".join(r) + "\n")

    print(f"kept {len(rows)}  dropped {dict(dropped)}")
    print("by split :", Counter(r[3] for r in rows))
    print("by source:", Counter(r[2] for r in rows))


if __name__ == "__main__":
    main()
