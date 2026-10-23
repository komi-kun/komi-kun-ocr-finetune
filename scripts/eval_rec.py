#!/usr/bin/env python3
"""Evaluate a recogniser ONNX on line crops (CER / exact match, overall + per source)."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import RecModel  # noqa: E402


def load_rows(path: Path) -> list[tuple[str, str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        p, text = parts[0], parts[1]
        source = parts[2] if len(parts) > 2 else "?"
        rows.append((p, text, source))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=ROOT / "models/ppocrv6_small_rec_manga.onnx")
    ap.add_argument("--dict", type=Path, default=ROOT / "models/ppocrv6_small_dict.txt")
    ap.add_argument("--labels", type=Path, default=ROOT / "data/manifests/test_tsv.txt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show", type=int, default=10, help="print N worst samples")
    args = ap.parse_args()

    rows = load_rows(args.labels)
    if args.limit:
        rows = rows[: args.limit]
    model = RecModel(onnx_path=args.model, dict_path=args.dict)

    stats = defaultdict(lambda: {"n": 0, "cer": 0.0, "em": 0})
    worst = []
    for i, (p, gt, source) in enumerate(rows, 1):
        img = cv2.imread(str(ROOT / p) if not Path(p).is_absolute() else p)
        if img is None:
            continue
        hyp, _ = model(img)
        d = Levenshtein.distance(hyp, gt)
        c = d / max(1, len(gt))
        for key in ("all", source):
            s = stats[key]
            s["n"] += 1
            s["cer"] += c
            s["em"] += int(hyp == gt)
        worst.append((c, gt, hyp, p))
        if i % 500 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    print(f"\nmodel: {args.model}")
    print(f"{'subset':22s} {'n':>7s} {'CER %':>8s} {'exact %':>8s}")
    for key, s in sorted(stats.items()):
        if not s["n"]:
            continue
        print(f"{key:22s} {s['n']:7d} {100*s['cer']/s['n']:8.2f} {100*s['em']/s['n']:8.2f}")

    worst.sort(reverse=True)
    print(f"\nworst {min(args.show, len(worst))}:")
    for c, gt, hyp, p in worst[: args.show]:
        print(f"  cer={c:.2f} src={Path(p).name}\n    gt ={gt}\n    hyp={hyp}")


if __name__ == "__main__":
    main()
