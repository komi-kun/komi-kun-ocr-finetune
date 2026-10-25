#!/usr/bin/env python3
"""Paired block comparison of two or more `eval_blocks.py --json-out` dumps.

Each dump is one `(detector, recogniser)` combination, i.e. one shipping release
(`ocr-models-v3` = stock det + stock rec, `ocr-manga-v2` = stock det + fine-tuned
rec, `ocr-models-v4` = fine-tuned det + stock rec, `ocr-manga-v3` = both
fine-tuned). Every run can evaluate a slightly different block set (a block is
skipped when the detector returns no boxes on its page), so the paired numbers
are computed on the intersection of the (book, page, block) keys across **all**
dumps, and each run's dropped blocks are reported.

Run 1 is the reference: every delta, flip count and p-value is run 1 vs run i.
The binomial ("McNemar") test is two-sided on the exact-match flips (fixed vs
broken) and treats them as a sign test, which is the right test for the same
blocks scored by two models.

    python scripts/compare_eval_blocks.py \\
        output/eval_blocks/stockdet_stockrec_test.json \\
        output/eval_blocks/stockdet_ftrec_test.json \\
        output/eval_blocks/exp2ep40det_stockrec_test.json \\
        output/eval_blocks/exp2ep40det_ftrec_test.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

COL = 11


def load(path: Path) -> tuple[dict[tuple[str, int, int], dict], dict]:
    doc = json.loads(path.read_text())
    return {(r["book"], r["page"], r["block"]): r for r in doc["records"]}, doc


def binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    runs = [load(Path(p)) for p in sys.argv[1:]]
    common = sorted(set.intersection(*[set(recs) for recs, _ in runs]))

    def cer(keys, d):
        num = sum(d[k]["cer"] * len(d[k]["gt"]) for k in keys)
        den = sum(len(d[k]["gt"]) for k in keys)
        return num / den if den else 0.0

    def exact(keys, d):
        return sum(d[k]["exact"] for k in keys) / len(keys) if keys else 0.0

    print("runs (run #1 is the reference for every delta):")
    for i, (recs, doc) in enumerate(runs, 1):
        print(
            f"  #{i} det {doc['det_sha256'][:8]} rec {doc['rec_sha256'][:8]}"
            f"  ({Path(doc['det_model']).name} + {Path(doc['rec_model']).name})"
        )
        print(
            f"      full set: blocks {len(recs)} / {doc['blocks_with_gt']} with GT text"
            f" ({doc['blocks_missing']} unread), CER {100 * doc['block_cer']:.2f} %"
            f", exact {100 * doc['block_exact']:.2f} %"
        )

    print(f"\npaired on the intersection of all {len(runs)} runs: {len(common)} blocks")
    for i, (recs, _) in enumerate(runs, 1):
        only = len(set(recs) - set(common))
        print(f"  #{i} blocks not in the paired set: {only}")

    head = f"{'':>22s}" + "".join(f"{'#' + str(i):>{COL}s}" for i in range(1, len(runs) + 1))
    print("\n" + head)
    print(f"{'block CER':>22s}" + "".join(f"{100 * cer(common, r):>{COL - 2}.2f} %" for r, _ in runs))
    print(f"{'block exact':>22s}" + "".join(f"{100 * exact(common, r):>{COL - 2}.2f} %" for r, _ in runs))

    print("")
    base = runs[0][0]
    for i, (recs, _) in enumerate(runs[1:], 2):
        deltas = [recs[k]["cer"] - base[k]["cer"] for k in common]
        improved = sum(1 for d in deltas if d < -1e-12)
        worsened = sum(1 for d in deltas if d > 1e-12)
        fixed = sum(1 for k in common if not base[k]["exact"] and recs[k]["exact"])
        broken = sum(1 for k in common if base[k]["exact"] and not recs[k]["exact"])
        print(
            f"#1 -> #{i}: block CER {100 * (cer(common, recs) - cer(common, base)):+.2f} pp"
            f"  exact {100 * (exact(common, recs) - exact(common, base)):+.2f} pp"
            f"  per-block CER improved {improved} / worsened {worsened} / equal {len(common) - improved - worsened}"
            f"  exact flips fixed {fixed} / broken {broken}"
            f"  p = {binom_two_sided(min(fixed, broken), fixed + broken):.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
