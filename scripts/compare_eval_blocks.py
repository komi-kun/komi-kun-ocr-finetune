#!/usr/bin/env python3
"""Paired block comparison of two `eval_blocks.py --json-out` dumps.

Both runs must use the same recogniser; only the detector changes.  The two runs
can evaluate slightly different block sets (a block is skipped when the detector
returns no boxes on its page), so the paired numbers are computed on the
intersection of (book, page, block) keys, and the dropped blocks are reported.

    python scripts/compare_eval_blocks.py output/eval_blocks/stock_test.json \
                                          output/eval_blocks/cand_test.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def load(path: Path) -> tuple[dict[tuple[str, int, int], dict], dict]:
    doc = json.loads(path.read_text())
    return {(r["book"], r["page"], r["block"]): r for r in doc["records"]}, doc


def binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    a, da = load(Path(sys.argv[1]))
    b, db = load(Path(sys.argv[2]))
    common = sorted(set(a) & set(b))

    def cer(keys, d):
        num = sum(d[k]["cer"] * len(d[k]["gt"]) for k in keys)
        den = sum(len(d[k]["gt"]) for k in keys)
        return num / den if den else 0.0

    def exact(keys, d):
        return sum(d[k]["exact"] for k in keys) / len(keys) if keys else 0.0

    only_a, only_b = len(set(a) - set(b)), len(set(b) - set(a))
    deltas = [b[k]["cer"] - a[k]["cer"] for k in common]
    improved = sum(1 for d in deltas if d < -1e-12)
    worsened = sum(1 for d in deltas if d > 1e-12)
    b01 = sum(1 for k in common if not a[k]["exact"] and b[k]["exact"])  # fixed
    c01 = sum(1 for k in common if a[k]["exact"] and not b[k]["exact"])  # broken

    print(f"base {da['det_model']}  rec {da['rec_model']}")
    print(f"cand {db['det_model']}")
    print(f"blocks base {len(a)} cand {len(b)} paired {len(common)} "
          f"(only-base {only_a}, only-cand {only_b})")
    print(f"block CER   base {100*cer(common, a):.2f} %  cand {100*cer(common, b):.2f} %  "
          f"delta {100*(cer(common, b)-cer(common, a)):+.2f} pp")
    print(f"block exact base {100*exact(common, a):.2f} %  cand {100*exact(common, b):.2f} %  "
          f"delta {100*(exact(common, b)-exact(common, a)):+.2f} pp")
    print(f"per-block CER improved {improved}  worsened {worsened}  "
          f"equal {len(common)-improved-worsened}  mean delta {100*sum(deltas)/max(1,len(deltas)):+.3f} pp")
    print(f"exact: fixed {b01}  broken {c01}  McNemar/binomial two-sided p = {binom_two_sided(min(b01, c01), b01 + c01):.4f}")
    print(f"full-set block CER base {100*da['block_cer']:.2f} %  cand {100*db['block_cer']:.2f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
