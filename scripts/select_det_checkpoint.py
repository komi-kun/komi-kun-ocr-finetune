#!/usr/bin/env python3
"""Pick a detector checkpoint on VALIDATION metrics with a constrained objective.

Experiment 1 showed that PaddleOCR's `DetMetric` hmean cannot see a dialogue
collapse: its dialogue GT is block/bubble-level, so suppressing line boxes raises
hmean while destroying dialogue.  This selector uses the independent
`scripts/eval_det.py` numbers instead and never looks at test.

Rule (documented, pre-registered before the sweep):

  1. constraint - dialogue block coverage >= stock_val - ``--max-dia-drop`` pp
     and art false positives/page <= stock_val * ``--max-art-ratio``;
  2. objective  - maximise SFX H at IoU 0.5, tie-break SFX crop recall @0.4.

Exit 0 and print the winner only if at least one candidate satisfies the
constraint; exit 1 otherwise.  Inputs are the ``--out`` JSONs of `eval_det.py`
run with ``--split val``.

    python scripts/select_det_checkpoint.py --stock output/eval_det/stock_val.json \
        output/eval_det/exp2_ep05_val.json output/eval_det/exp2_ep10_val.json ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    doc = json.loads(path.read_text())
    assert doc["split"] == "val", f"{path}: not a val evaluation ({doc['split']})"
    return {
        "path": str(path),
        "model": doc["model"],
        "sfx_h": doc["classes"]["sfx"]["hmean"],
        "sfx_r": doc["classes"]["sfx"]["recall"],
        "sfx_p": doc["classes"]["sfx"]["precision"],
        "crop": doc["sfx_crop_recall"]["recall"],
        "dia": doc["dialogue_block_coverage"]["recall"],
        "art": doc["false_positives"]["art_per_page"],
    }


def select(stock: dict, cands: list[dict], max_dia_drop: float, max_art_ratio: float) -> tuple[dict | None, list[dict]]:
    rows = []
    for c in cands:
        dia_ok = c["dia"] >= stock["dia"] - max_dia_drop / 100.0
        art_ok = c["art"] <= stock["art"] * max_art_ratio
        c["passes"] = dia_ok and art_ok
        c["dia_delta_pp"] = 100 * (c["dia"] - stock["dia"])
        c["sfx_h_delta_pp"] = 100 * (c["sfx_h"] - stock["sfx_h"])
        rows.append(c)
    ok = [r for r in rows if r["passes"]]
    best = max(ok, key=lambda r: (r["sfx_h"], r["crop"])) if ok else None
    return best, rows


def self_test() -> int:
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(("PASS  " if cond else "FAIL  ") + name + (f"  [{detail}]" if detail and not cond else ""))
        ok = ok and bool(cond)

    stock = {"dia": 0.94, "art": 7.0, "sfx_h": 0.40, "crop": 0.41}
    base = {"path": "", "model": "", "sfx_r": 0.5, "sfx_p": 0.6}
    good = dict(base, model="good", sfx_h=0.61, crop=0.68, dia=0.93, art=2.0)
    low_dia = dict(base, model="low_dia", sfx_h=0.66, crop=0.68, dia=0.25, art=1.4)
    high_art = dict(base, model="high_art", sfx_h=0.66, crop=0.68, dia=0.94, art=9.0)
    best, rows = select(stock, [good, low_dia, high_art], 1.5, 1.2)
    check("candidate that keeps dialogue and art FPs wins", best is not None and best["model"] == "good",
          str(best))
    check("higher SFX H with collapsed dialogue is rejected",
          not next(r for r in rows if r["model"] == "low_dia")["passes"])
    check("art FP growth above the ratio is rejected",
          not next(r for r in rows if r["model"] == "high_art")["passes"])
    check("dialogue allowance is a percentage-point margin",
          abs(next(r for r in rows if r["model"] == "good")["dia_delta_pp"] + 1.0) < 1e-9)
    none, _ = select(stock, [low_dia], 1.5, 1.2)
    check("no candidate satisfying the constraint -> no winner", none is None)
    print(f"\n{0 if ok else 1} failure(s)")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stock", type=Path, help="val eval JSON of the stock detector")
    ap.add_argument("candidates", nargs="*", type=Path)
    ap.add_argument("--max-dia-drop", type=float, default=1.5,
                    help="allowed dialogue block-coverage drop vs stock, in percentage points")
    ap.add_argument("--max-art-ratio", type=float, default=1.2,
                    help="allowed art FP/page relative to stock")
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.stock or not args.candidates:
        ap.error("--stock and at least one candidate JSON are required")

    stock = load(args.stock)
    cands = [load(p) for p in args.candidates]
    best, rows = select(stock, cands, args.max_dia_drop, args.max_art_ratio)

    print(f'stock: SFX H {100*stock["sfx_h"]:.2f} %  crop {100*stock["crop"]:.2f} %  '
          f'dialogue coverage {100*stock["dia"]:.2f} %  art {stock["art"]:.2f}/page')
    print(f'constraint: dialogue >= {100*(stock["dia"] - args.max_dia_drop/100):.2f} %  '
          f'art <= {stock["art"]*args.max_art_ratio:.2f}/page')
    print(f'{"checkpoint":34s} {"SFX H":>7s} {"dH":>7s} {"crop":>7s} {"diaCov":>7s} {"dDia":>7s} {"art/pg":>7s}  ok')
    for r in sorted(rows, key=lambda r: -r["sfx_h"]):
        print(f'{r["model"]:34s} {100*r["sfx_h"]:6.2f}% {r["sfx_h_delta_pp"]:+6.2f} '
              f'{100*r["crop"]:6.2f}% {100*r["dia"]:6.2f}% {r["dia_delta_pp"]:+6.2f} '
              f'{r["art"]:7.2f}  {"yes" if r["passes"] else "no"}')
    if best is None:
        print("\nNO checkpoint satisfies the constraint")
        return 1
    print(f'\nselected: {best["model"]}')
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({"selected": best, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
