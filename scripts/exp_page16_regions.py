#!/usr/bin/env python3
"""Region report: the 4-column bubble and the two undivided bubbles.

Both detectors run through the runtime postprocess (`scripts/db_det.py`, 960 px / 0.15 /
0.25 / 1.40) on the true raw page, and the fine-tuned recogniser reads every box
(`scripts/common.py`). For the two regions the user reported this prints:

  * the lower-right bubble: every box inside it, right-to-left, with the text column it
    covers, so "which of the 4 columns did each detector recover" is a table, not a claim;
  * the lower-left pair: whether any single box spans both bubbles, the paragraph gap /
    x-overlap, the order the app's `nextLine` rule produces inside each bubble, and the
    order that results if the app merges the two paragraphs into one (which is what the
    device overlay shows).

Usage
  python scripts/exp_page16_regions.py --page /tmp/opencode/page16/page16_raw.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import RecModel  # noqa: E402
from db_det import DBDetector, crop_line  # noqa: E402
from eval_det import APP_DET  # noqa: E402
from eval_line_level import FT, STOCK  # noqa: E402
from eval_page_lines import bbox, group_blocks  # noqa: E402

RIGHT_BUBBLE = (2580, 4930, 3330, 5860)
COLUMNS = {1: ("col1", 3120, 3340), 2: ("col2", 2950, 3160),
           3: ("col3", 2770, 3010), 4: ("col4", 2580, 2820)}
UPPER_BUBBLE = (580, 3680, 920, 4350)
LOWER_BUBBLE = (500, 4390, 1140, 5390)


def boxes_in(quads, region):
    """Indices of the boxes whose centre is inside `region`, right-to-left."""
    out = []
    for i, q in enumerate(quads):
        cx = (q[:, 0].min() + q[:, 0].max()) / 2
        cy = (q[:, 1].min() + q[:, 1].max()) / 2
        if region[0] <= cx <= region[2] and region[1] <= cy <= region[3]:
            out.append(i)
    return sorted(out, key=lambda i: -quads[i][:, 0].mean())


def col_of(q):
    cx = (q[:, 0].min() + q[:, 0].max()) / 2
    for col, (_t, x0, x1) in COLUMNS.items():
        if x0 <= cx <= x1:
            return col
    return 0


def spans(q, a, b) -> bool:
    x0, y0, x1, y1 = bbox(q)
    inside_a = max(0.0, min(x1, a[2]) - max(x0, a[0])) * max(0.0, min(y1, a[3]) - max(y0, a[1]))
    inside_b = max(0.0, min(x1, b[2]) - max(x0, b[0])) * max(0.0, min(y1, b[3]) - max(y0, b[1]))
    area = max(1e-6, (x1 - x0) * (y1 - y0))
    return inside_a / area > 0.10 and inside_b / area > 0.10


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--page", type=Path, required=True)
    args = ap.parse_args()

    img = cv2.imread(str(args.page))
    rec = RecModel(ROOT / "models/ppocrv6_small_rec_manga_real.onnx", ROOT / "models/ppocrv6_small_dict.txt")
    runs = {}
    for name, path in (("stock", STOCK), ("ft", FT)):
        det = DBDetector(str(path), **APP_DET)
        quads, texts = [], []
        for q, s in det(img):
            q = np.asarray(q, np.float32)
            crop = crop_line(img, q, 0.04)
            quads.append(q)
            texts.append((rec(crop)[0] if crop is not None else "", s))
        runs[name] = (quads, texts)

    for name in ("stock", "ft"):
        quads, texts = runs[name]
        print(f"\n=== {name}: {len(quads)} boxes on the page")
        print("-- lower-right 4-column bubble")
        inside = boxes_in(quads, RIGHT_BUBBLE)
        seen = {}
        for i in inside:
            q = quads[i]
            col = col_of(q)
            x0, y0, x1, y1 = bbox(q)
            text, score = texts[i]
            seen[col] = text
            print(f"   col {col or '?'}  q=[[{int(x0)},{int(y0)}],[{int(x1)},{int(y0)}],"
                  f"[{int(x1)},{int(y1)}],[{int(x0)},{int(y1)}]]  p={score:.2f}  {text}")
        print(f"   columns recovered: {sorted(c for c in seen if c)} / 4"
              + ("" if len(seen) == 4 else f"   missing {sorted(set(COLUMNS) - set(seen))}"))

        print("-- lower-left undivided pair")
        upper = boxes_in(quads, UPPER_BUBBLE)
        lower = boxes_in(quads, LOWER_BUBBLE)
        spanning = [i for i, q in enumerate(quads) if spans(q, UPPER_BUBBLE, LOWER_BUBBLE)]
        print(f"   boxes: upper bubble {len(upper)}, lower bubble {len(lower)}, "
              f"boxes spanning both: {len(spanning)}")
        if upper and lower:
            ux = (min(bbox(quads[i])[0] for i in upper), max(bbox(quads[i])[2] for i in upper))
            lx = (min(bbox(quads[i])[0] for i in lower), max(bbox(quads[i])[2] for i in lower))
            ug = max(bbox(quads[i])[3] for i in upper)
            lg = min(bbox(quads[i])[1] for i in lower)
            print(f"   paragraph gap (y): {int(lg - ug)} px   x-overlap: "
                  f"{int(max(0, min(ux[1], lx[1]) - max(ux[0], lx[0])))} px "
                  f"(upper x {int(ux[0])}-{int(ux[1])}, lower x {int(lx[0])}-{int(lx[1])})")
        for label, group in (("upper", upper), ("lower", lower)):
            blocks = group_blocks([quads[i] for i in group], vertical=True)
            for bi, g in enumerate(blocks):
                order = [texts[group[i]][0] for i in sorted(g, key=lambda k: -quads[group[k]][:, 0].mean())]
                print(f"   {label} block {bi}: " + " | ".join(order))
        both = upper + lower
        blocks = group_blocks([quads[i] for i in both], vertical=True)
        print(f"   nextLine groups over both bubbles: {len(blocks)}"
              f"  -> {'MERGED' if len(blocks) == 1 else 'separate'}")
        # what the app shows: one paragraph, ordered by descending right edge
        idx = sorted(both, key=lambda i: -quads[i][:, 0].mean())
        print("   order if the two paragraphs are merged (app behaviour, right edge desc): "
              + " | ".join(texts[i][0] for i in idx))
        correct = [texts[i][0] for i in upper] + [texts[i][0] for i in lower]
        print("   order two separate paragraphs should read in: " + " | ".join(correct))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
