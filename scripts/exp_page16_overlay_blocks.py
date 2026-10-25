#!/usr/bin/env python3
"""Read the app's *rendered* OCR blocks out of a reader screenshot.

The overlay is drawn as a semi-transparent fill with a light-blue border; there is no
dump of the block list on the device, so the screenshot is the only record of what the
app actually produced.  This finds the border colour, maps the rectangles back to raw-page
coordinates (the reader shows the page fit-height) and prints them, so the device result
can be compared with what the desktop pipeline produces for the same page.

Usage
  python scripts/exp_page16_overlay_blocks.py --shot /tmp/opencode/page16/page16_overlay_on.png \\
      --page-size <W>x<H> --json-out /tmp/opencode/page16/eval/overlay_blocks.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

BORDER_BGR = (255, 197, 179)  # sampled from the screenshot: light-blue block border
TOL = 12


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shot", type=Path, required=True)
    ap.add_argument("--page-size", required=True, help="raw page size as WxH")
    ap.add_argument("--min-area", type=int, default=150)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    img = cv2.imread(str(args.shot))
    h, w = img.shape[:2]
    pw, ph = (int(v) for v in args.page_size.lower().split("x"))
    scale = h / ph
    ox = (w - pw * scale) / 2

    b, g, r = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
    border = (
        (abs(b - BORDER_BGR[0]) <= TOL) & (abs(g - BORDER_BGR[1]) <= TOL) & (abs(r - BORDER_BGR[2]) <= TOL)
    ).astype(np.uint8)
    border = cv2.morphologyEx(border, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, _lab, stats, _cent = cv2.connectedComponentsWithStats(border, 8)

    blocks = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < args.min_area or bw < 10 or bh < 10:
            continue
        blocks.append(
            {
                "xmin": round((x - ox) / scale),
                "ymin": round(y / scale),
                "xmax": round((x + bw - ox) / scale),
                "ymax": round((y + bh) / scale),
                "screen_area": int(area),
            }
        )
    blocks.sort(key=lambda d: (d["ymin"], -d["xmax"]))
    print(f"{len(blocks)} overlay rectangles in {args.shot.name} ({w}x{h}, page scale {scale:.5f}):")
    for d in blocks:
        print(f"  ({d['xmin']},{d['ymin']})-({d['xmax']},{d['ymax']})")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(blocks, indent=2))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
