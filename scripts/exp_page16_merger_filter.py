#!/usr/bin/env python3
"""Port of the app's line filters, to explain the device result on that page.

`scripts/eval_page_lines.py` restates only the app's *grouping* rule
(`OcrMergeGeometry.nextLine`).  The app also **drops whole lines** before grouping, in
`OwOCRMerger` (komi-kun `core/ocr/.../merge/OwOCRMerger.kt`):

  1. `EngineLine.toLine()` - blank text, no Japanese, `isImplausibleJapaneseNoise()`
     (`OcrPreprocessor.kt`), line ~133;
  2. `filterSingleCharSfxNoise()` - line ~139;
  3. `globalFuriganaFilter()` - line 688: drops a line that is thinner than a nearby
     kanji-bearing line (<= 0.55x if it itself has kanji, else <= 0.75x) and sits to the
     right of it (vertical text) inside `0.30 * thickness`;
  4. `dedupeParagraphLines()` - line 221.

Only (1)-(3) can delete a correctly detected column, and (3) is *geometry-relative*, so it
can behave differently for two detectors whose boxes differ by a few percent.  This script
replays (1)-(3) on the real page for both detectors and reports which columns of the
lower-right 4-column bubble survive.

Diagnostic only.  `rotation` is derived from the detected quadrilateral (the engine gets it
from `minAreaRect`); `hasJpText` / kanji regexes are copied from the app.

Usage
  python scripts/exp_page16_merger_filter.py --page /tmp/opencode/page16/page16_raw.png
"""

from __future__ import annotations

import argparse
import re
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
from exp_page16_pipelines import dedup, tiles_2d  # noqa: E402

BUBBLE = (2560, 4930, 3340, 5850)
COLUMNS = {
    1: ("col1", 3130, 3300),
    2: ("col2", 2955, 3155),
    3: ("col3", 2780, 3010),
    4: ("col4", 2600, 2810),
}
LOWER_LEFT = (500, 3660, 1140, 5390)
MAX_OCR_PIXELS = 4_000_000

CJ_REGEX = re.compile("[\u3041-\u3096\u30a1-\u30fa\u4e01-\u9fff]")
KANJI_REGEX = re.compile("[\u4e00-\u9fff]")
KATAKANA_REGEX = re.compile("[\u30a1-\u30fa]")
ALPHA_REGEX = re.compile("[A-Za-z]")
DIGIT_REGEX = re.compile("[0-9]")
STROKE_LOOKALIKES = set("干手七十千イニハロトメソン出")


def is_cjk(c: str) -> bool:
    o = ord(c)
    return (0x3041 <= o <= 0x3096) or (0x30A1 <= o <= 0x30FA) or (o == 0x30FC) or (o == 0x3005) or (
        0x4E00 <= o <= 0x9FFF
    ) or (0x3400 <= o <= 0x4DBF)


def implausible_noise(text: str) -> bool:
    stripped = "".join(ch for ch in text if ch.isalnum())  # trims whitespace and punctuation
    if not stripped:
        return not any(is_cjk(c) for c in text)
    if len(stripped) > 4:
        return False
    cjk = sum(1 for c in stripped if is_cjk(c))
    alpha = sum(1 for c in stripped if c.isascii() and c.isalpha())
    digit = sum(1 for c in stripped if c.isascii() and c.isdigit())
    look = sum(1 for c in stripped if c in STROKE_LOOKALIKES and is_cjk(c))
    if cjk == 0:
        return True
    if alpha > 0:
        return True
    if digit > 0 and look > 0 and cjk == look:
        return True
    return False


def rotation_of(quad) -> float:
    q = np.asarray(quad, np.float32)
    dx, dy = q[1][0] - q[0][0], q[1][1] - q[0][1]
    return float(np.arctan2(dy, dx))


def filter_lines(lines, img_w, img_h):
    """Apply toLine gates, filterSingleCharSfxNoise and globalFuriganaFilter."""
    stage1 = []
    for ln in lines:
        if not ln["text"].strip():
            continue
        if not CJ_REGEX.search(ln["text"]) or implausible_noise(ln["text"]):
            continue
        stage1.append(ln)

    # filterSingleCharSfxNoise
    stage2 = list(stage1)
    if stage2:
        th = sorted(min(ln["w"], ln["h"]) for ln in stage2)
        med = th[len(th) // 2] if len(th) % 2 else (th[len(th) // 2 - 1] + th[len(th) // 2]) / 2
        kept = []
        for ln in stage2:
            stripped = "".join(ch for ch in ln["text"] if ch.isalnum())
            if len(stripped) == 1:
                thick = min(ln["w"], ln["h"])
                if abs(ln["rotation"]) > 0.03 or (len(stage2) >= 3 and thick > med * 1.55):
                    continue
            kept.append(ln)
        stage2 = kept

    # globalFuriganaFilter (vertical path)
    keep = [True] * len(stage2)
    for j, sub in enumerate(stage2):
        if not keep[j]:
            continue
        text = sub["text"]
        has_kanji = bool(KANJI_REGEX.search(text))
        if has_kanji and len(text) > 2:
            continue
        if ALPHA_REGEX.search(text) and not KATAKANA_REGEX.search(text):
            continue
        if DIGIT_REGEX.search(text) and not KATAKANA_REGEX.search(text) and not ALPHA_REGEX.search(text):
            continue
        sub_thickness = min(sub["w"], sub["h"])
        for i, main in enumerate(stage2):
            if i == j or not keep[i]:
                continue
            if not KANJI_REGEX.search(main["text"]):
                continue
            main_thickness = min(main["w"], main["h"])
            ratio = 0.55 if has_kanji else 0.75
            if sub_thickness > main_thickness * ratio:
                continue
            proximity = main_thickness * 0.30
            x_overlap = max(0.0, min(sub["x1"], main["x1"]) - max(sub["x0"], main["x0"]))
            x_gap = 0.0 if x_overlap > 0 else min(abs(sub["x0"] - main["x1"]), abs(sub["x1"] - main["x0"]))
            y_ov = max(0.0, min(sub["y1"], main["y1"]) - max(sub["y0"], main["y0"]))
            y_ratio = y_ov / max(0.001, min(sub["h"], main["h"]))
            is_vert_furi = x_gap < proximity and y_ratio > 0.05
            main_vertical = main["h"] > main["w"] * 1.2
            if main_vertical and is_vert_furi and main["cx"] < sub["cx"]:
                keep[j] = False
                break
    return stage1, stage2, [ln for i, ln in enumerate(stage2) if keep[i]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--page", type=Path, required=True)
    ap.add_argument("--path", default="4mp", choices=["4mp", "direct", "tile"],
                    help="detector input: the app's 4 MP image, the whole page in one resize (eval path), or 2400 px tiles")
    ap.add_argument("--side", type=int, default=0, help="force the detector input's long side (with --path direct)")
    args = ap.parse_args()

    img = cv2.imread(str(args.page))
    h, w = img.shape[:2]
    ar = w / h
    nw = int(np.sqrt(MAX_OCR_PIXELS * ar))
    if args.path == "4mp":
        small = cv2.resize(img, (nw, int(nw / ar)), interpolation=cv2.INTER_AREA)
        pieces = [(small, 0, 0, w / small.shape[1])]
    elif args.path == "direct" and not args.side:
        pieces = [(img, 0, 0, 1.0)]
    elif args.path == "direct":
        f = args.side / max(h, w)
        small = cv2.resize(img, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
        pieces = [(small, 0, 0, w / small.shape[1])]
    else:  # tile 2400 / 360, the app's tiler without the 4 MP pre-resize
        pieces = None
    print(f"page {w}x{h}; path={args.path}" + (f" side={args.side}" if args.side else ""))
    rec = RecModel(ROOT / "models/ppocrv6_small_rec_manga_real.onnx", ROOT / "models/ppocrv6_small_dict.txt")

    for name, path in (("stock", STOCK), ("ft", FT)):
        det = DBDetector(str(path), **APP_DET)
        if pieces is None:
            tile_pieces = [(c, x, y, 1.0) for c, x, y in tiles_2d(img)]
            raw = []
            for crop, ox, oy, sc in tile_pieces:
                for q, sc2 in det(crop):
                    raw.append((np.asarray(q, np.float32) * sc + np.array([ox, oy], np.float32), sc2))
            dets = dedup(raw)
        else:
            dets = [(np.asarray(q, np.float32) * sc + np.array([ox, oy], np.float32), s)
                    for crop, ox, oy, sc in pieces for q, s in det(crop)]
        lines = []
        for q, s in dets:
            crop = crop_line(img, q, 0.04)
            text = rec(crop)[0] if crop is not None else ""
            x0, y0, x1, y1 = q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()
            lines.append({"text": text, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                          "w": x1 - x0, "h": y1 - y0, "cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2,
                          "rotation": rotation_of(q)})
        s1, s2, s3 = filter_lines(lines, w, h)
        print(f"\n=== {name}: {len(lines)} engine lines -> toLine {len(s1)} -> singleChar {len(s2)} -> furigana {len(s3)}")
        for label, sel in (("dropped by toLine", [l for l in lines if l not in s1]),
                           ("dropped by singleChar", [l for l in s1 if l not in s2]),
                           ("dropped by furigana", [l for l in s2 if l not in s3])):
            for ln in sel:
                print(f"  {label}: '{ln['text']}'  bbox=({int(ln['x0'])},{int(ln['y0'])})-({int(ln['x1'])},{int(ln['y1'])})"
                      f"  {int(ln['w'])}x{int(ln['h'])}")
        for title, box in (("4-line bubble", BUBBLE), ("lower-left pair", LOWER_LEFT)):
            cols = {}
            for col, (_lbl, cx0, cx1) in COLUMNS.items():
                if box == BUBBLE:
                    hit = [ln for ln in s3 if cx0 <= ln["cx"] <= cx1 and ln["y1"] > BUBBLE[1] and ln["y0"] < BUBBLE[3]]
                    if hit:
                        cols[col] = hit[0]["text"]
            kept_in = [ln for ln in s3 if box[0] <= ln["cx"] <= box[2] and box[1] <= ln["cy"] <= box[3]]
            print(f"  {title}: {len(kept_in)} lines survive"
                  + (f"; columns surviving: " + " ".join(f"{k}:{v}" for k, v in sorted(cols.items())) if cols else ""))
            for ln in sorted(kept_in, key=lambda l: -l["cx"]):
                print(f"    '{ln['text']}'  ({int(ln['x0'])},{int(ln['y0'])})-({int(ln['x1'])},{int(ln['y1'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
