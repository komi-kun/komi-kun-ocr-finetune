#!/usr/bin/env python3
"""Which pipeline stage makes the device lose alternate columns? (regression-page forensics)

The reported page (`page16_raw.png`) is a large AI-upscaled page.  The
desktop path every eval script uses (`scripts/db_det.py`) downscales the whole page to
the detector's 960 px side in one step, so the detector sees 675x960.  The app stages it
(`core/ocr/.../ImageChunking.kt`): `prepareForOcr(bitmap)` first does
`resizeToMaxPixels(bitmap, MAX_OCR_PIXELS = 4 MP)` -> 1677x2385 and only then decides to
tile (`TILE_TRIGGER_DIMENSION = 2500`), so this page should be a single chunk; an older
or byte-path build instead uses `chunkImage` (3000 px strips) or `chunkImage2D` (2400 px
tiles).

This script replays those variants (plus a `box_thresh` sweep) on the one page and prints,
for the four text columns of the lower-right bubble, which columns each variant detects.
Diagnostic only, not part of the evaluation pipeline.

Usage
  python scripts/exp_page16_pipelines.py --page /tmp/opencode/page16/page16_raw.png
  python scripts/exp_page16_pipelines.py --page ... --json-out /tmp/opencode/page16/eval/pipelines.json
"""

from __future__ import annotations

import argparse
import json
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

EXP1 = ROOT / "models/ppocrv6_small_det_manga.onnx"  # experiment-1 detector (rejected)

# The lower-right bubble on the regression page, raw-page pixels. Column 1 is the RIGHTMOST column.
BUBBLE = (2560, 4930, 3340, 5850)
COLUMNS = {
    1: ("col1", 3130, 3300),
    2: ("col2", 2955, 3155),
    3: ("col3", 2780, 3000),
    4: ("col4", 2600, 2810),
}
MAX_OCR_PIXELS = 4_000_000
MAX_TOTAL_PIXELS = 3_000_000
TILE, TILE_OVERLAP = 2400, 360
CHUNK, CHUNK_OVERLAP = 3000, 96


def resize_max_pixels(img, max_pixels):
    h, w = img.shape[:2]
    if w * h <= max_pixels:
        return img
    ar = w / h
    nw = int(np.sqrt(max_pixels * ar))
    return cv2.resize(img, (nw, max(1, int(nw / ar))), interpolation=cv2.INTER_AREA)


def tiles_2d(img, tile=TILE, overlap=TILE_OVERLAP):
    h, w = img.shape[:2]
    step = tile - overlap
    out = []
    y = 0
    while True:
        x = 0
        while True:
            crop = img[y : min(y + tile, h), x : min(x + tile, w)]
            if crop.size:
                out.append((crop, x, y))
            if x + tile >= w:
                break
            x += step
        if y + tile >= h:
            break
        y += step
    return out


def strips(img, chunk=CHUNK, overlap=CHUNK_OVERLAP):
    h, w = img.shape[:2]
    out = []
    y = 0
    while True:
        crop = img[y : min(y + chunk, h), :]
        if crop.size:
            out.append((crop, 0, y))
        if y + chunk >= h:
            break
        y += chunk - overlap
    return out


def dedup(boxes, iou_threshold=0.5):
    """Drop duplicate detections that overlap across tile seams (IoU > 0.5)."""
    kept = []
    for q, s in sorted(boxes, key=lambda b: -b[1]):
        dup = False
        for kq, _ in kept:
            inter = cv2.intersectConvexConvex(np.asarray(q, np.float32), np.asarray(kq, np.float32))[0]
            if inter <= 0:
                continue
            union = cv2.contourArea(np.asarray(q, np.float32)) + cv2.contourArea(np.asarray(kq, np.float32)) - inter
            if union > 0 and inter / union > iou_threshold:
                dup = True
                break
        if not dup:
            kept.append((q, s))
    return kept


def run(det, pieces):
    """Detect on each piece; `pieces` are (crop, ox, oy, scale) with offsets in page px."""
    out = []
    for crop, ox, oy, scale in pieces:
        for q, s in det(crop):
            out.append((np.asarray(q, np.float32) * scale + np.array([ox, oy], np.float32), s))
    if len(pieces) == 1:
        return out
    kept = []
    for q, s in sorted(out, key=lambda b: -b[1]):
        dup = False
        for kq, _ in kept:
            inter = cv2.intersectConvexConvex(np.asarray(q, np.float32), np.asarray(kq, np.float32))[0]
            if inter <= 0:
                continue
            union = cv2.contourArea(q) + cv2.contourArea(kq) - inter
            if union > 0 and inter / union > 0.5:
                dup = True
                break
        if not dup:
            kept.append((q, s))
    return kept


def columns_hit(boxes):
    hit = {}
    for col, (_label, x0, x1) in COLUMNS.items():
        scores = []
        for q, s in boxes:
            cx = (q[:, 0].min() + q[:, 0].max()) / 2
            if x0 <= cx <= x1 and q[:, 1].max() > BUBBLE[1] and q[:, 1].min() < BUBBLE[3]:
                scores.append(s)
        hit[col] = max(scores) if scores else None
    return hit


def in_bubble(boxes):
    out = []
    for q, s in boxes:
        cx = (q[:, 0].min() + q[:, 0].max()) / 2
        cy = (q[:, 1].min() + q[:, 1].max()) / 2
        if BUBBLE[0] <= cx <= BUBBLE[2] and BUBBLE[1] <= cy <= BUBBLE[3]:
            out.append((q, s))
    return sorted(out, key=lambda b: -b[0][:, 0].mean())  # right to left


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--page", type=Path, required=True)
    ap.add_argument("--det", default="ft", choices=["stock", "ft", "exp1"])
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    img = cv2.imread(str(args.page))
    if img is None:
        raise SystemExit(f"cannot read {args.page}")
    h, w = img.shape[:2]
    det_path = {"stock": STOCK, "ft": FT, "exp1": EXP1}[args.det]
    rec = RecModel(ROOT / "models/ppocrv6_small_rec_manga_real.onnx", ROOT / "models/ppocrv6_small_dict.txt")
    print(f"page {args.page.name} {w}x{h} = {w * h / 1e6:.1f} MP   detector={args.det} {det_path.name}")

    four = resize_max_pixels(img, MAX_OCR_PIXELS)
    three = resize_max_pixels(img, MAX_TOTAL_PIXELS)
    s4, s3 = w / four.shape[1], w / three.shape[1]
    variants = {
        "direct960 (eval path: whole page, det 960)": [(img, 0, 0, 1.0)],
        f"4MP->det960 (app prepareForOcr, {four.shape[1]}x{four.shape[0]})": [(four, 0, 0, s4)],
        f"3MP->det960 (MAX_TOTAL_PIXELS cap, {three.shape[1]}x{three.shape[0]})": [(three, 0, 0, s3)],
        "tile2400/360 on page (chunkImage2D)": [(c, x, y, 1.0) for c, x, y in tiles_2d(img)],
        "tile2400/360 on 4MP image": [(c, x * s4, y * s4, s4) for c, x, y in tiles_2d(four)],
        "strip3000/96 on page (chunkImage)": [(c, x, y, 1.0) for c, x, y in strips(img)],
        "strip3000/96 on 4MP image": [(c, x * s4, y * s4, s4) for c, x, y in strips(four)],
    }

    report = {}
    for name, pieces in variants.items():
        det = DBDetector(str(det_path), **APP_DET)
        boxes = run(det, pieces)
        hit = columns_hit(boxes)
        inside = in_bubble(boxes)
        rows = []
        for q, s in inside:
            text = rec(crop_line(img, q, 0.04))[0]
            rows.append({"quad": [[int(round(float(x))), int(round(float(y)))] for x, y in q],
                         "score": round(float(s), 3), "text": text})
        report[name] = {"pieces": len(pieces), "boxes_in_bubble": len(inside),
                        "columns": {str(k): v for k, v in hit.items()}, "boxes": rows}
        print(f"\n{name}  [{len(pieces)} piece(s)]")
        print(f"  boxes in bubble: {len(inside)}   columns: "
              + "  ".join(f"{k}:{'yes' if v is not None else 'MISS'}" for k, v in hit.items()))
        for r in rows:
            print(f"    q={r['quad']}  p={r['score']:.2f}  {r['text']}")

    print("\nbox_thresh sweep on the 4MP->det960 path (APP_DET box_thresh = 0.25):")
    for bt in (0.25, 0.4, 0.5, 0.6):
        det = DBDetector(str(det_path), thresh=APP_DET["thresh"], box_thresh=bt,
                         unclip_ratio=APP_DET["unclip_ratio"], limit_side_len=APP_DET["limit_side_len"])
        hit = columns_hit(run(det, [(four, 0, 0, s4)]))
        report[f"box_thresh_{bt}"] = {str(k): v for k, v in hit.items()}
        print(f"  box_thresh {bt}: " + "  ".join(f"{k}:{'yes' if v is not None else 'MISS'}" for k, v in hit.items()))

    print("\ndetector input side sweep (page resized so max side = N, single step):")
    for side in (480, 640, 736, 960, 1280, 1600, 1920, 2560, 3200):
        sc = side / max(h, w)
        small = cv2.resize(img, (max(1, round(w * sc)), max(1, round(h * sc))), interpolation=cv2.INTER_AREA)
        det = DBDetector(str(det_path), **APP_DET)
        hit = columns_hit(run(det, [(small, 0, 0, w / small.shape[1])]))
        report[f"side_{side}"] = {str(k): v for k, v in hit.items()}
        print(f"  input {small.shape[1]}x{small.shape[0]}: "
              + "  ".join(f"{k}:{'yes' if v is not None else 'MISS'}" for k, v in hit.items()))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
