#!/usr/bin/env python3
"""Does upscaling hurt, and does pre-downscaling fix it?

Replays the app's chunking policy on real Manga109 pages:
  full       - whole page (det internally resizes max side to 960)
  app        - tile 2400px with 360 overlap when a side > 2500, else resize to 3MP
  downscale  - first shrink the (up)scaled page back to a target long side
  original   - reference on the untouched page

Upscaling mimics AI upscalers: cubic + unsharp mask.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import RecModel  # noqa: E402
from db_det import DBDetector, crop_line  # noqa: E402
from extract_page_lines import assign, parse_blocks, reading_order  # noqa: E402

TILE = 2400
OVERLAP = 360
TRIGGER = 2500
MAX_PIXELS = 3_000_000


def norm(s: str) -> str:
    return "".join(s.split())


def upscale(img, s):
    up = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(up, (0, 0), 1.0)
    return cv2.addWeighted(up, 1.6, blur, -0.6, 0)


def tiles(img):
    h, w = img.shape[:2]
    xs = list(range(0, w, TILE - OVERLAP))
    ys = list(range(0, h, TILE - OVERLAP))
    for y in ys:
        for x in xs:
            yield x, y, img[y : y + TILE, x : x + TILE]


def dedup(boxes, iou_thr=0.45):
    keep = []
    for q, s in sorted(boxes, key=lambda b: -b[1]):
        ok = True
        for k, _ in keep:
            inter = cv2.intersectConvexConvex(np.asarray(q, np.float32), np.asarray(k, np.float32))[0]
            if inter <= 0:
                continue
            a1 = cv2.contourArea(np.asarray(q, np.float32))
            a2 = cv2.contourArea(np.asarray(k, np.float32))
            if inter / max(1e-6, a1 + a2 - inter) > iou_thr:
                ok = False
                break
        if ok:
            keep.append((q, s))
    return keep


def detect(det, img, mode):
    """Return (image used for crops, boxes in its coords)."""
    h, w = img.shape[:2]
    if mode == "full":
        return img, det(img)
    if mode == "app":
        if w > TRIGGER or h > TRIGGER:
            out = []
            for x, y, tile in tiles(img):
                for q, s in det(tile):
                    out.append((q + np.array([x, y], np.float32), s))
            return img, dedup(out)
        if w * h > MAX_PIXELS:
            r = (MAX_PIXELS / (w * h)) ** 0.5
            small = cv2.resize(img, None, fx=r, fy=r, interpolation=cv2.INTER_AREA)
            return small, det(small)
        return img, det(img)
    raise ValueError(mode)


def eval_page(img, blocks, det, rec):
    used, lines = detect(det, img, "full")
    per_block = assign(lines, blocks)
    cer, n = 0.0, 0
    for bi, items in per_block.items():
        if not items:
            continue
        block = blocks[bi]
        x0, y0, x1, y1 = block["box"]
        ordered = reading_order(items, (y1 - y0) >= (x1 - x0))
        pred = ""
        for q, _ in ordered:
            c = crop_line(used, q, pad=0.04)
            if c is not None:
                pred += rec(c)[0]
        gt = norm(block["text"])
        if gt:
            cer += Levenshtein.distance(norm(pred), gt) / len(gt)
            n += 1
    return cer, n


def eval_app(img, blocks, det, rec):
    used = img
    if img.shape[1] > TRIGGER or img.shape[0] > TRIGGER:
        boxes = []
        for x, y, tile in tiles(img):
            for q, s in det(tile):
                boxes.append((q + np.array([x, y], np.float32), s))
        boxes = dedup(boxes)
    elif img.shape[0] * img.shape[1] > MAX_PIXELS:
        r = (MAX_PIXELS / (img.shape[0] * img.shape[1])) ** 0.5
        used = cv2.resize(img, None, fx=r, fy=r, interpolation=cv2.INTER_AREA)
        boxes = det(used)
        for q, _ in boxes:
            q /= r
    else:
        boxes = det(img)
    per_block = assign(boxes, blocks)
    cer, n = 0.0, 0
    for bi, items in per_block.items():
        if not items:
            continue
        block = blocks[bi]
        x0, y0, x1, y1 = block["box"]
        ordered = reading_order(items, (y1 - y0) >= (x1 - x0))
        pred = ""
        for q, _ in ordered:
            c = crop_line(used, q, pad=0.04)
            if c is not None:
                pred += rec(c)[0]
        gt = norm(block["text"])
        if gt:
            cer += Levenshtein.distance(norm(pred), gt) / len(gt)
            n += 1
    return cer, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=6)
    ap.add_argument("--target", type=int, default=2000, help="long side for the downscale test")
    args = ap.parse_args()

    root = next(p.parent for p in (ROOT / "data/raw/Manga109-s").rglob("annotations") if p.is_dir() and (p.parent / "images").is_dir())
    det = DBDetector(str(ROOT / "models/ppocrv6_small_det.onnx"))
    rec = RecModel(ROOT / "models/ppocrv6_small_rec_manga_real.onnx")

    pages = []
    for book in sorted((root / "annotations").glob("*.xml"))[30:34]:
        data = parse_blocks(book)
        for idx, blocks in sorted(data.items()):
            if 4 <= len(blocks) <= 30:
                img = cv2.imread(str(root / "images" / book.stem / f"{idx:03d}.jpg"))
                if img is not None:
                    pages.append((img, blocks))
            if len(pages) >= args.pages:
                break
        if len(pages) >= args.pages:
            break

    rows = []
    for img, blocks in pages:
        base_cer, base_n = eval_page(img, blocks, det, rec)
        r = args.target / max(img.shape[:2])
        small = cv2.resize(img, None, fx=r, fy=r, interpolation=cv2.INTER_AREA)
        small = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        for scale in (2.0, 3.0):
            up = upscale(img, scale)
            app_cer, n = eval_app(up, blocks, det, rec)
            # app-like tiles, but on the page downscaled to `target` first
            r2 = args.target / max(up.shape[:2])
            shrink = cv2.resize(up, None, fx=r2, fy=r2, interpolation=cv2.INTER_AREA)
            shrink = cv2.resize(shrink, (up.shape[1], up.shape[0]), interpolation=cv2.INTER_CUBIC)
            shrink_cer, _ = eval_app(shrink, blocks, det, rec)
            # downscale to the original page size (oracle), then app pipeline
            orig_cer, _ = eval_app(small, blocks, det, rec)
            rows.append((scale, app_cer, shrink_cer, orig_cer, n))
        print(f"page {img.shape[1]}x{img.shape[0]}: original full CER {100*base_cer/max(1,base_n):.1f}%", flush=True)

    print(f"\n{'scale':>5} {'app tiling':>12} {'->' + str(args.target):>10} {'->original':>12}  blocks")
    for s, a, d, o, n in rows:
        print(f"{s:5.1f} {100*a/max(1,n):11.1f}% {100*d/max(1,n):9.1f}% {100*o/max(1,n):11.1f}%  {n}")


if __name__ == "__main__":
    main()
