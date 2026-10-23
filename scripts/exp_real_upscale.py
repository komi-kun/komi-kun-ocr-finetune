#!/usr/bin/env python3
"""Evaluate real upscaled pages against their original counterparts.

Pairs come from the user's library (`original/` and `upscaled/` copies of the
same chapter). The original page is the reference: the current model is known to
read those well. For every variant we map detector boxes back to original
coordinates and compare the recognised text per matched line.

Modes:
  app     - exactly what the app does today (tile 2400 if a side > 2500, else 3 MP cap)
  cap3mp  - downscale to 3 MP *first*, then the app policy
  orig    - downscale to the original page size (oracle upper bound)
  native  - no tiling/resize at all (diagnostic)
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import RecModel  # noqa: E402
from db_det import DBDetector, crop_line  # noqa: E402

TILE = 2400
CAP = 3.0
OVERLAP = 360
TRIGGER = 2500
MAX_PIXELS = 3_000_000


def norm(s):
    return "".join(s.split())


def zoom(img, r):
    return cv2.resize(img, None, fx=r, fy=r, interpolation=cv2.INTER_AREA)


def tile_boxes(det, img):
    h, w = img.shape[:2]
    boxes = []
    for y in range(0, h, TILE - OVERLAP):
        for x in range(0, w, TILE - OVERLAP):
            tile = img[y : y + TILE, x : x + TILE]
            if tile.size == 0:
                continue
            for q, s in det(tile):
                boxes.append((q + np.array([x, y], np.float32), s))
    # dedup overlapping seam detections
    keep = []
    for q, s in sorted(boxes, key=lambda b: -b[1]):
        dup = False
        for k, _ in keep:
            inter = cv2.intersectConvexConvex(np.asarray(q, np.float32), np.asarray(k, np.float32))[0]
            if inter > 0:
                a1, a2 = cv2.contourArea(q), cv2.contourArea(k)
                if inter / max(1e-6, a1 + a2 - inter) > 0.45:
                    dup = True
                    break
        if not dup:
            keep.append((q, s))
    return keep


def run(det, rec, img, mode, orig_shape=None):
    h, w = img.shape[:2]
    if mode == "native":
        used, boxes = img, det(img)
    elif mode == "app":
        if w > TRIGGER or h > TRIGGER:
            used, boxes = img, tile_boxes(det, img)
        elif w * h > MAX_PIXELS:
            r = (MAX_PIXELS / (w * h)) ** 0.5
            used = zoom(img, r)
            boxes = det(used)
        else:
            used, boxes = img, det(img)
    elif mode.startswith("cap"):
        if w * h > CAP * 1_000_000:
            r = (CAP * 1_000_000 / (w * h)) ** 0.5
            img = zoom(img, r)
        return run(det, rec, img, "app")
    elif mode == "orig":
        r = orig_shape[1] / w
        img = zoom(img, r)
        return run(det, rec, img, "app")
    else:
        raise ValueError(mode)

    out = []
    for q, s in boxes:
        c = crop_line(used, q, pad=0.04)
        if c is None:
            continue
        text, _ = rec(c)
        out.append((q, text))
    return out, (used.shape[1], used.shape[0])


def match(ref, var, scale):
    """Match variant boxes (variant coords) to ref boxes (original coords)."""
    pairs = []
    used = set()
    for i, (rq, rt) in enumerate(ref):
        rc = rq.mean(0)
        best, best_d = -1, 1e9
        for j, (vq, vt) in enumerate(var):
            if j in used:
                continue
            vc = vq.mean(0) / np.array([scale, scale], np.float32)
            d = float(np.linalg.norm(vc - rc))
            if d < best_d:
                best, best_d = j, d
        if best >= 0 and best_d < 40:
            used.add(best)
            pairs.append((i, best))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("/mnt/datasets/komi-ocr"))
    ap.add_argument("--pages", type=int, default=6)
    ap.add_argument("--budgets", default="2.0,2.5,3.0,4.0")
    args = ap.parse_args()

    det = DBDetector(str(ROOT / "models/ppocrv6_small_det.onnx"))
    rec = RecModel(ROOT / "models/ppocrv6_small_rec_manga_real.onnx")

    zo = zipfile.ZipFile(args.dir / "original/ch1.cbz")
    zu = zipfile.ZipFile(args.dir / "upscaled/ch1.cbz")
    orig_names = {Path(n).stem: n for n in zo.namelist() if n.lower().endswith((".webp", ".jpg", ".png"))}
    up_names = {Path(n).stem: n for n in zu.namelist() if n.lower().endswith((".avif", ".webp", ".jpg", ".png"))}
    common = sorted(set(orig_names) & set(up_names))[: args.pages]
    print(f"paired pages: {len(common)}")

    global CAP
    budgets = [float(x) for x in args.budgets.split(",")]
    modes = ["app", "orig", "native"] + [f"cap{b:g}" for b in budgets]
    stats = {m: {"cer": 0.0, "n": 0, "exact": 0, "cov": 0, "ref": 0, "sec": 0.0} for m in modes}
    for name in common:
        o = cv2.cvtColor(np.array(Image.open(zo.open(orig_names[name])).convert("RGB")), cv2.COLOR_RGB2BGR)
        u = cv2.cvtColor(np.array(Image.open(zu.open(up_names[name])).convert("RGB")), cv2.COLOR_RGB2BGR)
        ref, _ = run(det, rec, o, "app")
        import time

        for mode in modes:
            t0 = time.time()
            CAP = float(mode[3:]) if mode.startswith("cap") else 3.0
            var, used_shape = run(det, rec, u, mode, orig_shape=o.shape)
            dt = time.time() - t0
            scale = used_shape[0] / o.shape[1]
            pairs = match(ref, var, scale)
            s = stats[mode]
            s["ref"] += len(ref)
            s["cov"] += len(pairs)
            s["sec"] += dt
            for i, j in pairs:
                gt, hyp = norm(ref[i][1]), norm(var[j][1])
                if not gt:
                    continue
                s["cer"] += Levenshtein.distance(hyp, gt) / len(gt)
                s["n"] += 1
                s["exact"] += int(hyp == gt)
        print(f"  {name}: ref lines={len(ref)}", flush=True)

    print(f"\n{'mode':8} {'CER %':>7} {'exact %':>8} {'coverage %':>11} {'sec/page':>9}")
    for m, s in stats.items():
        print(
            f"{m:8} {100*s['cer']/max(1,s['n']):7.1f} {100*s['exact']/max(1,s['n']):8.1f} "
            f"{100*s['cov']/max(1,s['ref']):11.1f} {s['sec']/max(1,len(common)):9.2f}"
        )


if __name__ == "__main__":
    main()
