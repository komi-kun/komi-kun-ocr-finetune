#!/usr/bin/env python3
"""Single-page line-level evaluator: stock det vs fine-tuned det, side by side.

Runs **both** detectors (the shipped stock DB detector and the fine-tuned
curved-SFX detector) plus the fine-tuned recogniser over one page image, through
the same runtime path the app uses (`scripts/db_det.py`: 960 max side, thresh
0.15, box_thresh 0.25, unclip 1.40; `scripts/common.py` RecModel: 3x48xW, CTC
greedy, 320 px clamp), and prints for each detector:

  * every detected box with its coordinates, detector score and recognised text
    (one line == one text column for vertical/tategaki text),
  * the block grouping it implies (lines merged into blocks with the app's
    `OcrMergeGeometry.nextLine` criterion, restated below),
  * a per-bubble breakdown: how many text columns each detector recovered inside
    the same bubble, so a bubble where one detector lost half its columns is
    visible next to the one that did not.

It writes one annotated PNG per detector (`--out`, default
`/tmp/opencode/page_lines/`) so the loss can be eyeballed.

Bubbles come from `--gt-xml`/`--gt-page` (Manga109 `<text>` blocks) when given -
that is exact and is what held-out pages should use - otherwise from the union of
both detectors' boxes expanded by a small margin, which is still detector
symmetric: a bubble one detector segments and the other merges is one cluster.

Weights are printed with their SHA-256 at start-up. Nothing is written to the
repo unless `--out` points inside it.

Usage
  python scripts/eval_page_lines.py --page /tmp/opencode/page.png --det ft
  python scripts/eval_page_lines.py --page data/raw/Manga109-s/.../Arisa/005.jpg --det ft \\
      --gt-xml data/raw/Manga109-s/.../annotations/Arisa.xml --gt-page 5 --out /tmp/opencode/page_lines
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import RecModel  # noqa: E402
from db_det import DBDetector, crop_line  # noqa: E402
from eval_det import APP_DET  # noqa: E402
from eval_line_level import DETECTORS  # noqa: E402

REC = ROOT / "models/ppocrv6_small_rec_manga_real.onnx"  # FT recogniser, ocr-manga-v3
DICT = ROOT / "models/ppocrv6_small_dict.txt"

# Restated from core/ocr `OcrMergeGeometry` (komi-kun): two boxes are successive
# lines of one block when they are stacked on the advance axis with a comparable
# thickness and a run overlap of >= 40 %.  The app's OwOCRMerger additionally
# filters furigana and merges close paragraphs; this keeps only the core rule.
MIN_THICKNESS_RATIO = 0.65
MAX_LINE_GAP_RATIO = 0.50
MAX_ADVANCE_OVERLAP = 0.45
MIN_RUN_OVERLAP = 0.40


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- geometry
def bbox(quad) -> tuple[float, float, float, float]:
    q = np.asarray(quad, np.float32)
    return float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())


def overlap_1d(a0, a1, b0, b1) -> float:
    lo, hi = max(a0, b0), min(a1, b1)
    return max(0.0, hi - lo)


def same_block(ba, bb, vertical: bool) -> bool:
    ax0, ay0, ax1, ay1 = ba
    bx0, by0, bx1, by1 = bb
    if vertical:  # text runs along y; successive columns advance along x
        ta, tb = ax1 - ax0, bx1 - bx0
        run_a, run_b = (ay0, ay1), (by0, by1)
        adv_a, adv_b = (ax0, ax1), (bx0, bx1)
    else:
        ta, tb = ay1 - ay0, by1 - by0
        run_a, run_b = (ax0, ax1), (bx0, bx1)
        adv_a, adv_b = (ay0, ay1), (by0, by1)
    larger = max(ta, tb)
    if larger <= 0 or min(ta, tb) / larger < MIN_THICKNESS_RATIO:
        return False
    run_ov = overlap_1d(*run_a, *run_b) / max(1e-6, min(run_a[1] - run_a[0], run_b[1] - run_b[0]))
    if run_ov < MIN_RUN_OVERLAP:
        return False
    adv_ov = overlap_1d(*adv_a, *adv_b) / max(1e-6, min(adv_a[1] - adv_a[0], adv_b[1] - adv_b[0]))
    if adv_ov > MAX_ADVANCE_OVERLAP:
        return False
    gap = max(0.0, max(adv_a[0], adv_b[0]) - min(adv_a[1], adv_b[1]))
    return gap <= MAX_LINE_GAP_RATIO * min(ta, tb)


def group_blocks(boxes, vertical: bool) -> list[list[int]]:
    parent = list(range(len(boxes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if same_block(bbox(boxes[i]), bbox(boxes[j]), vertical):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(boxes)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def cluster_bubbles(boxes, pad_ratio: float = 0.12, min_pad: float = 3.0) -> list[list[int]]:
    """Union of both detectors' boxes, expanded, as connected components."""
    if not boxes:
        return []
    expanded = []
    for q in boxes:
        x0, y0, x1, y1 = bbox(q)
        pad = max(min_pad, pad_ratio * min(x1 - x0, y1 - y0))
        expanded.append((x0 - pad, y0 - pad, x1 + pad, y1 + pad))
    parent = list(range(len(expanded)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(expanded)):
        for j in range(i + 1, len(expanded)):
            a, b = expanded[i], expanded[j]
            if overlap_1d(a[0], a[2], b[0], b[2]) > 0 and overlap_1d(a[1], a[3], b[1], b[3]) > 0:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(expanded)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def gt_blocks(xml_path: Path, page_idx: int) -> list[tuple[float, float, float, float]]:
    out = []
    for page in ET.parse(xml_path).getroot().iter("page"):
        if int(page.get("index", -1)) != page_idx:
            continue
        for el in page:
            if el.tag != "text":
                continue
            try:
                out.append((float(el.get("xmin")), float(el.get("ymin")),
                            float(el.get("xmax")), float(el.get("ymax"))))
            except (TypeError, ValueError):
                continue
    return out


def inside(quad, block) -> bool:
    x0, y0, x1, y1 = bbox(quad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bx0, by0, bx1, by1 = block
    return bx0 - 2 <= cx <= bx1 + 2 and by0 - 2 <= cy <= by1 + 2


# ---------------------------------------------------------------- run
def run_detector(name: str, path: Path, img, rec: RecModel):
    det = DBDetector(str(path), **APP_DET)
    quads = det(img)
    boxes, texts = [], []
    for quad, score in quads:
        crop = crop_line(img, quad, pad=0.04)
        text = rec(crop)[0] if crop is not None else ""
        boxes.append(np.asarray(quad, np.float32))
        texts.append((text, float(score)))
    return boxes, texts


def annotate(img, boxes, texts, blocks, out_path: Path, title: str) -> None:
    vis = img.copy()
    colours = [(0, 0, 255), (0, 160, 255), (0, 200, 0), (255, 0, 200), (255, 200, 0), (200, 0, 0)]
    for bi, group in enumerate(blocks):
        xs = [bbox(boxes[i]) for i in group]
        x0 = int(min(v[0] for v in xs)); y0 = int(min(v[1] for v in xs))
        x1 = int(max(v[2] for v in xs)); y1 = int(max(v[3] for v in xs))
        cv2.rectangle(vis, (x0, y0), (x1, y1), (180, 180, 180), 2)
        cv2.putText(vis, f"B{bi}", (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    for i, quad in enumerate(boxes):
        colour = colours[i % len(colours)]
        cv2.polylines(vis, [quad.astype(np.int32).reshape(-1, 1, 2)], True, colour, 2)
        x0, y0, _, _ = bbox(quad)
        cv2.putText(vis, str(i), (int(x0), int(y0) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
    cv2.putText(vis, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(vis, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    # a downscaled copy so a large page can be eyeballed without a huge file
    long_side = max(vis.shape[:2])
    if long_side > 1600:
        f = 1600 / long_side
        cv2.imwrite(str(out_path.with_name(out_path.stem + "_preview.png")),
                    cv2.resize(vis, None, fx=f, fy=f, interpolation=cv2.INTER_AREA))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--page", type=Path, required=True, help="page image")
    ap.add_argument("--det", default="ft", choices=["stock", "ft"],
                    help="which detector to detail first in the text output (both always run)")
    ap.add_argument("--out", type=Path, default=Path("/tmp/opencode/page_lines"),
                    help="directory for the annotated PNGs")
    ap.add_argument("--gt-xml", type=Path, default=None, help="Manga109 annotations XML (optional)")
    ap.add_argument("--gt-page", type=int, default=-1, help="page index for --gt-xml")
    ap.add_argument("--max-text", type=int, default=40, help="clip recognised text in the printout")
    args = ap.parse_args()

    img = cv2.imread(str(args.page))
    if img is None:
        raise SystemExit(f"cannot read {args.page}")
    h, w = img.shape[:2]

    print(f"page {args.page}  {w}x{h}  sha256 {sha(args.page)[:16]}…")
    print("weights:")
    for name, path in sorted(DETECTORS, key=lambda t: t[0] != args.det):
        print(f"  det {name:5s} {path}  {path.stat().st_size} B  sha256 {sha(path)}")
    print(f"  rec ft    {REC}  {REC.stat().st_size} B  sha256 {sha(REC)}")
    print(f"  dict      {DICT}  sha256 {sha(DICT)}")

    rec = RecModel(REC, DICT)
    runs = {}
    for name, path in DETECTORS:
        runs[name] = run_detector(name, path, img, rec)

    blocks = gt_blocks(args.gt_xml, args.gt_page) if args.gt_xml and args.gt_page >= 0 else []
    if blocks:
        print(f"\nbubbles: {len(blocks)} from {args.gt_xml.name} page {args.gt_page} (Manga109 <text>)")
        bubbles = [("gt", b) for b in blocks]
    else:
        all_boxes = [q for name, _ in DETECTORS for q in runs[name][0]]
        clusters = cluster_bubbles(all_boxes)
        bubbles = []
        for group in clusters:
            xs = [bbox(all_boxes[i]) for i in group]
            bubbles.append(("auto", (min(v[0] for v in xs), min(v[1] for v in xs),
                                     max(v[2] for v in xs), max(v[3] for v in xs))))
        bubbles.sort(key=lambda t: (t[1][1], -t[1][0]))
        print(f"\nbubbles: {len(bubbles)} from the union of both detectors' boxes (no --gt-xml)")

    order = [args.det] + [n for n, _ in DETECTORS if n != args.det]
    for name in order:
        path = dict(DETECTORS)[name]
        boxes, texts = runs[name]
        vertical_all = sum(1 for q in boxes if bbox(q)[3] - bbox(q)[1] >= bbox(q)[2] - bbox(q)[0])
        vertical = vertical_all * 2 >= max(1, len(boxes))
        groups = group_blocks(boxes, vertical) if boxes else []
        groups.sort(key=lambda g: (min(bbox(boxes[i])[1] for i in g), -max(bbox(boxes[i])[0] for i in g)))
        print(f"\n=== {name} — {len(boxes)} boxes, {len(groups)} blocks, "
              f"writing order {'vertical (tategaki)' if vertical else 'horizontal'} ===")
        for bi, group in enumerate(groups):
            xs = [bbox(boxes[i]) for i in group]
            print(f"  block {bi:02d}  ({int(min(v[0] for v in xs))},{int(min(v[1] for v in xs))})-"
                  f"({int(max(v[2] for v in xs))},{int(max(v[3] for v in xs))})  {len(group)} lines")
            for i in sorted(group, key=lambda i: -bbox(boxes[i])[0] if vertical else bbox(boxes[i])[1]):
                quad = boxes[i].astype(int).tolist()
                text, score = texts[i]
                print(f"    line box[{i:02d}] q={quad} p={score:.2f}  {text[: args.max_text]}")
        annotate(img, boxes, texts, groups, args.out / f"{args.page.stem}_{name}.png",
                 f"{args.page.stem}  det={name}  {len(boxes)} lines / {len(groups)} blocks")

    print("\n=== per-bubble columns recovered ===")
    print(f"{'bubble':>7s} {'source':>7s} {'bbox':>28s} {'stock':>6s} {'ft':>3s}  note")
    lost_total = 0
    for bi, (src, block) in enumerate(bubbles):
        count = {}
        text = {}
        for name, _ in DETECTORS:
            boxes, texts = runs[name]
            sel = [i for i, q in enumerate(boxes) if inside(q, block)]
            sel.sort(key=lambda i: (-bbox(boxes[i])[0], bbox(boxes[i])[1]))
            count[name] = len(sel)
            text[name] = "".join(texts[i][0] for i in sel)
        note = ""
        if count["ft"] < count["stock"]:
            note = f"!! ft recovered {count['stock'] - count['ft']} fewer column(s)"
            lost_total += count["stock"] - count["ft"]
        elif count["ft"] > count["stock"]:
            note = f"ft found {count['ft'] - count['stock']} more column(s)"
        bb = tuple(int(v) for v in block)
        print(f"{bi:7d} {src:>7s} {str(bb):>28s} {count['stock']:6d} {count['ft']:3d}  {note}")
        if note:
            print(f"        stock text: {text['stock'][: args.max_text]}")
            print(f"        ft    text: {text['ft'][: args.max_text]}")
    tot = {name: sum(1 for b in bubbles for q in runs[name][0] if inside(q, b[1])) for name, _ in DETECTORS}
    print(f"boxes inside bubbles: stock {tot['stock']}, ft {tot['ft']}"
          + (f"  (ft lost {lost_total} of {tot['stock']} stock columns)" if lost_total else ""))
    print(f"\nannotated PNGs: {args.out}/{args.page.stem}_stock.png, "
          f"{args.out}/{args.page.stem}_ft.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
