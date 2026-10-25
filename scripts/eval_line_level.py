#!/usr/bin/env python3
"""Line/box-level detector comparison on the held-out Manga109-s books.

`eval_det.py` scores dialogue at **block** level, and `eval_blocks.py` scores a
block's transcript after `extract_page_lines.assign()` has mapped detected lines
onto Manga109 `<text>` blocks.  Both can look healthy while a detector drops
alternate text columns inside a bubble, because the GT has no line instances for
dialogue: an under-segmented bubble still "covers" the block (area coverage) and
still has *some* line to recognise.

This script scores the two detectors at the level a reader sees:

  * **SFX** — COO polygons are genuine per-line GT (one polygon per
    onomatopoeia).  Absolute line recall, lines with no covering box, and how many
    GT lines each predicted box swallows.
  * **dialogue** — Manga109/MangaOCR only annotates whole bubbles, so there is no
    independent dialogue line GT.  Lines are therefore taken **from the stock
    detector** (`n =` reference lines) and the fine-tune is scored against them:
    a stock line is *lost* when no fine-tune box covers >= 50 % of it, and a
    fine-tune box *merges* when it covers >= 2 stock lines at >= 50 % each.
    This is a paired, stock-as-reference measure - it answers "does the fine-tune
    lose/merge the lines the shipping detector produced", not "what is the
    absolute number of lines on the page" (that is not annotated).

Both detectors run through the runtime postprocess (`scripts/db_det.py`, 960 max
side, 0.15 / 0.25 / 1.40), on the same held-out books and pages as
`eval_det.py --split test`.  Detections are cached so the table can be
recomputed without a GPU/CPU pass.

Usage
  python scripts/eval_line_level.py                       # full test split, cached
  python scripts/eval_line_level.py --limit-pages 50      # smoke
  python scripts/eval_line_level.py --recompute --json-out output/eval_lines/line_level_test.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from shapely.geometry import Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from db_det import DBDetector  # noqa: E402
from eval_det import APP_DET, greedy_match, image_path, iou, load_gt, to_poly  # noqa: E402

STOCK = ROOT / "models/ppocrv6_small_det_stock.onnx"
FT = ROOT / "output/exp2_onnx/exp2_ep40.onnx"  # == shipped ppocrv6_small_det.onnx
DETECTORS = (("stock", STOCK), ("ft", FT))

IOU_LINE = 0.5  # "this prediction is that GT line"
COVER = 0.5  # "this box holds / is held by >= 50 % of that region"
SPAN_COVER = 0.25  # a box with >= 25 % in two bubbles touches both


# ---------------------------------------------------------------- geometry
def area(p: Polygon) -> float:
    return float(p.area) if p is not None and not p.is_empty else 0.0


def cover(inner: Polygon, outer: Polygon) -> float:
    """Fraction of `inner` that lies inside `outer`."""
    a = area(inner)
    if a <= 0:
        return 0.0
    if not inner.intersects(outer):
        return 0.0
    return min(1.0, float(inner.intersection(outer).area) / a)


def best_cover(poly: Polygon, boxes) -> float:
    return max((cover(poly, b) for b in boxes), default=0.0)


def lines_per_box(boxes, lines) -> list[int]:
    """How many `lines` each box swallows (>= COVER of the line inside the box)."""
    return [sum(1 for line in lines if cover(line, box) >= COVER) for box in boxes]


def assign_blocks(boxes, blocks) -> dict[int, list[int]]:
    """Assign each box to the smallest block holding >= COVER of it."""
    out: dict[int, list[int]] = defaultdict(list)
    for i, box in enumerate(boxes):
        best, best_area = -1, 1e18
        for j, block in enumerate(blocks):
            if cover(box, block) >= COVER:
                if area(block) < best_area:
                    best, best_area = j, area(block)
        if best >= 0:
            out[best].append(i)
    return out


def spans_blocks(box, blocks) -> int:
    return sum(1 for b in blocks if cover(box, b) >= SPAN_COVER)


# ---------------------------------------------------------------- detection
def detect_pages(pages, cache_dir: Path, recompute: bool) -> dict[str, dict[str, list]]:
    """Run both detectors over every page; returns {det_name: {page_key: [quad]}}."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict[str, list]] = {}
    for name, path in DETECTORS:
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        cache = cache_dir / f"{name}_{sha[:8]}.json"
        quads: dict[str, list] = {}
        if cache.exists() and not recompute:
            quads = json.loads(cache.read_text())["pages"]
        todo = [(k, p) for k, p in pages if k not in quads]
        if todo:
            det = DBDetector(str(path), **APP_DET)
            for n, (key, img_path) in enumerate(todo, 1):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                quads[key] = [
                    [[round(float(x), 1), round(float(y), 1)] for x, y in q] for q, _ in det(img)
                ]
                if n % 200 == 0:
                    print(f"  {name}: {n}/{len(todo)} new pages", flush=True)
            cache.write_text(json.dumps({"model": str(path), "sha256": sha, "pages": quads}))
        out[name] = quads
        print(f"  {name}: {len(quads)} pages in cache, {len(todo)} run now ({sha[:12]}…)")
    return out


# ---------------------------------------------------------------- scoring
def score(pages_gt, dets) -> dict:
    sfx = Counter()
    dia = Counter()
    lines_per_box_hist: dict[str, Counter] = {name: Counter() for name, _ in DETECTORS}
    lines_per_ft_box_hist = Counter()
    lost_examples = []
    page_rows = []

    for key, gts in pages_gt.items():
        sfx_polys = [to_poly(g["points"]) for g in gts if g["class"] == "sfx" and g["source"] == "coo"]
        sfx_polys = [p for p in sfx_polys if p is not None]
        dia_polys = [to_poly(g["points"]) for g in gts if g["class"] == "dialogue"]
        dia_polys = [p for p in dia_polys if p is not None]

        boxes = {}
        for name, _ in DETECTORS:
            quads = dets[name].get(key, [])
            polys = [to_poly(q) for q in quads]
            boxes[name] = [p for p in polys if p is not None]

        # ---- SFX: COO polygons are per-line GT (absolute)
        for name, _ in DETECTORS:
            b = boxes[name]
            if sfx_polys:
                mat = [[iou(p, g) for g in sfx_polys] for p in b]
                matched = greedy_match(mat, IOU_LINE)
                sfx[f"{name}_tp"] += len(matched)
                sfx[f"{name}_gt"] += len(sfx_polys)
                # looser 0.4 match, the crop recall threshold of eval_det.py
                sfx[f"{name}_tp40"] += len(greedy_match(mat, 0.4))
                sfx[f"{name}_gt40"] += len(sfx_polys)
            sfx[f"{name}_pred"] += len(b)
            per_box = lines_per_box(b, sfx_polys)
            for n in per_box:
                lines_per_box_hist[name][n] += 1
            sfx[f"{name}_boxes_multi"] += sum(1 for n in per_box if n >= 2)
            sfx[f"{name}_lines_in_multi"] += sum(n for n in per_box if n >= 2)
            uncovered = sum(1 for g in sfx_polys if best_cover(g, b) < COVER)
            sfx[f"{name}_lines_uncovered"] += uncovered
            sfx[f"{name}_span_sfx"] += sum(1 for p in b if spans_blocks(p, sfx_polys) >= 2)

        # ---- dialogue: stock boxes are the reference line set (paired)
        for name, _ in DETECTORS:
            b = boxes[name]
            dia[f"{name}_boxes"] += len(b)
            dia[f"{name}_span_blocks"] += sum(1 for p in b if spans_blocks(p, dia_polys) >= 2)
            dia[f"{name}_bubbles"] += len(dia_polys)
            assigned = assign_blocks(b, dia_polys)
            n_assigned = 0
            for bi, _ in enumerate(dia_polys):
                idx = assigned.get(bi, [])
                n_assigned += len(idx)
                dia[f"{name}_boxes_in_bubbles"] += len(idx)
                if len(idx) == 0:
                    dia[f"{name}_bubbles_empty"] += 1
                elif len(idx) == 1:
                    dia[f"{name}_bubbles_one"] += 1
                else:
                    dia[f"{name}_bubbles_multi"] += 1
                dia[f"{name}_bubble_box_sum"] += len(idx)
            dia[f"{name}_assigned"] += n_assigned

        stock_assigned = assign_blocks(boxes["stock"], dia_polys)
        ft_assigned = assign_blocks(boxes["ft"], dia_polys)
        for bi, _ in enumerate(dia_polys):
            ref = [boxes["stock"][i] for i in stock_assigned.get(bi, [])]
            ftb = [boxes["ft"][i] for i in ft_assigned.get(bi, [])]
            if not ref:
                continue
            bucket = str(min(len(ref), 4))
            dia[f"size{bucket}_bubbles"] += 1
            dia[f"size{bucket}_ref_lines"] += len(ref)
            dia["ref_lines"] += len(ref)
            cov = 0
            for line in ref:
                hit = any(cover(line, b) >= COVER for b in ftb)
                dia["ref_covered_by_ft"] += int(hit)
                cov += int(hit)
                if not hit:
                    lost_examples.append((key, [round(v, 1) for v in line.bounds]))
            dia[f"size{bucket}_covered"] += cov
            if len(ftb) < len(ref):
                dia["bubbles_ft_fewer"] += 1
                dia["lines_lost_in_fewer_bubbles"] += len(ref) - cov
            elif len(ftb) == len(ref):
                dia["bubbles_ft_equal"] += 1
            else:
                dia["bubbles_ft_more"] += 1
            dia["bubbles_ft_empty_of_ref"] += int(len(ftb) == 0)
            dia["bubbles_ft_under_half"] += int(len(ftb) * 2 < len(ref))
            pieces = [p.intersection(dia_polys[bi]) for p in ftb if p.intersects(dia_polys[bi])]
            frac = min(1.0, unary_union(pieces).area / area(dia_polys[bi])) if pieces else 0.0
            if cov < len(ref):
                # does eval_det's block-coverage metric still call this bubble fine?
                dia["bubbles_any_loss"] += 1
                dia["any_loss_block_coverage_50"] += int(frac >= 0.5)
                dia["any_loss_coverage_sum"] += frac
            if len(ftb) * 2 < len(ref):
                dia["under_half_block_coverage_50"] += int(frac >= 0.5)
                dia["under_half_coverage_sum"] += frac
            n_in = lines_per_box(ftb, ref)
            for n in n_in:
                lines_per_ft_box_hist[n] += 1
                if n >= 2:
                    dia["ft_boxes_multi_ref"] += 1
            dia["ft_boxes_in_ref_bubbles"] += len(ftb)

        page_rows.append((key, len(sfx_polys), len(dia_polys)))

    return {
        "sfx": dict(sfx),
        "dialogue": dict(dia),
        "hist_sfx_lines_per_box": {k: dict(v) for k, v in lines_per_box_hist.items()},
        "hist_ft_ref_lines_per_box": dict(lines_per_ft_box_hist),
        "lost_examples": lost_examples[:5],
        "pages": len(page_rows),
    }


def pct(a: float, b: float) -> str:
    return f"{100 * a / b:.2f} %" if b else "n/a"


def report(res: dict, det_shas: dict[str, str], n_pages_gt: int) -> None:
    s, d = res["sfx"], res["dialogue"]
    print(f"\npages with GT {n_pages_gt}, evaluated {res['pages']}")
    print("detector sha256: " + ", ".join(f"{k} {v[:12]}…" for k, v in det_shas.items()))

    print("\n=== SFX — COO polygons are per-LINE ground truth (absolute) ===")
    print(f"{'det':6s} {'lines':>6s} {'recall@.5':>10s} {'recall@.4':>10s} {'lines no box':>13s} "
          f"{'GT/box':>7s} {'boxes>=2 GT':>12s} {'lines in them':>14s} {'boxes':>7s}")
    for name, _ in DETECTORS:
        gt = s.get(f"{name}_gt", 0)
        boxes = s.get(f"{name}_pred", 0)
        hist = res["hist_sfx_lines_per_box"][name]
        tot = sum(hist.values())
        mean = sum(int(k) * v for k, v in hist.items()) / tot if tot else 0.0
        print(f"{name:6s} {gt:6d} {pct(s.get(f'{name}_tp', 0), gt):>10s} "
              f"{pct(s.get(f'{name}_tp40', 0), gt):>10s} {s.get(f'{name}_lines_uncovered', 0):13d} "
              f"{mean:7.2f} {s.get(f'{name}_boxes_multi', 0):12d} {s.get(f'{name}_lines_in_multi', 0):14d} "
              f"{boxes:7d}")
    print("  GT lines per box distribution (n lines -> boxes): "
          + "; ".join(f"{k}:{v}" for k, v in sorted(res["hist_sfx_lines_per_box"]["stock"].items()))
          + " (stock) vs "
          + "; ".join(f"{k}:{v}" for k, v in sorted(res["hist_sfx_lines_per_box"]["ft"].items()))
          + " (ft)")
    for name, _ in DETECTORS:
        hist = res["hist_sfx_lines_per_box"][name]
        holding = sum(v for k, v in hist.items() if int(k) >= 1)
        lines_in = sum(int(k) * v for k, v in hist.items())
        print(f"  {name:5s}: boxes holding >=1 SFX line {holding}, "
              f"mean SFX lines per such box {lines_in / holding if holding else 0:.2f}")

    print("\n=== dialogue — reference lines = stock detector boxes inside MangaOCR bubbles (paired) ===")
    ref = d.get("ref_lines", 0)
    print(f"reference lines (stock boxes inside bubbles): {ref}")
    print(f"{'det':6s} {'boxes':>7s} {'in bubbles':>11s} {'bubbles':>8s} {'empty':>6s} {'1 box':>6s} "
          f"{'>1 box':>7s} {'boxes/bubble':>13s} {'boxes in >=2 bubbles':>21s}")
    for name, _ in DETECTORS:
        nb = d.get(f"{name}_bubbles", 0)
        print(f"{name:6s} {d.get(f'{name}_boxes', 0):7d} {d.get(f'{name}_boxes_in_bubbles', 0):11d} {nb:8d} "
              f"{d.get(f'{name}_bubbles_empty', 0):6d} {d.get(f'{name}_bubbles_one', 0):6d} "
              f"{d.get(f'{name}_bubbles_multi', 0):7d} "
              f"{(d.get(f'{name}_bubble_box_sum', 0) / nb if nb else 0):13.2f} "
              f"{d.get(f'{name}_span_blocks', 0):21d}")
    print(f"\n  fine-tune vs the stock reference lines:")
    print(f"    stock lines covered by an ft box        {d.get('ref_covered_by_ft', 0)} / {ref} "
          f"= {pct(d.get('ref_covered_by_ft', 0), ref)}")
    print(f"    stock lines with NO covering ft box     {ref - d.get('ref_covered_by_ft', 0)} "
          f"= {pct(ref - d.get('ref_covered_by_ft', 0), ref)}")
    n_paired = d.get("bubbles_ft_fewer", 0) + d.get("bubbles_ft_equal", 0) + d.get("bubbles_ft_more", 0)
    print(f"    bubbles where ft has FEWER boxes        {d.get('bubbles_ft_fewer', 0)} "
          f"= {pct(d.get('bubbles_ft_fewer', 0), n_paired)} of {n_paired} paired bubbles")
    print(f"    bubbles where ft has the same / MORE      {d.get('bubbles_ft_equal', 0)} / {d.get('bubbles_ft_more', 0)}")
    print(f"    bubbles ft leaves with < half the boxes   {d.get('bubbles_ft_under_half', 0)}"
          f"   (none at all: {d.get('bubbles_ft_empty_of_ref', 0)})")
    al = d.get("bubbles_any_loss", 0)
    if al:
        print(f"    bubbles that lost >=1 stock line         {al} of {n_paired} "
              f"({pct(al, n_paired)})")
        print(f"    of those, still counted as 'covered' by eval_det's block metric (>=50 % area): "
              f"{d.get('any_loss_block_coverage_50', 0)} = {pct(d.get('any_loss_block_coverage_50', 0), al)} "
              f"(mean area coverage {100 * d.get('any_loss_coverage_sum', 0.0) / al:.1f} %)")
    uh = d.get("bubbles_ft_under_half", 0)
    if uh:
        print(f"    of the < half ones, still 'covered' (>=50 % area): "
              f"{d.get('under_half_block_coverage_50', 0)} = {pct(d.get('under_half_block_coverage_50', 0), uh)} "
              f"(mean area coverage {100 * d.get('under_half_coverage_sum', 0.0) / uh:.1f} %)")

    print("\n  loss by bubble size (stock box count in the bubble):")
    print(f"    {'stock boxes':>11s} {'bubbles':>8s} {'stock lines':>11s} {'covered':>8s} {'lost':>6s} {'recall':>8s}")
    for bucket in ("1", "2", "3", "4"):
        nb = d.get(f"size{bucket}_bubbles", 0)
        if not nb:
            continue
        lines = d.get(f"size{bucket}_ref_lines", 0)
        cov = d.get(f"size{bucket}_covered", 0)
        label = bucket if bucket != "4" else "4+"
        print(f"    {label:>11s} {nb:8d} {lines:11d} {cov:8d} {lines - cov:6d} {pct(cov, lines):>8s}")

    tot = sum(res["hist_ft_ref_lines_per_box"].values())
    mean = sum(int(k) * v for k, v in res["hist_ft_ref_lines_per_box"].items()) / tot if tot else 0.0
    print(f"    stock lines per ft box (mean)           {mean:.2f}   "
          f"ft boxes covering >=2 stock lines: {d.get('ft_boxes_multi_ref', 0)} / "
          f"{d.get('ft_boxes_in_ref_bubbles', 0)} ft boxes in reference bubbles")
    print("    ft boxes by number of stock lines held: "
          + "; ".join(f"{k}:{v}" for k, v in sorted(res["hist_ft_ref_lines_per_box"].items())))
    if res["lost_examples"]:
        print("    sample stock lines the ft detector produced nothing for "
              "(page, bbox): " + str(res["lost_examples"][:3]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--det-labels", type=Path, default=ROOT / "data/det")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "output/eval_lines/det_cache")
    ap.add_argument("--recompute", action="store_true", help="ignore the detection cache")
    ap.add_argument("--limit-pages", type=int, default=0, help="first N pages only (smoke)")
    ap.add_argument("--cover", type=float, default=None,
                    help="fraction of a line a box must hold to count as covering it (default 0.5)")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    global COVER
    if args.cover is not None:
        COVER = args.cover

    pages_gt = load_gt(args.det_labels.resolve(), args.split)
    keys = sorted(pages_gt)
    if args.limit_pages:
        keys = keys[: args.limit_pages]
    pages = []
    for key in keys:
        p = image_path(args.det_labels.resolve(), pages_gt[key][0]["image"])
        if p.exists():
            pages.append((f"{key[0]}/{key[1]}", p))
    if not pages:
        raise SystemExit("no page images found")
    print(f"{len(pages)} pages, {len({k.split('/')[0] for k, _ in pages})} books")

    dets = detect_pages(pages, args.cache_dir, args.recompute)
    sub = {f"{b}/{p}": pages_gt[(b, p)] for b, p in keys}
    res = score(sub, dets)
    shas = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in DETECTORS}
    report(res, shas, len(keys))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        doc = dict(res)
        doc.update({"split": args.split, "pages": len(keys), "model_sha256": shas,
                    "det": APP_DET, "iou_line": IOU_LINE, "cover": COVER, "span_cover": SPAN_COVER})
        doc.pop("lost_examples", None)
        args.json_out.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
