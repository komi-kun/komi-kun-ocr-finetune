#!/usr/bin/env python3
"""Detector evaluation on held-out Manga109-s books.

Runs a DB detector ONNX through the *runtime* post-processing convention
(``scripts/db_det.py``: 960 max side, thresh 0.15, box_thresh 0.25, unclip 1.40 -
the app defaults) and scores its output against the detection ground truth in
``data/det/`` (``scripts/build_det_dataset.py``), split by class:

  * **SFX** ground truth  = COO onomatopoeia polygons (``source=coo``), arbitrary
    4-66 point geometry, matched as a polygon against the predicted quadrilateral.
  * **dialogue** ground truth = MangaOCR boxes (``class=dialogue``).  These are
    Manga109 ``<text>`` **blocks** (whole bubbles / 2-4 columns), so dialogue
    precision/recall here is *block/bubble level*, not line level.  Pair it with
    the end-to-end block CER of ``scripts/eval_blocks.py``.

Definitions (all reproducible, all recorded in the JSON):

* IoU is the **polygon** IoU (shapely intersection / union) between the predicted
  quadrilateral and the GT polygon - never a box-vs-box IoU, so a curved SFX is
  scored on its actual curve.
* Matching: greedy one-to-one by IoU, highest pair first, same threshold as the
  metric (0.5 for P/R/H, 0.4 additionally for the SFX crop recall).  A prediction
  can match at most one GT and a GT at most one prediction; extra predictions on
  an already matched GT are false positives (``duplicate``).
* Per-class precision attributes every prediction to a class: the class of the GT
  it was matched to at 0.5, else the class of the GT it overlaps most (any
  IoU > 0), else ``background``.  ``precision`` is ``tp / attributed``;
  ``precision_incl_background`` additionally counts background predictions as
  false positives of that class (the conservative number).  Recall is
  ``tp / gt``.
* False positives split by cause, per unmatched prediction:
  - ``duplicate``: IoU >= 0.5 with a GT that another prediction already matched;
  - ``excluded_sfx``: best overlap is a MangaOCR SFX box (labelled, but not part of
    the clean COO SFX GT); inside the annotation mask, just not scored;
  - ``localized_sfx`` / ``localized_dialogue``: overlaps a scored GT of that class
    with 0 < IoU < 0.5 (for dialogue this is the normal line-box-inside-bubble
    case, see below; for SFX it is a split fragment of a curved onomatopoeia);
  - ``unlabeled_text``: no annotation overlap, but >= 10 % of the prediction's
    area lies inside a Manga109 ``<text>`` block that is itself < 50 % covered by
    any positive label (one of the ~4 % blocks that neither COO nor MangaOCR
    annotated) - real text the detector found, not a blank-art detection;
  - ``art``: none of the above - a detection on unannotated artwork.
  Normalisations: counts per evaluated page and per scored GT instance, plus the
  share of the page area covered by ``art`` false positives.

Because dialogue GT is block-level, line detections cannot reach IoU 0.5 with a
bubble, so the dialogue IoU-0.5 P/R/H is reported for stock-vs-candidate
comparability only.  ``dialogue_block_coverage`` is the meaningful companion:
fraction of dialogue blocks whose area is >= 50 % covered by the union of det
boxes (plus the mean coverage and the share of predictions that lie >= 50 %
inside a dialogue block).

Val/test books are disjoint by construction (``build_real_dataset.book_split``,
the same hash split as the recogniser dataset) and never overlap train.  Nothing
here tunes on test: ``--split test`` only reports.

Usage
  python scripts/eval_det.py --model models/ppocrv6_small_det.onnx --split test \\
      --out output/eval_det/stock_test.json
  python scripts/eval_det.py --model models/ppocrv6_small_det_manga.onnx --split test \\
      --out output/eval_det/cand_test.json
  python scripts/eval_det.py --model models/ppocrv6_small_det.onnx --split test \\
      --overlay-pages 8 --overlay-dir /tmp/opencode/det_overlays
  python scripts/eval_det.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon, box as shapely_box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_det_dataset import book_split, parse_blocks  # noqa: E402
from db_det import DBDetector  # noqa: E402

# Runtime detector defaults - the app's and scripts/db_det.py's convention.
APP_DET = dict(thresh=0.15, box_thresh=0.25, unclip_ratio=1.40, limit_side_len=960)
IOU_MATCH = 0.5
IOU_CROP = 0.4
UNLABELED_TEXT_OVERLAP = 0.10  # >=10 % of a prediction inside an uncovered text block
BLOCK_COVERED = 0.50  # a block counts as covered when >=50 % of it is inside a label


# ---------------------------------------------------------------- geometry
def to_poly(points) -> Polygon | None:
    """Shapely polygon from a point list, repaired if needed; None if degenerate."""
    if len(points) < 3:
        return None
    poly = Polygon([(float(x), float(y)) for x, y in points])
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 0:
        return None
    return poly


def iou(a: Polygon, b: Polygon) -> float:
    """Polygon IoU (not a box IoU): curved GT polygons are scored on their curve."""
    if a.is_empty or b.is_empty:
        return 0.0
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    return float(inter / (a.area + b.area - inter))


def covered_fraction(block: Polygon, labels) -> float:
    if block.area <= 0:
        return 0.0
    inside = 0.0
    for lab in labels:
        if lab.intersects(block):
            inside += block.intersection(lab).area
    return min(1.0, inside / block.area)


def point_bucket(n: int) -> str:
    if n <= 4:
        return "4"
    if n <= 8:
        return "5-8"
    return "9+"


# ---------------------------------------------------------------- matching / scoring
def greedy_match(iou_matrix, threshold):
    """One-to-one greedy matching, highest IoU first. Returns [(pred_i, gt_j, iou)]."""
    pairs = [
        (iou_matrix[i][j], i, j)
        for i in range(len(iou_matrix))
        for j in range(len(iou_matrix[i]))
        if iou_matrix[i][j] >= threshold
    ]
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
    used_p, used_g, out = set(), set(), []
    for score, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        out.append((i, j, score))
    return out


def score_page(preds, gt, blocks, page_area, excluded=(), iou_match=IOU_MATCH, iou_crop=IOU_CROP):
    """Score one page.

    preds: [(quad 4x2 array-like, score)]
    gt:    [{"class": "sfx"|"dialogue", "source": str, "npoints": int, "poly": Polygon}]
    blocks: [Polygon]  Manga109 <text> blocks on the page (for FP attribution)
    excluded: [Polygon] labelled regions that are *not* scored (MangaOCR SFX boxes);
              they stay in the annotation mask so detections on them are not "art"
    """
    pred_polys, pred_scores = [], []
    for quad, score in preds:
        p = to_poly(quad)
        pred_polys.append(p)
        pred_scores.append(float(score))
    keep = [i for i, p in enumerate(pred_polys) if p is not None]
    pred_polys = [pred_polys[i] for i in keep]
    pred_scores = [pred_scores[i] for i in keep]

    gts = [g for g in gt if g["poly"] is not None]
    excluded = [p for p in excluded if p is not None]
    labels = [g["poly"] for g in gts] + excluded
    uncovered = [b for b in blocks if covered_fraction(b, labels) < BLOCK_COVERED]

    mat = [[iou(p, g["poly"]) for g in gts] for p in pred_polys]
    matched = greedy_match(mat, iou_match)
    matched_p = {i: (j, s) for i, j, s in matched}
    matched_g = {j for _, j, _ in matched}

    # crop-level matching for the SFX crop recall (looser threshold, independent)
    crop_matched = greedy_match(mat, iou_crop)
    crop_pairs = [(i, j, s) for i, j, s in crop_matched if gts[j]["class"] == "sfx"]

    page = {
        "preds": len(pred_polys),
        "matched": len(matched),
        "gt": Counter(),
        "gt_source": Counter(),
        "tp": Counter(),
        "attributed": Counter(),
        "fp": Counter(),
        "fp_art_area": 0.0,
        "page_area": float(page_area),
        "crop": Counter(),
        "crop_tp_by_points": Counter(),
        "crop_gt_by_points": Counter(),
        "sfx_gt_by_points": Counter(),
        "recall_by_points": Counter(),
        "dialogue_blocks": 0,
        "dialogue_blocks_covered": 0,
        "dialogue_coverage_sum": 0.0,
        "preds_on_dialogue": 0,
    }
    for g in gts:
        page["gt"][g["class"]] += 1
    for g in gts:
        if g["class"] == "sfx" and g["source"] == "coo":
            page["sfx_gt_by_points"][point_bucket(g["npoints"])] += 1
            page["crop_gt_by_points"][point_bucket(g["npoints"])] += 1

    # class attribution for precision, and TP at the metric threshold
    for i, p in enumerate(pred_polys):
        if i in matched_p:
            j, _ = matched_p[i]
            cls = gts[j]["class"]
            page["tp"][cls] += 1
            page["attributed"][cls] += 1
            if cls == "sfx" and gts[j]["source"] == "coo":
                page["recall_by_points"][point_bucket(gts[j]["npoints"])] += 1
            continue
        best_j, best = -1, 0.0
        for j, g in enumerate(gts):
            if mat[i][j] > best:
                best, best_j = mat[i][j], j
        if best_j >= 0 and best > 0:
            page["attributed"][gts[best_j]["class"]] += 1

    for i, j, _ in crop_pairs:
        if gts[j]["source"] == "coo":
            page["crop"]["tp"] += 1
            page["crop_tp_by_points"][point_bucket(gts[j]["npoints"])] += 1

    # dialogue is block-level GT: line boxes cannot reach IoU 0.5 with a bubble,
    # so also report area coverage of each block by the det boxes.
    dia = [g["poly"] for g in gts if g["class"] == "dialogue"]
    for b in dia:
        page["dialogue_blocks"] += 1
        pieces = [p.intersection(b) for p in pred_polys if p.intersects(b)]
        frac = (unary_union(pieces).area / b.area) if pieces else 0.0
        frac = min(1.0, max(0.0, frac))
        page["dialogue_coverage_sum"] += frac
        page["dialogue_blocks_covered"] += int(frac >= 0.5)
    for p in pred_polys:
        if any(p.intersection(b).area >= 0.5 * p.area for b in dia):
            page["preds_on_dialogue"] += 1

    # false-positive causes
    for i, p in enumerate(pred_polys):
        if i in matched_p:
            continue
        kind = None
        for j, g in enumerate(gts):
            if mat[i][j] >= iou_match and j in matched_g:
                kind = "duplicate"
                break
        if kind is None:
            best_j, best = -1, 0.0
            for j, g in enumerate(gts):
                if mat[i][j] > best:
                    best, best_j = mat[i][j], j
            if best_j >= 0 and best > 0:
                kind = f"localized_{gts[best_j]['class']}"
            elif any(iou(p, e) > 0 for e in excluded):
                # labelled SFX that is intentionally not part of the scored GT
                kind = "excluded_sfx"
        if kind is None:
            for b in uncovered:
                if b.intersects(p) and p.intersection(b).area >= UNLABELED_TEXT_OVERLAP * p.area:
                    kind = "unlabeled_text"
                    break
        if kind is None:
            kind = "art"
            page["fp_art_area"] += p.area
        page["fp"][kind] += 1
    return page


def prf(tp: int, n_pred: int, n_gt: int) -> dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gt if n_gt else 0.0
    h = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "hmean": h, "tp": tp, "pred": n_pred, "gt": n_gt}


def summarise(pages, iou_crop=IOU_CROP) -> dict:
    """Aggregate per-page counters into micro metrics + FP assessment."""
    gt, tp, attr, preds, matched = Counter(), Counter(), Counter(), 0, 0
    fp, art_area, page_area = Counter(), 0.0, 0.0
    crop_gt, crop_tp = Counter(), Counter()
    sfx_gt_pts, rec_pts, crop_tp_pts = Counter(), Counter(), Counter()
    sfx_gt_total = sfx_crop_tp = 0
    dia_blocks = dia_covered = on_dia = 0
    dia_sum = 0.0
    for pg in pages:
        gt.update(pg["gt"])
        tp.update(pg["tp"])
        attr.update(pg["attributed"])
        fp.update(pg["fp"])
        preds += pg["preds"]
        matched += pg["matched"]
        art_area += pg["fp_art_area"]
        page_area += pg["page_area"]
        crop_gt.update(pg["crop_gt_by_points"])
        crop_tp.update(pg["crop_tp_by_points"])
        sfx_gt_pts.update(pg["sfx_gt_by_points"])
        rec_pts.update(pg["recall_by_points"])
        crop_tp_pts.update(pg["crop_tp_by_points"])
        for k, v in pg["crop_gt_by_points"].items():
            sfx_gt_total += v
        for k in ("4", "5-8", "9+"):
            sfx_crop_tp += pg["crop_tp_by_points"].get(k, 0)
        dia_blocks += pg["dialogue_blocks"]
        dia_covered += pg["dialogue_blocks_covered"]
        dia_sum += pg["dialogue_coverage_sum"]
        on_dia += pg["preds_on_dialogue"]

    classes = {}
    for cls in ("sfx", "dialogue"):
        base = prf(tp[cls], attr[cls], gt[cls])
        base["attributed"] = attr[cls]
        base["precision_incl_background"] = tp[cls] / (attr[cls] + fp["art"]) if (attr[cls] + fp["art"]) else 0.0
        classes[cls] = base
    all_gt = gt["sfx"] + gt["dialogue"]
    classes["all"] = prf(matched, preds, all_gt)
    classes["all"]["attributed"] = preds
    classes["all"]["precision_incl_background"] = classes["all"]["precision"]

    n_pages = max(1, len(pages))
    n_gt = max(1, all_gt)
    fp_total = sum(fp.values())
    return {
        "classes": classes,
        "pages": len(pages),
        "predictions": preds,
        "false_positives": {
            "total": fp_total,
            "by_kind": dict(fp),
            "per_page": fp_total / n_pages,
            "per_gt": fp_total / n_gt,
            "art_per_page": fp["art"] / n_pages,
            "art_per_gt": fp["art"] / n_gt,
            "art_area_fraction": art_area / page_area if page_area else 0.0,
        },
        "sfx_crop_recall": {
            "iou": iou_crop,
            "recall": sfx_crop_tp / sfx_gt_total if sfx_gt_total else 0.0,
            "tp": sfx_crop_tp,
            "gt": sfx_gt_total,
            "by_points": {
                b: {
                    "recall": crop_tp.get(b, 0) / crop_gt[b] if crop_gt[b] else 0.0,
                    "tp": crop_tp.get(b, 0),
                    "gt": crop_gt[b],
                }
                for b in ("4", "5-8", "9+")
            },
        },
        "sfx_recall_by_points": {
            b: {
                "recall": rec_pts.get(b, 0) / sfx_gt_pts[b] if sfx_gt_pts[b] else 0.0,
                "tp": rec_pts.get(b, 0),
                "gt": sfx_gt_pts[b],
            }
            for b in ("4", "5-8", "9+")
        },
        "dialogue_block_coverage": {
            "blocks": dia_blocks,
            "blocks_covered": dia_covered,
            "recall": dia_covered / dia_blocks if dia_blocks else 0.0,
            "mean_coverage": dia_sum / dia_blocks if dia_blocks else 0.0,
            "preds_on_dialogue": on_dia,
            "preds_share": on_dia / preds if preds else 0.0,
        },
    }


def macro_by_book(per_page: dict[str, list[dict]]) -> dict:
    out = {}
    for book, pages in per_page.items():
        s = summarise(pages)
        out[book] = {
            "pages": s["pages"],
            "sfx": s["classes"]["sfx"],
            "dialogue": s["classes"]["dialogue"],
            "all": s["classes"]["all"],
            "false_positives": s["false_positives"],
            "sfx_crop_recall": s["sfx_crop_recall"],
            "dialogue_block_coverage": s["dialogue_block_coverage"],
        }
    mean = {}
    for cls in ("sfx", "dialogue", "all"):
        for metric in ("precision", "recall", "hmean"):
            vals = [b[cls][metric] for b in out.values() if b[cls]["gt"] or metric == "precision"]
            mean[f"{cls}_{metric}"] = sum(vals) / len(vals) if vals else 0.0
    mean["sfx_crop_recall"] = (
        sum(b["sfx_crop_recall"]["recall"] for b in out.values()) / len(out) if out else 0.0
    )
    mean["fp_per_page"] = (
        sum(b["false_positives"]["per_page"] for b in out.values()) / len(out) if out else 0.0
    )
    mean["dialogue_block_coverage"] = (
        sum(b["dialogue_block_coverage"]["recall"] for b in out.values()) / len(out) if out else 0.0
    )
    return {"books": out, "mean": mean}


# ---------------------------------------------------------------- data loading
def load_gt(det_dir: Path, split: str) -> dict[tuple[str, int], list[dict]]:
    """Positive labels per (book, page) from the builder's sidecar."""
    pages: dict[tuple[str, int], list[dict]] = defaultdict(list)
    with open(det_dir / f"{split}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("role") != "positive":
                continue
            pages[(rec["book"], int(rec["page"]))].append(
                {
                    "class": rec["class"],
                    "source": rec["source"],
                    "npoints": len(rec["points"]),
                    "points": rec["points"],
                    "image": rec["image"],
                }
            )
    return dict(pages)


def load_blocks(manga109: Path, book: str) -> dict[int, list]:
    root = next(p.parent for p in manga109.rglob("annotations") if p.is_dir() and (p.parent / "images").is_dir())
    return parse_blocks(root / "annotations" / f"{book}.xml")


def image_path(det_dir: Path, rel: str) -> Path:
    return ROOT / rel if not Path(rel).is_absolute() else Path(rel)


# ---------------------------------------------------------------- workers
_STATE: dict = {}


def _init_worker(onnx: str, det_kwargs: dict, det_dir: str) -> None:
    _STATE["det"] = DBDetector(onnx, **det_kwargs)
    _STATE["det_dir"] = Path(det_dir)


def _run_page(task) -> dict:
    book, page, image, gt, blocks, page_area = task
    img = cv2.imread(str(image))
    if img is None:
        return {"book": book, "page": page, "error": "unreadable"}
    h, w = img.shape[:2]
    preds = _STATE["det"](img)
    gt_polys, excluded_polys = [], []
    for g in gt:
        if g["class"] == "sfx" and g["source"] == "mangaocr":
            # labelled, but not part of the clean COO SFX GT: annotation mask only
            excluded_polys.append(to_poly(g["points"]))
            continue
        gt_polys.append(
            {
                "class": g["class"],
                "source": g["source"],
                "npoints": g["npoints"],
                "poly": to_poly(g["points"]),
            }
        )
    block_polys = [shapely_box(x0, y0, x1, y1) for x0, y0, x1, y1 in blocks]
    result = score_page(preds, gt_polys, block_polys, float(h * w), excluded=excluded_polys)
    result["book"] = book
    result["page"] = page
    result["quads"] = [np.asarray(q, float).round(1).tolist() for q, _ in preds]
    result["scores"] = [float(s) for _, s in preds]
    result["gt_polys"] = [g["points"] for g in gt if not (g["class"] == "sfx" and g["source"] == "mangaocr")]
    result["gt_classes"] = [g["class"] for g in gt if not (g["class"] == "sfx" and g["source"] == "mangaocr")]
    return result


def run(args) -> dict:
    det_dir = Path(args.det_labels).resolve()
    gt_pages = load_gt(det_dir, args.split)
    books = sorted({book for book, _ in gt_pages})
    stray = [b for b in books if book_split(b) != args.split]
    if stray:
        raise SystemExit(f"sidecar contains books outside the {args.split} split: {stray}")
    if args.book:
        books = [b for b in books if b in set(args.book)]
    if not books:
        raise SystemExit(f"no {args.split} books with labels in {det_dir}")

    block_cache = {b: load_blocks(args.manga109, b) for b in books}
    tasks = []
    for book in books:
        for (b, page), gt in sorted(gt_pages.items()):
            if b != book:
                continue
            img = image_path(det_dir, gt[0]["image"])
            if not img.exists():
                continue
            tasks.append(
                (
                    book,
                    page,
                    img,
                    gt,
                    block_cache[book].get(page, []),
                    0.0,
                )
            )
            if args.limit_pages and len(tasks) >= args.limit_pages:
                break
        if args.limit_pages and len(tasks) >= args.limit_pages:
            break

    det_kwargs = dict(
        thresh=args.thresh,
        box_thresh=args.box_thresh,
        unclip_ratio=args.unclip_ratio,
        limit_side_len=args.limit_side_len,
    )
    results = []
    if args.workers > 1:
        with mp.Pool(args.workers, initializer=_init_worker, initargs=(str(args.model), det_kwargs, str(det_dir))) as pool:
            for res in pool.imap_unordered(_run_page, tasks, chunksize=4):
                results.append(res)
                if len(results) % 100 == 0:
                    print(f"  {len(results)}/{len(tasks)} pages", flush=True)
    else:
        _init_worker(str(args.model), det_kwargs, str(det_dir))
        for task in tasks:
            results.append(_run_page(task))
            if len(results) % 100 == 0:
                print(f"  {len(results)}/{len(tasks)} pages", flush=True)

    results = [r for r in results if "error" not in r]
    per_book: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        per_book[r["book"]].append(r)
    summary = summarise(results)
    summary.update(macro_by_book(per_book))
    summary.update(
        {
            "model": str(args.model),
            "model_sha256": hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
            "split": args.split,
            "books": books,
            "book_disjoint": True,
            "det": {**det_kwargs, "iou_match": args.iou, "iou_crop": args.crop_iou},
            "gt": {
                "sfx": "COO polygons (source=coo, positive), matched as polygons",
                "dialogue": "MangaOCR boxes (class=dialogue) - Manga109 <text> blocks, block/bubble level",
                "excluded_from_gt": "MangaOCR SFX boxes (class=sfx, source=mangaocr), still counted in the FP mask as excluded_sfx",
            },
            "pages_skipped": len(tasks) - len(results),
        }
    )
    summary["results"] = [
        {
            "book": r["book"],
            "page": r["page"],
            "preds": r["preds"],
            "matched": r["matched"],
            "gt": dict(r["gt"]),
            "tp": dict(r["tp"]),
            "attributed": dict(r["attributed"]),
            "fp": dict(r["fp"]),
            "crop_tp": dict(r["crop_tp_by_points"]),
            "crop_gt": dict(r["crop_gt_by_points"]),
            "quads": r["quads"],
            "scores": r["scores"],
            "gt_polys": r["gt_polys"],
            "gt_classes": r["gt_classes"],
        }
        for r in results
    ]
    return summary


# ---------------------------------------------------------------- overlays
def write_overlays(args, summary) -> int:
    out_dir = Path(args.overlay_dir).resolve()
    if not str(out_dir).startswith("/tmp/"):
        raise SystemExit("--overlay-dir must live under /tmp (overlays are never committed)")
    out_dir.mkdir(parents=True, exist_ok=True)
    by_page = {(r["book"], r["page"]): r for r in summary["results"]}

    # Prefer pages with curved COO polygons: they are the ones this work targets.
    gt_pages = load_gt(Path(args.det_labels).resolve(), args.split)
    curved = []
    for (book, page), gts in sorted(gt_pages.items()):
        if book not in summary["books"]:
            continue
        if any(g["class"] == "sfx" and g["source"] == "coo" and g["npoints"] > 4 for g in gts):
            curved.append((book, page))
    picked = curved[: args.overlay_pages]

    n = 0
    for book, page in picked:
        r = by_page.get((book, page))
        if r is None:
            continue
        img = cv2.imread(str(image_path(Path(args.det_labels), gt_pages[(book, page)][0]["image"])))
        if img is None:
            continue
        vis = img.copy()
        for pts, cls in zip(r["gt_polys"], r["gt_classes"]):
            poly = np.asarray(pts, np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [poly], True, (255, 160, 0) if cls == "dialogue" else (0, 220, 0), 3)
        for quad, score in zip(r["quads"], r["scores"]):
            cv2.polylines(vis, [np.asarray(quad, np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 2)
            cv2.putText(vis, f"{score:.2f}", (int(quad[0][0]), int(quad[0][1]) - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, f"{book} p{page}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, f"{book} p{page}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"{book}_{page:03d}.jpg"), vis)
        n += 1
    print(f"wrote {n} overlays to {out_dir}")
    return n


# ---------------------------------------------------------------- self-test
def self_test() -> int:
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(("PASS  " if cond else "FAIL  ") + name + (f"  [{detail}]" if detail and not cond else ""))
        ok = ok and bool(cond)

    # polygon (not box) IoU: a half-disc curving inside a square has IoU ~ pi/8*... 
    square = to_poly([(0, 0), (100, 0), (100, 100), (0, 100)])
    inner = to_poly([(25, 25), (75, 25), (75, 75), (25, 75)])
    check("polygon IoU of a half-area box inside = 0.25", abs(iou(square, inner) - 0.25) < 1e-9, f"{iou(square, inner)}")
    same = to_poly([(100, 0), (100, 100), (0, 100), (0, 0)])
    check("polygon IoU ignores vertex order = 1.0", abs(iou(square, same) - 1.0) < 1e-9)
    tri = to_poly([(0, 0), (100, 0), (100, 100)])
    check("polygon IoU of a triangle = 0.5 of the square", abs(iou(square, tri) - 0.5) < 1e-9, f"{iou(square, tri)}")
    check("touching boxes have IoU 0", iou(to_poly([(0, 0), (10, 0), (10, 10), (0, 10)]),
                                          to_poly([(10, 0), (20, 0), (20, 10), (10, 10)])) == 0.0)

    # matching + per-class assignment
    g_sfx = {"class": "sfx", "source": "coo", "npoints": 9, "poly": to_poly([(0, 0), (100, 0), (100, 50), (0, 50)])}
    g_dia = {"class": "dialogue", "source": "mangaocr", "npoints": 4, "poly": to_poly([(0, 100), (100, 100), (100, 200), (0, 200)])}
    g_exc = {"class": "sfx", "source": "mangaocr", "npoints": 4, "poly": to_poly([(200, 0), (300, 0), (300, 50), (200, 50)])}
    gt = [g_sfx, g_dia]
    preds = [
        ([(0, 0), (100, 0), (100, 50), (0, 50)], 0.9),        # exact SFX
        ([(0, 100), (100, 100), (100, 200), (0, 200)], 0.9),  # exact dialogue
        ([(205, 0), (295, 0), (295, 50), (205, 50)], 0.8),    # MangaOCR-only SFX (excluded GT)
        ([(0, 0), (100, 0), (100, 50), (0, 50)], 0.7),        # duplicate SFX
        ([(0, 300), (40, 300), (40, 340), (0, 340)], 0.6),    # art
    ]
    blocks = [to_poly([(400, 0), (500, 0), (500, 50), (400, 50)]),
              to_poly([(0, 500), (100, 500), (100, 560), (0, 560)])]
    page = score_page(preds, gt, blocks, 1000.0 * 1000.0, excluded=[g_exc["poly"]])
    check("SFX TP = 1 of 1 scored SFX GT", page["tp"]["sfx"] == 1 and page["gt"]["sfx"] == 1,
          f"{dict(page['gt'])} {dict(page['tp'])}")
    check("dialogue TP = 1", page["tp"]["dialogue"] == 1)
    check("MangaOCR-only SFX is attributed to sfx but not TP",
          page["attributed"]["sfx"] == 2 and page["tp"]["sfx"] == 1)
    check("duplicate prediction counted as duplicate FP", page["fp"]["duplicate"] == 1, str(dict(page["fp"])))
    check("MangaOCR-only SFX detection counted as excluded_sfx FP", page["fp"]["excluded_sfx"] == 1, str(dict(page["fp"])))
    check("blank-art prediction counted as art FP", page["fp"]["art"] == 1, str(dict(page["fp"])))
    check("art FP area = 1600 px^2", abs(page["fp_art_area"] - 1600.0) < 1e-6, str(page["fp_art_area"]))
    check("SFX crop recall by point bucket keys", page["crop_gt_by_points"]["9+"] == 1, str(dict(page["crop_gt_by_points"])))
    check("SFX crop TP uses the looser 0.4 threshold", page["crop"]["tp"] == 1 and page["crop_tp_by_points"]["9+"] == 1)

    # sub-threshold overlap is a localization FP, not art / not a match
    gt2 = [{"class": "sfx", "source": "coo", "npoints": 4, "poly": to_poly([(0, 0), (100, 0), (100, 100), (0, 100)])}]
    page2 = score_page([([(40, 0), (140, 0), (140, 100), (40, 100)], 0.9)], gt2, [], 1e6)
    check("IoU 0.4286 is not a 0.5 match", page2["tp"]["sfx"] == 0 and abs(0.6 / 1.4 - 0.42857) < 1e-4)
    check("IoU 0.4286 is a localization FP", page2["fp"]["localized_sfx"] == 1, str(dict(page2["fp"])))
    page2b = score_page([([(40, 0), (140, 0), (140, 100), (40, 100)], 0.9)], gt2, [], 1e6,
                        iou_match=0.4, iou_crop=0.4)
    check("same prediction matches at threshold 0.4", page2b["tp"]["sfx"] == 1)

    # dialogue GT is block level: two line boxes inside one bubble cover it but
    # neither reaches IoU 0.5 with the block.
    bubble = {"class": "dialogue", "source": "mangaocr", "npoints": 4,
              "poly": to_poly([(0, 0), (100, 0), (100, 100), (0, 100)])}
    lines = [([(0, 0), (100, 0), (100, 45), (0, 45)], 0.9),
             ([(0, 45), (100, 45), (100, 90), (0, 90)], 0.9)]
    page5 = score_page(lines, [bubble], [], 1e6)
    check("line boxes do not match a dialogue block at IoU 0.5", page5["tp"]["dialogue"] == 0)
    check("line boxes both count as being on dialogue", page5["preds_on_dialogue"] == 2, str(page5["preds_on_dialogue"]))
    check("block coverage is 0.9 and the block counts as covered",
          abs(page5["dialogue_coverage_sum"] - 0.9) < 1e-9 and page5["dialogue_blocks_covered"] == 1,
          f"{page5['dialogue_coverage_sum']} {page5['dialogue_blocks_covered']}")
    page5b = score_page(lines, [bubble], [], 1e6)
    check("line boxes are localized_dialogue FPs, not art",
          page5b["fp"]["localized_dialogue"] == 2 and page5b["fp"]["art"] == 0, str(dict(page5b["fp"])))

    # unannotated Manga109 block: real text, not art
    block = to_poly([(0, 0), (100, 0), (100, 100), (0, 100)])
    page3 = score_page([([(0, 0), (100, 0), (100, 100), (0, 100)], 0.9)], [], [block], 1e6)
    check("detection on an uncovered text block is unlabeled_text, not art",
          page3["fp"]["unlabeled_text"] == 1 and page3["fp"]["art"] == 0, str(dict(page3["fp"])))
    page4 = score_page([([(0, 0), (100, 0), (100, 100), (0, 100)], 0.9)],
                       [{"class": "dialogue", "source": "mangaocr", "npoints": 4,
                         "poly": to_poly([(0, 0), (100, 0), (100, 100), (0, 100)])}],
                       [block], 1e6)
    check("a block covered by a label is not an unlabeled_text FP", page4["fp"]["unlabeled_text"] == 0)

    # metrics
    s = summarise([page, page2, page3])
    check("micro SFX recall = tp/gt", abs(s["classes"]["sfx"]["recall"] - 1 / 2) < 1e-9, str(s["classes"]["sfx"]))
    check("micro dialogue precision = tp/attributed",
          abs(s["classes"]["dialogue"]["precision"] - 1.0) < 1e-9, str(s["classes"]["dialogue"]))
    check("FP per page and per GT are normalised",
          abs(s["false_positives"]["per_page"] - s["false_positives"]["total"] / 3) < 1e-9)
    check("hmean of P=0.5 R=0.5 is 0.5", prf(1, 2, 2)["hmean"] == 0.5)

    # GT loading contract
    gt_dir = ROOT / "data/det"
    if (gt_dir / "test.jsonl").exists():
        test_gt = load_gt(gt_dir, "test")
        check("test sidecar has 9 books", len({b for b, _ in test_gt}) == 9, str(len({b for b, _ in test_gt})))
        check("test sidecar pages > 900", len(test_gt) > 900, str(len(test_gt)))
        check("every test book is a test book", all(book_split(b) == "test" for b, _ in test_gt))
    else:
        print("SKIP  sidecar assertions (data/det/test.jsonl missing)")

    print(f"\n{0 if ok else 1} failure(s)")
    return 0 if ok else 1


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, help="detector ONNX")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--det-labels", type=Path, default=ROOT / "data/det")
    ap.add_argument("--manga109", type=Path, default=ROOT / "data/raw/Manga109-s",
                    help="Manga109-s root (for Manga109 <text> blocks in FP attribution)")
    ap.add_argument("--out", type=Path, help="JSON output (keep it under output/ or /tmp)")
    ap.add_argument("--book", action="append", help="restrict to these books (repeatable)")
    ap.add_argument("--limit-pages", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--thresh", type=float, default=APP_DET["thresh"])
    ap.add_argument("--box-thresh", type=float, default=APP_DET["box_thresh"])
    ap.add_argument("--unclip-ratio", type=float, default=APP_DET["unclip_ratio"])
    ap.add_argument("--limit-side-len", type=int, default=APP_DET["limit_side_len"])
    ap.add_argument("--iou", type=float, default=IOU_MATCH)
    ap.add_argument("--crop-iou", type=float, default=IOU_CROP)
    ap.add_argument("--overlay-pages", type=int, default=0)
    ap.add_argument("--overlay-dir", type=Path, default=Path("/tmp/opencode/det_overlays"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.model:
        ap.error("--model is required (or use --self-test)")

    summary = run(args)

    c = summary["classes"]
    print(f"\nmodel {args.model}  split {args.split}  books {len(summary['books'])}  pages {summary['pages']}")
    for cls in ("sfx", "dialogue", "all"):
        m = c[cls]
        print(f"  {cls:8s} P {100*m['precision']:.2f} %  R {100*m['recall']:.2f} %  "
              f"H {100*m['hmean']:.2f} %  (tp {m['tp']} / pred {m['pred']} / gt {m['gt']})  "
              f"P_incl_bg {100*m['precision_incl_background']:.2f} %")
    cr = summary["sfx_crop_recall"]
    print(f"  SFX crop recall IoU>{cr['iou']}: {100*cr['recall']:.2f} % ({cr['tp']}/{cr['gt']})  "
          + "  ".join(f"{b} {100*v['recall']:.1f} %({v['tp']}/{v['gt']})" for b, v in cr["by_points"].items()))
    fp = summary["false_positives"]
    dc = summary["dialogue_block_coverage"]
    print(f"  dialogue blocks covered >=50 %: {100*dc['recall']:.2f} % ({dc['blocks_covered']}/{dc['blocks']})  "
          f"mean coverage {100*dc['mean_coverage']:.2f} %  predictions on dialogue {100*dc['preds_share']:.2f} %")
    print(f"  FPs {fp['total']} ({fp['per_page']:.2f}/page, {fp['per_gt']:.3f}/GT)  "
          f"art {fp['by_kind'].get('art', 0)} ({fp['art_per_page']:.2f}/page)  "
          f"art area {100*fp['art_area_fraction']:.4f} % of pages")
    print(f"  FP kinds {fp['by_kind']}")
    print("  macro " + "  ".join(f"{k} {100*v:.2f} %" for k, v in summary["mean"].items()
                                 if k.endswith(("_hmean", "_recall")) and k != "sfx_crop_recall"))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        slim = {k: v for k, v in summary.items() if k != "results"}
        args.out.write_text(json.dumps(slim, indent=2, ensure_ascii=False))
        print(f"wrote {args.out}")
    if args.overlay_pages:
        write_overlays(args, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
