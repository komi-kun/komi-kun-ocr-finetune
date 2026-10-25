#!/usr/bin/env python3
"""Dialogue-preserving detector training set (experiment 2).

Experiment 1 (``data/det``) trains SFX positives only and marks every MangaOCR
dialogue region as ``###`` ignore.  That removes the loss there but leaves nothing
anchoring the pretrained detector's *line* response: 5 epochs are enough to lose
dialogue detection (see RESULTS.md).  This builder keeps experiment 1 intact and
writes a second, separate dataset that adds **line-level dialogue supervision
distilled from the stock detector**:

  * **SFX positives** are copied unchanged from ``data/det/train.jsonl`` (COO
    arbitrary polygons 4–66 points + MangaOCR SFX boxes).  They are the only
    independent SFX ground truth here.
  * **Dialogue positives** are the *stock detector's own line quads*
    (``models/ppocrv6_small_det.onnx`` replayed through ``scripts/db_det.py``,
    the runtime postprocess), kept when >= ``--containment`` (default 0.8) of a
    quad's area lies inside a MangaOCR dialogue region, clipped to that region and
    to (region − SFX), and emitted as polygons with ``teacher=stock_det`` in the
    sidecar.  These are **teacher labels, not dialogue ground truth**: they encode
    "keep doing what the shipped detector does", not "this is where the lines
    are".  Manga109 / MangaOCR bubble boxes are never used as positive lines.
  * **Ignore regions** are the parts of a dialogue region that no kept line
    covers (and that are not SFX), plus regions where the stock detector fired
    nothing.  Ignore fragments are hole-free and subtracted around every
    positive, so no ignore can remove supervision.

Why distillation: the acceptance gate needs SFX detection to improve *without*
losing dialogue.  MangaOCR has no line-level dialogue labels (its boxes are
Manga109 ``<text>`` blocks), so the stock detector is the only available line-level
teacher.  Consequences are stated in RESULTS.md: dialogue quality is capped by the
stock detector, and any error the stock detector makes inside a bubble becomes a
training target.

Outputs (``--out``, default ``data/det_pseudo``; both ignored by git):

    train.txt        Paddle SimpleDataSet lines (relative image paths)
    train.jsonl      per-instance sidecar with provenance + evidence
    stock_det.jsonl  cached stock-detector quads per page (reused on re-runs)
    stats.json       counts, coverage, clip ratios
    validation_report.json

val/test are **not** written here: training keeps using ``data/det/val.txt`` and
evaluation uses ``data/det/{val,test}.txt`` with their unchanged, independent
ground truth.

Usage::

    python scripts/build_det_pseudo_dataset.py                 # detect + build
    python scripts/build_det_pseudo_dataset.py --reuse-cache   # rebuild from cache
    python scripts/build_det_pseudo_dataset.py --self-test
    python scripts/build_det_pseudo_dataset.py --validate-only
    python scripts/build_det_pseudo_dataset.py --overlay 6 --overlay-out /tmp/opencode/det_pseudo_overlays
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from shapely.geometry import Polygon, box as shapely_box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_det_dataset import book_split, hole_free_parts  # noqa: E402
from db_det import DBDetector  # noqa: E402

MIN_AREA = 4.0
MIN_EXTENT = 2.0
MIN_IGNORE_AREA = 16.0
MIN_IGNORE_EXTENT = 4.0
SFX_OVERLAP_TOLERANCE = 0.01
MAX_POINTS = 64
CONTAINMENT = 0.8
MIN_CLIP_RATIO = 0.5


# ---------------------------------------------------------------- geometry helpers
def to_poly(points) -> Polygon | None:
    if len(points) < 3:
        return None
    poly = Polygon([(float(x), float(y)) for x, y in points])
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.geom_type == "MultiPolygon":
        poly = largest_polygon(poly)
    if poly is None or poly.is_empty or poly.area <= 0:
        return None
    return poly


def ring_points(poly: Polygon) -> list[list[float]]:
    # 3 decimals: 1 dp rounding of a small fragment is enough to make an ignore
    # overlap a positive by more than the 1 % tolerance.
    return [[round(float(x), 3), round(float(y), 3)] for x, y in poly.exterior.coords[:-1]]


def cap_points(poly: Polygon, limit: int = MAX_POINTS) -> Polygon | None:
    """Simplify until the exterior ring fits PaddleOCR's padded point budget."""
    out = poly
    for tol in (0.5, 1.0, 2.0, 4.0, 8.0):
        if len(out.exterior.coords) - 1 <= limit:
            return out
        out = poly.simplify(tol, preserve_topology=True)
        if out.is_empty or out.geom_type != "Polygon":
            return None
    return out if len(out.exterior.coords) - 1 <= limit else None


def largest_polygon(geom) -> Polygon | None:
    if geom is None or geom.is_empty:
        return None
    parts = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
    polys = [p for p in parts if p.geom_type == "Polygon" and p.area > 0]
    return max(polys, key=lambda p: p.area) if polys else None


def degenerate(poly: Polygon, min_area: float, min_extent: float) -> bool:
    if poly.is_empty or poly.area < min_area:
        return True
    x0, y0, x1, y1 = poly.bounds
    return (x1 - x0) < min_extent or (y1 - y0) < min_extent


def clip_quad(quad: Polygon, region: Polygon, sfx: Polygon | None, cfg) -> tuple[Polygon | None, dict]:
    """Clip a stock-detection quad to a dialogue region minus SFX."""
    geom = quad.intersection(region)
    if sfx is not None and not sfx.is_empty:
        geom = geom.difference(sfx)
    poly = largest_polygon(geom)
    evidence = {"clip_ratio": 0.0, "points": 0}
    if poly is None:
        return None, evidence
    evidence["clip_ratio"] = poly.area / quad.area if quad.area else 0.0
    if evidence["clip_ratio"] < cfg.min_clip_ratio:
        return None, evidence
    poly = cap_points(poly)
    if poly is None or degenerate(poly, cfg.min_area, cfg.min_extent):
        return None, evidence
    if not Polygon(ring_points(poly)).is_valid:
        return None, evidence
    evidence["points"] = len(poly.exterior.coords) - 1
    return poly, evidence


# ---------------------------------------------------------------- page build (pure)
def build_page(quads, positives, regions, cfg) -> dict:
    """Return {"instances": [...], "counters": Counter} for one page.

    quads:    [(4x2 array-like, score)]  stock-detector output for the page
    positives: [{"class","source","points"}]  experiment-1 positives (SFX only)
    regions:  [[points]]  MangaOCR dialogue regions (experiment-1 ignore polygons)
    """
    counters = Counter()
    quad_polys = []
    for quad, score in quads:
        poly = to_poly(quad)
        if poly is None or degenerate(poly, cfg.min_area, cfg.min_extent):
            counters["quads_dropped_degenerate"] += 1
            continue
        quad_polys.append((poly, float(score)))

    sfx_polys = []
    for p in positives:
        poly = to_poly(p["points"])
        if poly is not None:
            sfx_polys.append(poly)
    sfx_union = unary_union(sfx_polys) if sfx_polys else None

    region_polys = []
    for pts in regions:
        poly = to_poly(pts)
        if poly is not None and not poly.is_empty:
            region_polys.append(poly)

    dialogue = []
    for poly, score in quad_polys:
        best_region, best_containment = None, 0.0
        for region in region_polys:
            if not poly.intersects(region):
                continue
            containment = poly.intersection(region).area / poly.area
            if containment > best_containment:
                best_region, best_containment = region, containment
        if best_region is None or best_containment < cfg.containment:
            counters["quads_outside_dialogue"] += 1
            continue
        clipped, evidence = clip_quad(poly, best_region, sfx_union, cfg)
        if clipped is None:
            counters["quads_clipped_away"] += 1
            continue
        dialogue.append(
            {
                "class": "dialogue",
                "source": "stock_det",
                "role": "positive",
                "teacher": "stock_det",
                "points": ring_points(clipped),
                "evidence": {
                    "containment": round(best_containment, 4),
                    "clip_ratio": round(evidence["clip_ratio"], 4),
                    "det_score": round(score, 4),
                    **evidence,
                },
            }
        )
        counters["dialogue_lines"] += 1

    line_union = unary_union([to_poly(d["points"]) for d in dialogue]) if dialogue else None
    pieces = [g for g in (line_union, sfx_union) if g is not None]
    covered = unary_union(pieces) if pieces else None

    ignores = []
    for region in region_polys:
        leftover = region if covered is None else region.difference(covered)
        for part in hole_free_parts(leftover):
            part = largest_polygon(part.buffer(0))
            if part is None or part.interiors:
                counters["ignore_fragments_dropped"] += 1
                continue
            if degenerate(part, cfg.min_ignore_area, cfg.min_ignore_extent):
                counters["ignore_fragments_dropped"] += 1
                continue
            if len(part.exterior.coords) - 1 < 4:
                counters["ignore_fragments_dropped"] += 1
                continue
            ring = Polygon(ring_points(part))
            if not ring.is_valid or ring.interiors:
                # a self-touching ring would be filled as one blob by cv2.fillPoly
                counters["ignore_fragments_dropped"] += 1
                continue
            ignores.append(
                {
                    "class": "dialogue",
                    "source": "mangaocr_region",
                    "role": "ignore",
                    "teacher": "stock_det",
                    "points": ring_points(part),
                    "evidence": {"reason": "dialogue_area_without_stock_line"},
                }
            )
            counters["ignore_fragments"] += 1
        counters["dialogue_regions"] += 1

    # ignore labels first: MakeShrinkMap zeroes the mask for them, so a positive
    # written before an ignore could be erased by an ignore that touches it.
    regions_union = unary_union(region_polys) if region_polys else None
    region_area = regions_union.area if regions_union is not None else 0.0
    covered_in_regions = (
        covered.intersection(regions_union).area
        if (covered is not None and regions_union is not None) else 0.0
    )
    ignore_area = sum(to_poly(i["points"]).area for i in ignores)
    counters["region_area"] = round(region_area, 1)
    counters["covered_in_regions_area"] = round(covered_in_regions, 1)
    counters["ignore_area"] = round(ignore_area, 1)
    # hole_free_parts cuts a band through every hole and drops the band, so a thin
    # strip beside each teacher line becomes *background* instead of ignore.  This
    # is deliberate (re-covering a hole would erase the positive) and reported.
    counters["background_area_in_regions"] = round(region_area - covered_in_regions - ignore_area, 1)

    instances = ignores + [
        {
            "class": p["class"],
            "source": p["source"],
            "role": "positive",
            "teacher": p.get("teacher", "none"),
            "points": p["points"],
            "evidence": p.get("evidence", {}),
        }
        for p in positives
    ] + dialogue
    for inst in instances:
        counters[f"emitted_{inst['role']}"] += 1
    return {"instances": instances, "counters": counters}


# ---------------------------------------------------------------- dataset I/O
def load_source(det_dir: Path, split: str):
    """Positives + dialogue regions per (book, page) from the experiment-1 sidecar."""
    positives: dict[tuple[str, int], list[dict]] = defaultdict(list)
    regions: dict[tuple[str, int], list[list[list[float]]]] = defaultdict(list)
    images: dict[tuple[str, int], str] = {}
    with open(det_dir / f"{split}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            key = (rec["book"], int(rec["page"]))
            images.setdefault(key, rec["image"])
            if rec["role"] == "positive":
                positives[key].append(
                    {"class": rec["class"], "source": rec["source"], "points": rec["points"]}
                )
            elif rec["role"] == "ignore" and rec["class"] == "dialogue":
                regions[key].append(rec["points"])
    return positives, regions, images


def load_transcriptions(det_dir: Path, split: str) -> dict[str, list[tuple[str, list]]]:
    """image -> [(transcription, points)] so SFX labels keep their real text."""
    out: dict[str, list[tuple[str, list]]] = defaultdict(list)
    with open(det_dir / f"{split}.txt", encoding="utf-8") as fh:
        for line in fh:
            image, payload = line.rstrip("\n").split("\t", 1)
            for item in json.loads(payload):
                out[image].append((item["transcription"], tuple(map(tuple, item["points"]))))
    return out


def text_for(image: str, points, texts) -> str:
    key = tuple(map(tuple, [[round(x, 1), round(y, 1)] for x, y in points]))
    for text, pts in texts.get(image, []):
        if tuple(map(tuple, [[round(x, 1), round(y, 1)] for x, y in pts])) == key:
            return text
    return "s"  # unused by the DB loss; must not be "###" or "*"


# ---------------------------------------------------------------- stock det replay
_W: dict = {}


def _init_worker(onnx: str, det_kwargs: dict) -> None:
    _W["det"] = DBDetector(onnx, **det_kwargs)


def _detect_one(task) -> dict:
    book, page, image = task
    img = cv2.imread(image)
    if img is None:
        return {"book": book, "page": page, "error": "unreadable"}
    preds = _W["det"](img)
    return {
        "book": book,
        "page": page,
        "quads": [np.asarray(q, float).round(2).tolist() for q, _ in preds],
        "scores": [round(float(s), 4) for _, s in preds],
    }


def replay(args, keys, images) -> dict[tuple[str, int], dict]:
    cache_path = Path(args.out) / "stock_det.jsonl"
    cache: dict[tuple[str, int], dict] = {}
    if cache_path.exists() and args.reuse_cache:
        for line in cache_path.open(encoding="utf-8"):
            rec = json.loads(line)
            cache[(rec["book"], int(rec["page"]))] = rec
        print(f"stock det cache: {len(cache)} pages from {cache_path}")
    todo = [k for k in keys if k not in cache]
    if todo:
        tasks = [(b, p, str(ROOT / images[(b, p)])) for b, p in todo]
        det_kwargs = dict(
            thresh=args.thresh, box_thresh=args.box_thresh,
            unclip_ratio=args.unclip_ratio, limit_side_len=args.limit_side_len,
        )
        print(f"stock det replay: {len(tasks)} pages with {args.workers} workers")
        with mp.Pool(args.workers, initializer=_init_worker,
                     initargs=(str(args.stock_det), det_kwargs)) as pool:
            for i, rec in enumerate(pool.imap_unordered(_detect_one, tasks, chunksize=8), 1):
                if "error" in rec:
                    print(f"  unreadable: {rec['book']}/{rec['page']}")
                    continue
                cache[(rec["book"], int(rec["page"]))] = rec
                if i % 500 == 0:
                    print(f"  {i}/{len(tasks)}", flush=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as fh:
            for key in sorted(cache):
                fh.write(json.dumps(cache[key], ensure_ascii=False) + "\n")
    return cache


# ---------------------------------------------------------------- build / validate
def build(args) -> int:
    det_dir = Path(args.det_labels).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    positives, regions, images = load_source(det_dir, "train")
    texts = load_transcriptions(det_dir, "train")

    books = sorted({b for b, _ in images})
    stray = [b for b in books if book_split(b) != "train"]
    assert not stray, f"train sidecar contains non-train books: {stray}"
    keys = sorted(images)
    if args.limit_pages:
        keys = keys[: args.limit_pages]

    det = replay(args, keys, images)

    cfg = argparse.Namespace(
        containment=args.containment,
        min_clip_ratio=args.min_clip_ratio,
        min_area=args.min_area,
        min_extent=args.min_extent,
        min_ignore_area=args.min_ignore_area,
        min_ignore_extent=args.min_ignore_extent,
    )
    stats = Counter()
    lines_out, sidecar_out = [], []
    pages_with_dialogue = 0
    for key in keys:
        book, page = key
        rec = det.get(key, {"quads": [], "scores": []})
        quads = list(zip(rec["quads"], rec["scores"]))
        result = build_page(quads, positives.get(key, []), regions.get(key, []), cfg)
        stats.update(result["counters"])
        insts = result["instances"]
        if any(i["class"] == "dialogue" and i["role"] == "positive" for i in insts):
            pages_with_dialogue += 1
        if not insts:
            stats["pages_empty"] += 1
            continue
        payload = []
        for inst in insts:
            if inst["role"] == "ignore":
                text = "###"
            elif inst["class"] == "dialogue":
                text = "d"  # teacher line; DB ignores the transcription
            else:
                text = text_for(images[key], inst["points"], texts)
            payload.append({"transcription": text, "points": inst["points"]})
        lines_out.append(f"{images[key]}\t{json.dumps(payload, ensure_ascii=False)}")
        for inst in insts:
            sidecar_out.append(
                {
                    "book": book,
                    "page": page,
                    "image": images[key],
                    "class": inst["class"],
                    "source": inst["source"],
                    "role": inst["role"],
                    "teacher": inst["teacher"],
                    "points": inst["points"],
                    "evidence": inst["evidence"],
                }
            )
        stats["pages_with_labels"] += 1

    (out_dir / "train.txt").write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    with (out_dir / "train.jsonl").open("w", encoding="utf-8") as fh:
        for rec in sidecar_out:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    dialogue_total = sum(1 for r in sidecar_out if r["class"] == "dialogue" and r["role"] == "positive")
    summit = {
        "generator": "scripts/build_det_pseudo_dataset.py",
        "role": "experiment 2 - dialogue-preserving distillation from the stock detector",
        "source_dataset": str(det_dir / "train.txt"),
        "stock_det": str(args.stock_det),
        "stock_det_sha256": _sha256(Path(args.stock_det)),
        "containment": args.containment,
        "min_clip_ratio": args.min_clip_ratio,
        "config": {
            "min_area": args.min_area,
            "min_extent": args.min_extent,
            "min_ignore_area": args.min_ignore_area,
            "min_ignore_extent": args.min_ignore_extent,
            "max_points": MAX_POINTS,
        },
        "books": {"total": len(books), "names": books},
        "pages": {
            "with_image": len(keys),
            "with_labels": stats["pages_with_labels"],
            "empty": stats["pages_empty"],
            "with_dialogue_lines": pages_with_dialogue,
        },
        "instances": {
            "positive_sfx": sum(1 for r in sidecar_out if r["role"] == "positive" and r["class"] == "sfx"),
            "positive_dialogue": dialogue_total,
            "ignore": sum(1 for r in sidecar_out if r["role"] == "ignore"),
        },
        "provenance": {
            "sfx": "COO polygons + MangaOCR SFX boxes, copied unchanged from data/det/train.jsonl",
            "dialogue": "stock detector line quads, clipped to a MangaOCR dialogue region",
            "ignore": "dialogue region area not covered by a kept stock line",
            "teacher": "models/ppocrv6_small_det.onnx via scripts/db_det.py (runtime postprocess)",
            "warning": "teacher labels distil the shipped detector; they are not independent dialogue GT",
        },
        "counters": dict(stats),
    }
    (out_dir / "stats.json").write_text(json.dumps(summit, indent=2, ensure_ascii=False))
    print(json.dumps({k: summit[k] for k in ("pages", "instances")}, indent=2))
    print(f"wrote {out_dir}/train.txt ({len(lines_out)} pages), train.jsonl ({len(sidecar_out)} instances)")
    return 0


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(args) -> int:
    out_dir = Path(args.out).resolve()
    det_dir = Path(args.det_labels).resolve()
    sidecar = [json.loads(l) for l in (out_dir / "train.jsonl").open(encoding="utf-8")]
    exp1 = {
        (r["book"], r["page"], r["role"], tuple(map(tuple, r["points"])))
        for r in (json.loads(l) for l in (det_dir / "train.jsonl").open(encoding="utf-8"))
        if r["role"] == "positive"
    }
    cache = {}
    with (out_dir / "stock_det.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            cache[(rec["book"], int(rec["page"]))] = rec

    checks = Counter()
    by_page: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in sidecar:
        by_page[(r["book"], r["page"])].append(r)
    for (book, page), insts in by_page.items():
        image = ROOT / insts[0]["image"]
        if not image.exists():
            checks["missing_images"] += 1
            continue
        h, w = _image_size(image)
        roles = [i["role"] for i in insts]
        pos_idx = [i for i, r in enumerate(roles) if r == "positive"]
        ign_idx = [i for i, r in enumerate(roles) if r == "ignore"]
        if pos_idx and ign_idx and min(ign_idx) > max(pos_idx):
            checks["ignore_after_positive"] += 1
        polys = {id(i): to_poly(i["points"]) for i in insts}
        for inst in insts:
            poly = polys[id(inst)]
            if poly is None:
                checks["degenerate"] += 1
                continue
            if any(x < 0 or y < 0 or x > w - 1 or y > h - 1 for x, y in inst["points"]):
                checks["out_of_bounds"] += 1
            if inst["role"] == "ignore" and poly.interiors:
                checks["ignore_with_hole"] += 1
            if inst["class"] == "dialogue" and inst["role"] == "positive":
                if inst["teacher"] != "stock_det":
                    checks["dialogue_without_teacher"] += 1
                rec = cache.get((book, page))
                if rec is None:
                    checks["dialogue_missing_cache"] += 1
                    continue
                ok = False
                for quad in rec["quads"]:
                    q = to_poly(quad)
                    if q is None:
                        continue
                    # points are rounded to 0.1 px, so allow 2 % slack on small crops
                    if poly.intersection(q).area / poly.area >= 0.98 and q.area > 0:
                        # the emitted polygon is inside one of the teacher quads
                        ok = True
                        break
                if not ok:
                    checks["dialogue_not_in_teacher"] += 1
            if inst["role"] == "positive" and inst["class"] == "sfx":
                key = (book, page, "positive", tuple(map(tuple, inst["points"])))
                if key not in exp1:
                    checks["sfx_differs_from_experiment1"] += 1
        # ignore/positive overlap
        positives = [polys[id(i)] for i in insts if i["role"] == "positive"]
        for inst in insts:
            if inst["role"] != "ignore":
                continue
            poly = polys[id(inst)]
            if poly is None or not positives:
                continue
            overlap = unary_union([poly.intersection(p) for p in positives if poly.intersects(p)]).area
            if overlap / max(poly.area, 1e-9) > SFX_OVERLAP_TOLERANCE:
                checks["ignore_covers_positive"] += 1

    # teacher quads themselves must not be MangaOCR/Manga109 block geometry
    dialogue_shapes = Counter()
    for r in sidecar:
        if r["class"] == "dialogue" and r["role"] == "positive":
            pts = r["points"]
            dialogue_shapes[len(pts)] += 1
            if len(pts) < 4:
                checks["dialogue_degenerate_points"] += 1

    report = {
        "sidecar_instances": len(sidecar),
        "pages": len(by_page),
        "dialogue_positive_point_counts": dict(sorted(dialogue_shapes.items())),
        "checks": dict(checks),
    }
    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2))
    return 0 if not checks else 1


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.height, im.width


def overlay(args) -> int:
    out = Path(args.overlay_out).resolve()
    if not str(out).startswith("/tmp/"):
        raise SystemExit("--overlay-out must live under /tmp")
    out.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out).resolve()
    sidecar = [json.loads(l) for l in (out_dir / "train.jsonl").open(encoding="utf-8")]
    cache = {}
    with (out_dir / "stock_det.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            cache[(rec["book"], int(rec["page"]))] = rec
    by_page: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in sidecar:
        by_page[(r["book"], r["page"])].append(r)

    picked = []
    for (book, page), insts in sorted(by_page.items()):
        curved = any(len(i["points"]) > 4 for i in insts if i["class"] == "sfx" and i["role"] == "positive")
        if curved and any(i["class"] == "dialogue" and i["role"] == "positive" for i in insts):
            picked.append((book, page))
        if len(picked) >= args.overlay:
            break
    n = 0
    for book, page in picked:
        insts = by_page[(book, page)]
        img = cv2.imread(str(ROOT / insts[0]["image"]))
        if img is None:
            continue
        vis = img.copy()
        for inst in insts:
            pts = np.asarray(inst["points"], np.int32).reshape(-1, 1, 2)
            if inst["role"] == "ignore":
                cv2.polylines(vis, [pts], True, (160, 160, 160), 2)
            elif inst["class"] == "sfx":
                cv2.polylines(vis, [pts], True, (0, 220, 0), 3)
            else:
                cv2.polylines(vis, [pts], True, (0, 0, 255), 2)
        rec = cache.get((book, page))
        for quad in (rec or {}).get("quads", []):
            cv2.polylines(vis, [np.asarray(quad, np.int32).reshape(-1, 1, 2)], True, (255, 0, 255), 1)
        cv2.putText(vis, f"{book} p{page}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, f"{book} p{page}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / f"{book}_{page:03d}.jpg"), vis)
        n += 1
    print(f"wrote {n} overlays to {out} (green=COO SFX, red=dialogue teacher line, grey=ignore, magenta=raw teacher quads)")
    return 0


# ---------------------------------------------------------------- self-test
def self_test() -> int:
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(("PASS  " if cond else "FAIL  ") + name + (f"  [{detail}]" if detail and not cond else ""))
        ok = ok and bool(cond)

    cfg = argparse.Namespace(
        containment=CONTAINMENT, min_clip_ratio=MIN_CLIP_RATIO, min_area=MIN_AREA,
        min_extent=MIN_EXTENT, min_ignore_area=MIN_IGNORE_AREA, min_ignore_extent=MIN_IGNORE_EXTENT,
    )
    region = [[(0, 0), (100, 0), (100, 100), (0, 100)]]
    sfx = [{"class": "sfx", "source": "coo", "points": [(0, 80), (20, 80), (20, 100), (0, 100)]}]
    quads = [
        ([(10, 10), (40, 10), (40, 30), (10, 30)], 0.9),    # line inside
        ([(50, 10), (90, 10), (90, 30), (50, 30)], 0.8),    # line inside
        ([(74, 60), (104, 60), (104, 80), (74, 80)], 0.7),  # 86.7 % inside -> keep, clipped
        ([(200, 10), (260, 10), (260, 30), (200, 30)], 0.6),  # outside -> dropped
        ([(2, 82), (18, 82), (18, 98), (2, 98)], 0.5),      # inside the SFX -> dropped
    ]
    page = build_page(quads, sfx, region, cfg)
    insts = page["instances"]
    positives = [i for i in insts if i["role"] == "positive"]
    dialogue = [i for i in positives if i["class"] == "dialogue"]
    ignores = [i for i in insts if i["role"] == "ignore"]

    check("3 dialogue teacher lines kept (2 inside + 1 clipped)", len(dialogue) == 3, str(len(dialogue)))
    check("quad outside the region is not a positive", page["counters"]["quads_outside_dialogue"] == 1,
          str(dict(page["counters"])))
    check("clipped line is narrower than its teacher quad",
          max(x for x, _ in dialogue[2]["points"]) - min(x for x, _ in dialogue[2]["points"]) == 26,
          str(dialogue[2]["points"]))
    check("quad on the SFX is clipped away", page["counters"]["quads_clipped_away"] == 1,
          str(dict(page["counters"])))
    check("clipped line stays inside the region",
          all(0 <= x <= 100 and 0 <= y <= 100 for i in dialogue for x, y in i["points"]))
    check("dialogue positives carry teacher provenance",
          all(i["teacher"] == "stock_det" and i["source"] == "stock_det" for i in dialogue))
    check("SFX positive preserved unchanged", any(i["class"] == "sfx" for i in positives))
    check("ignore labels come before positives", insts and insts[0]["role"] == "ignore", str([i["role"] for i in insts]))
    check("ignore regions are hole-free", all(len(to_poly(i["points"]).interiors) == 0 for i in ignores))
    # region 10,000 px^2; lines cover 30*20*2 + clipped 10*20 = 1400; SFX 400
    ignore_area = sum(to_poly(i["points"]).area for i in ignores)
    check("leftover dialogue area becomes ignore (hole-free cuts dropped)",
          abs(ignore_area - 7080.0) < 1e-6, str(ignore_area))
    check("dropped hole-free band is reported as background",
          abs(page["counters"]["background_area_in_regions"] - 600.0) < 1e-6,
          str(page["counters"]["background_area_in_regions"]))
    check("no ignore overlaps a positive",
          all(
              unary_union([g.intersection(p) for p in [to_poly(x["points"]) for x in positives]
                           if g.intersects(p)]).area / g.area <= SFX_OVERLAP_TOLERANCE
              for g in [to_poly(i["points"]) for i in ignores]
          ))

    # a region fully covered by a teacher line leaves no ignore
    full = build_page([([(0, 0), (100, 0), (100, 100), (0, 100)], 0.9)], [], region, cfg)
    check("fully covered region leaves no ignore",
          [i["role"] for i in full["instances"]] == ["positive"], str([i["role"] for i in full["instances"]]))

    # a low-containment quad is not assigned to a neighbouring bubble
    near = build_page([([(20, 20), (80, 20), (80, 40), (20, 40)], 0.9)], [], region, cfg)
    check("containment 1.0 inside is accepted", near["counters"]["dialogue_lines"] == 1, str(dict(near["counters"])))

    # cap_points keeps the ring within the budget
    many = Polygon([(x, 50 + 10 * np.sin(x / 3.0)) for x in range(0, 300)] + [(300, 100), (0, 100)])
    capped = cap_points(many)
    check("cap_points enforces <= 64 points", capped is not None and len(capped.exterior.coords) - 1 <= 64,
          str(None if capped is None else len(capped.exterior.coords) - 1))

    print(f"\n{0 if ok else 1} failure(s)")
    return 0 if ok else 1


# ---------------------------------------------------------------- CLI
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--det-labels", type=Path, default=ROOT / "data/det", help="experiment-1 dataset")
    ap.add_argument("--out", type=Path, default=ROOT / "data/det_pseudo")
    ap.add_argument("--stock-det", type=Path, default=ROOT / "models/ppocrv6_small_det.onnx")
    ap.add_argument("--containment", type=float, default=CONTAINMENT)
    ap.add_argument("--min-clip-ratio", type=float, default=MIN_CLIP_RATIO)
    ap.add_argument("--min-area", type=float, default=MIN_AREA)
    ap.add_argument("--min-extent", type=float, default=MIN_EXTENT)
    ap.add_argument("--min-ignore-area", type=float, default=MIN_IGNORE_AREA)
    ap.add_argument("--min-ignore-extent", type=float, default=MIN_IGNORE_EXTENT)
    ap.add_argument("--thresh", type=float, default=0.15)
    ap.add_argument("--box-thresh", type=float, default=0.25)
    ap.add_argument("--unclip-ratio", type=float, default=1.40)
    ap.add_argument("--limit-side-len", type=int, default=960)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reuse-cache", action="store_true", help="reuse stock_det.jsonl when present")
    ap.add_argument("--limit-pages", type=int, default=0)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--overlay", type=int, default=0)
    ap.add_argument("--overlay-out", type=Path, default=Path("/tmp/opencode/det_pseudo_overlays"))
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    if args.validate_only:
        return validate(args)
    if args.overlay:
        return overlay(args)
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
