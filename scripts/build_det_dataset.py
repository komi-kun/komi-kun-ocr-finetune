#!/usr/bin/env python3
"""Build PaddleOCR DB (SimpleDataSet) *detection* labels from Manga109-s pages.

Two complementary, page-aligned sources:

  * **COO** ``annotations_COO/<book>.xml`` ``<onomatopoeia>`` polygons -> ``class=sfx``.
    Every polygon is kept, whatever its point count (4..66 in the 2026 release);
    DB trains on polygon masks (``cv2.fillPoly``) and ``DetLabelEncode`` pads short
    polygons, so curved SFX are supervised as one region instead of being split.
  * **hal-utokyo/MangaOCR** ``bbox_2d`` instances (annotations only, no images) ->
    ``class=dialogue`` when the box is a Manga109 text *instance*, ``class=sfx``
    when it is not.  Aligned by book + page to the Manga109-s images.

**MangaOCR does not provide line-level dialogue.**  Its boxes are the Manga109
``<text>`` instances, i.e. whole bubbles / multi-column text regions (verified by
rendering them: 1:1 boxes hold 2-4 columns).  Training DB on them as positives
would teach it to merge lines, so:

  * **train**: every dialogue box is emitted as an ``###`` **ignore region**
    (``role=ignore``); only COO polygons and MangaOCR SFX boxes are positives.
  * **val / test**: dialogue stays a **positive** label so detection metrics can be
    reported per class - but those metrics are *block/bubble level*, not line level,
    and must be complemented by an end-to-end block CER (e.g. ``eval_blocks.py``).

Labelling rules (all of them are reported in ``stats.json``):

1. COO polygons keep their original geometry (never replaced by a box).  Polygons
   with < ``--min-points`` points, < ``--min-area`` px^2 or a bounding-box side
   < ``--min-extent`` px are dropped as degenerate; every point is clipped to the
   *image* bounds (``[0, w-1] x [0, h-1]``, the same clip PaddleOCR applies in
   ``MakeShrinkMap.validate_polygons``).
2. MangaOCR ``bbox_2d [x1,y1,x2,y2]`` becomes the four-point box
   ``[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]``.  Rows with an empty assistant message
   and pages without a Manga109-s image are skipped.
3. Class-aware de-duplication, in priority order, comparing a MangaOCR box with the
   COO polygons of the same page using exact geometry (shapely):
   a. *polygon-vs-box IoU* > ``--dedupe-iou`` (default 0.5) - the box **is** that
      SFX region, so the box is dropped and the COO polygon (exact curve) is kept;
   b. otherwise, if the box is a Manga109 text instance (see rule 4) it is
      dialogue and is kept - a bubble never loses its lines to a neighbouring SFX;
   c. otherwise, if the box covers more than ``--dedupe-containment`` (default
      0.5) of a COO polygon's area, it is that SFX again drawn as a wide
      axis-aligned box, and it is dropped in favour of the polygon.
   The comparison is SFX-vs-SFX: COO only holds onomatopoeia, and dialogue is
   never de-duplicated against it beyond rule 3a.
4. Class tags for the remaining MangaOCR boxes need Manga109 ``annotations/``
   *only as evidence* - never as label geometry.  A box is ``dialogue`` when it is
   the same instance as a Manga109 ``<text>`` box (IoU > 0.5) or when it absorbs two
   or more of them (MangaOCR merges neighbouring columns), otherwise ``sfx``.
   Manga109 block dialogue boxes are therefore never emitted as labels.
5. Label text is never used by the DB loss; it is carried in ``transcription``
   because PaddleOCR's loader requires the key (``*`` / ``###`` are rewritten so a
   real label can never become an ignore region).
6. Ignore-region construction (train only).  A dialogue box becomes an ignore
   region only after de-duplication, and it is trimmed so it cannot remove SFX
   supervision: when more than ``--ignore-sfx-tolerance`` (default 1 %) of the box
   overlaps the page's SFX ground truth (COO polygons + MangaOCR SFX boxes), the
   SFX geometry is subtracted, the remainder is split into hole-free polygons
   (``cv2.fillPoly`` only fills exterior rings) and every fragment below
   ``--min-ignore-area`` / ``--min-extent`` / ``--min-points`` is discarded; a box
   entirely swallowed by an SFX is dropped instead of being written over it.
   Ignore labels are emitted **before** positives inside the line, so
   ``MakeShrinkMap`` (which zeroes the mask for an ignore) cannot erase a positive
   that overlaps it by the tolerance.

Ignore regions: **only the block-level dialogue regions above**.  Nothing is
inferred from the stock detector, and no unlabelled Manga109 text block is turned
into an ignore region - text that neither COO nor MangaOCR annotated stays
negative supervision.  Consequence for train: dialogue areas are excluded from the
DB loss (no merging pressure and no pressure to suppress them either) while SFX and
background are supervised, so dialogue behaviour is inherited from the pretrained
detector and must be measured on val/test.

Outputs (``--out``, default ``data/det``; ``data/`` is git-ignored):

    {train,val,test}.txt      Paddle SimpleDataSet lines
                              "<image>\\t[{transcription, points}, ...]"
                              train mixes "###" ignore regions (block dialogue) and
                              SFX positives; val/test are all positive GT
    {train,val,test}.jsonl    one record per instance: book, page, class, source,
                              role, granularity, points, match evidence
    stats.json                book/page/instance counts, drops, de-dupe + alignment
    mangaocr_boxes.jsonl      cached MangaOCR parse (reused by --validate-only)
    validation_report.json    written by --validate-only

Splits come from ``build_real_dataset.book_split`` - the exact hash-based book
split the recogniser dataset uses (74/4/9 books over Manga109-s's 87 books;
``val`` = JijiBabaFight, MemorySeijin, MukoukizuNoChonbo, UltraEleven, ``test`` =
Arisa, GarakutayaManta, Hamlet, HanzaiKousyouninMinegishiEitarou, ParaisoRoad,
SaladDays_vol18, SeisinkiVulnus, TouyouKidan, YamatoNoHane).

Usage::

    python scripts/build_det_dataset.py                       # build
    python scripts/build_det_dataset.py --validate-only       # check the outputs
    python scripts/build_det_dataset.py --overlay 8 \\
        --overlay-out /tmp/opencode/det_overlays              # visual spot-check

Run training from the repository root: label image paths are repo-relative
(``--path-mode relative``, the default) and the training config uses
``data_dir: ./``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image
from shapely.geometry import Polygon, box as shapely_box
from shapely.ops import unary_union
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_MANGA109 = ROOT / "data/raw/Manga109-s"
DEFAULT_OUT = ROOT / "data/det"
MANGAOCR_DATASET = "hal-utokyo/MangaOCR"
IMAGE_SUFFIX = ".jpg"
SPLITS = ("train", "val", "test")
CLASSES = ("dialogue", "sfx")
SOURCES = ("coo", "mangaocr")
# DetLabelEncode turns these transcriptions into ignore regions.
IGNORE_TOKENS = ("*", "###")
IGNORE_TOKEN = "###"
# Counters reported per split (zero-filled, so the schema is stable).
COUNTER_KEYS = (
    "coo_polygons",
    "coo_clipped",
    "coo_dropped_few_points",
    "coo_dropped_degenerate",
    "coo_dropped_empty",
    "coo_invalid_repaired",
    "boxes",
    "boxes_clipped",
    "boxes_dropped_degenerate",
    "boxes_deduped_coo",
    "boxes_deduped_by_containment",
    "boxes_deduped_conflicting_block",
    "boxes_class_dialogue",
    "boxes_class_sfx",
    "dialogue_merged_blocks",
    "dialogue_dropped_merged",
    "pages_with_labels",
    "pages_empty",
)


# ----------------------------------------------------------------------------- utils
def book_split(book: str) -> str:
    """Book-level split, imported from build_real_dataset so det and rec agree.

    The import is lazy so that ``--help`` and cache-only modes do not pull in
    onnxruntime (``build_real_dataset`` imports ``common``, which loads it).
    """
    from build_real_dataset import book_split as _book_split

    return _book_split(book)


def num(value: float):
    """Round for compact, stable JSON: ints stay ints, the rest keep 2 decimals."""
    rounded = round(float(value), 2)
    return int(rounded) if rounded == int(rounded) else rounded


def clean_text(text: str | None) -> str:
    return " ".join((text or "").split())


def find_release_dir(root: Path) -> Path:
    """Directory holding ``images/`` and ``annotations_COO/``."""
    for cand in [root, *sorted(p.parent for p in root.rglob("annotations_COO"))]:
        if (cand / "annotations_COO").is_dir() and (cand / "images").is_dir():
            return cand
    raise SystemExit(f"could not find images/ + annotations_COO/ under {root}")


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:  # noqa: BLE001 - an unreadable page is reported, not fatal
        return None


def shoelace_area(points: Sequence[tuple[float, float]]) -> float:
    area = 0.0
    q = points[-1]
    for p in points:
        area += p[0] * q[1] - p[1] * q[0]
        q = p
    return abs(area) / 2.0


def bbox_of(points: Sequence[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def rect_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def contains_ratio(outer: Sequence[float], inner: Sequence[float]) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer``."""
    ix1, iy1 = max(outer[0], inner[0]), max(outer[1], inner[1])
    ix2, iy2 = min(outer[2], inner[2]), min(outer[3], inner[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return (ix2 - ix1) * (iy2 - iy1) / area if area > 0 else 0.0


def is_degenerate(points: Sequence[tuple[float, float]], min_area: float, min_extent: float) -> bool:
    x1, y1, x2, y2 = bbox_of(points)
    if (x2 - x1) < min_extent or (y2 - y1) < min_extent:
        return True
    if len(set(points)) < 3:
        return True
    return shoelace_area(points) < min_area


def clip_points(
    points: Iterable[tuple[float, float]], width: int, height: int
) -> tuple[list[tuple[float, float]], bool]:
    """Clip to the pixel range PaddleOCR accepts, ``[0, w-1] x [0, h-1]``."""
    xmax, ymax = float(width - 1), float(height - 1)
    out = []
    clipped = False
    for x, y in points:
        cx = min(max(x, 0.0), xmax)
        cy = min(max(y, 0.0), ymax)
        clipped = clipped or cx != x or cy != y
        out.append((cx, cy))
    return out, clipped


# ------------------------------------------------------------------------------ coo
@dataclass
class CooInstance:
    page: int
    points: list[tuple[float, float]]
    text: str


def read_polygon(el: ET.Element) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    i = 0
    while el.get(f"x{i}") is not None and el.get(f"y{i}") is not None:
        pts.append((float(el.get(f"x{i}")), float(el.get(f"y{i}"))))
        i += 1
    return pts


def parse_coo(path: Path, counters: Counter) -> dict[int, list[CooInstance]]:
    """``{page: [CooInstance]}`` with every ``<onomatopoeia>`` polygon of a book."""
    pages: dict[int, list[CooInstance]] = defaultdict(list)
    for page in ET.parse(path).getroot().iter("page"):
        idx = int(page.get("index", 0))
        for el in page.iter("onomatopoeia"):
            pts = read_polygon(el)
            counters["polygons"] += 1
            counters[f"points_{len(pts)}"] += 1
            if pts:
                pages[idx].append(CooInstance(idx, pts, clean_text(el.text)))
    return dict(pages)


def parse_blocks(path: Path) -> dict[int, list[tuple[float, float, float, float]]]:
    """Manga109 ``<text>`` *block* boxes - class evidence only, never labels.

    A block covers a whole bubble (possibly several columns); using it as a line
    label would teach the detector to merge lines.
    """
    pages: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for page in ET.parse(path).getroot().iter("page"):
        idx = int(page.get("index", 0))
        for el in page:
            if el.tag != "text":
                continue
            pages[idx].append(
                (
                    float(el.get("xmin")),
                    float(el.get("ymin")),
                    float(el.get("xmax")),
                    float(el.get("ymax")),
                )
            )
    return dict(pages)


# ------------------------------------------------------------------------- mangaocr
@dataclass
class Box:
    rect: tuple[float, float, float, float]  # x1, y1, x2, y2
    text: str


# Fallback scanner for rows whose JSON is broken (unescaped quotes and raw
# newlines inside text_content, ~1.2% of rows).  Anchoring the text on the quote
# that precedes "}" keeps embedded quotes inside the string.
_TOLERANT_BOX_RE = re.compile(
    r'"bbox_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,'
    r'\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
    r'\s*(?:,\s*"text_content"\s*:\s*"(.*?)"\s*\})?',
    re.S,
)
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


def _box(x1: float, y1: float, x2: float, y2: float, text: str) -> Box:
    return Box((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), text)


def parse_mangaocr_row(row: dict) -> tuple[list[Box], str]:
    """Parse one MangaOCR row -> (boxes, mode) with mode empty/json/tolerant."""
    raw = ""
    for msg in row.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        for content in msg.get("content", []):
            if content.get("text"):
                raw += content["text"]
    raw = raw.strip()
    if not raw:
        return [], "empty"
    fenced = _FENCE_RE.match(raw)
    payload = fenced.group(1) if fenced else raw
    try:
        items = json.loads(payload, strict=False)
    except Exception:  # noqa: BLE001 - falls back to the tolerant scanner
        items = None
    if isinstance(items, list):
        boxes: list[Box] = []
        for item in items:
            bbox = item.get("bbox_2d") if isinstance(item, dict) else None
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = (float(v) for v in bbox)
            boxes.append(_box(x1, y1, x2, y2, clean_text(item.get("text_content"))))
        return boxes, "json"
    boxes = [
        _box(float(g[0]), float(g[1]), float(g[2]), float(g[3]), clean_text(g[4]))
        for g in _TOLERANT_BOX_RE.findall(payload)
    ]
    return boxes, "tolerant"


def coordinate_mismatch(
    book: str,
    images: dict[int, Path],
    mangaocr: dict[tuple[str, int], list[Box]],
    sizes: dict[int, tuple[int, int]],
) -> tuple[int, int]:
    """Count (boxes, boxes outside the page) for one book.

    MangaOCR boxes are in the coordinate space of the page image they were
    annotated on.  For most books that is the Manga109-s canvas, but at least one
    book (``PrayerHaNemurenai``) was annotated on a ~2x larger canvas, so its
    boxes cannot be placed on the Manga109-s pages by book+page alone.
    """
    total = outside = 0
    for page in images:
        size = sizes.get(page)
        if size is None:
            continue
        width, height = size
        for box in mangaocr.get((book, page), []):
            total += 1
            if (
                box.rect[0] < -2
                or box.rect[1] < -2
                or box.rect[2] > width + 2
                or box.rect[3] > height + 2
            ):
                outside += 1
    return total, outside


def split_row_id(row_id: str) -> tuple[str, int] | None:
    """``"<book>_<page:06d>"`` -> (book, page); book names may contain underscores."""
    if "_" not in row_id:
        return None
    book, page = row_id.rsplit("_", 1)
    try:
        return book, int(page)
    except ValueError:
        return None


def load_mangaocr(dataset: str, cache: Path | None, counters: Counter) -> dict[tuple[str, int], list[Box]]:
    """Read the MangaOCR annotations from a local cache or from the Hub."""
    pages: dict[tuple[str, int], list[Box]] = defaultdict(list)
    if cache is not None and cache.exists():
        with open(cache, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                pages[(rec["book"], int(rec["page"]))].extend(Box(tuple(b), t) for b, t in rec["boxes"])
                counters["pages"] += 1
        counters["source"] = str(cache)
        return dict(pages)

    from datasets import load_dataset

    ds = load_dataset(dataset)
    for split_name in ds:
        for row in ds[split_name]:
            counters["rows"] += 1
            counters[f"rows_split_{split_name}"] += 1
            parsed = split_row_id(str(row.get("id", "")))
            if parsed is None:
                counters["rows_bad_id"] += 1
                continue
            boxes, mode = parse_mangaocr_row(row)
            counters[f"rows_{mode}"] += 1
            if boxes:
                pages[parsed].extend(boxes)
            elif mode != "empty":
                counters["rows_without_boxes"] += 1
    counters["source"] = dataset
    return dict(pages)


def write_mangaocr_cache(path: Path, pages: dict[tuple[str, int], list[Box]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for key in sorted(pages):
            rec = {
                "book": key[0],
                "page": key[1],
                "boxes": [[[num(v) for v in b.rect], b.text] for b in pages[key]],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------------ page build
@dataclass
class Config:
    dedupe_iou: float = 0.5
    dedupe_containment: float = 0.5
    min_area: float = 4.0
    min_extent: float = 2.0
    min_points: int = 4
    drop_merged_dialogue: bool = False
    train_dialogue: str = "ignore"
    ignore_sfx_tolerance: float = 0.01
    min_ignore_area: float = 16.0


@dataclass
class Instance:
    page: int
    cls: str
    source: str
    points: list[tuple[float, float]]
    text: str
    evidence: dict = field(default_factory=dict)
    role: str = "positive"  # "positive" | "ignore" (detector is not trained there)

    @property
    def rect(self) -> tuple[float, float, float, float]:
        return bbox_of(self.points)


@dataclass
class PageResult:
    labels: list[Instance] = field(default_factory=list)
    dropped: list[Instance] = field(default_factory=list)
    counters: Counter = field(default_factory=Counter)


def hole_free_parts(geom, limit: int = 64) -> list[Polygon]:
    """Split a (multi)polygon into hole-free polygons by cutting bands through holes.

    Ignore regions are rasterised by ``cv2.fillPoly`` on their exterior ring only,
    so a polygon with an interior ring would silently re-cover the hole - which is
    exactly where the SFX ground truth sits.  Cutting horizontal bands through each
    hole leaves pieces whose exterior is the whole piece.
    """
    out: list[Polygon] = []
    stack = [geom]
    guard = 0
    while stack:
        guard += 1
        if guard > limit:  # pathological geometry: drop rather than emit a wrong mask
            return []
        current = stack.pop()
        if current.is_empty:
            continue
        parts = list(current.geoms) if current.geom_type.startswith("Multi") else [current]
        for part in parts:
            if part.geom_type != "Polygon" or part.is_empty or part.area <= 0:
                continue
            if not part.interiors:
                out.append(part)
                continue
            x1, y1, x2, y2 = part.bounds
            _, hy1, _, hy2 = part.interiors[0].bounds
            band = shapely_box(x1 - 1, hy1, x2 + 1, hy2)
            rest = part.difference(band)
            if not rest.is_empty:
                stack.append(rest)
    return out


def build_page(
    page: int,
    size: tuple[int, int],
    coo: Sequence[CooInstance],
    boxes: Sequence[Box],
    blocks: Sequence[tuple[float, float, float, float]],
    cfg: Config,
    dialogue_role: str = "positive",
) -> PageResult:
    """Label one page: COO polygons + de-duplicated MangaOCR boxes.

    ``dialogue_role="ignore"`` (train) turns every MangaOCR dialogue box into a
    ``###`` ignore region instead of a positive label, because MangaOCR's dialogue
    instances are Manga109 text *blocks* (multi-column bubbles), not lines: DB would
    otherwise be trained to merge lines into blocks.  Ignore geometry is trimmed so
    it never swallows SFX ground truth.
    """
    width, height = size
    res = PageResult()
    c = res.counters

    # -- COO polygons ------------------------------------------------------
    polys: list[Polygon] = []
    for inst in coo:
        c["coo_polygons"] += 1
        if len(inst.points) < cfg.min_points:
            c["coo_dropped_few_points"] += 1
            res.dropped.append(Instance(page, "sfx", "coo", inst.points, inst.text, {"drop": "few_points"}))
            continue
        pts, clipped = clip_points(inst.points, width, height)
        if clipped:
            c["coo_clipped"] += 1
        if is_degenerate(pts, cfg.min_area, cfg.min_extent):
            c["coo_dropped_degenerate"] += 1
            res.dropped.append(Instance(page, "sfx", "coo", pts, inst.text, {"drop": "degenerate"}))
            continue
        poly = Polygon(pts)
        if not poly.is_valid:  # self-intersecting annotation artefact
            poly = poly.buffer(0)
            c["coo_invalid_repaired"] += 1
        if poly.is_empty or poly.area <= 0:
            c["coo_dropped_empty"] += 1
            res.dropped.append(Instance(page, "sfx", "coo", pts, inst.text, {"drop": "empty"}))
            continue
        polys.append(poly)
        res.labels.append(Instance(page, "sfx", "coo", pts, inst.text, {"points": len(pts)}))

    tree = STRtree(polys) if polys else None

    # -- MangaOCR boxes ----------------------------------------------------
    for box in boxes:
        c["boxes"] += 1
        rect, clipped = clip_points(
            [(box.rect[0], box.rect[1]), (box.rect[2], box.rect[3])], width, height
        )
        rect = (rect[0][0], rect[0][1], rect[1][0], rect[1][1])
        if clipped:
            c["boxes_clipped"] += 1
        points = [(rect[0], rect[1]), (rect[2], rect[1]), (rect[2], rect[3]), (rect[0], rect[3])]
        if is_degenerate(points, cfg.min_area, cfg.min_extent):
            c["boxes_dropped_degenerate"] += 1
            res.dropped.append(Instance(page, "sfx", "mangaocr", points, box.text, {"drop": "degenerate"}))
            continue

        # Polygon-vs-box IoU (and polygon coverage) against this page's COO polygons.
        coo_iou = 0.0
        coo_cover = 0.0
        if tree is not None:
            probe = shapely_box(*rect)
            for hit in tree.query(probe):
                poly = polys[int(hit)]
                inter = poly.intersection(probe).area
                union = poly.area + probe.area - inter
                if union > 0:
                    coo_iou = max(coo_iou, inter / union)
                if poly.area > 0:
                    coo_cover = max(coo_cover, inter / poly.area)

        block_iou = max((rect_iou(rect, b) for b in blocks), default=0.0)
        blocks_inside = sum(1 for b in blocks if contains_ratio(rect, b) > 0.5)

        merged = blocks_inside >= 2
        is_dialogue = block_iou > 0.5 or merged

        # Class-aware de-dupe, in priority order:
        #   1. overlap with a COO polygon (IoU) -> the box *is* that SFX, COO wins;
        #   2. dialogue evidence (a Manga109 text instance, possibly merged) ->
        #      dialogue, a bubble never loses its lines to a neighbouring SFX;
        #   3. a looser box that covers most of a COO polygon -> that SFX again
        #      (MangaOCR draws a wide box around curved lettering), drop the box.
        duplicate = coo_iou > cfg.dedupe_iou or (
            not is_dialogue and coo_cover > cfg.dedupe_containment
        )
        if duplicate:
            c["boxes_deduped_coo"] += 1
            if coo_iou <= cfg.dedupe_iou:
                c["boxes_deduped_by_containment"] += 1
            if block_iou > cfg.dedupe_iou:
                c["boxes_deduped_conflicting_block"] += 1
            res.dropped.append(
                Instance(
                    page,
                    "sfx",
                    "mangaocr",
                    points,
                    box.text,
                    {
                        "drop": "duplicate_of_coo",
                        "coo_iou": round(coo_iou, 4),
                        "coo_cover": round(coo_cover, 4),
                    },
                )
            )
            continue

        cls = "dialogue" if is_dialogue else "sfx"
        if merged:
            c["dialogue_merged_blocks"] += 1
        if cls == "dialogue" and cfg.drop_merged_dialogue and merged:
            c["dialogue_dropped_merged"] += 1
            res.dropped.append(
                Instance(page, cls, "mangaocr", points, box.text, {"drop": "merged_dialogue"})
            )
            continue
        res.labels.append(
            Instance(
                page,
                cls,
                "mangaocr",
                points,
                box.text,
                {
                    "block_iou": round(block_iou, 3),
                    "blocks_inside": blocks_inside,
                    "coo_iou": round(coo_iou, 3),
                    "coo_cover": round(coo_cover, 3),
                    "merged": merged,
                },
            )
        )
        c[f"boxes_class_{cls}"] += 1

    if dialogue_role == "ignore":
        res.labels = _dialogue_to_ignore(res.labels, polys, page, cfg, c)

    # Ignore regions first: MakeShrinkMap fills the supervision mask with 0 for an
    # ignore label and leaves positives alone, so a positive that shares a few
    # tolerance pixels with an ignore region stays supervised when it comes last.
    res.labels.sort(key=lambda inst: 0 if inst.role == "ignore" else 1)
    return res


def _dialogue_to_ignore(
    labels: list[Instance],
    coo_polys: Sequence[Polygon],
    page: int,
    cfg: Config,
    counters: Counter,
) -> list[Instance]:
    """Replace dialogue labels with `###` ignore regions that avoid SFX ground truth.

    Inputs are the positives of one page, where ``coo_polys`` is the COO polygon of
    each ``source == "coo"`` label, in order.  The SFX ground truth (COO polygons +
    MangaOCR-only SFX boxes) is subtracted from every ignore candidate, so an ignore
    region can never remove supervision from an SFX target.
    """
    sfx_geoms = list(coo_polys)
    sfx_geoms += [
        shapely_box(*inst.rect) for inst in labels if inst.cls == "sfx" and inst.source == "mangaocr"
    ]
    union = unary_union(sfx_geoms) if sfx_geoms else None

    out: list[Instance] = []
    for inst in labels:
        if inst.cls != "dialogue":
            out.append(inst)
            continue
        counters["dialogue_as_ignore_candidates"] += 1
        candidate = shapely_box(*inst.rect)
        overlap = 0.0
        if union is not None and not union.is_empty and candidate.area > 0:
            overlap = candidate.intersection(union).area / candidate.area
        if overlap <= cfg.ignore_sfx_tolerance:
            counters["ignore_direct"] += 1
            out.append(
                Instance(
                    inst.page,
                    inst.cls,
                    inst.source,
                    inst.points,
                    "###",
                    {**inst.evidence, "sfx_overlap": round(overlap, 5), "ignore_parts": 1},
                    role="ignore",
                )
            )
            continue

        counters["ignore_conflicting"] += 1
        parts = hole_free_parts(candidate.difference(union))
        keep = [
            part
            for part in parts
            if part.area >= cfg.min_ignore_area
            and (part.bounds[2] - part.bounds[0]) >= cfg.min_extent
            and (part.bounds[3] - part.bounds[1]) >= cfg.min_extent
            and len(part.exterior.coords) - 1 >= cfg.min_points
        ]
        if not keep:
            counters["ignore_dropped_no_room"] += 1
            continue
        counters["ignore_trimmed"] += 1
        counters["ignore_parts_emitted"] += len(keep)
        for part in keep:
            ring = [(float(x), float(y)) for x, y in list(part.exterior.coords)[:-1]]
            out.append(
                Instance(
                    page,
                    inst.cls,
                    inst.source,
                    ring,
                    "###",
                    {
                        **inst.evidence,
                        "sfx_overlap": round(overlap, 5),
                        "ignore_parts": len(keep),
                        "trimmed": True,
                    },
                    role="ignore",
                )
            )
    return out


# ---------------------------------------------------------------------------- tests
def self_test() -> int:
    """Assertions for the geometry/classification rules that the data may not exercise.

    The release happens to need no clipping and no de-dupe conflicts on most pages,
    so the pure helpers are pinned here instead of relying on coverage.
    """
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + detail}")
        if not ok:
            failures.append(name)

    clipped, was_clipped = clip_points([(-5, 10), (2000, 10), (2000, 500), (-5, 500)], 100, 50)
    check(
        "clip_points clamps to [0, w-1] x [0, h-1]",
        clipped[0] == (0.0, 10.0) and clipped[1] == (99.0, 10.0) and clipped[2] == (99.0, 49.0),
        str(clipped),
    )
    check("clip_points reports whether it clipped", was_clipped)
    _, untouched = clip_points([(1, 2), (3, 4)], 100, 50)
    check("clip_points leaves in-bounds points alone", not untouched)

    check("is_degenerate: zero area", is_degenerate([(5, 5)] * 4, 4.0, 2.0))
    check("is_degenerate: 1 px wide", is_degenerate([(0, 0), (1, 0), (1, 40), (0, 40)], 4.0, 2.0))
    check(
        "is_degenerate: a wide, thin line is a label",
        not is_degenerate([(0, 0), (50, 0), (50, 4), (0, 4)], 4.0, 2.0),
    )
    check("is_degenerate: area below --min-area", is_degenerate([(0, 0), (2, 0), (2, 1), (0, 1)], 4.0, 2.0))

    check("rect_iou: identical boxes", rect_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0)
    check("rect_iou: half-covered box", abs(rect_iou((0, 0, 10, 10), (0, 0, 5, 10)) - 0.5) < 1e-9)
    check("rect_iou: disjoint boxes", rect_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0)

    cfg = Config()
    sfx = CooInstance(0, [(10, 10), (60, 10), (60, 40), (10, 40)], "ズバッ")
    block = [(10.0, 10.0, 60.0, 40.0)]
    page = build_page(0, (200, 100), [sfx], [Box((12.0, 11.0, 58.0, 39.0), "ズバッ")], block, cfg)
    check(
        "de-dupe keeps the COO polygon and drops the MangaOCR box",
        len(page.labels) == 1
        and page.labels[0].source == "coo"
        and page.counters["boxes_deduped_coo"] == 1
        and page.counters["boxes_deduped_conflicting_block"] == 1,
        str(page.counters),
    )
    check("de-dupe keeps 4-point box geometry as a polygon", page.labels[0].evidence["points"] == 4)

    near = build_page(
        0, (200, 100), [sfx], [Box((100.0, 10.0, 150.0, 40.0), "セリフ")], block, cfg
    )
    check("a box outside the COO polygon survives", len(near.labels) == 2, str(near.counters))

    loose = build_page(0, (200, 100), [sfx], [Box((0.0, 0.0, 100.0, 60.0), "ズバッ")], [], cfg)
    check(
        "a looser box covering the COO polygon is de-duplicated by containment",
        len(loose.labels) == 1
        and loose.counters["boxes_deduped_coo"] == 1
        and loose.counters["boxes_deduped_by_containment"] == 1,
        str(loose.counters),
    )

    dialogue = build_page(
        0, (200, 100), [], [Box((70.0, 10.0, 90.0, 40.0), "セリフ")], [(70.0, 10.0, 90.0, 40.0)], cfg
    )
    check("block-matched box -> dialogue", dialogue.labels[0].cls == "dialogue")
    merged = build_page(
        0,
        (200, 100),
        [],
        [Box((70.0, 10.0, 90.0, 75.0), "あ\nい")],
        [(70.0, 10.0, 90.0, 40.0), (70.0, 45.0, 90.0, 75.0)],
        cfg,
    )
    check(
        "box absorbing two blocks -> dialogue (merged)",
        merged.labels[0].cls == "dialogue" and merged.labels[0].evidence["merged"] is True,
    )
    unlabelled = build_page(
        0, (200, 100), [], [Box((10.0, 60.0, 40.0, 90.0), "ドン")], [(70.0, 10.0, 90.0, 40.0)], cfg
    )
    check("box matching no block and no COO polygon -> sfx", unlabelled.labels[0].cls == "sfx")

    outside = build_page(
        0, (100, 100), [CooInstance(0, [(-20, -20), (60, -20), (60, 40), (-20, 40)], "x")], [], [], cfg
    )
    check(
        "out-of-bounds COO polygon is clipped, not dropped",
        len(outside.labels) == 1 and outside.labels[0].points[0] == (0.0, 0.0),
        str(outside.labels[0].points if outside.labels else None),
    )

    # ---- training labels: block-level dialogue must be an ignore region, never
    #      a positive line label, and ignore regions must not swallow SFX GT.
    dialogue_block = [(100.0, 10.0, 150.0, 40.0)]
    train_page = build_page(
        0, (300, 100), [sfx], [Box((100.0, 10.0, 150.0, 40.0), "セリフ")], dialogue_block, cfg,
        dialogue_role="ignore",
    )
    check(
        "train: a block-level dialogue box becomes a ### ignore region",
        [i for i in train_page.labels if i.cls == "dialogue" and i.role == "ignore"]
        and all(
            i.role == "ignore" for i in train_page.labels if i.cls == "dialogue"
        ),
        str(train_page.counters),
    )
    check(
        "train: COO SFX on an ignored page stays positive",
        [i.role for i in train_page.labels if i.cls == "sfx"] == ["positive"],
        str([(i.cls, i.source, i.role) for i in train_page.labels]),
    )
    check(
        "train: ignore regions are emitted before positives",
        [i.role for i in train_page.labels] == sorted(
            [i.role for i in train_page.labels], key=lambda r: 0 if r == "ignore" else 1
        ),
    )
    check(
        "train: ### is the transcription written for an ignore region",
        payload_of(train_page.labels)[0]["transcription"] == IGNORE_TOKEN
        and payload_of(train_page.labels)[-1]["transcription"] != IGNORE_TOKEN,
        str(payload_of(train_page.labels)),
    )
    eval_page = build_page(
        0, (300, 100), [sfx], [Box((100.0, 10.0, 150.0, 40.0), "セリフ")], dialogue_block, cfg,
        dialogue_role="positive",
    )
    check(
        "val/test: dialogue stays a positive block-level label",
        all(i.role == "positive" for i in eval_page.labels)
        and [i.cls for i in eval_page.labels] == ["sfx", "dialogue"],
        str([(i.cls, i.role) for i in eval_page.labels]),
    )

    # An ignore candidate that overlaps an SFX polygon is trimmed, never written
    # over the SFX: the difference of the box and the polygon has no interior rings.
    wide_box = Box((40.0, 20.0, 120.0, 70.0), "セリフ")  # covers sfx x=10..60
    trimmed = build_page(
        0, (300, 100), [sfx], [wide_box], [(40.0, 20.0, 120.0, 70.0)], cfg, dialogue_role="ignore"
    )
    ignore_polys = [Polygon(i.points) for i in trimmed.labels if i.role == "ignore"]
    sfx_poly = Polygon([i.points for i in trimmed.labels if i.cls == "sfx"][0])
    check(
        "train: an ignore region overlapping SFX is trimmed away from it",
        ignore_polys
        and sum(p.area for p in ignore_polys) > 0
        and all(p.intersection(sfx_poly).area <= 1e-6 for p in ignore_polys)
        and trimmed.counters["ignore_trimmed"] == 1,
        f"{len(ignore_polys)} parts, counters={dict(trimmed.counters)}",
    )
    check(
        "train: trimmed ignore parts have no interior rings",
        all(not p.interiors and p.is_valid for p in ignore_polys),
        str([len(p.interiors) for p in ignore_polys]),
    )
    check(
        "train: SFX from the trimmed page is still a positive label",
        [i.cls for i in trimmed.labels if i.role == "positive"] == ["sfx"],
        str([(i.cls, i.role) for i in trimmed.labels]),
    )
    # A bubble drawn inside a big SFX region (IoU < 0.5, fully covered): the ignore
    # candidate has no room left after subtracting the SFX, so it is dropped.
    fully_covered = build_page(
        0, (300, 100), [sfx], [Box((20.0, 15.0, 40.0, 35.0), "セリフ")], [(20.0, 15.0, 40.0, 35.0)], cfg,
        dialogue_role="ignore",
    )
    check(
        "train: an ignore candidate swallowed by SFX is dropped, not written over it",
        not [i for i in fully_covered.labels if i.role == "ignore"]
        and fully_covered.counters["ignore_dropped_no_room"] == 1,
        str(fully_covered.counters),
    )
    holed = shapely_box(0, 0, 100, 80).difference(sfx_poly)
    check(
        "hole_free_parts removes interior rings",
        bool(holed.interiors or all(not p.interiors for p in hole_free_parts(holed)))
        and all(not p.interiors for p in hole_free_parts(holed)),
        f"source interiors={len(holed.interiors)}",
    )

    check("book_split: Arisa -> test (matches the rec dataset)", book_split("Arisa") == "test")
    check("book_split: ARMS -> train", book_split("ARMS") == "train")
    check("book_split: UltraEleven -> val", book_split("UltraEleven") == "val")

    print(f"\n{len(failures)} failure(s)")
    return 0 if not failures else 1


# -------------------------------------------------------------------------- driver
def label_path(image: Path, path_mode: str) -> str:
    resolved = image.resolve()
    if path_mode == "absolute":
        return str(resolved)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def payload_of(instances: Sequence[Instance]) -> list[dict]:
    """Paddle SimpleDataSet payload; ignore labels carry ``###`` (class unchanged)."""
    payload = []
    for inst in instances:
        if inst.role == "ignore":
            text = IGNORE_TOKEN
        else:
            text = inst.text if inst.text not in IGNORE_TOKENS else "text"
        payload.append(
            {"transcription": text, "points": [[num(x), num(y)] for x, y in inst.points]}
        )
    return payload


def sidecar_of(book: str, split: str, image_path: str, instances: Sequence[Instance]) -> list[dict]:
    return [
        {
            "book": book,
            "page": inst.page,
            "split": split,
            "image": image_path,
            "class": inst.cls,
            "source": inst.source,
            "role": inst.role,
            "granularity": "block" if inst.cls == "dialogue" else "instance",
            "points": [[num(x), num(y)] for x, y in inst.points],
            "evidence": inst.evidence,
        }
        for inst in instances
    ]


def build(args: argparse.Namespace) -> int:
    release = find_release_dir(args.manga109)
    images_dir = args.images or release / "images"
    coo_dir = args.coo or release / "annotations_COO"
    blocks_dir = release / "annotations"
    out = args.out.resolve()
    cfg = Config(
        dedupe_iou=args.dedupe_iou,
        dedupe_containment=args.dedupe_containment,
        min_area=args.min_area,
        min_extent=args.min_extent,
        min_points=args.min_points,
        drop_merged_dialogue=args.drop_merged_dialogue,
        train_dialogue=args.train_dialogue,
        ignore_sfx_tolerance=args.ignore_sfx_tolerance,
        min_ignore_area=args.min_ignore_area,
    )

    books = sorted(p.stem for p in coo_dir.glob("*.xml"))
    if not books:
        raise SystemExit(f"no COO annotations under {coo_dir}")
    if args.limit_books:
        books = books[: args.limit_books]
    if not blocks_dir.is_dir():
        raise SystemExit(
            f"{blocks_dir} is missing: Manga109 <text> blocks are needed to tag "
            "MangaOCR boxes as dialogue vs SFX (they are never used as labels)"
        )

    src = Counter()
    cache_path = Path(args.mangaocr_jsonl) if args.mangaocr_jsonl else out / "mangaocr_boxes.jsonl"
    # A build always re-reads the annotations (the Hub cache makes that cheap) so
    # the row-level provenance in stats.json is complete; --mangaocr-jsonl opts
    # into a frozen cache instead, and the cache is refreshed after every build.
    mangaocr = load_mangaocr(args.mangaocr_dataset, cache_path if args.mangaocr_jsonl else None, src)
    if src.get("rows"):
        write_mangaocr_cache(cache_path, mangaocr)

    out.mkdir(parents=True, exist_ok=True)
    counters = Counter()
    per_split: dict[str, Counter] = {s: Counter() for s in SPLITS}
    split_books: dict[str, list[str]] = defaultdict(list)
    page_geometry = Counter()
    geometry_examples: list[dict] = []
    zero_overlap_pages: dict[str, int] = defaultdict(int)
    compared_pages: dict[str, int] = defaultdict(int)
    instance_counts: Counter = Counter()
    role_counts: Counter = Counter()
    positive_counts: Counter = Counter()
    lines: dict[str, list[str]] = {s: [] for s in SPLITS}
    sidecars: dict[str, list[str]] = {s: [] for s in SPLITS}
    excluded: dict[str, dict] = {}
    skip_books = {b.strip() for b in (args.exclude_books or "").split(",") if b.strip()}

    for n, book in enumerate(books, 1):
        split = book_split(book)
        img_dir = images_dir / book
        images = {int(p.stem): p for p in img_dir.glob(f"*{IMAGE_SUFFIX}")} if img_dir.is_dir() else {}
        if not images:
            counters["books_without_images"] += 1
            continue
        if book in skip_books:
            excluded[book] = {"reason": "requested with --exclude-books", "pages": len(images)}
            continue
        sizes = {page: image_size(image) for page, image in images.items()}
        book_boxes, book_outside = coordinate_mismatch(book, images, mangaocr, sizes)
        if book_boxes >= args.mangaocr_min_boxes and book_outside / book_boxes > args.mangaocr_max_oob_ratio:
            if not args.include_misaligned_books:
                excluded[book] = {
                    "reason": (
                        "MangaOCR bboxes are in a different (larger) coordinate space than the "
                        "Manga109-s page images; book+page alignment is not possible without "
                        "rescaling, and an unlabelled dialogue book would become negative "
                        "supervision"
                    ),
                    "pages": len(images),
                    "mangaocr_boxes": book_boxes,
                    "boxes_outside_page": book_outside,
                }
                continue
            mangaocr = {k: v for k, v in mangaocr.items() if k[0] != book}
        split_books[split].append(book)
        coo = parse_coo(coo_dir / f"{book}.xml", counters)
        blocks = parse_blocks(blocks_dir / f"{book}.xml")

        for page in sorted(images):
            size = sizes[page]
            if size is None:
                page_geometry["unreadable"] += 1
                continue
            page_geometry["checked"] += 1
            # COO <page width height> are unreliable for a few pages
            # (GinNoChimera: 24 pages at 0x0), so the JPEG header is authoritative.
            boxes = mangaocr.get((book, page), [])
            dialogue_role = "ignore" if (split == "train" and cfg.train_dialogue == "ignore") else "positive"
            res = build_page(
                page,
                size,
                coo.get(page, []),
                boxes,
                blocks.get(page, []),
                cfg,
                dialogue_role=dialogue_role,
            )
            counters.update(res.counters)
            per_split[split].update(res.counters)

            page_blocks = blocks.get(page, [])
            if page_blocks and boxes:
                compared_pages[book] += 1
                matched = any(
                    max(rect_iou(b.rect, blk) for blk in page_blocks) > 0.5 for b in boxes
                )
                if not matched:
                    zero_overlap_pages[book] += 1
            if page_blocks:
                # Consequence of emitting no ignore regions: blocks nobody labelled
                # are background/negative supervision for the detector.
                counters["manga109_blocks"] += len(page_blocks)
                counters["manga109_blocks_not_covered"] += sum(
                    1
                    for blk in page_blocks
                    if not any(contains_ratio(lbl.rect, blk) > 0.5 for lbl in res.labels)
                )

            if not res.labels:
                per_split[split]["pages_empty"] += 1
                continue
            per_split[split]["pages_with_labels"] += 1
            image_path = label_path(images[page], args.path_mode)
            lines[split].append(
                f"{image_path}\t{json.dumps(payload_of(res.labels), ensure_ascii=False)}"
            )
            sidecars[split].extend(
                json.dumps(rec, ensure_ascii=False)
                for rec in sidecar_of(book, split, image_path, res.labels)
            )
            for inst in res.labels:
                instance_counts[(split, inst.cls, inst.source)] += 1
                role_counts[(split, inst.role)] += 1
                if inst.role == "positive":
                    positive_counts[(split, inst.cls)] += 1
                else:
                    role_counts[(split, "ignore", inst.cls)] += 1
        if n % 20 == 0 or n == len(books):
            print(f"[{n}/{len(books)}] {book}", flush=True)

    for split in SPLITS:
        (out / f"{split}.txt").write_text("\n".join(lines[split]) + "\n", encoding="utf-8")
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for rec in sidecars[split]:
                f.write(rec + "\n")

    # ---- page geometry cross-check against the COO <page> attributes -----
    coo_attr_zero = 0
    for book in books:
        for page in ET.parse(coo_dir / f"{book}.xml").getroot().iter("page"):
            width = int(float(page.get("width", 0)))
            height = int(float(page.get("height", 0)))
            idx = int(page.get("index", 0))
            if (width, height) == (0, 0) and (images_dir / book / f"{idx:03d}{IMAGE_SUFFIX}").exists():
                coo_attr_zero += 1
                if len(geometry_examples) < 8:
                    geometry_examples.append({"book": book, "page": idx, "coo_page_size": [width, height]})

    # ---- alignment --------------------------------------------------------
    book_pages = set()
    for book in books:
        img_dir = images_dir / book
        if img_dir.is_dir():
            book_pages |= {(book, int(p.stem)) for p in img_dir.glob(f"*{IMAGE_SUFFIX}")}
    rows_in_books = sum(1 for key in mangaocr if key in book_pages)
    rows_other_books = sum(1 for key in mangaocr if key[0] not in set(books))
    rows_page_missing = sum(1 for key in mangaocr if key[0] in set(books) and key not in book_pages)

    by_class = {s: {cls: 0 for cls in CLASSES} for s in SPLITS}
    by_source = {s: {src_name: 0 for src_name in SOURCES} for s in SPLITS}
    for (split, cls, source), count in instance_counts.items():
        by_class[split][cls] += count
        by_source[split][source] += count

    suspicious = {
        book: {
            "compared_pages": compared_pages[book],
            "pages_without_block_match": zero_overlap_pages[book],
        }
        for book in compared_pages
        if compared_pages[book] >= 5 and zero_overlap_pages[book] / compared_pages[book] > 0.25
    }

    total_instances = sum(instance_counts.values())
    report = {
        "generator": "scripts/build_det_dataset.py",
        "config": {
            "manga109": str(release),
            "images": str(images_dir),
            "coo": str(coo_dir),
            "out": str(out),
            "mangaocr_source": src["source"],
            "dedupe_iou": cfg.dedupe_iou,
            "dedupe_containment": cfg.dedupe_containment,
            "min_area": cfg.min_area,
            "min_extent": cfg.min_extent,
            "min_points": cfg.min_points,
            "path_mode": args.path_mode,
            "drop_merged_dialogue": cfg.drop_merged_dialogue,
            "train_dialogue": cfg.train_dialogue,
            "ignore_sfx_tolerance": cfg.ignore_sfx_tolerance,
            "min_ignore_area": cfg.min_ignore_area,
        },
        "books": {
            "total": sum(len(v) for v in split_books.values()),
            "per_split": {s: len(split_books[s]) for s in SPLITS},
            "names": {s: sorted(split_books[s]) for s in SPLITS},
            "without_images": counters["books_without_images"],
            "excluded": len(excluded),
            "excluded_detail": excluded,
        },
        "pages": {
            "images": page_geometry["checked"],
            "unreadable": page_geometry["unreadable"],
            "with_labels": sum(per_split[s]["pages_with_labels"] for s in SPLITS),
            "without_labels": sum(per_split[s]["pages_empty"] for s in SPLITS),
        },
        "instances": {
            "total": total_instances,
            "pages_with_labels": {s: per_split[s]["pages_with_labels"] for s in SPLITS},
            "total_by_split": {s: sum(by_class[s].values()) for s in SPLITS},
            "by_class": by_class,
            "by_role": {
                split: {
                    "positive": role_counts[(split, "positive")],
                    "ignore": role_counts[(split, "ignore")],
                }
                for split in SPLITS
            },
            "positive_by_class": {
                split: {cls: positive_counts[(split, cls)] for cls in CLASSES} for split in SPLITS
            },
            "by_source": by_source,
            "counters_by_split": {
                s: {key: per_split[s].get(key, 0) for key in COUNTER_KEYS} for s in SPLITS
            },
            "by_class_and_source": {
                cls: {
                    source: sum(
                        v for (_, c2, s2), v in instance_counts.items() if c2 == cls and s2 == source
                    )
                    for source in SOURCES
                }
                for cls in CLASSES
            },
        },
        "coo": {
            "polygons": counters["polygons"],
            "used": sum(
                v for (_, cls, source), v in instance_counts.items() if cls == "sfx" and source == "coo"
            ),
            "point_count_histogram": {
                int(k.split("_")[1]): v for k, v in counters.items() if k.startswith("points_")
            },
            "clipped_polygons": counters["coo_clipped"],
            "dropped_few_points": counters["coo_dropped_few_points"],
            "dropped_degenerate": counters["coo_dropped_degenerate"],
            "dropped_empty": counters["coo_dropped_empty"],
            "invalid_repaired": counters["coo_invalid_repaired"],
        },
        "mangaocr": {
            "rows": src["rows"],
            "rows_by_split": {s: src[f"rows_split_{s}"] for s in ("train", "validation", "test")},
            "rows_parsed_json": src["rows_json"],
            "rows_parsed_tolerant": src["rows_tolerant"],
            "rows_empty_assistant": src["rows_empty"],
            "rows_without_boxes": src["rows_without_boxes"],
            "cached_pages": src["pages"],
            "pages_matched_to_images": rows_in_books,
            "pages_in_books_outside_manga109s": rows_other_books,
            "pages_without_image": rows_page_missing,
            "boxes": counters["boxes"],
            "boxes_clipped": counters["boxes_clipped"],
            "boxes_dropped_degenerate": counters["boxes_dropped_degenerate"],
            "boxes_deduped_into_coo": counters["boxes_deduped_coo"],
            "boxes_deduped_by_containment_only": counters["boxes_deduped_by_containment"],
            "boxes_deduped_but_matching_block": counters["boxes_deduped_conflicting_block"],
            "boxes_class_dialogue": counters["boxes_class_dialogue"],
            "boxes_class_sfx": counters["boxes_class_sfx"],
            "dialogue_merging_multiple_blocks": counters["dialogue_merged_blocks"],
            "dialogue_dropped_merged": counters["dialogue_dropped_merged"],
        },
        "alignment": {
            "manga109_blocks": counters["manga109_blocks"],
            "manga109_blocks_not_covered_by_a_label": counters["manga109_blocks_not_covered"],
            "not_covered_definition": (
                "Manga109 <text> block with less than half its area inside any label "
                "(positive or ignore)"
            ),
            "pages_with_blocks_and_boxes": sum(compared_pages.values()),
            "pages_without_any_block_match": sum(zero_overlap_pages.values()),
            "suspicious_books": suspicious,
        },
        "page_geometry": {
            "source": "JPEG header via Pillow; COO <page> attributes are not trusted",
            "checked": page_geometry["checked"],
            "unreadable": page_geometry["unreadable"],
            "coo_page_size_zero": coo_attr_zero,
            "examples": geometry_examples,
        },
        "splits": {
            "function": "build_real_dataset.book_split",
            "book_disjoint": len(set(split_books["train"]) & set(split_books["val"]))
            + len(set(split_books["train"]) & set(split_books["test"]))
            + len(set(split_books["val"]) & set(split_books["test"]))
            == 0,
        },
        "ignore_regions": {
            "mode": "dialogue_blocks_ignored_in_train"
            if cfg.train_dialogue == "ignore"
            else "none",
            "count": {split: role_counts[(split, "ignore")] for split in SPLITS},
            "dialogue_ground_truth": {
                "train": "ignore (block-level, not a line label)",
                "val": "positive (block-level, for detection metrics)",
                "test": "positive (block-level, for detection metrics)",
            },
            "granularity_note": (
                "MangaOCR did not provide line-level dialogue: its bbox_2d instances are "
                "the Manga109 <text> blocks, i.e. whole bubbles / columns. Training on "
                "them as positives would teach DB to merge lines, so train emits them as "
                "### ignore regions and val/test keep them as block-level positives."
            ),
            "sfx_overlap_tolerance": cfg.ignore_sfx_tolerance,
            "candidates": counters["dialogue_as_ignore_candidates"],
            "emitted_without_trim": counters["ignore_direct"],
            "conflicting_with_sfx": counters["ignore_conflicting"],
            "trimmed": counters["ignore_trimmed"],
            "parts_emitted_for_trimmed": counters["ignore_parts_emitted"],
            "dropped_no_room": counters["ignore_dropped_no_room"],
            "consequence": (
                "Dialogue regions are excluded from the DB loss in train (no merging "
                "pressure, and no pressure to suppress them either). Text annotated by "
                "neither COO nor MangaOCR is still trained as background, and because "
                "nothing was inferred from the stock detector there is no silent ignore."
            ),
        },
    }
    (out / "stats.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in ("books", "pages", "instances")}, ensure_ascii=False, indent=2))
    print(f"wrote {{train,val,test}}.txt/.jsonl + stats.json under {out}")
    return 0


# ----------------------------------------------------------------------- validation
def validate(args: argparse.Namespace) -> int:
    """Re-check the emitted labels against the sources; exit 1 on any failure."""
    out = args.out.resolve()
    release = find_release_dir(args.manga109)
    images_dir = args.images or release / "images"
    coo_dir = args.coo or release / "annotations_COO"
    blocks_dir = release / "annotations"

    cache_path = Path(args.mangaocr_jsonl) if args.mangaocr_jsonl else out / "mangaocr_boxes.jsonl"
    if not cache_path.exists():
        raise SystemExit(f"missing MangaOCR cache {cache_path}; build the dataset first")
    mangaocr = load_mangaocr(MANGAOCR_DATASET, cache_path, Counter())
    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))

    failures: list[str] = []
    totals = Counter()
    class_totals: Counter = Counter()
    books_by_split: dict[str, set[str]] = {s: set() for s in SPLITS}
    seen_books: set[str] = set()
    numeric = {
        "missing_images": 0,
        "out_of_bounds": 0,
        "degenerate": 0,
        "provenance": 0,
        "dedupe_leaks": 0,
        "role_violations": 0,
        "ignore_covers_sfx": 0,
        "ignore_with_hole": 0,
        "ignore_order_violations": 0,
    }
    role_totals: Counter = Counter()
    ignore_overlap_max = 0.0
    ignore_overlap_sum = 0.0
    ignore_regions_seen = 0
    # Informational: MangaOCR is derived from the Manga109 annotations, so a box
    # is occasionally byte-identical to a <text> block.  Geometry provenance is
    # proven above against the MangaOCR cache, so equality is not a failure.
    block_geometry = 0
    book_cache: dict[str, dict] = {}
    sizes: dict[Path, tuple[int, int] | None] = {}

    for split in SPLITS:
        labels_path = out / f"{split}.txt"
        sidecar_path = out / f"{split}.jsonl"
        if not labels_path.exists() or not sidecar_path.exists():
            failures.append(f"{split}: missing label or sidecar file")
            continue
        by_page: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for line in sidecar_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                by_page[(rec["book"], int(rec["page"]))].append(rec)
        per_split = 0
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                image_field, payload = line.split("\t", 1)
                instances = json.loads(payload)
            except ValueError:
                failures.append(f"{split}: unparsable label line: {line[:80]}")
                continue
            image = Path(image_field) if Path(image_field).is_absolute() else ROOT / image_field
            if image not in sizes:
                sizes[image] = image_size(image)
            size = sizes[image]
            book, page = image.parent.name, int(image.stem)
            books_by_split[split].add(book)
            seen_books.add(book)
            records = by_page.pop((book, page), None)
            if records is None or len(records) != len(instances):
                failures.append(f"{split}:{image_field}: sidecar/label count mismatch")
                records = records or []
            if size is None:
                numeric["missing_images"] += 1
                continue
            width, height = size
            if book not in book_cache:
                book_cache[book] = {
                    "coo": parse_coo(coo_dir / f"{book}.xml", Counter()),
                    "blocks": parse_blocks(blocks_dir / f"{book}.xml"),
                }
            coo_page = book_cache[book]["coo"].get(page, [])
            blocks = book_cache[book]["blocks"].get(page, [])
            polys = []
            for inst in coo_page:
                if len(inst.points) < args.min_points:
                    continue
                pts, _ = clip_points(inst.points, width, height)
                if not is_degenerate(pts, args.min_area, args.min_extent):
                    poly = Polygon(pts)
                    polys.append(poly if poly.is_valid else poly.buffer(0))
            tree = STRtree(polys) if polys else None

            # Ignore regions must come first so MakeShrinkMap's mask keeps the
            # positives that follow (see build_page).
            roles = [rec["role"] for rec in records]
            if roles != sorted(roles, key=lambda r: 0 if r == "ignore" else 1):
                numeric["ignore_order_violations"] += 1

            for instance, rec in zip(instances, records):
                per_split += 1
                totals["instances"] += 1
                points = [(float(p[0]), float(p[1])) for p in instance["points"]]
                cls, source = rec["class"], rec["source"]
                role = rec["role"]
                role_totals[(split, role)] += 1
                class_totals[(split, cls)] += 1
                if role == "ignore":
                    if split != "train":
                        numeric["role_violations"] += 1
                    if instance["transcription"] != IGNORE_TOKEN:
                        numeric["role_violations"] += 1
                elif split == "train" and cls == "dialogue":
                    # A block-level dialogue box must never be a train positive.
                    numeric["role_violations"] += 1
                elif role not in ("positive",):
                    numeric["role_violations"] += 1
                if rec["points"] != instance["points"]:
                    failures.append(f"{split}:{image_field}: sidecar points differ from labels")
                if cls not in CLASSES or source not in SOURCES:
                    failures.append(f"{split}:{image_field}: bad class/source {cls}/{source}")
                if book_split(rec["book"]) != split:
                    failures.append(f"{split}:{image_field}: book {rec['book']} not in split")
                if rec["book"] != book or int(rec["page"]) != page:
                    failures.append(f"{split}:{image_field}: sidecar page/book mismatch")
                if len(points) < 4:
                    failures.append(f"{split}:{image_field}: {len(points)}-point instance")
                if is_degenerate(points, args.min_area, args.min_extent):
                    numeric["degenerate"] += 1
                if any(x < 0 or y < 0 or x > width - 1 or y > height - 1 for x, y in points):
                    numeric["out_of_bounds"] += 1
                rect = bbox_of(points)
                if source == "mangaocr":
                    boxes = mangaocr.get((book, page), [])
                    if role == "ignore":
                        # Trimmed ignore fragments are derived geometry, so prove they
                        # lie inside an annotated MangaOCR box of this page instead of
                        # matching one exactly.
                        if not any(
                            b.rect[0] - 1 <= x <= b.rect[2] + 1 and b.rect[1] - 1 <= y <= b.rect[3] + 1
                            for b in boxes
                            for x, y in points
                        ):
                            numeric["provenance"] += 1
                    elif len(points) != 4 or not any(
                        abs(b.rect[0] - rect[0]) < 0.5
                        and abs(b.rect[1] - rect[1]) < 0.5
                        and abs(b.rect[2] - rect[2]) < 0.5
                        and abs(b.rect[3] - rect[3]) < 0.5
                        for b in boxes
                    ):
                        numeric["provenance"] += 1
                else:
                    matches = 0
                    for inst in coo_page:
                        if len(inst.points) != len(points):
                            continue
                        clipped, _ = clip_points(inst.points, width, height)
                        if all(
                            abs(a[0] - b[0]) < 0.01 and abs(a[1] - b[1]) < 0.01
                            for a, b in zip(clipped, points)
                        ):
                            matches += 1
                    if matches != 1:
                        numeric["provenance"] += 1
                if source == "mangaocr" and tree is not None and role == "positive":
                    probe = shapely_box(*rect)
                    best_iou = best_cover = 0.0
                    for hit in tree.query(probe):
                        poly = polys[int(hit)]
                        inter = poly.intersection(probe).area
                        union = poly.area + probe.area - inter
                        if union > 0:
                            best_iou = max(best_iou, inter / union)
                        if poly.area > 0:
                            best_cover = max(best_cover, inter / poly.area)
                    if best_iou > args.dedupe_iou or (
                        cls == "sfx" and best_cover > args.dedupe_containment
                    ):
                        numeric["dedupe_leaks"] += 1
                if cls == "dialogue" and blocks:
                    if any(
                        abs(rect[0] - b[0]) < 0.5
                        and abs(rect[1] - b[1]) < 0.5
                        and abs(rect[2] - b[2]) < 0.5
                        and abs(rect[3] - b[3]) < 0.5
                        for b in blocks
                    ):
                        block_geometry += 1

            # Ignore regions (train) must not remove SFX supervision: their overlap
            # with the SFX ground truth of the page stays below --ignore-sfx-tolerance,
            # and they are hole-free so fillPoly cannot re-cover an SFX polygon.
            ignore_recs = [rec for rec in records if rec["role"] == "ignore"]
            if ignore_recs:
                sfx_geoms = list(polys)
                for rec in records:
                    if rec["class"] == "sfx" and rec["source"] == "mangaocr":
                        xs = [float(p[0]) for p in rec["points"]]
                        ys = [float(p[1]) for p in rec["points"]]
                        sfx_geoms.append(shapely_box(min(xs), min(ys), max(xs), max(ys)))
                sfx_union = unary_union(sfx_geoms) if sfx_geoms else None
                for rec in ignore_recs:
                    ignore_regions_seen += 1
                    pts = [(float(p[0]), float(p[1])) for p in rec["points"]]
                    poly = Polygon(pts)
                    if not poly.is_valid or poly.interiors or poly.area <= 0:
                        numeric["ignore_with_hole"] += 1
                        continue
                    if sfx_union is None or sfx_union.is_empty:
                        continue
                    frac = poly.intersection(sfx_union).area / poly.area
                    ignore_overlap_sum += frac
                    ignore_overlap_max = max(ignore_overlap_max, frac)
                    if frac > args.ignore_sfx_tolerance + 1e-9:
                        numeric["ignore_covers_sfx"] += 1
        left = sum(len(v) for v in by_page.values())
        if left:
            failures.append(f"{split}: {left} sidecar rows without a label")
        totals[f"{split}_instances"] = per_split

    overlap = (
        books_by_split["train"] & books_by_split["val"]
        | books_by_split["train"] & books_by_split["test"]
        | books_by_split["val"] & books_by_split["test"]
    )
    if overlap:
        failures.append(f"books in more than one split: {sorted(overlap)}")
    excluded = set(stats["books"].get("excluded_detail", {}))
    leaked = sorted(seen_books & excluded)
    if leaked:
        failures.append(f"excluded books appear in the label files: {leaked}")
    if any(numeric.values()):
        failures.append("numeric failures: " + json.dumps(numeric))
    per_split_books = {s: len(books_by_split[s]) for s in SPLITS}
    if stats["books"]["per_split"] != per_split_books:
        failures.append(
            f"book counts differ from stats.json: {stats['books']['per_split']} vs {per_split_books}"
        )
    recomputed = {
        s: {cls: sum(v for (sp, c), v in class_totals.items() if sp == s and c == cls) for cls in CLASSES}
        for s in SPLITS
    }
    if stats["instances"]["by_class"] != recomputed:
        failures.append(
            f"per-class counts differ from stats.json: {stats['instances']['by_class']} vs {recomputed}"
        )
    recomputed_roles = {
        split: {role: role_totals[(split, role)] for role in ("positive", "ignore")} for split in SPLITS
    }
    if stats["instances"]["by_role"] != recomputed_roles:
        failures.append(
            f"role counts differ from stats.json: {stats['instances']['by_role']} vs {recomputed_roles}"
        )
    if stats["ignore_regions"]["count"] != {
        split: role_totals[(split, "ignore")] for split in SPLITS
    }:
        failures.append("ignore-region counts differ from stats.json")

    report = {
        "ok": not failures,
        "failures": failures,
        "splits": {
            s: {"books": len(books_by_split[s]), "instances": totals[f"{s}_instances"]} for s in SPLITS
        },
        "books_disjoint": not overlap,
        "instances": totals["instances"],
        "per_class": recomputed,
        "checks": numeric,
        "dialogue_geometry_equal_to_manga109_block": block_geometry,
        "roles": {split: {role: role_totals[(split, role)] for role in ("positive", "ignore")} for split in SPLITS},
        "ignore_regions": {
            "count": {split: role_totals[(split, "ignore")] for split in SPLITS},
            "regions_checked": ignore_regions_seen,
            "sfx_overlap_max": round(ignore_overlap_max, 6),
            "sfx_overlap_mean": round(ignore_overlap_sum / ignore_regions_seen, 6)
            if ignore_regions_seen
            else 0.0,
            "tolerance": args.ignore_sfx_tolerance,
        },
    }
    (out / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


# ------------------------------------------------------------------------- overlays
def overlay(args: argparse.Namespace) -> int:
    """Draw labels on a few representative pages (never inside the repository)."""
    import cv2
    import numpy as np

    out = args.out.resolve()
    dest = Path(args.overlay_out).resolve()
    if dest.is_relative_to(ROOT):
        raise SystemExit(
            f"refusing to write overlays inside the repository ({dest}); pass e.g. /tmp/opencode/det_overlays"
        )
    dest.mkdir(parents=True, exist_ok=True)

    release = find_release_dir(args.manga109)
    images_dir = args.images or release / "images"
    coo_dir = args.coo or release / "annotations_COO"
    blocks_dir = release / "annotations"
    cfg = Config(
        args.dedupe_iou,
        args.dedupe_containment,
        args.min_area,
        args.min_extent,
        args.min_points,
        args.drop_merged_dialogue,
        args.train_dialogue,
        args.ignore_sfx_tolerance,
        args.min_ignore_area,
    )
    cache_path = Path(args.mangaocr_jsonl) if args.mangaocr_jsonl else out / "mangaocr_boxes.jsonl"
    if not cache_path.exists():
        raise SystemExit(f"missing MangaOCR cache {cache_path}; build the dataset first")
    mangaocr = load_mangaocr(MANGAOCR_DATASET, cache_path, Counter())

    candidates: list[tuple[int, str, int]] = []
    for book in sorted(p.stem for p in coo_dir.glob("*.xml")):
        if args.overlay_split != "all" and book_split(book) != args.overlay_split:
            continue
        coo = parse_coo(coo_dir / f"{book}.xml", Counter())
        for page, insts in coo.items():
            curved = sum(1 for i in insts if len(i.points) > 8)
            if curved:
                candidates.append((curved, book, page))
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
    picked = candidates[: args.overlay]
    if not picked:
        raise SystemExit("no pages with curved COO polygons matched the selection")

    index = []
    for curved, book, page in picked:
        image_path = images_dir / book / f"{page:03d}{IMAGE_SUFFIX}"
        size = image_size(image_path)
        if size is None:
            continue
        coo = parse_coo(coo_dir / f"{book}.xml", Counter())
        blocks = parse_blocks(blocks_dir / f"{book}.xml")
        dialogue_role = (
            "ignore" if (args.overlay_split == "train" and cfg.train_dialogue == "ignore") else "positive"
        )
        res = build_page(
            page,
            size,
            coo.get(page, []),
            mangaocr.get((book, page), []),
            blocks.get(page, []),
            cfg,
            dialogue_role=dialogue_role,
        )
        img = cv2.imread(str(image_path))
        if args.overlay_dedupe:
            for inst in res.dropped:
                if inst.evidence.get("drop") == "duplicate_of_coo":
                    outline = np.array(inst.points, np.int32).reshape(-1, 1, 2)
                    cv2.polylines(img, [outline], True, (255, 0, 255), 1)
        for inst in res.labels:
            if inst.role == "ignore":
                colour, thickness = (0, 255, 255), 1  # ### ignore region: yellow
            elif inst.source == "coo":
                colour, thickness = (0, 0, 255), 3  # COO SFX polygon: red
            elif inst.cls == "dialogue":
                colour, thickness = (0, 200, 0), 2  # dialogue box: green
            else:
                colour, thickness = (0, 165, 255), 2  # MangaOCR-only SFX: orange
            cv2.polylines(img, [np.array(inst.points, np.int32).reshape(-1, 1, 2)], True, colour, thickness)
        name = f"{book}_{page:03d}.png"
        cv2.imwrite(str(dest / name), img)
        ignores = sum(1 for inst in res.labels if inst.role == "ignore")
        entry = (
            f"{name}\t{book}\t{page}\tcurved_polygons={curved}\t"
            f"labels={len(res.labels)}\tignore_regions={ignores}"
        )
        index.append(entry)
        print(entry)
    (dest / "index.txt").write_text("\n".join(index) + "\n", encoding="utf-8")
    print(f"overlays -> {dest}")
    return 0


# ----------------------------------------------------------------------------- cli
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ignore regions: none are emitted by default, so unannotated text becomes "
            "negative supervision (see the module docstring).\n"
            "Manga109 <text> blocks are used only to tag MangaOCR boxes as dialogue "
            "vs SFX - they are never emitted as label geometry."
        ),
    )
    ap.add_argument("--manga109", type=Path, default=DEFAULT_MANGA109, help="Manga109-s root")
    ap.add_argument("--images", type=Path, default=None, help="page image dir (default <release>/images)")
    ap.add_argument(
        "--coo", type=Path, default=None, help="COO annotation dir (default <release>/annotations_COO)"
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output dir (default data/det)")
    ap.add_argument(
        "--mangaocr-dataset", default=MANGAOCR_DATASET, help="HF dataset for the MangaOCR annotations"
    )
    ap.add_argument(
        "--mangaocr-jsonl",
        type=Path,
        default=None,
        help="MangaOCR cache to read (default <out>/mangaocr_boxes.jsonl); read from the Hub if absent",
    )
    ap.add_argument("--dedupe-iou", type=float, default=0.5, help="polygon-vs-box IoU treated as a duplicate")
    ap.add_argument(
        "--dedupe-containment",
        type=float,
        default=0.5,
        help="duplicate when a box covers this much of a COO polygon's area (0 disables)",
    )
    ap.add_argument("--min-area", type=float, default=4.0, help="drop labels below this polygon area (px^2)")
    ap.add_argument(
        "--min-extent", type=float, default=2.0, help="drop labels with a bbox side below this (px)"
    )
    ap.add_argument("--min-points", type=int, default=4, help="drop COO polygons with fewer points")
    ap.add_argument(
        "--drop-merged-dialogue",
        action="store_true",
        help="drop dialogue boxes that merge >=2 Manga109 text blocks (val/test positives only; "
        "train ignores all block-level dialogue anyway)",
    )
    ap.add_argument(
        "--train-dialogue",
        choices=("ignore", "positive"),
        default="ignore",
        help="train role for the block-level MangaOCR dialogue boxes: 'ignore' (default, emits "
        "### regions so DB cannot learn to merge lines) or 'positive' (legacy, unsafe)",
    )
    ap.add_argument(
        "--ignore-sfx-tolerance",
        type=float,
        default=0.01,
        help="max share of an ignore region that may overlap SFX ground truth before it is trimmed",
    )
    ap.add_argument(
        "--min-ignore-area",
        type=float,
        default=16.0,
        help="minimum area of an ignore fragment kept after trimming (px^2)",
    )
    ap.add_argument(
        "--path-mode",
        choices=("relative", "absolute"),
        default="relative",
        help="image paths in the label files (default: relative to the repo root)",
    )
    ap.add_argument("--limit-books", type=int, default=0, help="only the first N books (debug)")
    ap.add_argument(
        "--exclude-books",
        default="",
        help="comma-separated book names to leave out of the dataset",
    )
    ap.add_argument(
        "--mangaocr-max-oob-ratio",
        type=float,
        default=0.25,
        help="exclude a book when more than this fraction of its MangaOCR boxes lie outside the page",
    )
    ap.add_argument(
        "--mangaocr-min-boxes",
        type=int,
        default=20,
        help="only run the coordinate-space check on books with at least this many boxes",
    )
    ap.add_argument(
        "--include-misaligned-books",
        action="store_true",
        help="keep books whose MangaOCR boxes are in another coordinate space (COO labels only)",
    )
    ap.add_argument("--validate-only", action="store_true", help="check the emitted dataset, build nothing")
    ap.add_argument(
        "--self-test", action="store_true", help="assert the geometry/classification rules and exit"
    )
    ap.add_argument("--overlay", type=int, default=0, help="draw this many curved-SFX pages and exit")
    ap.add_argument("--overlay-out", type=str, default="/tmp/opencode/det_overlays", help="overlay dir")
    ap.add_argument("--overlay-split", choices=("train", "val", "test", "all"), default="test")
    ap.add_argument("--overlay-dedupe", action="store_true", help="also draw de-duplicated MangaOCR boxes")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    if args.overlay:
        return overlay(args)
    if args.validate_only:
        return validate(args)
    return build(args)


if __name__ == "__main__":
    sys.exit(main())
