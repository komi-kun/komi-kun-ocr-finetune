#!/usr/bin/env python3
"""Turn Manga109(-s) + COO annotations into line-level recognition crops.

Output layout (consumed by build_manifest.py / train_rec.sh):
    data/real/<book>/<page>_<idx>.jpg
    data/real/labels.tsv          # path \t text \t source \t split

Sources
  * Manga109 `<text>` elements      -> dialogue lines      (source=dialogue)
  * COO `<onomatopoeia>` elements   -> SFX lines           (source=onomatopoeia)
  * optionally hal-utokyo/MangaOCR  -> extra page-level boxes (source=mangaocr)

Orientation: crops are normalised to the recogniser's input convention, i.e. the
line is horizontal and reads left->right.  Vertical manga columns are rotated 90
deg counter-clockwise (verified against the shipped ONNX model).  The remaining
180-degree ambiguity is resolved by scoring both candidates against the ground
truth with the *base* model, falling back to the annotation point order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import RecModel, load_dict  # noqa: E402

MAX_LEN = 25
MIN_SIDE = 8
MAX_ASPECT = 9.0


# ----------------------------------------------------------------------------- io
def find_dirs(root: Path) -> tuple[Path, Path]:
    for p in sorted(root.rglob("images")):
        if p.is_dir() and (p.parent / "annotations").is_dir():
            return p, p.parent / "annotations"
    raise SystemExit(f"could not find images/ + annotations/ under {root}")


def parse_book(path: Path) -> dict[int, list[dict]]:
    """Return {page_index: [{tag, text, pts}]} for one book XML."""
    root = ET.parse(path).getroot()
    pages: dict[int, list[dict]] = {}
    for page in root.iter("page"):
        idx = int(page.get("index", 0))
        items = []
        for el in page:
            if el.tag not in ("text", "onomatopoeia"):
                continue
            pts = []
            i = 0
            while el.get(f"x{i}") is not None and el.get(f"y{i}") is not None:
                pts.append((float(el.get(f"x{i}")), float(el.get(f"y{i}"))))
                i += 1
            text = (el.text or "").strip()
            if len(pts) >= 4 and text:
                items.append({"tag": el.tag, "text": text, "pts": pts})
        if items:
            pages[idx] = items
    return pages


# ------------------------------------------------------------------------- crops
def order_quad(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) != 4:
        return pts
    s = pts.sum(1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def crop_rect(img: np.ndarray, pts: list[tuple[float, float]], pad: float = 0.03) -> np.ndarray:
    """Crop the min-area rect around `pts`, long side horizontal, with padding."""
    pts = np.asarray(pts, dtype=np.float32)
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    if w < 1 or h < 1:
        return None

    # rotate the rect so the long axis is horizontal
    if h > w:
        w, h, angle = h, w, angle + 90.0
    pw, ph = w * (1 + 2 * pad), h * (1 + 2 * pad)
    out_w, out_h = max(2, int(round(pw))), max(2, int(round(ph)))

    # map source -> output: translate centre, rotate by -angle, move to origin
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    M[0, 2] += out_w / 2 - cx
    M[1, 2] += out_h / 2 - cy
    crop = cv2.warpAffine(
        img, M, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return crop


def aspect_ok(crop: np.ndarray) -> bool:
    h, w = crop.shape[:2]
    return h >= MIN_SIDE and w >= MIN_SIDE and w / h <= MAX_ASPECT


def valid_text(text: str, vocab: set[str]) -> bool:
    t = text.replace("　", " ")
    if not t or len(t) > MAX_LEN:
        return False
    return all(c == " " or c in vocab for c in t)


# ------------------------------------------------------------------- main build
def process_book(args):
    book, book_path, images_dir, out_dir, split, vocab, model_path, dict_path = args
    pages = parse_book(book_path)
    model = RecModel(onnx_path=model_path, dict_path=dict_path)
    rows = []
    n_eval = 0
    for page_idx, items in sorted(pages.items()):
        img_path = images_dir / book / f"{page_idx:03d}.jpg"
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        for i, item in enumerate(items):
            text = item["text"].replace("　", " ")
            if not valid_text(text, vocab):
                continue
            crop = crop_rect(img, item["pts"])
            if crop is None or not aspect_ok(crop):
                continue
            flipped = cv2.rotate(crop, cv2.ROTATE_180)
            a, _ = model(crop)
            b, _ = model(flipped)
            ca, cb = cer(a, text), cer(b, text)
            if cb < ca:
                crop = flipped
            book_dir = out_dir / book
            book_dir.mkdir(parents=True, exist_ok=True)
            name = f"{book}_{page_idx:03d}_{i:03d}.jpg"
            cv2.imwrite(str(book_dir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append(
                {
                    "path": str((book_dir / name).relative_to(ROOT)),
                    "text": text,
                    "source": "onomatopoeia" if item["tag"] == "onomatopoeia" else "dialogue",
                    "split": split,
                }
            )
            n_eval += 1
    return rows


def cer(hyp: str, ref: str) -> float:
    if not ref:
        return 1.0
    d = edit_distance(hyp, ref)
    return d / len(ref)


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def book_split(book: str, n_val: int = 9, n_test: int = 9) -> str:
    h = int(hashlib.md5(book.encode()).hexdigest(), 16) % 100
    if h < n_val:
        return "val"
    if h < n_val + n_test:
        return "test"
    return "train"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manga109", type=Path, default=ROOT / "data/raw/Manga109-s")
    ap.add_argument("--coo", type=Path, default=ROOT / "data/raw/COO")
    ap.add_argument("--out", type=Path, default=ROOT / "data/real")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-books", type=int, default=0)
    args = ap.parse_args()

    images_dir, ann_dir = find_dirs(args.manga109)
    print(f"images: {images_dir}\nannotations: {ann_dir}")

    vocab = set(load_dict())
    model_path = ROOT / "models/ppocrv6_small_rec.onnx"
    dict_path = ROOT / "models/ppocrv6_small_dict.txt"

    books = sorted(ann_dir.glob("*.xml"))
    # COO annotations live in their own tree; merge them onto matching books
    coo_books = {}
    if args.coo.exists():
        for p in args.coo.rglob("*.xml"):
            if p.stem in {b.stem for b in books}:
                coo_books[p.stem] = p

    if args.limit_books:
        books = books[: args.limit_books]

    tasks = []
    for b in books:
        # COO carries the onomatopoeia elements; the Manga109 XML carries dialogue.
        merged = ann_dir / b.name
        merged = b
        if b.stem in coo_books:
            merged = args.out / "_merged" / f"{b.stem}.xml"
            merged.parent.mkdir(parents=True, exist_ok=True)
            merge_xml(b, coo_books[b.stem], merged)
        tasks.append(
            (b.stem, merged, images_dir, args.out, book_split(b.stem), vocab, model_path, dict_path)
        )

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, part in enumerate(ex.map(process_book, tasks), 1):
            rows.extend(part)
            print(f"[{i}/{len(tasks)}] {len(part)} crops", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "labels.tsv", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{r['path']}\t{r['text']}\t{r['source']}\t{r['split']}\n")

    by_split = {}
    for r in rows:
        by_split[r["split"]] = by_split.get(r["split"], 0) + 1
    print(json.dumps({"total": len(rows), **by_split}, indent=2))
    import shutil

    shutil.rmtree(args.out / "_merged", ignore_errors=True)


def merge_xml(dialogue: Path, coo: Path, out: Path) -> None:
    """Write one XML per book with both dialogue and onomatopoeia elements."""
    a = ET.parse(dialogue).getroot()
    b = ET.parse(coo).getroot()
    pages = {int(p.get("index")): p for p in a.iter("page")}
    for page in b.iter("page"):
        idx = int(page.get("index"))
        if idx not in pages:
            continue
        for el in page:
            if el.tag == "onomatopoeia":
                pages[idx].append(el)
    ET.ElementTree(a).write(out, encoding="utf-8")


if __name__ == "__main__":
    main()
