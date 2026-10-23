#!/usr/bin/env python3
"""Extract onomatopoeia crops from the COO annotations.

COO gives a polygon + transcript per SFX.  Only 4-point (straight) polygons are
used: they are what the detector produces at runtime, whereas a curved 14-point
polygon would train the recogniser on an input it never sees.  The 180-degree
ambiguity is resolved against the transcript with the base model.
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
from common import RecModel, load_dict, edit_distance  # noqa: E402


def parse(path: Path):
    for page in ET.parse(path).getroot().iter("page"):
        idx = int(page.get("index", 0))
        for el in page:
            if el.tag != "onomatopoeia":
                continue
            pts = []
            i = 0
            while el.get(f"x{i}") is not None and el.get(f"y{i}") is not None:
                pts.append((float(el.get(f"x{i}")), float(el.get(f"y{i}"))))
                i += 1
            text = "".join((el.text or "").split())
            if text and len(pts) in (4,):
                yield idx, text, np.array(pts, np.float32)


def crop_quad(img, quad, pad=0.05):
    s = quad.sum(1)
    d = np.diff(quad, axis=1).ravel()
    pts = np.array([quad[np.argmin(s)], quad[np.argmin(d)], quad[np.argmax(s)], quad[np.argmax(d)]], np.float32)
    w = max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3]))
    h = max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2]))
    if w < 6 or h < 6:
        return None
    pw, ph = w * (1 + 2 * pad), h * (1 + 2 * pad)
    dst = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    crop = cv2.warpPerspective(img, M, (int(pw), int(ph)), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return crop


def book_split(book: str, n_val: int = 9, n_test: int = 9) -> str:
    h = int(hashlib.md5(book.encode()).hexdigest(), 16) % 100
    return "val" if h < n_val else "test" if h < n_val + n_test else "train"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manga109", type=Path, default=ROOT / "data/raw/Manga109-s")
    ap.add_argument("--out", type=Path, default=ROOT / "data/real")
    ap.add_argument("--limit-books", type=int, default=0)
    args = ap.parse_args()

    root = next(p.parent for p in args.manga109.rglob("annotations") if p.is_dir() and (p.parent / "images").is_dir())
    img_dir, coo_dir = root / "images", root / "annotations_COO"
    args.out = args.out.resolve()
    vocab = set(load_dict())
    model = RecModel()

    rows = []
    xmls = sorted(coo_dir.glob("*.xml"))
    if args.limit_books:
        xmls = xmls[: args.limit_books]
    for xml in xmls:
        book = xml.stem
        split = book_split(book)
        by_page: dict[int, list] = {}
        for page_idx, text, quad in parse(xml):
            if len(text) > 25 or any(c != " " and c not in vocab for c in text):
                continue
            by_page.setdefault(page_idx, []).append((text, quad))
        for page_idx, items in sorted(by_page.items()):
            img = cv2.imread(str(img_dir / book / f"{page_idx:03d}.jpg"))
            if img is None:
                continue
            for text, quad in items:
                crop = crop_quad(img, quad)
                if crop is None:
                    continue
                h, w = crop.shape[:2]
                if h > w:
                    crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
                if min(crop.shape[:2]) < 8 or crop.shape[1] / crop.shape[0] > 10:
                    continue
                # resolve 180-degree ambiguity against the transcript
                a, _ = model(crop)
                b, _ = model(cv2.rotate(crop, cv2.ROTATE_180))
                ca = edit_distance(a, text) / len(text)
                cb = edit_distance(b, text) / len(text)
                if cb < ca:
                    crop = cv2.rotate(crop, cv2.ROTATE_180)
                bdir = args.out / book
                bdir.mkdir(parents=True, exist_ok=True)
                name = f"{book}_{page_idx:03d}_sfx_{len(rows):06d}.jpg"
                cv2.imwrite(str(bdir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                full = (bdir / name).resolve()
                rel = str(full.relative_to(ROOT)) if full.is_relative_to(ROOT) else str(full)
                rows.append((rel, text, "onomatopoeia", split))

    with open(args.out / "labels_sfx.tsv", "w", encoding="utf-8") as f:
        for r in rows:
            f.write("\t".join(r) + "\n")
    print(f"{len(rows)} SFX crops -> {args.out/'labels_sfx.tsv'}")


if __name__ == "__main__":
    main()
