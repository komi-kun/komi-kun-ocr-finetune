#!/usr/bin/env python3
"""Extract line-level crops from Manga109-s dialogue blocks.

Manga109 `<text>` annotations are *block* boxes (a whole multi-column bubble)
with the full transcript, while the recogniser needs single lines.  This script
detects lines inside each page, assigns them to the block that contains them,
filters furigana, orders them in manga reading order, and writes one crop per
line plus a manifest carrying the block transcript.

`label_lines.py` (teacher venv) then splits the transcript across the lines and
validates it, producing the final labels.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from db_det import DBDetector, crop_line  # noqa: E402


def parse_blocks(xml_path: Path) -> dict[int, list[dict]]:
    pages: dict[int, list[dict]] = {}
    for page in ET.parse(xml_path).getroot().iter("page"):
        idx = int(page.get("index", 0))
        blocks = []
        for el in page:
            if el.tag != "text":
                continue
            try:
                x0, y0 = float(el.get("xmin")), float(el.get("ymin"))
                x1, y1 = float(el.get("xmax")), float(el.get("ymax"))
            except (TypeError, ValueError):
                continue
            text = "".join((el.text or "").split())
            if not text or x1 <= x0 or y1 <= y0:
                continue
            blocks.append({"id": el.get("id"), "box": (x0, y0, x1, y1), "text": text})
        if blocks:
            pages[idx] = blocks
    return pages


def assign(lines, blocks):
    """Assign each detected line bbox to the smallest block that contains its centre."""
    out = {i: [] for i in range(len(blocks))}
    for quad, score in lines:
        bx0, by0 = quad[:, 0].min(), quad[:, 1].min()
        bx1, by1 = quad[:, 0].max(), quad[:, 1].max()
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        best, best_area = -1, 1e18
        for i, b in enumerate(blocks):
            x0, y0, x1, y1 = b["box"]
            if x0 - 2 <= cx <= x1 + 2 and y0 - 2 <= cy <= y1 + 2:
                area = (x1 - x0) * (y1 - y0)
                if area < best_area:
                    best, best_area = i, area
        if best >= 0:
            out[best].append((quad, score))
    return out


def reading_order(items, vertical: bool):
    if vertical:
        return sorted(items, key=lambda t: (-round(t[0][:, 0].mean() / 12), t[0][:, 1].mean()))
    return sorted(items, key=lambda t: (round(t[0][:, 1].mean() / 12), t[0][:, 0].mean()))


def process_book(args):
    book, xml_path, img_dir, out_dir, split, det_path = args
    det = DBDetector(det_path)
    pages = parse_blocks(xml_path)
    rows = []
    for page_idx, blocks in sorted(pages.items()):
        img_path = img_dir / book / f"{page_idx:03d}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        lines = det(img)
        if not lines:
            continue
        per_block = assign(lines, blocks)
        for bi, items in per_block.items():
            if not items:
                continue
            block = blocks[bi]
            x0, y0, x1, y1 = block["box"]
            vertical = (y1 - y0) >= (x1 - x0)
            ordered = reading_order(items, vertical)

            # furigana: much shorter than the block's median line
            longs = [max(float(q[:, 0].max() - q[:, 0].min()), float(q[:, 1].max() - q[:, 1].min())) for q, _ in ordered]
            med = float(np.median(longs))
            if len(ordered) >= 2:
                ordered = [t for t, L in zip(ordered, longs) if L >= 0.55 * med]
            if not ordered:
                continue

            for li, (quad, score) in enumerate(ordered):
                h = max(np.linalg.norm(quad[0] - quad[3]), np.linalg.norm(quad[1] - quad[2]))
                w = max(np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[2] - quad[3]))
                rot = int(h > w * 1.2)
                crop = crop_line(img, quad, pad=0.04)
                if crop is None or min(crop.shape[:2]) < 8:
                    continue
                if crop.shape[1] / crop.shape[0] > 10:
                    continue
                bdir = out_dir / book
                bdir.mkdir(parents=True, exist_ok=True)
                name = f"{book}_{page_idx:03d}_{bi:03d}_{li:02d}.jpg"
                cv2.imwrite(str(bdir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append(
                    (
                        str((bdir / name).relative_to(ROOT)),
                        book,
                        page_idx,
                        bi,
                        li,
                        rot,
                        round(float(score), 4),
                        block["text"],
                        split,
                    )
                )
    return rows


def book_split(book: str, n_val: int = 9, n_test: int = 9) -> str:
    h = int(hashlib.md5(book.encode()).hexdigest(), 16) % 100
    return "val" if h < n_val else "test" if h < n_val + n_test else "train"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manga109", type=Path, default=ROOT / "data/raw/Manga109-s")
    ap.add_argument("--out", type=Path, default=ROOT / "data/real_lines")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit-books", type=int, default=0)
    args = ap.parse_args()

    args.out = args.out.resolve()
    args.manga109 = args.manga109.resolve()
    root = next(p.parent for p in args.manga109.rglob("annotations") if p.is_dir() and (p.parent / "images").is_dir())
    img_dir, ann_dir = root / "images", root / "annotations"
    books = sorted(ann_dir.glob("*.xml"))
    if args.limit_books:
        books = books[: args.limit_books]
    print(f"{len(books)} books under {root}")

    det_path = str(ROOT / "models/ppocrv6_small_det.onnx")
    tasks = [
        (b.stem, b, img_dir, args.out, book_split(b.stem), det_path) for b in books
    ]
    allrows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, rows in enumerate(ex.map(process_book, tasks), 1):
            allrows.extend(rows)
            print(f"[{i}/{len(tasks)}] {len(rows)} lines (total {len(allrows)})", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "lines.tsv", "w", encoding="utf-8") as f:
        for r in allrows:
            f.write("\t".join(map(str, r)) + "\n")
    print(f"total {len(allrows)} -> {args.out/'lines.tsv'}")


if __name__ == "__main__":
    main()
