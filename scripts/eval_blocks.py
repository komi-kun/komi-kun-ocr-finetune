#!/usr/bin/env python3
"""Page-level evaluation on held-out Manga109-s books.

Runs the same det -> crop -> rec path the app uses, then compares the
concatenated per-block predictions against the block transcript.  This avoids
both per-line boundary assumptions and teacher-based selection bias, so it is the
honest "does this help on real manga" number.

Usage:
  python scripts/eval_blocks.py --model models/ppocrv6_small_rec.onnx --split test
  python scripts/eval_blocks.py --model models/ppocrv6_small_rec_manga_real.onnx --split test \
      --det models/ppocrv6_small_det_manga.onnx --json-out output/eval_blocks/cand_test.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import RecModel  # noqa: E402
from db_det import DBDetector, crop_line  # noqa: E402
from extract_page_lines import assign, parse_blocks, reading_order  # noqa: E402


def book_split(book: str, n_val: int = 9, n_test: int = 9) -> str:
    h = int(hashlib.md5(book.encode()).hexdigest(), 16) % 100
    return "val" if h < n_val else "test" if h < n_val + n_test else "train"


def norm(s: str) -> str:
    return "".join(s.split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--det", type=Path, default=ROOT / "models/ppocrv6_small_det.onnx",
                    help="detector ONNX (default: the shipped stock detector)")
    ap.add_argument("--manga109", type=Path, default=ROOT / "data/raw/Manga109-s")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit-pages", type=int, default=0)
    ap.add_argument("--show", type=int, default=5)
    ap.add_argument("--json-out", type=Path, default=None,
                    help="per-block predictions (for paired stock-vs-candidate deltas)")
    args = ap.parse_args()

    root = next(p.parent for p in args.manga109.rglob("annotations") if p.is_dir() and (p.parent / "images").is_dir())
    img_dir, ann_dir = root / "images", root / "annotations"
    books = [b for b in sorted(ann_dir.glob("*.xml")) if book_split(b.stem) == args.split]
    det = DBDetector(str(args.det))
    rec = RecModel(args.model)

    records = []
    total_cer, total_n, exact, blocks_done = 0.0, 0, 0, 0
    total_gt_blocks, missing_blocks = 0, 0
    worst = []
    pages = 0
    for book in books:
        pages_data = parse_blocks(book)
        for page_idx, blines in sorted(pages_data.items()):
            img = cv2.imread(str(img_dir / book.stem / f"{page_idx:03d}.jpg"))
            if img is None:
                continue
            lines = det(img)
            pages += 1
            per_block = assign(lines, blines) if lines else {}
            for bi, block in enumerate(blines):
                gt = norm(block["text"])
                if not gt:
                    continue
                total_gt_blocks += 1
                items = per_block.get(bi, []) if per_block else []
                if not items:
                    missing_blocks += 1
                    continue
                x0, y0, x1, y1 = block["box"]
                ordered = reading_order(items, (y1 - y0) >= (x1 - x0))
                pred = ""
                for quad, _ in ordered:
                    crop = crop_line(img, quad, pad=0.04)
                    if crop is None:
                        continue
                    text, _ = rec(crop)
                    pred += text
                pred = norm(pred)
                d = Levenshtein.distance(pred, gt)
                total_cer += d / len(gt)
                total_n += 1
                exact += int(pred == gt)
                blocks_done += 1
                worst.append((d / len(gt), gt, pred))
                if args.json_out:
                    records.append(
                        {
                            "book": book.stem,
                            "page": page_idx,
                            "block": bi,
                            "gt": gt,
                            "pred": pred,
                            "cer": d / len(gt),
                            "exact": pred == gt,
                        }
                    )
            if args.limit_pages and pages >= args.limit_pages:
                break
        print(f"  {book.stem}: blocks={blocks_done} pages={pages}", flush=True)
        if args.limit_pages and pages >= args.limit_pages:
            break

    print(f"\nmodel {args.model}")
    print(f"det   {args.det}")
    print(f"books={len(books)} pages={pages} blocks={blocks_done}")
    print(f"block CER {100*total_cer/max(1,total_n):.2f} %   exact {100*exact/max(1,total_n):.2f} %")
    print(f"blocks with GT text {total_gt_blocks}  evaluated {blocks_done}  "
          f"no det lines {missing_blocks} ({100*missing_blocks/max(1,total_gt_blocks):.2f} % of blocks unread)")
    worst.sort(reverse=True)
    for c, gt, pred in worst[: args.show]:
        print(f"  cer={c:.2f}\n    gt  ={gt}\n    pred={pred}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {
                    "rec_model": str(args.model),
                    "det_model": str(args.det),
                    "rec_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
                    "det_sha256": hashlib.sha256(args.det.read_bytes()).hexdigest(),
                    "split": args.split,
                    "books": [b.stem for b in books],
                    "pages": pages,
                    "blocks": blocks_done,
                    "blocks_with_gt": total_gt_blocks,
                    "blocks_missing": missing_blocks,
                    "block_cer": total_cer / max(1, total_n),
                    "block_exact": exact / max(1, total_n),
                    "records": records,
                },
                ensure_ascii=False,
            )
        )
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
