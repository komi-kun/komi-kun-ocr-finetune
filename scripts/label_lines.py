#!/usr/bin/env python3
"""Teacher-label the Manga109-s line crops and split each block transcript.

Runs in `.venv-teacher` (torch + manga-ocr). For every block:

  1. the teacher reads each detected line;
  2. the block transcript is split across the lines proportionally to the
     teacher's lengths (so boundaries follow the actual layout);
  3. a line is kept only if the teacher agrees with its transcript slice,
     which filters furigana leftovers, over-detected art and bad crops.

Output: data/real/labels.tsv (path, text, source, split) + a rejection report.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rapidfuzz.distance import Levenshtein  # noqa: E402


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("…", "...").replace("‥", "..")
    s = re.sub(r"\.{3,}", "...", s)
    return "".join(s.split())


def split_transcript(gt: str, lengths: list[int]) -> list[str]:
    """Deprecated proportional splitter; kept for reference."""
    return _split_proportional(gt, lengths)


def align_segments(teacher_texts: list[str], gt: str) -> list[str]:
    """Split `gt` across lines using a character alignment with the teacher output.

    Proportional splitting drifts by a character whenever a line is misread; an
    alignment pins boundaries to actual matched characters instead.
    """
    concat = "".join(teacher_texts)
    n, m = len(concat), len(gt)
    out = [""] * len(teacher_texts)
    if n == 0 or m == 0:
        return out

    boundary = [0] * (n + 1)
    sm = difflib.SequenceMatcher(None, concat, gt, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        li, lj = i2 - i1, j2 - j1
        if tag == "equal":
            for k in range(li):
                boundary[i1 + k] = j1 + k
        else:
            for k in range(li + 1):
                boundary[i1 + k] = j1 + (k * lj // li if li else 0)
    boundary[n] = m
    for i in range(1, n + 1):
        boundary[i] = max(boundary[i], boundary[i - 1])

    pos = 0
    for idx, txt in enumerate(teacher_texts):
        start = boundary[pos]
        pos += len(txt)
        end = boundary[min(pos, n)]
        out[idx] = gt[start:end]
    return out


def _split_proportional(gt: str, lengths: list[int]) -> list[str]:
    n = len(lengths)
    if n == 0:
        return []
    if len(gt) < n:
        return [""] * n
    total = sum(max(1, L) for L in lengths)
    out, prev, acc = [], 0, 0
    for i, L in enumerate(lengths):
        acc += max(1, L)
        if i == n - 1:
            end = len(gt)
        else:
            end = int(round(len(gt) * acc / total))
            end = max(prev + 1, min(end, len(gt) - (n - 1 - i)))
        out.append(gt[prev:end])
        prev = end
    return out


def load_crop(path: Path, rot: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    if rot:
        im = im.transpose(Image.Transpose.ROTATE_270)  # undo the student-side CCW rotation
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", type=Path, default=ROOT / "data/real_lines/lines.tsv")
    ap.add_argument("--out", type=Path, default=ROOT / "data/real")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-cer", type=float, default=0.40)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from manga_ocr import MangaOcr
    from manga_ocr.ocr import post_process

    rows = [l.rstrip("\n").split("\t") for l in args.lines.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    groups = defaultdict(list)
    for r in rows:
        if len(r) != 9:
            continue
        path, book, page, bi, li, rot, score, text, split = r
        groups[(book, int(page), int(bi))].append((int(li), path, int(rot), text, split))
    for g in groups.values():
        g.sort()
    print(f"{len(rows)} lines in {len(groups)} blocks")

    m = MangaOcr()
    dev = m.model.device
    tok = m.tokenizer

    def run_batch(images):
        pix = [m._preprocess(im) for im in images]
        x = torch.stack(pix).to(dev)
        with torch.no_grad():
            out = m.model.generate(x, max_length=300)
        return [post_process(t) for t in tok.batch_decode(out.cpu(), skip_special_tokens=True)]

    accepted, rejected = [], {"disagree": 0, "empty": 0, "short": 0}
    keys = list(groups)
    for gi, key in enumerate(keys):
        items = groups[key]
        imgs = [load_crop(ROOT / it[1], it[2]) for it in items]
        texts = []
        for i in range(0, len(imgs), args.batch):
            texts.extend(run_batch(imgs[i : i + args.batch]))
        segs = align_segments(texts, items[0][3])
        for (li, path, rot, gt, split), hyp, seg in zip(items, texts, segs):
            if not seg:
                rejected["short"] += 1
                continue
            if not norm(hyp):
                rejected["empty"] += 1
                continue
            cer = Levenshtein.distance(norm(hyp), norm(seg)) / max(1, len(norm(seg)))
            if cer > args.max_cer:
                rejected["disagree"] += 1
                continue
            accepted.append((path, seg, "dialogue", split))
        if (gi + 1) % 200 == 0:
            print(f"  {gi+1}/{len(keys)} blocks, {len(accepted)} kept", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "labels.tsv", "w", encoding="utf-8") as f:
        for p, t, s, sp in accepted:
            f.write(f"{p}\t{t}\t{s}\t{sp}\n")
    print(f"kept {len(accepted)} / {len(rows)}  {rejected}")


if __name__ == "__main__":
    main()
