#!/usr/bin/env python3
"""Render manga-style single-line training images for the PP-OCRv6 recogniser.

Two layouts, matching what the runtime feeds the recogniser:
  * horizontal  -> upright glyphs, left-to-right
  * vertical    -> upright glyphs stacked top-to-bottom, then rotated 90 deg CCW
                   (this is exactly what a vertical manga column becomes after
                   the engine rotates the crop; verified against the base model)

Everything is rendered from OFL fonts, so the generated images are freely
redistributable and contain no third-party artwork.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import load_dict  # noqa: E402

FONT_DIR = ROOT / "fonts"


def load_fonts(
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    font_dir: Path | None = None,
) -> list[Path]:
    font_dir = font_dir or FONT_DIR
    fonts = sorted(font_dir.glob("*.ttf")) + sorted(font_dir.glob("*.otf"))
    if include:
        fonts = [f for f in fonts if any(s in f.stem for s in include)]
    if exclude:
        fonts = [f for f in fonts if not any(s in f.stem for s in exclude)]
    return fonts


def load_vocab() -> set[str]:
    return set(load_dict())


def pick_text(rng: random.Random, corpus: list[str], onomatopoeia: list[str]) -> tuple[str, bool]:
    if rng.random() < 0.30:
        txt = rng.choice(onomatopoeia)
        return txt, True
    txt = rng.choice(corpus)
    # trim to a manga-like line length
    n = min(len(txt), rng.choice([3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]))
    start = rng.randint(0, max(0, len(txt) - n))
    return txt[start : start + n].strip(), False


def render_line(rng, text, font_path, vertical, vocab) -> tuple[np.ndarray, bool] | None:
    size = rng.randint(40, 76)
    try:
        font = ImageFont.truetype(str(font_path), size)
    except Exception:
        return None
    chars = [c for c in text if c == " " or c in vocab]
    if len(chars) < 1:
        return None

    fill = rng.choices(
        ["black", "white", "color"], weights=[0.78, 0.12, 0.10]
    )[0]
    if fill == "black":
        color = (0, 0, 0)
        stroke = rng.choice([0, 0, 1, 2, 3])
        stroke_color = (255, 255, 255)
    elif fill == "white":
        color = (255, 255, 255)
        stroke = rng.choice([2, 3, 4, 5])
        stroke_color = (0, 0, 0)
    else:
        color = rng.choice([(200, 30, 30), (30, 60, 180), (20, 120, 60), (140, 40, 160)])
        stroke = rng.choice([0, 1, 2, 3])
        stroke_color = (255, 255, 255)

    pad = size // 3 + stroke * 2
    text_s = "".join(chars)

    if not vertical:
        w = int(font.getlength(text_s)) + pad * 2
        h = size + pad * 2
        if w > 1600 or h < 16:
            return None
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.text(
            (pad, pad),
            text_s,
            font=font,
            fill=color + (255,),
            stroke_width=stroke,
            stroke_fill=stroke_color + (255,),
        )
    else:
        step = int(size * rng.uniform(0.98, 1.12))
        w = size + pad * 2
        h = step * len(text_s) + pad * 2
        if h > 1600 or h <= w:
            return None
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        for i, ch in enumerate(text_s):
            d.text(
                (pad, pad + i * step),
                ch,
                font=font,
                fill=color + (255,),
                stroke_width=stroke,
                stroke_fill=stroke_color + (255,),
            )

    arr = np.array(im)
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0:
        return None
    arr = arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    if min(arr.shape[:2]) < 6:
        return None

    if vertical:
        # upright column -> what the engine passes to the recogniser
        arr = np.rot90(arr, k=1).copy()
    return arr, fill == "white"


def make_background(rng, w, h, light_text: bool) -> np.ndarray:
    kind = rng.choices(["bubble", "paper", "tone", "dark"], weights=[0.62, 0.20, 0.12, 0.06])[0]
    if kind == "dark" and not light_text:
        kind = "bubble"
    if kind in ("bubble", "paper"):
        base = rng.randint(238, 255) if kind == "bubble" else rng.randint(225, 245)
        bg = np.full((h, w, 3), base, np.uint8)
        if kind == "paper":
            nprng = np.random.default_rng(rng.randrange(1 << 30))
            bg = np.clip(
                bg.astype(np.int16) + nprng.integers(-12, 12, (h, w, 1), dtype=np.int16), 0, 255
            ).astype(np.uint8)
    elif kind == "tone":
        bg = np.full((h, w, 3), 245, np.uint8)
        r = max(1, int(min(h, w) / rng.uniform(25, 60)))
        for y in range(0, h, r * 3):
            for x in range(0, w, r * 3):
                if rng.random() < 0.7:
                    cv2.circle(bg, (x, y), r, (rng.randint(120, 200),) * 3, -1)
    else:
        bg = np.full((h, w, 3), rng.randint(15, 70), np.uint8)
        for _ in range(rng.randint(3, 10)):
            c = rng.randint(50, 190)
            p1 = (rng.randint(0, w), rng.randint(0, h))
            p2 = (rng.randint(0, w), rng.randint(0, h))
            cv2.line(bg, p1, p2, (c, c, c), rng.randint(1, 4))
    return bg


def compose(rng, line: np.ndarray, light_text: bool) -> np.ndarray:
    lh, lw = line.shape[:2]
    mx = int(lw * rng.uniform(0.04, 0.16)) + 2
    my = int(lh * rng.uniform(0.08, 0.30)) + 2
    w, h = lw + mx * 2, lh + my * 2
    if max(w, h) > 1600:
        return None
    bg = make_background(rng, w, h, light_text)

    if rng.random() < 0.35:
        # a sliver of the bubble outline / neighbouring art at one edge
        c = rng.randint(0, 90)
        if rng.random() < 0.5:
            cv2.line(bg, (rng.randint(0, w), 0), (rng.randint(0, w), rng.randint(0, h // 4)), (c,) * 3, rng.randint(1, 3))
        else:
            cv2.line(bg, (0, rng.randint(0, h)), (rng.randint(0, w // 4), rng.randint(0, h)), (c,) * 3, rng.randint(1, 3))

    alpha = (line[:, :, 3:4].astype(np.float32) / 255.0)
    patch = line[:, :, :3].astype(np.float32)
    roi = bg[my : my + lh, mx : mx + lw].astype(np.float32)
    bg[my : my + lh, mx : mx + lw] = (roi * (1 - alpha) + patch * alpha).astype(np.uint8)
    return bg


def augment(rng, img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if rng.random() < 0.12:
        # mild perspective: a curved SFX unwrapped by the detector looks like this
        d = 0.04
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + np.float32(
            [
                [rng.uniform(-d, d) * w, rng.uniform(-d, d) * h],
                [rng.uniform(-d, d) * w, rng.uniform(-d, d) * h],
                [rng.uniform(-d, d) * w, rng.uniform(-d, d) * h],
                [rng.uniform(-d, d) * w, rng.uniform(-d, d) * h],
            ]
        )
        M = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.2:
        # condensed / stretched lettering
        sx = rng.uniform(0.8, 1.2)
        img = cv2.resize(img, (max(2, int(w * sx)), h), interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]
    if rng.random() < 0.12:
        # bolder / thinner strokes
        k = np.ones((rng.choice([2, 3]), rng.choice([2, 3])), np.uint8)
        img = cv2.erode(img, k) if rng.random() < 0.5 else cv2.dilate(img, k)
    if rng.random() < 0.8:
        ang = rng.uniform(-2.5, 2.5)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.3, 1.1))
    if rng.random() < 0.35:
        nprng = np.random.default_rng(rng.randrange(1 << 30))
        noise = nprng.normal(0, rng.uniform(2, 8), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        a = rng.uniform(0.85, 1.2)
        b = rng.uniform(-15, 15)
        img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    return img


def worker(job):
    idx, n, seed = job
    rng = random.Random(seed)
    fonts = FONTS
    corpus = CORPUS
    ono = ONO
    vocab = VOCAB
    produced = 0
    attempts = 0
    out_dir = OUT_DIR / f"{idx // 10000:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    while produced < n and attempts < n * 4:
        attempts += 1
        text, is_ono = pick_text(rng, corpus, ono)
        vertical = (not HORIZONTAL_ONLY) and rng.random() < 0.62
        font_path = rng.choice(fonts)
        rendered = render_line(rng, text, font_path, vertical, vocab)
        if rendered is None:
            continue
        line, light_text = rendered
        img = compose(rng, line, light_text=light_text)
        if img is None:
            continue
        img = augment(rng, img)
        name = f"{idx:07d}_{produced:05d}.jpg"
        cv2.imwrite(str(out_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(78, 96)])
        split = "val" if rng.random() < VAL_P else "train"
        rows.append((str((out_dir / name).relative_to(ROOT)), text, "onomatopoeia" if is_ono else "synthetic", split))
        produced += 1
    return rows


CORPUS: list[str] = []
ONO: list[str] = []
VOCAB: set[str] = set()
FONTS: list[Path] = []
VAL_P = 0.01
HORIZONTAL_ONLY = False
OUT_DIR = ROOT / "data" / "synth" / "img"


def main() -> None:
    global CORPUS, ONO, VOCAB, OUT_DIR, FONTS, VAL_P, HORIZONTAL_ONLY
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120_000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "synth")
    ap.add_argument("--corpus-dir", type=Path, default=ROOT / "data" / "synth")
    ap.add_argument("--include-fonts", default="", help="comma-separated font-name substrings")
    ap.add_argument("--exclude-fonts", default="", help="comma-separated font-name substrings")
    ap.add_argument("--val-p", type=float, default=0.01)
    ap.add_argument("--font-dir", type=Path, default=None)
    ap.add_argument("--horizontal-only", action="store_true")
    args = ap.parse_args()

    CORPUS = (args.corpus_dir / "corpus.txt").read_text(encoding="utf-8").splitlines()
    ONO = (args.corpus_dir / "onomatopoeia.txt").read_text(encoding="utf-8").splitlines()
    VOCAB = load_vocab()
    global HORIZONTAL_ONLY
    FONTS = load_fonts(
        [s for s in args.include_fonts.split(",") if s],
        [s for s in args.exclude_fonts.split(",") if s],
        args.font_dir,
    )
    HORIZONTAL_ONLY = args.horizontal_only
    VAL_P = args.val_p
    OUT_DIR = (args.out / "img").resolve()
    print(f"corpus={len(CORPUS)} onomatopoeia={len(ONO)} fonts={[f.stem for f in FONTS]}")

    per = args.n // args.workers
    jobs = [(i * per, per if i < args.workers - 1 else args.n - per * (args.workers - 1), 1000 + i) for i in range(args.workers)]

    all_rows = []
    with Pool(args.workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(worker, jobs), 1):
            all_rows.extend(rows)
            print(f"worker {i}/{args.workers}: {len(rows)}", flush=True)

    with open(args.out / "labels.tsv", "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write("\t".join(r) + "\n")
    print(f"total {len(all_rows)} -> {args.out/'labels.tsv'}")


if __name__ == "__main__":
    main()
