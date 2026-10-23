#!/usr/bin/env python3
"""Self-test for build_real_dataset.py without the gated Manga109 download.

Builds a tiny synthetic "book" (page image + Manga109-style XML + COO-style
XML) and runs the real parser/cropper over it, then reports what it produced.
"""

from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "data" / "selftest"
FONT = ROOT / "fonts" / "ZenMaruGothic-Regular.ttf"


def draw_vertical(d, x, y, text, size=64) -> tuple[int, int, int, int]:
    font = ImageFont.truetype(str(FONT), size)
    for i, ch in enumerate(text):
        d.text((x, y + i * size), ch, font=font, fill="black")
    return x, y, x + size, y + size * len(text)


def draw_horizontal(d, x, y, text, size=64) -> tuple[int, int, int, int]:
    font = ImageFont.truetype(str(FONT), size)
    d.text((x, y), text, font=font, fill="black")
    return d.textbbox((x, y), text, font=font)


def box_attrs(x0, y0, x1, y1) -> dict[str, str]:
    return {
        "x0": str(x0), "y0": str(y0),
        "x1": str(x1), "y1": str(y0),
        "x2": str(x1), "y2": str(y1),
        "x3": str(x0), "y3": str(y1),
    }


def main() -> None:
    page_dir = TMP / "Manga109s" / "images" / "TestBook"
    ann_dir = TMP / "Manga109s" / "annotations"
    coo_dir = TMP / "COO" / "TestBook"
    for p in (page_dir, ann_dir, coo_dir):
        p.mkdir(parents=True, exist_ok=True)

    page = Image.new("RGB", (1200, 1600), "white")
    d = ImageDraw.Draw(page)
    dialogue = [
        ("これはテストです", draw_vertical(d, 900, 100, "これはテストです")),
        ("マンガの縦書き", draw_vertical(d, 780, 150, "マンガの縦書き")),
    ]
    sfx = ("ドドド", draw_horizontal(d, 100, 1200, "ドドド"))
    page.save(page_dir / "000.jpg")

    root = ET.Element("book", title="TestBook")
    pages = ET.SubElement(root, "pages")
    pg = ET.SubElement(pages, "page", index="0", width="1200", height="1600")
    for i, (text, b) in enumerate(dialogue):
        ET.SubElement(pg, "text", id=f"{i:08x}", **box_attrs(*b)).text = text
    ET.ElementTree(root).write(ann_dir / "TestBook.xml", encoding="utf-8")

    coo = ET.Element("book", title="TestBook")
    cpages = ET.SubElement(coo, "pages")
    cpg = ET.SubElement(cpages, "page", index="0", width="1200", height="1600")
    ET.SubElement(cpg, "onomatopoeia", id="10000000", **box_attrs(*sfx[1])).text = sfx[0]
    ET.ElementTree(coo).write(coo_dir / "TestBook.xml", encoding="utf-8")

    out = TMP / "real"
    cmd = [
        sys.executable,
        str(ROOT / "scripts/build_real_dataset.py"),
        "--manga109", str(TMP / "Manga109s"),
        "--coo", str(TMP / "COO"),
        "--out", str(out),
        "--workers", "1",
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    rows = (out / "labels.tsv").read_text(encoding="utf-8").splitlines()
    print(f"\n{len(rows)} crops:")
    for r in rows:
        print(" ", r)

    ims = [Image.open(ROOT / r.split("\t")[0]) for r in rows]
    W = max(i.width for i in ims) + 20
    H = sum(i.height + 30 for i in ims) + 20
    sheet = Image.new("RGB", (W, H), "#dddddd")
    y = 10
    for im in ims:
        sheet.paste(im, (10, y))
        y += im.height + 30
    sheet.save(TMP / "crops.png")
    print(f"sheet: {TMP/'crops.png'}")


if __name__ == "__main__":
    main()
