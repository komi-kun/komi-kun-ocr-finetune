#!/usr/bin/env python3
"""End-to-end COO onomatopoeia (SFX) recognition on held-out Manga109-s books.

`eval_blocks.py` measures Manga109 ``<text>`` **dialogue** blocks, which is not
what the detector fine-tune was trained for: it targets COO onomatopoeia, and in
particular the curved (5-66 point) ones.  This script closes that gap by running
the full runtime path on the COO SFX polygons themselves:

    page image -> detector -> predicted quads -> crop_line() -> recogniser

and scoring each COO ground-truth SFX against its transcription:

* **missed** (no predicted quad reaches IoU 0.4 with the polygon): the
  transcription is scored as a full deletion, i.e. ``len(gt)`` edits, which is
  what a reader sees - the onomatopoeia is simply not read.  This is the metric
  that punishes a detector which boxes the curve as several fragments and a
  recogniser that cannot read either fragment.
* **matched**: CER of the recognised text of the *detected* crop - the runtime
  quadrilateral, not the GT polygon - against the GT transcription.

Both numbers are reported, because they answer different questions: the overall
CER is the end-to-end user-visible quality (det + rec), the matched-only CER is
the recogniser quality *given* that the SFX was found.  ``exact`` counts SFX read
character-for-character correctly, over all GT (an unmatched SFX is never
exact).  Every figure is also broken down by GT polygon complexity (4-point,
5-8-point, 9+-point = strongly curved), which is the axis the fine-tune targets.

Matching is the same one-to-one greedy polygon IoU at 0.4 that
``eval_det.py --crop-iou`` uses for its "SFX crop recall @0.4", so
``matched / gt`` here is bit-for-bit that recall and the detector table and this
one describe the same predictions.

Usage
  python scripts/eval_sfx.py --det models/ppocrv6_small_det.onnx \\
      --rec models/ppocrv6_small_rec.onnx --split test \\
      --json-out output/eval_sfx/stockdet_stockrec_test.json
  # paired comparison of two such dumps (same GT keys, so CER is paired)
  python scripts/eval_sfx.py --compare output/eval_sfx/a.json output/eval_sfx/b.json
  python scripts/eval_sfx.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_det_dataset import Config, clean_text, clip_points, is_degenerate, parse_coo  # noqa: E402
from common import RecModel, load_dict  # noqa: E402
from compare_eval_blocks import binom_two_sided  # noqa: E402
from db_det import DBDetector, crop_line  # noqa: E402
from eval_det import greedy_match, iou, point_bucket, to_poly  # noqa: E402

IOU_CROP = 0.4  # identical to eval_det.IOU_CROP ("SFX crop recall @0.4")
CROP_PAD = 0.04  # identical to eval_blocks.py (the runtime selection path)
BUCKETS = ("4", "5-8", "9+")


def book_split(book: str, n_val: int = 9, n_test: int = 9) -> str:
    """Same book-disjoint hash split as the recogniser/detector datasets."""
    h = int(hashlib.md5(book.encode()).hexdigest(), 16) % 100
    return "val" if h < n_val else "test" if h < n_val + n_test else "train"


def norm(s: str) -> str:
    return "".join(s.split())


# ------------------------------------------------------------------------------ GT
def page_gt(coo: dict, page: int, size: tuple[int, int], cfg: Config) -> tuple[list[dict], Counter]:
    """COO SFX ground truth on one page, filtered exactly like ``build_det_dataset``.

    The same ``min_points`` / clip / degeneracy rules the detector GT went through
    are applied here, so the SFX scored by this script are the SFX the detector was
    scored on.  Polygons whose transcription is empty after whitespace
    normalisation are dropped: there is nothing to recognise.
    """
    width, height = size
    out, counters = [], Counter()
    for inst in coo.get(page, []):
        counters["sfx_polygons"] += 1
        if len(inst.points) < cfg.min_points:
            counters["dropped_few_points"] += 1
            continue
        pts, _ = clip_points(inst.points, width, height)
        if is_degenerate(pts, cfg.min_area, cfg.min_extent):
            counters["dropped_degenerate"] += 1
            continue
        poly = to_poly(pts)
        if poly is None:
            counters["dropped_empty"] += 1
            continue
        text = norm(clean_text(inst.text))
        if not text:
            counters["dropped_no_text"] += 1
            continue
        out.append({"poly": poly, "points": len(pts), "bucket": point_bucket(len(pts)), "gt": text})
    counters["sfx"] += len(out)
    return out, counters


# -------------------------------------------------------------------------- scoring
def score_page(gt: list[dict], preds, img: np.ndarray, rec: RecModel, iou_crop: float = IOU_CROP):
    """Match predicted quads to COO SFX, recognise the detected crops, score both.

    Returns ``(records, counters)``; one record per COO SFX GT instance.
    """
    pred_polys = [to_poly(quad) for quad, _ in preds]
    keep = [i for i, p in enumerate(pred_polys) if p is not None]
    pred_polys = [pred_polys[i] for i in keep]
    quads = [np.asarray(preds[i][0], np.float32) for i in keep]

    mat = [[iou(p, g["poly"]) for g in gt] for p in pred_polys]
    matched = {j: i for i, j, _ in greedy_match(mat, iou_crop)}

    records, counters = [], Counter()
    for j, g in enumerate(gt):
        n_gt = len(g["gt"])
        counters["gt"] += 1
        counters[f"gt_{g['bucket']}"] += 1
        counters["gt_chars"] += n_gt
        counters[f"gt_chars_{g['bucket']}"] += n_gt
        pred = ""
        if j in matched:
            crop = crop_line(img, quads[matched[j]], pad=CROP_PAD)
            if crop is not None:
                pred = norm(rec(crop)[0])
        else:
            counters["missed"] += 1
            counters[f"missed_{g['bucket']}"] += 1
        dist = Levenshtein.distance(pred, g["gt"])
        exact = pred == g["gt"] and j in matched
        if j in matched:
            counters["matched"] += 1
            counters[f"matched_{g['bucket']}"] += 1
            counters["matched_chars"] += n_gt
            counters["matched_chars_" + g["bucket"]] += n_gt
            counters["matched_dist"] += dist
            counters["matched_dist_" + g["bucket"]] += dist
            counters["matched_exact"] += int(exact)
            counters["matched_exact_" + g["bucket"]] += int(exact)
        counters["dist"] += dist
        counters["dist_" + g["bucket"]] += dist
        counters["exact"] += int(exact)
        counters["exact_" + g["bucket"]] += int(exact)
        records.append(
            {
                "book": None,  # filled in by run()
                "page": None,
                "sfx": j,
                "bucket": g["bucket"],
                "points": g["points"],
                "gt": g["gt"],
                "pred": pred,
                "matched": j in matched,
                "dist": dist,
                "cer": dist / n_gt,
                "exact": exact,
            }
        )
    counters["preds"] = len(preds)
    return records, counters


def summarise(records: list[dict], counters: Counter) -> dict:
    def ratio(a: str, b: str) -> float:
        return counters[a] / counters[b] if counters[b] else 0.0

    by_bucket = {}
    for b in BUCKETS:
        by_bucket[b] = {
            "gt": counters[f"gt_{b}"],
            "matched": counters[f"matched_{b}"],
            "missed": counters[f"missed_{b}"],
            "recall": ratio(f"matched_{b}", f"gt_{b}"),
            "cer": ratio(f"dist_{b}", f"gt_chars_{b}"),
            "matched_cer": ratio(f"matched_dist_{b}", f"matched_chars_{b}"),
            "exact": counters[f"exact_{b}"],
            "exact_rate": ratio(f"exact_{b}", f"gt_{b}"),
        }
    return {
        "gt": counters["gt"],
        "gt_chars": counters["gt_chars"],
        "matched": counters["matched"],
        "missed": counters["missed"],
        "recall": ratio("matched", "gt"),
        "cer": ratio("dist", "gt_chars"),
        "matched_cer": ratio("matched_dist", "matched_chars"),
        "exact": counters["exact"],
        "exact_rate": ratio("exact", "gt"),
        "matched_exact": counters["matched_exact"],
        "matched_exact_rate": ratio("matched_exact", "matched"),
        "preds": counters["preds"],
        "by_bucket": by_bucket,
    }


# ------------------------------------------------------------------------------ run
def find_root(manga109: Path) -> Path:
    for p in manga109.rglob("annotations"):
        if p.is_dir() and (p.parent / "images").is_dir() and (p.parent / "annotations_COO").is_dir():
            return p.parent
    raise SystemExit(f"no Manga109-s release with annotations_COO/ under {manga109}")


def load_det_cache(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def run(args) -> dict:
    root = find_root(args.manga109)
    img_dir, ann_dir, coo_dir = root / "images", root / "annotations", root / "annotations_COO"
    books = [b for b in sorted(ann_dir.glob("*.xml")) if book_split(b.stem) == args.split]
    if args.book:
        books = [b for b in books if b.stem in set(args.book)]
    cfg = Config()
    det_sha = hashlib.sha256(args.det.read_bytes()).hexdigest()
    rec_sha = hashlib.sha256(args.rec.read_bytes()).hexdigest()

    cache_path = None
    cache = {}
    if args.det_cache:
        args.det_cache.mkdir(parents=True, exist_ok=True)
        cache_path = args.det_cache / f"{det_sha[:8]}_{args.split}.json"
        cache = load_det_cache(cache_path)

    det = DBDetector(str(args.det))
    rec = RecModel(args.rec)
    vocab = set(load_dict())

    records, counters, pages = [], Counter(), 0
    for book in books:
        xml = coo_dir / f"{book.stem}.xml"
        if not xml.exists():
            print(f"  {book.stem}: no COO annotations, skipped", flush=True)
            continue
        coo = parse_coo(xml, Counter())
        for page_idx in sorted(coo):
            img = cv2.imread(str(img_dir / book.stem / f"{page_idx:03d}.jpg"))
            if img is None:
                continue
            pages += 1
            key = f"{book.stem}/{page_idx}"
            gt, page_counters = page_gt(coo, page_idx, (img.shape[1], img.shape[0]), cfg)
            for k, v in page_counters.items():
                counters[k] += v
            for g in gt:
                counters["gt_oov_char"] += int(any(c not in vocab for c in g["gt"]))
            if not gt:
                continue
            if key in cache:
                quads = [np.asarray(q, np.float32) for q in cache[key]]
                preds = [(q, 1.0) for q in quads]
            else:
                preds = det(img)
                if cache_path is not None:
                    cache[key] = [[[float(x), float(y)] for x, y in quad] for quad, _ in preds]
            page_records, page_score = score_page(gt, preds, img, rec, args.iou_crop)
            counters.update(page_score)
            for r in page_records:
                r["book"], r["page"] = book.stem, page_idx
            records.extend(page_records)
            if args.limit_pages and pages >= args.limit_pages:
                break
        print(f"  {book.stem}: pages={pages} sfx={counters['gt']}", flush=True)
        if args.limit_pages and pages >= args.limit_pages:
            break

    if cache_path is not None:
        cache_path.write_text(json.dumps(cache))

    summary = summarise(records, counters)
    summary.update(
        {
            "det_model": str(args.det),
            "rec_model": str(args.rec),
            "det_sha256": det_sha,
            "rec_sha256": rec_sha,
            "split": args.split,
            "books": [b.stem for b in books],
            "pages": pages,
            "gt_oov_char": counters["gt_oov_char"],
            "iou_crop": args.iou_crop,
            "crop_pad": CROP_PAD,
            "records": records,
        }
    )
    return summary


def report(s: dict) -> None:
    print(f"\ndet   {s['det_model']}  {s['det_sha256'][:8]}")
    print(f"rec   {s['rec_model']}  {s['rec_sha256'][:8]}")
    print(f"books={len(s['books'])} pages={s['pages']} COO SFX={s['gt']} ({s['gt_chars']} chars)")
    print(
        f"found {s['matched']} / {s['gt']} ({100 * s['recall']:.2f} %), missed {s['missed']} "
        f"(scored as full deletions)"
    )
    print(
        f"end-to-end SFX CER {100 * s['cer']:.2f} %   exact {s['exact']} / {s['gt']} "
        f"({100 * s['exact_rate']:.2f} %)"
    )
    print(
        f"matched-only SFX CER {100 * s['matched_cer']:.2f} %   exact {s['matched_exact']} / "
        f"{s['matched']} ({100 * s['matched_exact_rate']:.2f} %)"
    )
    print(f"\n{'bucket':8s} {'gt':>6s} {'found':>6s} {'recall':>8s} {'CER all':>9s} "
          f"{'CER matched':>12s} {'exact':>7s}")
    for b in BUCKETS:
        d = s["by_bucket"][b]
        print(
            f"{b:8s} {d['gt']:6d} {d['matched']:6d} {100 * d['recall']:7.2f}% {100 * d['cer']:8.2f}% "
            f"{100 * d['matched_cer']:11.2f}% {d['exact']:7d}"
        )
    if s.get("gt_oov_char"):
        print(f"\n{s['gt_oov_char']} GT SFX contain a character outside the dictionary "
              f"(unreadable by any model here)")


# -------------------------------------------------------------------------- compare
def compare(a: dict, b: dict) -> int:
    ra = {(r["book"], r["page"], r["sfx"]): r for r in a["records"]}
    rb = {(r["book"], r["page"], r["sfx"]): r for r in b["records"]}
    common = sorted(set(ra) & set(rb))

    def cer(keys, d):
        num = sum(d[k]["dist"] for k in keys)
        den = sum(len(d[k]["gt"]) for k in keys)
        return num / den if den else 0.0

    fixed = sum(1 for k in common if not ra[k]["exact"] and rb[k]["exact"])
    broken = sum(1 for k in common if ra[k]["exact"] and not rb[k]["exact"])
    print(f"base det {a['det_sha256'][:8]} rec {a['rec_sha256'][:8]}  ({a['det_model']} + {a['rec_model']})")
    print(f"cand det {b['det_sha256'][:8]} rec {b['rec_sha256'][:8]}  ({b['det_model']} + {b['rec_model']})")
    print(f"SFX base {len(ra)} cand {len(rb)} paired {len(common)}")
    print(f"end-to-end SFX CER base {100 * cer(common, ra):.2f} %  cand {100 * cer(common, rb):.2f} %  "
          f"delta {100 * (cer(common, rb) - cer(common, ra)):+.2f} pp")
    print(f"found base {sum(ra[k]['matched'] for k in common)} cand {sum(rb[k]['matched'] for k in common)}")
    print(f"exact base {sum(ra[k]['exact'] for k in common)} cand {sum(rb[k]['exact'] for k in common)}")
    print(f"exact flips: fixed {fixed}  broken {broken}  "
          f"binomial two-sided p = {binom_two_sided(min(fixed, broken), fixed + broken):.4f}")
    for bucket in BUCKETS:
        keys = [k for k in common if ra[k]["bucket"] == bucket]
        if not keys:
            continue
        print(f"  {bucket:4s} n={len(keys):5d}  CER {100 * cer(keys, ra):6.2f} % -> "
              f"{100 * cer(keys, rb):6.2f} %  ({100 * (cer(keys, rb) - cer(keys, ra)):+.2f} pp)")
    return 0


# ------------------------------------------------------------------------ self-test
def self_test() -> int:
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(("PASS  " if cond else "FAIL  ") + name + (f"  [{detail}]" if detail and not cond else ""))
        ok = ok and bool(cond)

    class FakeRec:
        """Returns a fixed string, so the CER arithmetic is testable without a model."""

        def __init__(self, text: str):
            self.text = text

        def __call__(self, crop):
            return self.text, 1.0

    img = np.zeros((400, 400, 3), np.uint8)
    # a square GT SFX at (0..100, 0..100), 4 points
    gt = [{"poly": to_poly([(0, 0), (100, 0), (100, 100), (0, 100)]), "points": 4,
           "bucket": "4", "gt": "ドン"}]

    # 1. exact match: found, read exactly -> CER 0, exact
    recs, c = score_page(gt, [([(0, 0), (100, 0), (100, 100), (0, 100)], 0.9)], img, FakeRec("ドン"))
    s = summarise(recs, c)
    check("exact crop: matched, CER 0, exact 1", s["matched"] == 1 and s["cer"] == 0.0 and s["exact"] == 1,
          f"{s['matched']} {s['cer']} {s['exact']}")

    # 2. found but misread: matched-only CER counts the edits, no exact
    recs, c = score_page(gt, [([(0, 0), (100, 0), (100, 100), (0, 100)], 0.9)], img, FakeRec("ドソ"))
    s = summarise(recs, c)
    check("misread crop: matched-only CER = 1/2, not exact",
          abs(s["matched_cer"] - 0.5) < 1e-12 and s["exact"] == 0 and s["cer"] == 0.5,
          f"{s['matched_cer']} {s['cer']}")

    # 3. missed SFX is a FULL deletion: CER 1.0, counted in the end-to-end number
    #    but NOT in the matched-only number (nothing was recognised).
    recs, c = score_page(gt, [], img, FakeRec("ドン"))
    s = summarise(recs, c)
    check("missed SFX: CER 1.0 end-to-end", s["missed"] == 1 and s["cer"] == 1.0, f"{s['cer']}")
    check("missed SFX: excluded from matched-only CER", s["matched_cer"] == 0.0 and s["matched"] == 0,
          f"{s['matched_cer']} {s['matched']}")
    check("missed SFX: never exact", s["exact"] == 0)

    # 4. a prediction that only grazes the GT (IoU < 0.4) does not count as found
    recs, c = score_page(gt, [([(80, 0), (180, 0), (180, 100), (80, 100)], 0.9)], img, FakeRec("ドン"))
    s = summarise(recs, c)
    check("IoU 0.111 prediction is not a match", s["matched"] == 0 and s["cer"] == 1.0, f"{s['matched']}")

    # 5. bucket attribution uses the GT polygon's point count, not the prediction's
    curved = [{"poly": to_poly([(0, 0), (50, -5), (100, 0), (105, 50), (100, 100), (50, 105),
                                (0, 100), (-5, 50)]), "points": 8, "bucket": "5-8", "gt": "ゴゴゴ"}]
    recs, c = score_page(curved, [([(0, 0), (100, 0), (100, 100), (0, 100)], 0.9)], img, FakeRec("ゴゴゴ"))
    s = summarise(recs, c)
    check("8-point GT lands in the '5-8' bucket", s["by_bucket"]["5-8"]["gt"] == 1
          and s["by_bucket"]["4"]["gt"] == 0 and s["by_bucket"]["9+"]["gt"] == 0,
          str({b: s["by_bucket"][b]["gt"] for b in BUCKETS}))
    check("the 5-8 matched CER is the one populated", s["by_bucket"]["5-8"]["matched_cer"] == 0.0
          and s["by_bucket"]["5-8"]["exact"] == 1)
    check("bucket CERs do not leak into each other", s["by_bucket"]["9+"]["cer"] == 0.0)

    # 6. one-to-one matching: two predictions on one GT cannot double-count recall
    recs, c = score_page(gt, [([(0, 0), (100, 0), (100, 100), (0, 100)], 0.9),
                              ([(0, 0), (100, 0), (100, 100), (0, 100)], 0.8)], img, FakeRec("ドン"))
    s = summarise(recs, c)
    check("duplicate predictions do not inflate recall",
          s["matched"] == 1 and s["recall"] == 1.0 and s["preds"] == 2, f"{s['matched']} {s['recall']}")

    # 7. per-page GT construction applies the dataset's filters
    class Inst:
        def __init__(self, points, text):
            self.points, self.text = points, text

    coo = {
        0: [
            Inst([(0, 0), (50, 0), (50, 50), (0, 50)], "ドン"),  # kept
            Inst([(0, 0), (50, 0), (50, 50)], "ドン"),  # 3 points -> dropped
            Inst([(0, 0), (50, 0), (0, 1), (1, 1)], "ドン"),  # degenerate -> dropped
            Inst([(0, 0), (50, 0), (50, 50), (0, 50)], "   "),  # no text -> dropped
        ]
    }
    kept, counters = page_gt(coo, 0, (200, 200), Config())
    check("page_gt keeps 1 of 4: 3-point, degenerate and text-less polygons dropped",
          len(kept) == 1 and counters["dropped_few_points"] == 1 and counters["dropped_degenerate"] == 1
          and counters["dropped_no_text"] == 1, f"{len(kept)} {dict(counters)}")
    check("page_gt normalises whitespace out of the transcript", kept[0]["gt"] == "ドン")

    print(f"\n{0 if ok else 1} failure(s)")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--det", type=Path, help="detector ONNX")
    ap.add_argument("--rec", type=Path, help="recogniser ONNX")
    ap.add_argument("--manga109", type=Path, default=ROOT / "data/raw/Manga109-s")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--book", action="append", help="restrict to these books (repeatable)")
    ap.add_argument("--limit-pages", type=int, default=0)
    ap.add_argument("--iou-crop", type=float, default=IOU_CROP, help="match threshold (eval_det uses 0.4)")
    ap.add_argument("--det-cache", type=Path, default=None,
                    help="directory of per-detector quad caches, so a recogniser swap reuses the detection pass")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("BASE", "CAND"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if args.compare:
        return compare(*(json.loads(p.read_text()) for p in args.compare))
    if not args.det or not args.rec:
        ap.error("--det and --rec are required (or use --compare / --self-test)")

    summary = run(args)
    report(summary)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, ensure_ascii=False))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
