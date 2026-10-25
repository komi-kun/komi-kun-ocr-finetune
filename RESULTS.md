# Results

All inference uses `scripts/eval_rec.py` / `scripts/eval_blocks.py`, i.e. the
runtime path (ONNX Runtime, 3×48×W, BGR, [-1,1], CTC greedy, 320 px clamp).
`base` = shipped `ppocrv6_small_rec`, `stage-1` = synthetic-only fine-tune,
`stage-2` = synthetic + real Manga109-s/COO (`ppocrv6_small_rec_manga_real`).

## Released combinations — the four shipping models, head to head

Every number in this document is one model fine-tune or one detector experiment.
This section is the shipping view: **four release tags, four (detector,
recogniser) combinations**, all measured on the same held-out test split. Read it
first; the experiment sections below are the detailed record behind it.

| Release | Detector | Recogniser | What it is |
|---|---|---|---|
| [`ocr-models-v3`](https://github.com/komi-kun/komi-kun-ocr-models/releases/tag/ocr-models-v3) | **stock** `d73e0058…` (9 880 512 B) | **stock** `5435fd74…` (21 159 378 B) | previous **stock** release — the baseline everything is compared to |
| [`ocr-manga-v2`](https://github.com/komi-kun/komi-kun-ocr-models/releases/tag/ocr-manga-v2) | **stock** `d73e0058…` | **fine-tuned** `3e898ae2…` (21 159 418 B) | previous **fine-tune** release (recogniser only) |
| [`ocr-models-v4`](https://github.com/komi-kun/komi-kun-ocr-models/releases/tag/ocr-models-v4) | **fine-tuned** `bf5ed74a…` (9 929 569 B) | **stock** `5435fd74…` | new **stock-rec** release (curved-SFX detector) |
| [`ocr-manga-v3`](https://github.com/komi-kun/komi-kun-ocr-models/releases/tag/ocr-manga-v3) | **fine-tuned** `bf5ed74a…` | **fine-tuned** `3e898ae2…` | new **fine-tune** release (both stages) |

All four live on `komi-kun/komi-kun-ocr-models`; the code host
`komi-kun/komi-kun-ocr-finetune` publishes **no** releases (`gh release list` on
both repos, 2026-10-25). The dictionary (`ppocrv6_small_dict.txt`,
`b5f2bfe2…`) is byte-identical in all four, so the pairs differ in exactly one
model file: `ocr-models-v3` ↔ `ocr-manga-v2` differ only in the **recogniser**,
`ocr-manga-v2` ↔ `ocr-manga-v3` differ only in the **detector**.

Test split for all three tables: 9 books, held out by book, never used for
training or for checkpoint selection.

### 1. Detector — `scripts/eval_det.py`, 9 books / 982 pages

Two detector weights cover the four releases: `ocr-models-v3` and `ocr-manga-v2`
ship the stock detector, `ocr-models-v4` and `ocr-manga-v3` ship the fine-tune.
Polygon IoU (never box IoU), greedy one-to-one at 0.5, app postprocess (960 max
side, 0.15 / 0.25 / 1.40). GT: 6 639 COO onomatopoeia polygons, 12 209 Manga109
`<text>` blocks.

| metric | stock det — `ocr-models-v3`, `ocr-manga-v2` | fine-tuned det — `ocr-models-v4`, `ocr-manga-v3` | Δ |
|---|---|---|---|
| SFX precision / recall / H @0.5 | 49.62 / 24.48 / 32.78 % | **70.50 / 60.36 / 65.03 %** | H +32.25 pp |
| SFX crop recall @0.4 | 32.07 % (2 129/6 639) | **69.91 % (4 641/6 639)** | +37.84 pp |
| — 4-point SFX | 34.80 % (1 057/3 037) | **67.57 % (2 052/3 037)** | +32.77 pp |
| — 5–8-point SFX | 30.43 % (846/2 780) | **71.26 % (1 981/2 780)** | +40.83 pp |
| — 9+-point SFX (curved) | 27.49 % (226/822) | **73.97 % (608/822)** | +46.48 pp |
| dialogue block coverage ≥50 % | 92.10 % (11 245/12 209) | **94.94 % (11 591/12 209)** | +2.84 pp |
| dialogue mean block coverage | 81.78 % | 82.81 % | +1.03 pp |
| dialogue P / R / H @0.5 (block level) | 7.99 / 21.42 / 11.64 % | 14.05 / 37.48 / 20.44 % | +8.80 pp H |
| per-book macro block coverage | 92.45 % | 95.25 % | +2.80 pp |
| predictions | 41 758 | 40 107 | −1 651 |
| false positives / page | 38.21 | **32.10** | −6.11 |
| — art false positives / page | 5.35 | **1.35** | −75 % |
| — art area share | 0.900 % | 0.266 % | −0.634 pp |
| — split SFX fragments (`localized_sfx`) | 1 650 | 1 677 | +27 |
| — line-in-bubble (`localized_dialogue`) | 30 085 | 27 950 | −2 135 |

Dialogue P/R/H at IoU 0.5 is *not* a line metric — the GT is a whole bubble, so a
line box can never match it — and is listed only for comparability; block
coverage is the meaningful dialogue number.

### 2. End-to-end COO SFX recognition — `scripts/eval_sfx.py` (new)

The detector fine-tune targets COO onomatopoeia, which `eval_blocks.py` does not
measure (it scores Manga109 `<text>` **dialogue** blocks). `scripts/eval_sfx.py`
runs the real runtime path on the COO polygons themselves — page image → detector
→ predicted quads → `crop_line()` → recogniser — and scores every one of the
6 639 held-out COO SFX against its transcription:

* a **missed** SFX (no predicted quad reaches IoU 0.4 with the polygon) is scored
  as a full deletion, i.e. `len(gt)` edits — what a reader actually experiences;
* a **matched** SFX is scored on the text the recogniser returns for the
  *detected* crop (the runtime quadrilateral, not the GT polygon).

So `end-to-end CER` is det + rec together, `matched-only CER` is the recogniser
given that the SFX was found, and `exact` counts SFX read character-for-character
(an unmatched SFX is never exact). Matching is the same greedy polygon IoU at 0.4
that `eval_det.py` uses, so **`found` here is bit-for-bit the SFX crop recall
above** — 2 129 / 4 641, 32.07 % / 69.91 %, identical per point bucket.

| Release (det + rec) | SFX found | **end-to-end CER** | exact | matched-only CER | matched exact |
|---|---|---|---|---|---|
| `ocr-models-v3` (stock + stock) | 32.07 % (2 129) | 91.37 % | 3.71 % (246) | 75.69 % | 11.55 % (246) |
| `ocr-manga-v2` (stock + FT rec) | 32.07 % (2 129) | 75.53 % | 16.87 % (1 120) | **31.05 %** | 52.61 % (1 120) |
| `ocr-models-v4` (FT det + stock) | 69.94 % (4 643) | 86.64 % | 5.95 % (395) | 81.82 % | 8.51 % (395) |
| `ocr-manga-v3` (FT det + FT rec) | **69.94 % (4 643)** | **51.38 %** | **36.95 % (2 453)** | 33.86 % | **52.83 % (2 453)** |

Broken down by the GT polygon's complexity (end-to-end CER / exact count):

| Release | 4-point (3 037) | 5–8-point (2 780) | 9+-point curved (822) |
|---|---|---|---|
| `ocr-models-v3` | 87.13 % / 176 | 93.12 % / 63 | 95.89 % / 7 |
| `ocr-manga-v2` | 68.73 % / 705 | 77.46 % / 357 | 84.93 % / 58 |
| `ocr-models-v4` | 81.07 % / 257 | 88.60 % / 127 | 93.39 % / 11 |
| `ocr-manga-v3` | **43.69 % / 1 425** | **51.76 % / 876** | **66.50 % / 152** |

Paired over the same 6 639 SFX (all four runs score the identical GT set, so
`scripts/eval_sfx.py --compare` is exact, not an intersection):

| pair | end-to-end CER | exact | exact flips | p |
|---|---|---|---|---|
| `ocr-models-v3` → `ocr-manga-v2` (recogniser only) | +15.84 pp better: 91.37 → 75.53 % | 246 → 1 120 | 886 fixed / 12 broken | < 1e-4 |
| `ocr-models-v3` → `ocr-models-v4` (detector only) | 91.37 → 86.64 % (−4.73 pp) | 246 → 395 | 230 fixed / 81 broken | < 1e-4 |
| **`ocr-manga-v2` → `ocr-manga-v3`** (the upgrade) | **75.53 → 51.38 % (−24.15 pp)** | **1 120 → 2 453** | 1 535 fixed / 202 broken | < 1e-4 |
| `ocr-models-v3` → `ocr-manga-v3` (both) | 91.37 → 51.38 % (−39.99 pp) | 246 → 2 453 | 2 250 fixed / 43 broken | < 1e-4 |

Four things this table says that the dialogue block metric cannot:

* **The detector alone barely moves SFX reading** (91.37 → 86.64 %, +149 exact):
  finding 2 514 more onomatopoeia is worthless while the stock recogniser still
  cannot read hand-lettered display text — and 81 SFX that the stock pair read
  exactly stop being exact, because the new detector hands the recogniser a
  different (tighter, curved) crop.
* **The recogniser alone is capped by what the detector finds**: −15.84 pp CER,
  but 4 510 of the 6 639 SFX still contribute their full length as deletions.
* **Together they multiply**: end-to-end CER 91.37 → 51.38 % and 246 → 2 453
  character-exact onomatopoeia (×9.97), and over the previous fine-tune release
  `ocr-manga-v2` it is −24.15 pp CER with 1 333 more SFX read exactly.
* `matched-only CER` is deliberately *not* compared across the two detectors: the
  fine-tuned detector's matched set is a different, harder population (2 514 more
  SFX, including 384 of the 822 strongly curved ones), which is why it reads
  31.05 → 33.86 % there while the same-SFX paired CER falls in **every** bucket
  (4-pt −25.04 pp, 5–8-pt −25.70 pp, 9+ −18.43 pp).

### 3. End-to-end Manga109 dialogue blocks — `scripts/eval_blocks.py`

Same runtime path, but the GT is the Manga109 `<text>` block transcript, so this
is the "does the app still read the speech bubbles" number. `blocks read` counts
the blocks with at least one detected box inside them; a block the detector
ignores entirely is skipped by the tool, which is why the paired set below is
smaller than any single full set.

| Release (det + rec) | blocks read / 13 868 | block CER | block exact |
|---|---|---|---|
| `ocr-models-v3` (stock + stock) | 13 083 (785 unread, 5.66 %) | 28.05 % | 38.42 % |
| `ocr-manga-v2` (stock + FT rec) | 13 083 (785 unread) | 25.45 % | 43.63 % |
| `ocr-models-v4` (FT det + stock) | **13 523** (345 unread, 2.49 %) | 28.42 % | 39.88 % |
| `ocr-manga-v3` (FT det + FT rec) | **13 523** (345 unread) | 26.16 % | **44.48 %** |

`python scripts/compare_eval_blocks.py <dump>…` takes any number of
`eval_blocks.py --json-out` dumps and pairs them on the **intersection** of all of
them — 12 945 blocks are read by all four releases, and 440 more blocks are read
only by the two `ocr-models-v4` / `ocr-manga-v3` runs (they are the harder,
SFX-dominated ones, which is why the full-set CER can rise while the paired CER
falls):

| paired on the same 12 945 blocks | `ocr-models-v3` | `ocr-manga-v2` | `ocr-models-v4` | `ocr-manga-v3` |
|---|---|---|---|---|
| block CER | 29.76 % | 27.95 % | 28.54 % | **26.92 %** |
| block exact | 38.51 % | 43.72 % | 41.07 % | **45.64 %** |

| pair (same blocks) | block CER | block exact | exact flips | p |
|---|---|---|---|---|
| `ocr-models-v3` → `ocr-manga-v2` (recogniser only, 13 083 paired) | −1.84 pp | +5.21 pp | 1 095 fixed / 414 broken | < 1e-4 |
| `ocr-models-v3` → `ocr-models-v4` (detector only, 12 945 paired) | −1.22 pp | +2.56 pp | 668 fixed / 336 broken | < 1e-4 |
| **`ocr-manga-v2` → `ocr-manga-v3`** (the upgrade, 12 945 paired) | **−1.03 pp** | **+1.92 pp** | 530 fixed / 281 broken | < 1e-4 |
| `ocr-models-v3` → `ocr-manga-v3` (both, 12 945 paired) | −2.84 pp | +7.13 pp | 1 386 fixed / 463 broken | < 1e-4 |

### What a user upgrading from `ocr-manga-v2` to `ocr-manga-v3` gets

| | `ocr-manga-v2` | `ocr-manga-v3` | Δ |
|---|---|---|---|
| COO SFX found | 32.07 % | 69.94 % | +37.87 pp |
| COO SFX end-to-end CER | 75.53 % | **51.38 %** | **−24.15 pp** |
| COO SFX read exactly | 16.87 % (1 120) | **36.95 % (2 453)** | +20.08 pp (×2.19) |
| Manga109 block CER (paired) | 27.95 % | **26.92 %** | −1.03 pp |
| Manga109 block exact (paired) | 43.72 % | **45.64 %** | +1.92 pp |
| dialogue blocks unreadable | 785 | **345** | −440 |
| art false positives / page | 5.35 | **1.35** | −75 % |

Nothing regresses: SFX detection, SFX reading, dialogue block CER/exact, art
false positives and unreadable dialogue blocks all move the right way, every
change is significant on the paired set, and the detector metrics are the same
in both releases' recogniser variants (the detector is shared).

Reproduce (all four combinations; `scripts/eval_sfx.py --det-cache` makes the two
recogniser variants reuse one detection pass):

```bash
# 1. detector table (two detector weights cover the four releases)
python scripts/eval_det.py --model models/ppocrv6_small_det_stock.onnx --split test --out output/eval_det/stock_test_v2.json
python scripts/eval_det.py --model output/exp2_onnx/exp2_ep40.onnx     --split test --out output/eval_det/exp2_ep40_test.json

# 2. end-to-end COO SFX table, one dump per release
S="--split test --det-cache output/eval_sfx/det_cache"
python scripts/eval_sfx.py --det models/ppocrv6_small_det_stock.onnx   --rec models/ppocrv6_small_rec.onnx \
    $S --json-out output/eval_sfx/stockdet_stockrec_test.json         # ocr-models-v3
python scripts/eval_sfx.py --det models/ppocrv6_small_det_stock.onnx   --rec models/ppocrv6_small_rec_manga_real.onnx \
    $S --json-out output/eval_sfx/stockdet_ftrec_test.json            # ocr-manga-v2
python scripts/eval_sfx.py --det output/exp2_onnx/exp2_ep40.onnx       --rec models/ppocrv6_small_rec.onnx \
    $S --json-out output/eval_sfx/exp2ep40det_stockrec_test.json      # ocr-models-v4
python scripts/eval_sfx.py --det output/exp2_onnx/exp2_ep40.onnx       --rec models/ppocrv6_small_rec_manga_real.onnx \
    $S --json-out output/eval_sfx/exp2ep40det_ftrec_test.json         # ocr-manga-v3
# paired deltas (v2 -> v3 shown; any two dumps work, the GT keys are identical)
python scripts/eval_sfx.py --compare output/eval_sfx/stockdet_ftrec_test.json \
    output/eval_sfx/exp2ep40det_ftrec_test.json

# 3. end-to-end dialogue table, one dump per release
python scripts/eval_blocks.py --model models/ppocrv6_small_rec.onnx            --det models/ppocrv6_small_det_stock.onnx \
    --split test --json-out output/eval_blocks/stockdet_stockrec_test.json    # ocr-models-v3
python scripts/eval_blocks.py --model models/ppocrv6_small_rec_manga_real.onnx --det models/ppocrv6_small_det.onnx \
    --split test --json-out output/eval_blocks/stock_test.json                # ocr-manga-v2
python scripts/eval_blocks.py --model models/ppocrv6_small_rec.onnx            --det output/exp2_onnx/exp2_ep40.onnx \
    --split test --json-out output/eval_blocks/exp2ep40det_stockrec_test.json # ocr-models-v4
python scripts/eval_blocks.py --model models/ppocrv6_small_rec_manga_real.onnx --det output/exp2_onnx/exp2_ep40.onnx \
    --split test --json-out output/eval_blocks/exp2ep40det_ftrec_test.json    # ocr-manga-v3
# 4-way paired intersection (any number of dumps; run #1 is the reference)
python scripts/compare_eval_blocks.py output/eval_blocks/stockdet_stockrec_test.json \
    output/eval_blocks/stock_test.json output/eval_blocks/exp2ep40det_stockrec_test.json \
    output/eval_blocks/exp2ep40det_ftrec_test.json
```

Coverage of each metric, so the denominators are not a mystery: `eval_det.py`
scores all 982 test pages; `eval_blocks.py` visits the 969 pages that carry a
Manga109 `<text>` block (all 13 868 GT blocks are on those pages); `eval_sfx.py`
visits the 858 pages that carry a COO annotation (all 6 639 COO SFX). Ten of the
6 639 COO transcriptions contain a character outside the dictionary, so they
cannot be produced by any of these models — a ~0.06 % floor that is present in
all four rows equally.

## Real manga — Manga109-s, held out **by book**

9 test books, 969 pages, 13,083 text blocks, 24,567 line crops
(dialogue 21,535 / onomatopoeia 3,032).

### Line level (recogniser only)

| Model | all CER | all exact | dialogue CER | dialogue exact | SFX CER | SFX exact |
|---|---|---|---|---|---|---|
| base | 17.23 % | 70.72 % | 8.73 % | 78.75 % | 77.56 % | 13.69 % |
| stage-1 (synthetic) | 16.04 % | 66.15 % | 11.07 % | 70.11 % | 51.33 % | 37.99 % |
| **stage-2 (real)** | **7.82 %** | **82.22 %** | **6.51 %** | **83.15 %** | **17.12 %** | **75.63 %** |

### Page level (det + crop + rec, whole-block transcript)

This is the unbiased end-to-end view: every detected block is compared with the
Manga109 transcript, no teacher-based selection.

| Model | block CER | block exact | blocks |
|---|---|---|---|
| base | 28.05 % | 38.42 % | 13,083 |
| stage-1 (synthetic) | 30.21 % | 30.45 % | 13,083 |
| **stage-2 (real)** | **25.45 %** | **43.63 %** | 13,083 |

The synthetic-only stage **hurt** real dialogue (block CER 28.05 → 30.21 %) while
helping SFX; the real-data stage fixes that and improves both.

## Held-out typography — synthetic, 1,200 lines, 4 unseen display fonts

| Model | all CER | all exact | SFX CER | SFX exact |
|---|---|---|---|---|
| base | 17.86 % | 52.58 % | 31.29 % | 45.43 % |
| stage-1 | **2.81 %** | **86.42 %** | 2.98 % | 92.29 % |
| stage-2 | 2.69 % | 87.42 % | 3.19 % | 91.14 % |

## General-text guard — 500 clean lines, 2 unseen print fonts

| Model | all CER | all exact |
|---|---|---|
| base | 3.98 % | 84.60 % |
| stage-1 | **0.43 %** | **97.00 %** |
| stage-2 | 0.53 % | 96.60 % |

## Training (stage 2)

- 252,366 train lines = 192,964 real (210,723 dialogue + 16,797 COO SFX before
  the split) + 59,402 synthetic; 11,787 validation.
- Initialised from stage-1, Adam, cosine LR 1e-4, 8 epochs, batch 128,
  MultiScaleSampler, RecAug + RecConAug. Best validation exact 78.36 % (epoch 7).
- 969 test pages / 13k blocks; ~1 h on the RTX 5070 Ti.

## Artifact

| | |
|---|---|
| file | `models/ppocrv6_small_rec_manga_real.onnx` |
| sha256 | `3e898ae24a566d3606f459eaf9ee69b3110e8ea50f200701dc8abb5403467a5d` |
| size | 21 159 418 bytes |
| input / output | `x [N,3,48,W]` / `[N,T,18710]` |
| opset | 11 |
| contract check | same I/O as the shipped model, `verify_onnx.py` passes |

## Detector fine-tune — dataset

`scripts/build_det_dataset.py` (see README §2.3). Reproduced 2026-10-24 on
Manga109-s v2026 + `hal-utokyo/MangaOCR`; `python scripts/build_det_dataset.py --validate-only`
is green (0 out-of-bounds points, 0 degenerate labels, 0 provenance failures,
0 de-dupe leaks, 0 role violations, 0 ignore regions with holes, 0 ignore regions
covering SFX beyond the 1 % tolerance, book-disjoint splits).

**Training roles.** MangaOCR provides no line-level dialogue: its boxes are the
Manga109 `<text>` blocks (bubbles / 2–4 columns), so train uses them as `###`
ignore regions instead of positives (POS = positive label):

| | train | val | test |
|---|---|---|---|
| books | 73 | 4 | 9 |
| pages with labels | 6,738 | 333 | 982 |
| POS total | 39,391 | 7,401 | 19,318 |
| — POS dialogue | 0 | 4,796 | 12,209 |
| — POS SFX | 39,391 | 2,605 | 7,109 |
| — ignore regions (`###`) | 87,908 | 0 | 0 |

Ignore construction on train (87,759 dialogue boxes): 86,700 emitted as-is,
1,036 trimmed around SFX ground truth into 1,208 hole-free fragments, 23 dropped
(box almost entirely covered by an SFX). Overlap of ignore regions with SFX GT:
max 1.0 %, mean 0.002 % (tolerance 1 %). Ignore labels come first in every line so
`MakeShrinkMap`'s mask zeroing cannot erase a positive; verified with PaddleOCR
itself: `mask=0` covers 100 % of every ignore region, positives keep `mask=1`
coverage 0.25–0.36 (the shrunk GT), and `MakeBorderMap` produces no threshold
target for ignore labels.

Source accounting (86 of 87 books; `PrayerHaNemurenai` excluded, see below):

- COO: 45,422 polygons in the release, 44,858 in the kept books, 44,857 emitted
  (1 dropped: three points). 16,720 four-point, 28,137 with 5–66 points.
- MangaOCR: 10,602 rows (478 with an empty assistant message, 124 rows with
  malformed JSON parsed by the tolerant scanner), 157,291 boxes; 48,278 duplicate
  a COO polygon and are dropped (40,829 at IoU > 0.5, 7,449 by covering > 50 % of
  the polygon from a looser box; 99.4 % of all those pairs agree on the
  transcription), 1 degenerate, 104,764 dialogue + 4,248 MangaOCR-only SFX kept.
  Of the kept SFX boxes, 1 still overlaps a COO polygon at IoU > 0.5.
- Page alignment: 7,983 pages compared against Manga109 `<text>` blocks, 22 pages
  (0.3 %) with no box matching any block, no book flagged.
- Coverage: 4,854 of 121,763 Manga109 text blocks (4.0 %) are covered by no label
  at all - neither a positive nor an ignore region - so they supervise the detector
  as background. Nothing is inferred from the stock detector; ignore regions come
  only from the block-level dialogue boxes described above.

Known limits of this dataset:

- MangaOCR dialogue instances are the Manga109 block instances (93,463 match one
  block at IoU > 0.5, 11,290 merge ≥ 2), so **dialogue detection metrics on
  val/test are block/bubble-level**, not line-level (4,796 val + 12,209 test
  instances). Quote them next to an end-to-end block CER (`scripts/eval_blocks.py`).
  The 1:1 boxes are block-level too: rendered 1:1 boxes hold 2–4 text columns, so
  train cannot use any of them as a positive line label.
- `PrayerHaNemurenai` is annotated by MangaOCR on a ~2× larger canvas (1,238 of
  1,745 boxes outside the Manga109-s page), so the book is excluded instead of
  being mislabelled; 24 `GinNoChimera` COO `<page>` records have `width=0 height=0`
  (the builder reads the JPEG header instead).
- For per-class eval use `source=coo` for the SFX ground truth (44,857 exact
  polygons). 394 dialogue boxes overlap a COO polygon (a bubble sitting on an
  SFX), and 96 dialogue boxes are numerically identical to their `<text>` block
  because MangaOCR copied those coordinates - geometry provenance is still
  MangaOCR, which `--validate-only` checks against the annotation cache.

## Detector fine-tune — training and held-out evaluation

`configs/manga_det.yml` is `PaddleOCR/configs/det/PP-OCRv6/PP-OCRv6_small_det.yml`
with the same Architecture / Loss / Optimizer and the manga data pipeline;
`scripts/train_det.sh` writes the exact per-run config to
`output/det_configs/<tag>.yml` beside its log (`output/logs/<tag>.log`).

| | |
|---|---|
| initialisation | `models/PP-OCRv6_small_det_pretrained.pdparams` (official), 10,757,224 B, sha256 `13e7072d…75043f` |
| train | `data/det/train.txt` — 6,738 pages, 39,391 SFX positives, 87,908 `###` dialogue ignores |
| val | `data/det/val.txt` — 333 pages, 7,401 positives (2,605 SFX + 4,796 block-level dialogue) |
| batch | 8 / card, 8 workers, `distributed: false`, stock 640×640 `RandomCrop` |
| optimiser | stock Adam + L2 1e-5, cosine peak LR 5e-4 (half the stock 1e-3 for the 1/8 effective batch), warmup 2 epochs, 40 epochs = 33,720 steps |
| EMA | stock (`ema_decay 0.9997`, threshold) |
| eval | every epoch on val, `DetMetric` hmean with the runtime postprocess (960 max side, 0.15 / 0.25 / 1.40) → `best_accuracy` |
| augmentation | stock: CopyPaste, IaaAugment (Fliplr measured below, Affine ±45°, Resize 0.1–2), RandomCrop 640 |
| GPU memory | 8,626 MB reserved at batch 8; batch 16 peaks at 15.5 GB and OOMs with the desktop session resident, so 8 is the largest batch that fits |
| duration | 2 h 37 min (21:00:59 → 23:38:27) on the RTX 5070 Ti |
| best checkpoint | val hmean 0.32114 (precision 0.34134 / recall 0.30320) at epoch 40 |
| selection | val only; the test split was evaluated once, after training |

Val hmean per epoch (full curve in `output/logs/full.log`): 1 16.48, 5 24.69,
10 26.69, 15 27.35, 20 30.83, 25 29.87, 30 30.47, 35 31.60, 40 32.11.

**Tiny smoke run** (pipeline only, not a result): 20 ARMS pages / 8
`JijiBabaFight` pages, batch 8, 2 epochs. It exercised pretrained loading,
`best_accuracy` saving, per-epoch eval, resume (`--resume output/det_tiny8/latest`
restarted at `start_epoch 2` with optimiser state) and export (opset 14, one
`[N,1,H,W]` map, Paddle↔ONNX max |Δ| 7.2e-4, 0 % binarised-mask disagreement at
thresh 0.15). On 20 pages the loss moves (10.6 → 9.6) but val hmean does not
(0.145); learning is established by the 5-epoch pilot and the full run above.

### Fliplr ablation (val only)

Two 5-epoch arms with identical seed (1024), data, batch, LR and schedule; only
the stock `IaaAugment` Fliplr probability differs. Selection metric: val
`DetMetric` hmean; test untouched.

| val hmean (%) | ep 1 | ep 2 | ep 3 | ep 4 | ep 5 |
|---|---|---|---|---|---|
| p = 0.5 (stock) | 18.28 | 24.25 | 24.20 | 25.31 | **27.02** |
| p = 0.0 | 20.44 | 23.80 | 21.73 | 24.38 | 26.49 |

p=0.5 wins the last epoch by 0.53 pp, p=0.0 wins epoch 1 by 1.6 pp. With one seed
the arms are not separable at this budget, so the pre-stated rule (keep stock
unless the alternative wins by ≥1 pp) keeps **Fliplr p=0.5**; no Fliplr effect is
claimed. A rigorous comparison would need several seeds × the full 40-epoch
schedule (≈5 h GPU per seed pair) and would still not be a significance test
here.

Both arms were also scored on val (never test) with the detector metrics below:
SFX recall 51.68 % (p=0.5) vs 53.74 % (p=0.0), dialogue block coverage 25.23 % vs
26.90 % (stock detector: 93.79 %), and val block CER 49.86 % vs the stock 29.17 %.
The dialogue collapse reported below is therefore already present at 5 epochs in
both arms and is not a Fliplr artefact.

### Detection metrics (polygon IoU, app postprocess)

`scripts/eval_det.py`: `models/ppocrv6_small_det.onnx` (stock) and
`models/ppocrv6_small_det_manga.onnx` (40-epoch fine-tune), both through
`scripts/db_det.py` (960 max side, thresh 0.15, box 0.25, unclip 1.40). GT is
`data/det/{val,test}.jsonl`: SFX = COO polygons (`source=coo`), dialogue =
MangaOCR boxes (Manga109 `<text>` blocks). IoU is polygon-vs-predicted-quad,
greedy one-to-one at 0.5; per-class precision attributes each prediction to the
GT it matched, else to the GT it overlaps most, else background.

**Test (9 books, 982 pages, 6,639 COO SFX + 12,209 dialogue blocks; never seen
in training or selection)**

| | stock | fine-tune |
|---|---|---|
| SFX P / R / H @0.5 | 49.62 / 24.48 / 32.78 % | **66.99 / 57.34 / 61.79 %** |
| dialogue P / R / H @0.5 (block level) | 7.99 / 21.42 / 11.64 % | 23.68 / 18.05 / 20.49 % |
| SFX crop recall @0.4 | 32.07 % (2,129/6,639) | **68.11 % (4,522/6,639)** |
| — 4-point SFX | 34.8 % (1,057/3,037) | 65.0 % (1,973/3,037) |
| — 5–8-point SFX | 30.4 % (846/2,780) | 70.5 % (1,961/2,780) |
| — 9+-point SFX (highly curved) | 27.5 % (226/822) | 71.5 % (588/822) |
| dialogue blocks ≥50 % covered | **92.10 %** (11,245/12,209) | 25.99 % (3,173/12,209) |
| dialogue mean block coverage | 81.78 % | 27.42 % |
| predictions | 41,758 | 16,732 |
| false positives / page | 38.21 | 10.92 |
| — art / page | 5.35 | 1.34 |
| — art area share | 0.900 % | 0.221 % |
| — split fragments (`localized_sfx`) | 1,650 | 1,876 |
| — line-in-bubble (`localized_dialogue`) | 30,085 | 7,104 |
| per-book macro SFX H | 33.16 % | 61.62 % |
| per-book macro dialogue block coverage | 92.45 % | 28.03 % |

**Val (4 books, 333 pages, 2,378 SFX + 4,796 blocks)** — used for checkpoint
selection, shown for completeness: SFX P/R/H 53.11/32.67/40.46 → 73.05/60.30/66.07 %;
SFX crop recall 41.04 → 67.96 % (9+: 40.2 → 66.5 %); dialogue block coverage
93.79 → 22.50 %; art FP 7.56 → 1.36 /page; 18,636 → 6,583 predictions.

Dialogue P/R/H at IoU 0.5 is *not* a line metric: the GT is a whole bubble, so a
line box can never match it. It is reported for stock-vs-fine-tune comparability
only; the meaningful dialogue number is block coverage, which collapses.

### End-to-end block CER (test, same recogniser)

`scripts/eval_blocks.py --split test` with
`models/ppocrv6_small_rec_manga_real.onnx` and either detector; paired with
`scripts/compare_eval_blocks.py`. The stock run reproduces the handoff baseline
exactly (25.45 % CER / 43.63 % exact, 969 pages, 13,083 blocks), so the comparison
is paired on identical inputs. `eval_blocks.py` reports block-level CER only; the
line-level dialogue/SFX figures come from `scripts/eval_rec.py` on the recogniser,
which is unchanged here, so they are not re-measured for a detector swap.

| | stock det | fine-tuned det |
|---|---|---|
| block CER | **25.45 %** | 44.74 % |
| block exact | **43.63 %** | 29.58 % |
| blocks read / with GT text | 13,083 / 13,868 (5.66 % unread) | 7,217 / 13,868 (**47.96 % unread**) |
| paired set: blocks both read | 6,881 | 6,881 |
| paired block CER | 29.91 % | 60.38 % (+30.46 pp) |
| paired block exact | 47.10 % | 29.70 % (−17.40 pp) |
| per-block CER | improved 579, worsened 3,040, equal 3,262 | |
| exact flips | 146 fixed, 1,343 broken (McNemar p < 1e-4) | |

6,202 blocks the stock detector reads have no detection at all with the
fine-tune, and 336 the other way; the 44.74 % CER is computed on the 52 % of
blocks that survive, so it understates the loss.

### Qualitative (test, curved SFX)

Overlays are written only to `/tmp` (`--overlay-pages 6 --overlay-dir
/tmp/opencode/det_overlays/<tag>`), e.g. `Arisa` page 5: the stock detector misses
the curved ゴゴゴ entirely and boxes every dialogue line; the fine-tune puts one
tight quadrilateral around the curve and on the vertical だめーっ, and leaves most
bubbles untouched. This matches the numbers and was inspected after the metrics,
not used to pick the model.

### Detector artifact

| | |
|---|---|
| file | `models/ppocrv6_small_det_manga.onnx` (not shipped, decision below) |
| sha256 | `4f9af517d3acd12762f720a9ebbae7708babe282af28a12928ac99f433f34e7e` |
| size | 9,929,569 B (stock 9,880,512 B, sha256 `d73e0058…c9410e`) |
| I/O | `x [N,3,H,W]` dynamic → `[N,1,H,W]`, opset 14 |
| export | `scripts/export_det_onnx.sh`: `.venv-export` (Paddle 3.3.1) → paddle2onnx 2.1.0; needs `scripts/pir_scale_f32.py` (paddle2onnx cannot read the PIR `interpolate.scale` f64 attribute) and `scripts/finalize_det_onnx.py` (declares the output's channel dim) |
| parity | Paddle vs ONNX on 3 test pages: max |Δ| 8.3e-4, binarised-mask disagreement at 0.15 = 0.0000 % |

### Decision (experiment 1): REJECT the detector

> **Superseded by [experiment 2](#experiment-2--dialogue-preserving-detector-shipped)**,
> which adds line-level dialogue supervision and is accepted and shipped.

- SFX detection improves materially on held-out books: SFX H 32.78 → 61.79 %,
  crop recall 32.07 → 68.11 %, curved 9+ recall 27.5 → 71.5 %, and the false
  positives *fall* (art 5.35 → 1.34 per page), so it is not an FP trade.
- Dialogue detection collapses: block coverage 92.10 → 25.99 %, 48 % of text
  blocks get no detection at all, and the paired block CER rises by 30.46 pp
  (exact −17.40 pp, p < 1e-4) with the recogniser held fixed.
- Cause: the dataset trains SFX positives only and marks block-level dialogue as
  `###` ignore regions, which removes the loss there but leaves nothing anchoring
  the pretrained model's line response; 40 epochs of SFX-only supervision drift
  the backbone away from dialogue lines. "Dialogue behaviour is inherited" is
  therefore false as measured. The configured selection metric (`DetMetric`
  hmean over block-level dialogue GT) cannot see the regression - it *rises*,
  because suppressing line boxes improves block-level precision.
- Next step is supervision, not thresholds or more epochs: line-level dialogue
  positives (pseudo-labels from the stock detector, split MangaOCR blocks, or the
  380-page MIT `manga109s-line-annotations` set) or a distillation term that
  keeps the stock detector's line map; then re-run this evaluation. Nothing is
  exported to a release and the model is not integrated.
- Detection alone cannot fix curved SFX end-to-end in any case: the recogniser
  still receives the min-area **rectangular** crop from `crop_line()`, so a
  polygon→straight-strip unwarp in the native shim is a separate prerequisite for
  curved SFX *text* accuracy (the detector can only stop splitting them).

## Experiment 2 — dialogue-preserving detector (shipped)

Experiment 1 failed on supervision, not on the objective: it trained SFX
positives only and marked every dialogue region `###` ignore, which removed the
loss without anchoring the pretrained detector's *line* response. Experiment 2
keeps experiment 1 (`data/det/`) intact and trains on a second dataset,
`data/det_pseudo/` (`scripts/build_det_pseudo_dataset.py`), that adds **line-level
dialogue supervision distilled from the stock detector**: the stock detector's
own line quads are kept as positives when ≥ 0.8 of their area lies inside a
MangaOCR dialogue region, clipped to (region − SFX), and emitted with
`teacher=stock_det`. Everything else is unchanged: COO polygons are still the only
independent SFX ground truth, val/test keep their untouched independent GT.

Consequence, stated up front: dialogue quality is now **capped by the stock
detector**, and any error the stock detector makes inside a bubble becomes a
training target. That is the price of "do not lose dialogue" without line-level
dialogue labels, which MangaOCR does not provide.

### Compute profiling and the training fix

The resumed run was preceded by a controlled sweep: 160 logged real training
steps per trial from the pretrained init with eval disabled, `nvidia-smi` sampled
at 2 Hz (`output/bench_compute/`). All numbers are `output/bench_compute/results.jsonl`.

| trial | batch | workers | OMP=1 | AMP | ips (samples/s) | batch cost | reader cost | peak VRAM (pool) | VRAM (smi) | GPU util mean | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `a_base8` | 8 | 8 | – | – | 38.79 | 0.2082 s | 0.00146 s | 8,663 MB | 10,960 MB | 84.3 % | baseline |
| `b_bs12` | 12 | 8 | – | – | 42.67 | 0.2813 s | 0.00147 s | 13,033 MB | 15,401 MB | 90.7 % | ok |
| `c_bs14` | 14 | 8 | – | – | — | — | — | — | — | — | **OOM** |
| `d_bs16` | 16 | 8 | – | – | — | — | — | — | — | — | **OOM** |
| `e_bs8_omp1` | 8 | 8 | ✓ | – | 41.29 | 0.1946 s | 0.00147 s | 8,626 MB | 10,797 MB | 81.7 % | ok |
| `f_bs12_omp1` | 12 | 8 | ✓ | – | 44.65 | 0.2713 s | 0.00143 s | 13,033 MB | 15,285 MB | 88.9 % | ok |
| `g_bs14_omp1` | 14 | 8 | ✓ | – | — | — | — | — | — | — | **OOM** |
| `h_bs14_w12omp1` | 14 | 12 | ✓ | – | — | — | — | — | — | — | **OOM** |
| `i_bs12_w12omp1` | 12 | 12 | ✓ | – | 44.65 | 0.2688 s | 0.00137 s | 13,033 MB | 15,292 MB | 89.7 % | ok (contaminated) |
| `j_amp_bs12` | 12 | 8 | ✓ | O1 | 48.00 | 0.2445 s | 0.00143 s | 10,896 MB | 13,109 MB | 80.5 % | ok |
| `k_amp_bs14` | 14 | 8 | ✓ | O1 | 50.91 | 0.2738 s | 0.00152 s | 12,728 MB | 14,788 MB | 85.4 % | ok |
| `l_bs13_omp1` | 13 | 8 | ✓ | – | 45.22 | 0.2876 s | 0.00136 s | 13,424 MB | 15,528 MB | 91.2 % | ok |
| `n_bs14_naive` | 14 | 8 | ✓ | – | — | — | — | — | — | — | **OOM** (naive allocator) |

Diagnosis:

- **Input is not the bottleneck.** `avg_reader_cost` is 1.4–1.5 ms against a
  200–290 ms step — 0.5–0.7 % of the step — so the pipeline is solidly
  GPU-compute-bound, and raising workers 8 → 12 changed nothing (`i` ≡ `f`,
  within noise). OMP/MKL/OpenBLAS pinning to 1 still buys +6.4 % at batch 8
  (38.79 → 41.29) and +4.6 % at batch 12 (42.67 → 44.65) by stopping the loader
  workers from oversubscribing BLAS threads. That is the only host-side win.
- **Batch is capped by VRAM, hard.** Batch 14 and 16 OOM at 16 GB with the
  desktop session resident (`c`, `d`, `g`, `h`), and forcing the naive allocator
  OOMs too (`n`). Batch 12 peaks at 13,033 MB pool / 15.4 GB device, batch 13 at
  13,424 MB / 15.53 GB — 13 fits but leaves ~470 MB of headroom, so **batch 12
  was chosen**. Going 8 → 12 is worth +15.1 % (`a` → `f`), the largest available
  gain, and it comes from occupancy: GPU utilisation 84.3 % → 88.9 %.
- **Eval was the second cost, and it does not scale.** The original schedule
  evaluated every epoch at `batch_size_per_card: 1`; each pass took 31.2 s
  (mean of 31 passes, 966 s = **16.1 min = 13.6 %** of a 1.98 h segment), and
  `--workers 2/4/8` measured 36.6 / 31.2 / 32.6 s — more loader workers do not
  help a batch-1 GPU-idle pass, so the only lever is doing it less often.
- **AMP is the remaining headroom, not taken.** `j`/`k` reach 48.0 / 50.9 ips
  (+7.5 % / +14 % over `f`) *and* halve pool VRAM (10,896 MB at batch 12), which
  would allow batch 14. It was left off for this run so the resumed segment keeps
  the same numeric path as the first 32 epochs (EMA + Dice/Focal); it is the
  first thing to try if more throughput is needed.

**What the real run did.** `exp2` (batch 8) was killed mid-epoch 32; the resume
config (`output/det_configs/exp2_resume.yml`) re-ran from `latest` at **batch 12**
with the cosine `t_max` pinned to 33,720 steps so the schedule does not end early,
and ran epochs 32–40.

| | before (`exp2`, batch 8) | after (`exp2_resume`, batch 12) |
|---|---|---|
| median ips | 36.37 samples/s | **40.51 samples/s** (+11.4 %) |
| median batch cost | 0.2193 s | 0.2958 s |
| median reader cost | 0.00151 s (0.69 %) | 0.00140 s (0.47 %) |
| epoch time | 843 it × 0.2193 = **184.9 s** | 562 it × 0.2958 = **166.2 s** (−10.1 %) |
| peak VRAM (pool) | 8,626 MB | 13,074 MB |
| in-training evals | 31 (16.1 min of stall) | **0** |
| wall clock | 32 epochs / 1.98 h | 9 epochs / 26 min 1 s |

The 40-epoch projection for pure training is 110.7 min instead of 123.3 min
(−10.2 %). Add the per-epoch eval that the original schedule ran — 31.2 s per
pass, measured over 31 passes — and a 40-epoch run costs 123.3 + 20.8 = 144.1 min
uninterrupted versus 110.7 min here: ≈ 23 % shorter end to end, of which two
thirds is simply not evaluating every epoch.

**Gotcha worth recording:** `eval_batch_step` is compared against the *run-local*
step counter, which restarts at 0 on resume. The pinned absolute values
`[25571, 2810]` therefore never fired (the resume logged `global_step: 20` on its
first line and stopped at 5,058), so the resumed segment evaluated **zero** times
and `best_accuracy` still holds epoch 31 of the pre-resume segment
(`best_epoch: 31`). Checkpoint selection was done afterwards, offline, on val —
which is what `scripts/select_det_checkpoint.py` exists for.

### Validation (checkpoint selection)

`scripts/eval_det.py --split val` — 4 books, 333 pages, 2,605 COO SFX, 4,796
dialogue blocks. Selection metric is **not** PaddleOCR's `DetMetric` hmean (it
cannot see a dialogue collapse: it rose to 0.2937 at epoch 31 and then never
improved); it is the pre-registered constrained rule in
`scripts/select_det_checkpoint.py`: dialogue block coverage ≥ stock − 1.5 pp and
art FP ≤ 2× stock, then maximise SFX H.

| epoch | SFX P / R / H @0.5 | SFX crop R @0.4 | dialogue block cov. | mean coverage | art FP / page | predictions |
|---|---|---|---|---|---|---|
| stock | 53.11 / 32.67 / 40.46 % | 41.04 % | 93.79 % | 84.3 % | 7.56 | 18,636 |
| exp1 (40 ep) | 73.05 / 60.30 / 66.07 % | 67.96 % | 22.50 % | 24.3 % | 1.36 | 6,583 |
| exp2 ep 5 | 69.8 / 55.8 / 62.0 % | 61.9 % | 96.2 % | 85.8 % | 3.04 | 17,469 |
| exp2 ep 10 | 71.7 / 59.0 / 64.7 % | 65.7 % | 96.6 % | 86.2 % | 2.28 | 17,273 |
| exp2 ep 15 | 72.9 / 60.4 / 66.1 % | 67.2 % | 96.6 % | 85.8 % | 1.79 | 17,163 |
| exp2 ep 20 | 74.0 / 62.1 / 67.5 % | 68.9 % | 96.5 % | 85.9 % | 1.80 | 17,070 |
| exp2 ep 25 | 75.1 / 62.8 / 68.4 % | 69.5 % | 96.3 % | 85.4 % | 1.54 | 17,013 |
| exp2 ep 30 | 76.3 / 63.2 / 69.2 % | 69.2 % | 96.1 % | 85.3 % | 1.38 | 16,816 |
| exp2 ep 31 | 76.0 / 63.2 / 69.0 % | 69.2 % | 96.1 % | 85.3 % | 1.37 | 16,818 |
| exp2 ep 35 | 76.7 / 63.7 / 69.6 % | 69.5 % | 96.1 % | 85.3 % | 1.38 | 16,823 |
| **exp2 ep 40** | **76.9 / 64.1 / 70.0 %** | **69.8 %** | 96.1 % | 85.3 % | 1.39 | 16,825 |

The distillation worked from the first epoch: dialogue coverage is back at
stock level (96.2 % at epoch 5, *above* stock's 93.79 %) while SFX H climbs
steadily, and art false positives fall 5.4×. Every epoch passes the constraint;
the rule picks **epoch 40** (highest SFX H, tie-break SFX crop recall).

### Held-out test (never used for training or selection)

9 books, 982 pages, 6,639 COO SFX polygons, 12,209 dialogue blocks, book-disjoint.
Polygon IoU, greedy one-to-one at 0.5, app postprocess (960 max side, 0.15 /
0.25 / 1.40).

| | stock | exp1 (SFX-only) | **exp2 ep 40** |
|---|---|---|---|
| SFX P / R / H @0.5 | 49.62 / 24.48 / 32.78 % | 66.99 / 57.34 / 61.79 % | **70.50 / 60.36 / 65.03 %** |
| SFX crop recall @0.4 | 32.07 % (2,129) | 68.11 % (4,522) | **69.91 % (4,641)** |
| — 4-point SFX | 34.8 % | 65.0 % | **67.6 %** |
| — 5–8-point SFX | 30.4 % | 70.5 % | **71.3 %** |
| — 9+-point (highly curved) | 27.5 % | 71.5 % | **74.0 %** |
| dialogue blocks ≥50 % covered | 92.10 % (11,245) | 25.99 % (3,173) | **94.94 % (11,591)** |
| dialogue mean block coverage | 81.78 % | 27.42 % | **82.81 %** |
| predictions | 41,758 | 16,732 | 40,107 |
| false positives / page | 38.21 | 10.92 | **32.10** |
| — art / page | 5.35 | 1.34 | **1.35** |
| — art area share | 0.900 % | 0.221 % | **0.266 %** |
| — split SFX fragments (`localized_sfx`) | 1,650 | 1,876 | 1,677 |
| — line-in-bubble (`localized_dialogue`) | 30,085 | 7,104 | 27,950 |

Versus stock, on the split that was never touched: **SFX H +32.25 pp**, SFX
recall +35.88 pp, SFX crop recall +37.84 pp, curved 9+ recall +46.5 pp,
dialogue block coverage **+2.84 pp**, art false positives −75 %. Nothing the
acceptance criteria care about moved backwards.

### End-to-end block CER (test, same recogniser)

`scripts/eval_blocks.py --split test` with `models/ppocrv6_small_rec_manga_real.onnx`
and either detector, paired with `scripts/compare_eval_blocks.py`. This is the
detector experiment with the fine-tuned recogniser held fixed; the same table with
all four shipping releases (and the COO SFX end-to-end numbers) is in
§"Released combinations" at the top of this document, and the
`ocr-manga-v2` → `ocr-manga-v3` pair is the bottom row there (−1.03 pp CER,
+1.92 pp exact on the same 12 945 blocks).

| | stock det | exp2 ep 40 det |
|---|---|---|
| block CER (full set) | **25.45 %** | 26.16 % |
| block exact (full set) | 43.63 % | **44.48 %** |
| blocks read / with GT text | 13,083 / 13,868 (5.66 % unread) | **13,523** / 13,868 (**2.49 % unread**) |
| paired set: blocks both read | 12,945 | 12,945 |
| paired block CER | 27.95 % | **26.92 % (−1.03 pp)** |
| paired block exact | 43.72 % | **45.64 % (+1.92 pp)** |
| per-block CER | improved 1,771, worsened 1,215, equal 9,959 | |
| exact flips | 530 fixed, 281 broken (McNemar p < 1e-4) | |

The full-set CER is 0.71 pp *higher* and that is not a regression: 440 more text
blocks now get a detection at all, and the ones that appear are the harder,
SFX-dominated blocks. On the identical block set the new detector reads *better*
(−1.03 pp CER, +1.92 pp exact, binomial p < 1e-4), and 440 fewer blocks are
unreadable (785 → 345).

### Acceptance

| criterion | result |
|---|---|
| material SFX detection gain (COO P/R/H @0.5, crop recall @0.4) | ✅ H 32.78 → 65.03 % (+32.25 pp), crop 32.07 → 69.91 % (+37.84 pp) |
| no dialogue recall / block-coverage regression | ✅ 92.10 → 94.94 % (+2.84 pp) |
| no end-to-end block CER regression | ✅ paired −1.03 pp CER, +1.92 pp exact, unread 785 → 345 |
| no false-positive explosion | ✅ art 5.35 → 1.35 /page, total 38.21 → 32.10 /page |

**ACCEPTED and shipped.**

### Shipped artifact

| | |
|---|---|
| file | `ppocrv6_small_det.onnx` (export: `output/exp2_onnx/exp2_ep40.onnx`) |
| sha256 | `bf5ed74a98dbbb16f6f4122fd1660692eeccc651933cc7582e38b112dbf01677` |
| size | 9,929,569 B (stock 9,880,512 B) |
| I/O | `x [N,3,H,W]` dynamic batch **and** H/W → `[N,1,H,W]`, opset 14 |
| parity | Paddle vs ONNX on 3 test pages: worst map \|Δ\| 5.06e-4 (limit 2e-3), binarised-mask disagreement at 0.15 = 0.0000 % (limit 0.01 %) |
| releases | [`ocr-manga-v3`](https://github.com/komi-kun/komi-kun-ocr-models/releases/tag/ocr-manga-v3) (this detector + fine-tuned rec) and [`ocr-models-v4`](https://github.com/komi-kun/komi-kun-ocr-models/releases/tag/ocr-models-v4) (this detector + stock rec + the unchanged native `.so`s) |

Both releases' `manifest.json` carry an explicit flat `"path"` on every entry,
because `ModelDownloader` falls back to `$baseUrl/models/<name>` when it is
absent and the release assets are flat, so the fetch 404s. `ocr-manga-v2` shipped
without it and is superseded for that reason alone.

### Remaining limitation: the crop is still a rectangle

The detector now puts one tight quadrilateral around a curved onomatopoeia
instead of splitting it, and box-level recall on the 9+-point (highly curved)
class goes 27.5 → 74.0 %. But the runtime still feeds the recogniser the
`minAreaRect` **rectangular** crop from `crop_line()`, so a strong arc is
straightened by axis-aligned cropping, not by unwarping the curve. Curved SFX
*text* accuracy therefore stays limited: this stage fixes *where* the onomatopoeia
is and stops the fragments, and a polygon→straight-strip unwarp in the native
shim is still the prerequisite for reading it correctly. The polygon the detector
emits is already the input that unwarp would need — it is simply not consumed
yet.

## Caveats

- The **line-level** dialogue numbers are measured on lines that the teacher
  could read consistently with the transcript, so they are optimistic in
  absolute terms; the **block-level** table has no such selection and is the
  number to quote for the app.
- 969 pages is a good but not huge sample; the 9 test books are disjoint from the
  69 training books.
- SFX gains are **not** limited to straight onomatopoeia any more: crop recall on
  the 9+-point (curved) class goes 27.49 → 73.97 %, and the end-to-end SFX CER on
  that class 84.93 → 66.50 % over `ocr-manga-v2`. What still caps curved SFX is
  the **rectangular** crop (`crop_line()`'s `minAreaRect`) handed to the
  recogniser, not the detector — see "Remaining limitation" above. The end-to-end
  SFX numbers in §"Released combinations" inherit that limitation: they measure
  what the app actually does today, unwarp not included.
- The end-to-end SFX metric scores an undetected onomatopoeia as a full deletion,
  so it is dominated by detection recall (32 % → 70 % of 6 639). That is
  deliberate — it is the reader-visible error — but it means a detector change and
  a recogniser change cannot be read off one column; both the paired bucket table
  and `matched-only CER` are printed for that reason.
- 10 of the 6 639 COO test transcriptions contain a character outside the shipped
  dictionary, so no model here can read them exactly (a ~0.06 % floor shared by
  every row).
- On synthetic uniform images wider than ~180 px the model can emit a lone `タ`
  (base returns empty). Real detector boxes rarely look like this: of 2,210
  sampled boxes only 1 was near-blank, and it produced no text.
