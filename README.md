# PP-OCRv6-small manga recogniser fine-tune

Fine-tunes the **PP-OCRv6_small_rec** recogniser that `:core:ocr` ships
(`ppocrv6_small_rec.onnx`, 3×48×W, CTC, 18 710 classes) on manga text:
speech bubbles, hand-lettered dialogue, and onomatopoeia / SFX.

The output is a **drop-in ONNX replacement**: same input signature, same
character dictionary, same CTC decode. Only the weights change, so the app-side
integration is a manifest entry plus one constant, not a code change.

---

## 1. Why fine-tune at all

The shipped model is a general 50-language scene-text recogniser. On manga it
loses accuracy on four things:

| Failure mode | Example | Why |
|---|---|---|
| Hand-lettered / display fonts | ゴゴゴ, ズバッ | training fonts are mostly print/UI faces |
| Heavy stroke outlines | white text with thick black outline on dark art | scene text rarely has 3–5 px outlines |
| Rotated / curved SFX | ドドン drawn along an arc, diagonal | rec sees a rotated rect, not the curve |
| Screentone / texture behind text | halftone dots bleeding into glyph edges | scene text backgrounds are cleaner |

Detection already isolates the lines; the recogniser is where these turn into
wrong characters. Fine-tuning targets exactly that.

## 2. Data

### 2.1 Public labelled manga datasets (surveyed 2026-10)

| Dataset | Contents | Size | Licence | Access |
|---|---|---|---|---|
| **Manga109-s v2026** | 87 books, page images + dialogue `text` boxes with transcriptions | ~4k pages | academic use, **auto-gated** | [HF `hal-utokyo/Manga109-s`](https://huggingface.co/datasets/hal-utokyo/Manga109-s) |
| **Manga109 v2026** | 109 books, 21 142 pages, frames/text/faces/bodies | ~21k pages | academic use, **manual gate** | [HF `hal-utokyo/Manga109`](https://huggingface.co/datasets/hal-utokyo/Manga109) |
| **COO** (Comic Onomatopoeia) | onomatopoeia polygons + transcriptions on Manga109 pages | ~9k SFX | annotations **CC BY 4.0** | [GitHub](https://github.com/manga109/public-annotations) |
| **MangaOCR** | page-level bbox+text for dialogue **and** SFX (Manga109 + COO consolidated) | 10 602 pages | **CC BY 4.0** | [HF `hal-utokyo/MangaOCR`](https://huggingface.co/datasets/hal-utokyo/MangaOCR) (annotations only) |
| **Manga109-s line annotations** | line-level boxes + 8-pt polygons, furigana excluded | 380 pages / 14 182 lines | **MIT** | [HF `bluolightning/manga109s-line-annotations`](https://huggingface.co/datasets/bluolightning/manga109s-line-annotations) |
| **Manga109Dialog** | speaker↔text links | 109 books | CC BY 4.0 | [GitHub](https://github.com/manga109/public-annotations) |
| **MangaUB** | manga understanding benchmark prompts | 109 books | CC BY 4.0 | GitHub `woodrush/manga-ub` |
| MangaSeg / Manga-Text-Segmentation | pixel masks for text regions | — | research | CVPR'25 / Zenodo (needs Manga109 images) |
| COMICS / COMICS Text+ | **English** comic bubbles + transcripts | 24k bubbles | research agreement | request |
| eBDtheque | **French** bande dessinée, 100 pages | 100 pages | research | request |
| DCM772 / DCM895 | **English** public-domain comics | ~1k pages | public domain | public |

**What we use:** Manga109-s images + Manga109 `<text>` dialogue boxes + COO
onomatopoeia polygons. The COO and MangaOCR annotations are CC BY 4.0 and can be
kept in this repo; the page images cannot be redistributed (academic licence), so
they live in `data/raw/` and are never committed.

The annotation-only sets are combined per page by
`scripts/build_real_dataset.py` (COO merged into the Manga109 XML, cropped with
`minAreaRect`, oriented against the model, filtered to the shipped dictionary).

### 2.2 Synthetic supplement

`scripts/render_synth.py` renders single-line images from **OFL fonts**
(`fonts/`, 16 Japanese families incl. display faces) and a corpus built from
Japanese Wikipedia (CC BY-SA) + a hand-written onomatopoeia list. Layouts:

* horizontal — upright glyphs, left→right;
* vertical — upright glyphs stacked, then rotated 90° **CCW**, which is exactly
  the orientation the engine hands the recogniser for a vertical manga column
  (verified empirically against the shipped ONNX).

Styles vary fill (black / white outline / colour), stroke width, bubble vs
screentone vs dark-art background, blur, rotation ±2.5°, noise and JPEG quality.
The generator deliberately **holds out four display fonts** (`RampartOne`,
`PottaOne`, `TrainOne`, `ZenAntique`) to produce a leakage-free validation set.

### 2.3 Detector dataset (COO polygons + MangaOCR)

`scripts/build_det_dataset.py` builds PaddleOCR `SimpleDataSet` labels for the
*detector* (DB), so curved onomatopoeia stop being split into pieces.

**MangaOCR has no line-level dialogue GT.** Its `bbox_2d` instances are the
Manga109 `<text>` blocks — whole bubbles of 2–4 columns, even where one MangaOCR
box maps 1:1 to one block (checked by rendering them). Training DB on them as
positives would teach merging, so the two roles differ by split:

| | train | val / test |
|---|---|---|
| COO polygons (4–66 points, `class=sfx`) | positive | positive |
| MangaOCR SFX boxes | positive | positive |
| MangaOCR dialogue boxes (block-level) | **`###` ignore region** | positive GT |

| | |
|---|---|
| Source A | COO `annotations_COO/<book>.xml` polygons, **4–66 points**, kept as polygons |
| Source B | `hal-utokyo/MangaOCR` `bbox_2d` → four-point boxes (`class=dialogue` / `sfx`) |
| Alignment | MangaOCR `id = <book>_<page>` against Manga109-s `images/<book>/<page>.jpg` |
| De-dupe | MangaOCR box dropped when it is the same SFX as a COO polygon: polygon-vs-box IoU > 0.5, or (only without dialogue evidence) the box covers > 50 % of the polygon; COO wins. 48,278 boxes dropped, 99.4 % of the pairs agree on the transcription |
| Class tags | `dialogue` when the box is a Manga109 `<text>` instance (IoU > 0.5) or merges ≥ 2 of them, else `sfx` |
| Ignore regions | train: block-level dialogue only, trimmed off the SFX GT (subtract → hole-free fragments, tolerance 1 %); a box swallowed by SFX is dropped, not ignored over it |
| Split | the exact `build_real_dataset.book_split()` books: 74 / 4 / 9 of 87 (73 train after the exclusion below) |

Outputs (`data/det/`, git-ignored): `{train,val,test}.txt` (8,053 page lines /
154,018 instances: train 39,391 positives + 87,908 `###` ignores, val 7,401,
test 19,318), `{train,val,test}.jsonl` (sidecar: book, page, `class`, `source`,
`role`, `granularity`, match evidence — enough for per-class eval), `stats.json`,
`mangaocr_boxes.jsonl` cache.

```bash
python scripts/build_det_dataset.py                # build data/det/*
python scripts/build_det_dataset.py --self-test    # rule-level assertions
python scripts/build_det_dataset.py --validate-only  # re-check the outputs (exit 1 on failure)
python scripts/build_det_dataset.py --overlay 8 --overlay-out /tmp/det_overlays
```

Consequences for measurement and training (all reported in `stats.json` and
printed by `--validate-only`):

* **Dialogue detection metrics are block/bubble-level**, not line-level: 87,759
  train and 17,005 eval dialogue instances are Manga109 blocks (93,463 of the
  104,764 match exactly one block at IoU > 0.5, 11,290 merge ≥ 2). Report them
  together with an end-to-end block CER (`scripts/eval_blocks.py`), which is the
  metric that actually reflects what the app reads.
* Train has **no positive dialogue label**: dialogue regions are excluded from the
  DB loss, so the fine-tune cannot learn to merge *or* to suppress them; dialogue
  behaviour is inherited from the pretrained detector. `--train-dialogue positive`
  restores the old (unsafe) behaviour for comparison only. **This is experiment 1
  and it is rejected** (dialogue block coverage 92.10 → 25.99 % on test): see below.
* `PrayerHaNemurenai` (a train book) is annotated by MangaOCR on a ~2× larger
  canvas (1,238 of 1,745 boxes fall outside the Manga109-s page), so it is
  excluded from the detector dataset rather than mislabelled.

### 2.4 Dialogue-preserving detector (experiment 2, shipped)

`scripts/build_det_pseudo_dataset.py` writes a **second** dataset, `data/det_pseudo/`,
that keeps `data/det/` intact and fixes the missing supervision instead of
retreating from it: it replays the stock detector over every train page and keeps
its **line quads** as dialogue positives where ≥ 80 % of a quad's area lies inside
a MangaOCR dialogue region, clipped to (region − SFX). The rest of each dialogue
region becomes hole-free `###` ignore fragments, so no ignore can cancel a
positive.

These are **teacher labels, not ground truth** — they encode "keep doing what the
shipped detector does", and the evaluator's independent GT is untouched. Two
consequences to keep in mind: dialogue quality is now capped by the stock
detector, and any error the stock detector makes inside a bubble becomes a
training target. val/test still use `data/det/{val,test}` with their own labels.

```bash
python scripts/build_det_pseudo_dataset.py                     # writes data/det_pseudo/*
python scripts/build_det_pseudo_dataset.py --self-test         # rule-level assertions
```

Result on held-out test: SFX H 32.78 → 65.03 %, SFX crop recall 32.07 → 69.91 %,
dialogue block coverage 92.10 → **94.94 %** (no collapse this time), art false
positives 5.35 → 1.35 per page, paired block CER −1.03 pp. Full tables in
`RESULTS.md` §"Experiment 2".

## 3. Pipeline

```bash
# 0. environment (Python 3.12, Paddle GPU for sm_120/Blackwell)
uv venv --python 3.12 .venv
uv pip install -r requirements.txt            # see file for the paddle index url

# 1. real data (needs an HF account that accepted the dataset terms)
python scripts/fetch_manga109.py --repo hal-utokyo/Manga109-s
git clone https://github.com/manga109/public-annotations data/raw/COO-public
python scripts/build_real_dataset.py --coo data/raw/COO-public/COO-Comic-Onomatopoeia/v2022.07.07

# 1b. detector labels (COO polygons + MangaOCR boxes, page-aligned; see §2.3)
python scripts/build_det_dataset.py

# 2. synthetic data
python scripts/make_corpus.py --n-wiki 40000
python scripts/render_synth.py --n 60000 --exclude-fonts RampartOne,PottaOne,TrainOne,ZenAntique
python scripts/render_synth.py --n 1200 --out data/synth_eval --include-fonts RampartOne,PottaOne,TrainOne,ZenAntique --val-p 0

# 3. manifests + train + export
python scripts/build_manifest.py --synth-ratio 0.5
bash scripts/train_rec.sh
bash scripts/export_onnx.sh

# 4. evaluate base vs fine-tuned on held-out real pages
python scripts/eval_rec.py --model models/ppocrv6_small_rec.onnx     --labels data/manifests/test_tsv.txt
python scripts/eval_rec.py --model models/ppocrv6_small_rec_manga.onnx --labels data/manifests/test_tsv.txt
```

Training recipe: PP-OCRv6_small_rec config (PPLCNetV4-small backbone, LightSVTR
neck, CTC+NRTR multi-head) initialised from the official
`PP-OCRv6_small_rec_pretrained.pdparams`; Adam, cosine LR 2e-4, warmup 2 epochs,
batch 128, MultiScaleSampler `[320×32, 320×48, 320×64]`, RecAug + RecConAug.
Only the recogniser is fine-tuned in this step — the detector is unchanged
*here*; §2.4 fine-tunes it separately.

## 4. Results

Real Manga109-s, held out **by book** (9 test books, 13 083 readable blocks).
`base` = shipped model, `stage-1` = synthetic only, `stage-2` = synthetic + real
(stage-2 is what ships, in `ocr-manga-v2` / `ocr-manga-v3`).

### 4.1 Recogniser file (stage-2 vs base)

Page level (det + crop + rec; no teacher selection):

| Model | block CER | block exact |
|---|---|---|
| base | 28.05 % | 38.42 % |
| stage-1 (synthetic) | 30.21 % | 30.45 % |
| **stage-2 (synthetic + real)** | **25.45 %** | **43.63 %** |

Line level (recogniser only):

| Model | all CER | all exact | dialogue CER | SFX CER | SFX exact |
|---|---|---|---|---|---|
| base | 17.23 % | 70.72 % | 8.73 % | 77.56 % | 13.69 % |
| stage-1 | 16.04 % | 66.15 % | 11.07 % | 51.33 % | 37.99 % |
| **stage-2** | **7.82 %** | **82.22 %** | **6.51 %** | **17.12 %** | **75.63 %** |

Synthetic-only training helped SFX but *hurt* real dialogue; the real-data stage
is the one to ship. General Japanese text also improves (3.98 % → 0.53 % CER).

### 4.2 The four shipping releases, compared

Four tags on `komi-kun/komi-kun-ocr-models`, four (detector, recogniser) pairs.
`komi-kun/komi-kun-ocr-finetune` (this repo) publishes **no** releases. The
dictionary is byte-identical in all four.

| Release | Detector | Recogniser | Role |
|---|---|---|---|
| `ocr-models-v3` | stock `d73e0058…` | stock `5435fd74…` | previous **stock** — baseline |
| `ocr-manga-v2` | stock `d73e0058…` | FT `3e898ae2…` | previous **fine-tune** (rec only) |
| `ocr-models-v4` | FT `bf5ed74a…` | stock `5435fd74…` | new **stock-rec** (FT detector) |
| `ocr-manga-v3` | FT `bf5ed74a…` | FT `3e898ae2…` | new **fine-tune** (both) |

Same held-out test split for all three axes (9 books; 982 pages for the detector,
6 639 COO onomatopoeia and 12 209 dialogue blocks for the end-to-end runs).

**Detector metrics** (`scripts/eval_det.py`, polygon IoU):

| metric | stock det (`ocr-models-v3`, `ocr-manga-v2`) | FT det (`ocr-models-v4`, `ocr-manga-v3`) |
|---|---|---|
| SFX P / R / H @0.5 | 49.62 / 24.48 / 32.78 % | **70.50 / 60.36 / 65.03 %** |
| SFX crop recall @0.4 | 32.07 % (2 129) | **69.91 % (4 641)** |
| — 4-point | 34.80 % | **67.57 %** |
| — 5–8-point | 30.43 % | **71.26 %** |
| — 9+-point (curved) | 27.49 % | **73.97 %** |
| dialogue block coverage ≥50 % | 92.10 % | **94.94 %** |
| art false positives / page | 5.35 | **1.35** |

**End-to-end COO SFX recognition** (`scripts/eval_sfx.py` — detector, crop and
recogniser on the 6 639 held-out COO onomatopoeia; a missed SFX is scored as a
full deletion, so this is the reader-visible error):

| Release | SFX found | end-to-end CER | read exactly | matched-only CER |
|---|---|---|---|---|
| `ocr-models-v3` | 32.07 % | 91.37 % | 3.71 % (246) | 75.69 % |
| `ocr-manga-v2` | 32.07 % | 75.53 % | 16.87 % (1 120) | **31.05 %** |
| `ocr-models-v4` | 69.94 % | 86.64 % | 5.95 % (395) | 81.82 % |
| `ocr-manga-v3` | **69.94 %** | **51.38 %** | **36.95 % (2 453)** | 33.86 % |

By GT complexity (end-to-end CER): 4-point 87.13 → 68.73 → 81.07 → **43.69 %**,
5–8-point 93.12 → 77.46 → 88.60 → **51.76 %**, 9+-point curved 95.89 → 84.93 →
93.39 → **66.50 %** (same release order as above).

**End-to-end Manga109 dialogue blocks** (`scripts/eval_blocks.py`; `ocr-manga-v2`
→ `ocr-manga-v3` paired on the same 12 945 blocks is −1.03 pp CER / +1.92 pp
exact, exact flips 530 fixed vs 281 broken, p < 1e-4):

| Release | blocks read / 13 868 | block CER | block exact | paired CER / exact (12 945) |
|---|---|---|---|---|
| `ocr-models-v3` | 13 083 | 28.05 % | 38.42 % | 29.76 % / 38.51 % |
| `ocr-manga-v2` | 13 083 | 25.45 % | 43.63 % | 27.95 % / 43.72 % |
| `ocr-models-v4` | 13 523 | 28.42 % | 39.88 % | 28.54 % / 41.07 % |
| `ocr-manga-v3` | **13 523** | 26.16 % | **44.48 %** | **26.92 % / 45.64 %** |

Reading the two end-to-end tables together: the recogniser fixes *reading* but not
*finding* (4 510 of 6 639 SFX are still undetected in `ocr-manga-v2`), the
detector fixes *finding* but not *reading* (SFX CER only 91.37 → 86.64 % with the
stock recogniser), and only the pair makes onomatopoeia actually readable. Every
paired change is significant (binomial p < 1e-4) and nothing regresses.

Full tables, per-bucket breakdowns, the reproduce commands and the caveats are in
[`RESULTS.md`](RESULTS.md) §"Released combinations"; the model card with the
required Manga109-s attribution is [`models/MODEL_CARD.md`](models/MODEL_CARD.md).


## 5. Notes

* **What ships.** Two immutable releases on `komi-kun/komi-kun-ocr-models`:
  [`ocr-manga-v3`](https://github.com/komi-kun/komi-kun-ocr-models/releases/tag/ocr-manga-v3)
  (curved-SFX detector + fine-tuned recogniser + dict, 9 929 569 B / 21 159 418 B /
  74 947 B) and
  [`ocr-models-v4`](https://github.com/komi-kun/komi-kun-ocr-models/releases/tag/ocr-models-v4)
  (the same detector + stock recogniser + the unchanged native `.so`s). Detector
  sha256 `bf5ed74a…f01677`, recogniser `3e898ae2…03467a5d`, dict `b5f2bfe2…401c5d`.
  Every manifest entry carries an explicit flat `path`; `models/manifest.ocr-manga-v3.json`
  and `models/manifest.ocr-models-v4.json` are tracked copies of the published ones.
  The two they supersede — `ocr-models-v3` (stock) and `ocr-manga-v2` (recogniser
  only) — stay published and are the baselines in §4.2, so each release is
  attributable to exactly one changed model file.
* **Curved SFX are read, but not unwarped.** The detector now puts one tight
  quadrilateral on a curved onomatopoeia instead of splitting it (9+-point crop
  recall 27.49 → 73.97 %), and with the fine-tuned recogniser that class goes from
  84.93 % to **66.50 %** end-to-end CER over `ocr-manga-v2`. What is still left on
  the table is the crop: the recogniser is handed the `minAreaRect`
  *rectangular* region, so a strong arc is straightened by axis-aligned cropping.
  A polygon→straight-strip unwarp in the native shim is the remaining
  prerequisite, and the polygon the detector already emits is its input.
* **Two evaluation axes for SFX.** `scripts/eval_blocks.py` scores Manga109
  `<text>` dialogue blocks; the detector fine-tune targets COO onomatopoeia, which
  is why `scripts/eval_sfx.py` exists — it runs det → crop → rec on the COO
  polygons and scores a missed SFX as a full deletion. Quote both; a detector gain
  shows up as recall in the former and as CER in the latter.

* **Dictionary is frozen.** Training uses `models/ppocrv6_small_dict.txt`
  verbatim (identical to PaddleOCR's `ppocrv6_dict.txt`), so the exported ONNX
  keeps the `['#'] + dict + [' ']` layout the app already assumes.
* **Long lines.** The runtime clamps the recogniser input to 320 px wide at 48 px
  high; the pipeline evaluates with the same clamp, so numbers are comparable to
  on-device behaviour.
* **Licensing.** The weights are published (Apache-2.0) with the Manga109-s
  attribution required by its licence; the page images themselves are never
  redistributed. See `models/MODEL_CARD.md` and `NOTICE`.
* **Fonts** are OFL and are not redistributed in the app; synthetic images are
  generated locally and also not committed.
