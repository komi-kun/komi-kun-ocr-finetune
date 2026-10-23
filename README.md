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

## 3. Pipeline

```bash
# 0. environment (Python 3.12, Paddle GPU for sm_120/Blackwell)
uv venv --python 3.12 .venv
uv pip install -r requirements.txt            # see file for the paddle index url

# 1. real data (needs an HF account that accepted the dataset terms)
python scripts/fetch_manga109.py --repo hal-utokyo/Manga109-s
git clone https://github.com/manga109/public-annotations data/raw/COO-public
python scripts/build_real_dataset.py --coo data/raw/COO-public/COO-Comic-Onomatopoeia/v2022.07.07

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
Only the recogniser is fine-tuned — the detector is unchanged.

## 4. Results

Real Manga109-s, held out **by book** (9 test books, 969 pages, 13,083 blocks).
`base` = shipped model, `stage-1` = synthetic only, `stage-2` = synthetic + real.

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

Full tables, caveats, artifact hash and the Paddle↔ONNX agreement check are in
[`RESULTS.md`](RESULTS.md); the model card with the required Manga109-s
attribution is [`models/MODEL_CARD.md`](models/MODEL_CARD.md).

## 5. Notes

* **Dictionary is frozen.** Training uses `models/ppocrv6_small_dict.txt`
  verbatim (identical to PaddleOCR's `ppocrv6_dict.txt`), so the exported ONNX
  keeps the `['#'] + dict + [' ']` layout the app already assumes.
* **Long lines.** The runtime clamps the recogniser input to 320 px wide at 48 px
  high; the pipeline evaluates with the same clamp, so numbers are comparable to
  on-device behaviour.
* **Licensing.** This workspace is tooling + weights derived from CC-BY
  annotations and academically-licensed images; the resulting weights are meant
  for this fork's personal use. Do not redistribute Manga109 page images.
* **Fonts** are OFL and are not redistributed in the app; synthetic images are
  generated locally and also not committed.
