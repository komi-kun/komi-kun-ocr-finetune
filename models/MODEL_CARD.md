# Model card — `ppocrv6_small_rec_manga_real.onnx`

| | |
|---|---|
| Base | PP-OCRv6_small_rec (`PPLCNetV4` small + LightSVTR + CTC/NRTR multi-head, 7.7 M params) |
| Task | Japanese line recognition (CTC head used at inference) |
| Input | `x` `[N, 3, 48, W]`, BGR, normalised `(p-0.5)/0.5`, `W` dynamic (runtime clamps to 320) |
| Output | `[N, T, 18710]` logits — `['#'] + ppocrv6_dict(18708) + [' ']` |
| Opset | 11 |
| Size / sha256 | 21 159 418 B / `3e898ae24a566d3606f459eaf9ee69b3110e8ea50f200701dc8abb5403467a5d` |
| Init | PP-OCRv6_small_rec_pretrained → synthetic stage-1 → this checkpoint |

## Training

Two stages, recogniser only (detector unchanged):

1. **Synthetic** — 60k manga-style lines from OFL fonts + Wikipedia-ja corpus,
   vertical/horizontal, bubbles/screentone/dark art, stroke outlines, rotation,
   perspective, blur, noise (4 display fonts held out).
2. **Real** — 210,723 dialogue lines and 16,797 onomatopoeia crops mined from
   Manga109-s v2026 (87 books / 8,519 pages) + COO. Blocks are split into lines
   with the detector, the transcript is allocated via a character alignment with
   the `manga-ocr` teacher, and a line is kept only when the teacher agrees with
   its slice (77 % acceptance). Split **by book**: 69 train / 9 val / 9 test.
   Mixed with 59,402 synthetic lines (252,366 train total).

Recipe: Adam, cosine LR 1e-4, warmup 1 epoch, 8 epochs, batch 128,
MultiScaleSampler `[320×32, 320×48, 320×64]`, RecAug + RecConAug, CTC+NRTR loss.

## Evaluation

See `RESULTS.md` §"Released combinations" for the four shipping releases head to
head. Held-out Manga109-s books, recogniser only (stock detector, i.e.
`ocr-models-v3` → `ocr-manga-v2`): block-level end-to-end CER 28.05 % → **25.45 %**,
exact 38.42 % → **43.63 %**. Line-level SFX CER 77.56 % → **17.12 %**, dialogue
8.73 % → **6.51 %**. General Japanese text 3.98 % → 0.53 % CER.

## Released combinations — this recogniser vs stock, with and without the new detector

Four release tags on `komi-kun/komi-kun-ocr-models`, four (detector, recogniser)
pairs, all on the same held-out test split (9 books, never trained on, never used
for selection). `komi-kun/komi-kun-ocr-finetune` publishes no releases.

| Release | Detector | Recogniser |
|---|---|---|
| `ocr-models-v3` (previous stock) | stock `d73e0058…` | stock `5435fd74…` |
| `ocr-manga-v2` (previous fine-tune) | stock `d73e0058…` | **this** recogniser `3e898ae2…` |
| `ocr-models-v4` (new stock-rec) | fine-tuned `bf5ed74a…` | stock `5435fd74…` |
| `ocr-manga-v3` (new fine-tune) | fine-tuned `bf5ed74a…` | **this** recogniser `3e898ae2…` |

**Recogniser quality, detector held fixed**
(`ocr-models-v3` → `ocr-manga-v2`): Manga109 dialogue block CER 28.05 → 25.45 %
(exact 38.42 → 43.63 %), and on the COO onomatopoeia end-to-end CER 91.37 →
75.53 % with 246 → 1 120 SFX read character-for-character.

**Detector quality, recogniser held fixed** (`ocr-manga-v2` → `ocr-manga-v3`):
COO SFX found 32.07 → 69.91 %, dialogue block coverage 92.10 → 94.94 %, art false
positives 5.35 → 1.35 per page, and end-to-end (COO SFX) CER **75.53 → 51.38 %**
with exact **1 120 → 2 453**; Manga109 dialogue block CER 27.95 → 26.92 % and
exact 43.72 → 45.64 % on the same 12 945 blocks. Every paired change is
significant (binomial p < 1e-4); nothing regresses.

The recogniser alone fixes *reading* but not *finding* (4 510 of 6 639 held-out
SFX still get no detection), and the detector alone fixes *finding* but not
*reading* (SFX CER only 91.37 → 86.64 % with the stock recogniser). The pair is
what makes curved and straight onomatopoeia actually readable; the numbers are in
`RESULTS.md` §"Released combinations".

## Intended use / limits

Drop-in for the recogniser `:core:ocr` ships; no API change.

The model was trained on detector-style crops. On artificial uniform images
wider than ~180 px it can emit a lone `タ`; on 2,210 sampled real detector boxes
only 1 was near-blank and it produced no text, so this does not occur in the app
pipeline.

## Companion detector — `ppocrv6_small_det.onnx`

Shipped alongside this recogniser since `ocr-manga-v3` / `ocr-models-v4`; it is a
fine-tune of the same PP-OCRv6_small_det backbone.

| | |
|---|---|
| Task | DB text detection, manga SFX + dialogue lines |
| Input | `x` `[N, 3, H, W]`, dynamic batch **and** H/W, BGR `(p-0.5)/0.5` |
| Output | `[N, 1, H, W]` probability map (postprocess: 960 max side, thresh 0.15, box thresh 0.25, unclip 1.40) |
| Opset | 14 |
| Size / sha256 | 9 929 569 B / `bf5ed74a98dbbb16f6f4122fd1660692eeccc651933cc7582e38b112dbf01677` |
| Init | `PP-OCRv6_small_det_pretrained` → 40 epochs on Manga109-s |

Training set: COO onomatopoeia polygons as SFX positives **plus** line-level
dialogue supervision distilled from the shipped (stock) detector, because
MangaOCR provides block-level dialogue boxes only. Held-out test (9 books,
982 pages): SFX H 32.78 → **65.03 %**, SFX crop recall 32.07 → **69.91 %**,
dialogue block coverage 92.10 → **94.94 %**, art false positives 5.35 → **1.35**
per page; end to end, COO SFX CER 75.53 → **51.38 %** and exact 1 120 → **2 453**
over the previous fine-tune release, with the paired dialogue block CER improving
by 1.03 pp (12 945 blocks). Full tables in `RESULTS.md` §"Experiment 2".

**Limit that remains:** the runtime still hands this recogniser the `minAreaRect`
*rectangular* crop of a detected SFX, so a strongly curved onomatopoeia is boxed
as one region but not unwarped. This detector stops the curve being split into
fragments; reading it needs a polygon→straight-strip unwarp in the native shim.

## Licensing and attribution

Weights are fine-tuned from Apache-2.0 PP-OCRv6 weights.

**This model was trained using the Manga109-s dataset**, provided by the
Aizawa–Yamakata–Matsui Laboratory, The University of Tokyo. Manga109-s permits
machine-learning experiments and commercial use of the results, but requires
attribution when results (including pre-trained models) are published, and
forbids redistribution of the images (none are redistributed here).

The onomatopoeia annotations come from **COO** (Baek et al., ECCV 2022),
licensed CC BY 4.0.

Cite:

```bibtex
@inproceedings{baek2022COO,
  title={COO: Comic Onomatopoeia Dataset for Recognizing Arbitrary or Truncated Texts},
  author={Baek, Jeonghun and Matsui, Yusuke and Aizawa, Kiyoharu},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022},
}
@inproceedings{baek2026mangav26,
  title={{Manga109-v2026: Revisiting Manga109 Annotations for Modern Manga Understanding}},
  author={Baek, Jeonghun and Miyai, Atsuyuki and Onohara, Shota and Ikuta, Hikaru and Aizawa, Kiyoharu},
  booktitle={Culture × AI Workshop at ICML 2026},
  year={2026},
}
@article{multimedia_aizawa_2020,
  author={Kiyoharu Aizawa and Azuma Fujimoto and Atsushi Otsubo and Toru Ogawa and Yusuke Matsui and Koki Tsubota and Hikaru Ikuta},
  title={Building a Manga Dataset ``Manga109'' with Annotations for Multimedia Applications},
  journal={IEEE MultiMedia},
  volume={27}, number={2}, pages={8--18}, year={2020},
}
```
