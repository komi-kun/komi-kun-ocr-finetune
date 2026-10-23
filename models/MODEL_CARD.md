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

See `RESULTS.md`. Held-out books, block-level (end-to-end): CER 28.05 % → **25.45 %**,
exact 38.42 % → **43.63 %**. Line-level SFX CER 77.56 % → **17.12 %**, dialogue
8.73 % → **6.51 %**. General Japanese text 3.98 % → 0.53 % CER.

## Intended use / limits

Drop-in for the recogniser `:core:ocr` ships; no API change. Curved onomatopoeia
is still split by the detector before recognition — that needs a detector
fine-tune (COO's polygons are the labels).

The model was trained on detector-style crops. On artificial uniform images
wider than ~180 px it can emit a lone `タ`; on 2,210 sampled real detector boxes
only 1 was near-blank and it produced no text, so this does not occur in the app
pipeline.

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
