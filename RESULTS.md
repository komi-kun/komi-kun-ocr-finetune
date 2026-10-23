# Results

All inference uses `scripts/eval_rec.py` / `scripts/eval_blocks.py`, i.e. the
runtime path (ONNX Runtime, 3×48×W, BGR, [-1,1], CTC greedy, 320 px clamp).
`base` = shipped `ppocrv6_small_rec`, `stage-1` = synthetic-only fine-tune,
`stage-2` = synthetic + real Manga109-s/COO (`ppocrv6_small_rec_manga_real`).

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

## Caveats

- The **line-level** dialogue numbers are measured on lines that the teacher
  could read consistently with the transcript, so they are optimistic in
  absolute terms; the **block-level** table has no such selection and is the
  number to quote for the app.
- 969 pages is a good but not huge sample; the 9 test books are disjoint from the
  69 training books.
- SFX gains come from COO's straight (4-point) onomatopoeia. Curved SFX are still
  split by the detector before the recogniser sees them — a detector fine-tune is
  the remaining step.
- On synthetic uniform images wider than ~180 px the model can emit a lone `タ`
  (base returns empty). Real detector boxes rarely look like this: of 2,210
  sampled boxes only 1 was near-blank, and it produced no text.
