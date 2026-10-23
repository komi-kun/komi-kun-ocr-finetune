# Status: complete (2026-10-23)

The full offline run finished. Final artifact:

**`models/ppocrv6_small_rec_manga_real.onnx`** · 21 159 418 B ·
sha256 `3e898ae24a566d3606f459eaf9ee69b3110e8ea50f200701dc8abb5403467a5d`

## What was produced

| | |
|---|---|
| Dialogue lines kept | 211,710 (from 272,853 candidate crops, 77 %) |
| COO SFX crops | 16,797 (straight 4-point polygons) |
| Merged real labels | 227,520 (train 192,964 / val 9,989 / test 24,567) |
| Training set | 252,366 lines (real + 59,402 synthetic), 11,787 val |
| Stage-2 training | 8 epochs, best val exact 78.36 % (epoch 7) |
| Eval | 9 held-out books, 969 pages, 13,083 blocks |

## Headline numbers (held-out books)

| | base | stage-1 (synthetic) | **stage-2 (real)** |
|---|---|---|---|
| block CER (end-to-end) | 28.05 % | 30.21 % | **25.45 %** |
| block exact | 38.42 % | 30.45 % | **43.63 %** |
| line CER — dialogue | 8.73 % | 11.07 % | **6.51 %** |
| line CER — SFX | 77.56 % | 51.33 % | **17.12 %** |
| SFX exact | 13.69 % | 37.99 % | **75.63 %** |

Synthetic-only training helped SFX but hurt real dialogue; stage-2 fixes both.

## Next steps

1. **Ship it** — follow `INTEGRATION.md`: publish release `ocr-manga-v2` with
   `models/manifest.ocr-manga-v2.json`, bump `REC_MODEL` + `DEFAULT_RELEASE` in
   `:core:ocr`, add the required Manga109-s attribution (see `MODEL_CARD.md`).
2. **Optional**: fine-tune the detector on COO's multi-point polygons — curved
   onomatopoeia is still split before recognition.
3. Everything is documented in `RESULTS.md`, `README.md`, `INTEGRATION.md`.
