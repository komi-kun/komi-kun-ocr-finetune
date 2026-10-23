# Shipping the fine-tuned recogniser in komi-kun

Ship **one file**: `models/ppocrv6_small_rec_manga_real.onnx` (21 159 418 B,
sha256 `3e898ae24a566d3606f459eaf9ee69b3110e8ea50f200701dc8abb5403467a5d`).
Detector and dictionary are unchanged, and the ONNX contract is identical
(`x [N,3,48,W]` → `[N,T,18710]`, opset 11), so the app needs no pipeline change.

## 1. Publish the artifact

Create a GitHub release on `komi-kun/komi-kun-ocr-models`, tag `ocr-manga-v2`,
containing `manifest.json` plus:

```
ppocrv6_small_det.onnx         (unchanged)
ppocrv6_small_rec_manga_real.onnx
ppocrv6_small_dict.txt         (unchanged)
```

A ready-to-use manifest is `models/manifest.ocr-manga-v2.json`, already filled
with the hashes/sizes:

```bash
cp models/manifest.ocr-manga-v2.json /path/to/release/manifest.json
```

The runtime downloader only requires `det`, `rec`, `dict`; the `.so`s stay on the
build-time `ocr-models-v3` release.

## 2. App changes (`core/ocr`)

All inside `// KUN` code.

**`paddle/PaddleOcrEngine.kt`**

```kotlin
private const val REC_MODEL = "ppocrv6_small_rec_manga_real.onnx"
```

**`paddle/ModelDownloader.kt`**

```kotlin
private const val DEFAULT_RELEASE = "ocr-manga-v2"
```

and add the stock weights to the superseded list so upgraded installs drop them:

```kotlin
"ppocrv6_small_rec.onnx",
```

**`core/ocr/paddle-manifest.json`** — update the documented model entry (the
build-time `.so` fetch keeps pointing at `ocr-models-v3`; entries without `abi`
are not fetched).

## 3. Attribution (required)

Manga109-s must be credited when the model is published. Add the paragraph from
`models/MODEL_CARD.md` §"Licensing and attribution" to:

* `NOTICE` / `LICENSES` in the app repo, and
* the OCR settings "about models" text (or the model download dialog).

Minimum wording:

> This OCR model was trained using the Manga109-s dataset (Aizawa–Yamakata–Matsui
> Laboratory, The University of Tokyo) and the COO onomatopoeia annotations
> (Baek et al., ECCV 2022, CC BY 4.0). Manga109-s images are not redistributed.

plus the BibTeX from the model card.

## 4. Verify

```bash
./gradlew verifyGplFree          # untouched, must stay green
./gradlew spotlessCheck
./gradlew :core:ocr:testDebugUnitTest
```

Install, enable OCR, let it download `ocr-manga-v2`, and compare a page with a
lot of SFX against the previous build.

## 5. Rollback

Revert `REC_MODEL` + `DEFAULT_RELEASE`; the downloader re-fetches `ocr-models-v3`.
Delete `filesDir/paddle_ocr/` if a stale manifest sticks.
