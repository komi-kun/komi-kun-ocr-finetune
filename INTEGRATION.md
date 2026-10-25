# Shipping the fine-tuned recogniser + detector in komi-kun

Ship **two files** (both already published):

| file | size | sha256 | release |
|---|---|---|---|
| `ppocrv6_small_rec_manga_real.onnx` | 21 159 418 B | `3e898ae24a566d3606f459eaf9ee69b3110e8ea50f200701dc8abb5403467a5d` | `ocr-manga-v3` |
| `ppocrv6_small_det.onnx` | 9 929 569 B | `bf5ed74a98dbbb16f6f4122fd1660692eeccc651933cc7582e38b112dbf01677` | `ocr-manga-v3` / `ocr-models-v4` |

The dictionary (`ppocrv6_small_dict.txt`, `b5f2bfe2…401c5d`) and the native
`libpaddle_ocr.so` / `libonnxruntime.so` / `libomp.so` are **unchanged** from
`ocr-models-v3`, so the build-time fetch keeps pointing there. Both ONNX
contracts are identical to what the app already assumes — `rec`
`x [N,3,48,W]` → `[N,T,18710]` opset 11, `det` `x [N,3,H,W]` → `[N,1,H,W]`
opset 14 — so the app needs no pipeline change.

**Model selection.** `RecModelVariant` (`:core/ocr`, default `FINE_TUNED`) switches
between the fine-tuned recogniser (`ocr-manga-v3`) and the stock PP-OCRv6
(`ocr-models-v4`). Both releases carry the **same** curved-SFX detector, so the
switch changes recognition only. Selection is stored in `pref_ocr_rec_model`; the
downloader resolves release, manifest and rec file per choice, the engine reloads
weights on the next page, and the OCR cache key includes the variant. Both rec
files may coexist.

## 1. Published artifacts

Two immutable releases on `komi-kun/komi-kun-ocr-models`, each with `manifest.json`:

- **`ocr-manga-v3`** — `ppocrv6_small_det.onnx` + `ppocrv6_small_rec_manga_real.onnx`
  + `ppocrv6_small_dict.txt`. The default (`FINE_TUNED`) download target.
- **`ocr-models-v4`** — `manifest.json` + `app-paddle-manifest.json` +
  `ppocrv6_small_det.onnx` + `ppocrv6_small_rec.onnx` + `ppocrv6_small_dict.txt`
  + the twelve `.so`s, re-published byte-for-byte from `ocr-models-v3`. The
  `STOCK` download target and a self-contained successor to v3.

Ready-to-use copies live beside the model card:

```bash
cp models/manifest.ocr-manga-v3.json   /path/to/ocr-manga-v3/manifest.json
cp models/manifest.ocr-models-v4.json  /path/to/ocr-models-v4/manifest.json
```

**Every manifest entry must carry an explicit flat `"path"`.** `ModelDownloader`
resolves `$baseUrl/${entry.path ?: "models/${entry.name}"}`
(`core/ocr/.../ModelDownloader.kt`), and release assets are published flat, so an
entry without `path` requests `$baseUrl/models/<name>` and 404s. `ocr-manga-v2`
shipped without it; that is why it is superseded.

The runtime downloader only requires `det`, `rec`, `dict`; the `.so`s stay on the
build-time `ocr-models-v3` release (entries without `abi` are not fetched by
`fetchPaddleNativeLibs`).

## 2. App changes (`core/ocr`)

On `feat/curved-sfx-detector`. For a future model release, update the enum
rather than scattering constants:

```kotlin
// RecModelVariant.kt
enum class RecModelVariant(val recModel: String, val release: String) {
    FINE_TUNED("ppocrv6_small_rec_manga_real.onnx", "ocr-manga-v3"),
    STOCK("ppocrv6_small_rec.onnx", "ocr-models-v4"),
}
```

The detector and dictionary names stay in `ModelDownloader`; `REQUIRED_MODELS`
is derived per variant. `core/ocr/paddle-manifest.json` is deliberately **not**
bumped: it drives the native `.so` fetch only, and those bytes are identical in
v3 and v4, so moving it would re-download ~142 MB per machine for no change.
`RecModelVariantTest` pins the two release tags so a half-finished bump fails.

## 3. Attribution (required)

Manga109-s must be credited when the model is published. The paragraph from
`models/MODEL_CARD.md` §"Licensing and attribution" applies to the detector too
and is already in:

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

Install, enable OCR, let it download `ocr-manga-v3`, and compare a page with a
lot of SFX against the previous build. The detector change is visible in two
ways: a curved onomatopoeia gets **one** box instead of several fragments, and
`art`-class false positives drop ~4× per page.

## 5. Rollback

Revert the two release strings in `RecModelVariant`; the downloader re-fetches
`ocr-manga-v2` / `ocr-models-v3` (stock detector included). Delete
`filesDir/paddle_ocr/` if a stale manifest sticks — the release id in the cached
manifest is what stops an old model set being reused.
