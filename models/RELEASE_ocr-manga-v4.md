# Release `ocr-manga-v4` — stock detector + fine-tuned manga recogniser

Published: <https://github.com/komi-kun/komi-kun-ocr-models/releases/tag/ocr-manga-v4>

- **Release name:** `OCR models v4 — stock detector + fine-tuned manga recogniser (PP-OCRv6 small)`
- **Prerelease:** no. **Draft:** no. **Marked latest:** yes.
  This matches the other four model releases (`ocr-models-v3`, `ocr-manga-v2`,
  `ocr-manga-v3`, `ocr-models-v4` are all `prerelease: false`, `draft: false`), and
  GitHub's single "Latest" pointer moves to the newest model release — it sat on
  `ocr-manga-v3` before this one.
- **Why it exists:** it is the repaired `ocr-manga-v2`. v2's *weights* were fine;
  its `manifest.json` had no `"path"` on any entry, so the app fell back to
  `$baseUrl/models/<name>` and 404'd on all three files. `ocr-manga-v2` was left
  untouched (never re-tagged, edited or re-uploaded) — this is a new tag.

## Contents (all PP-OCRv6 small)

| asset | size (B) | SHA-256 |
|---|---|---|
| `manifest.json` | 846 | `f179e2b7e86057fde3ef580ef7f24e0501b4f67457af2ee63193fd3b8ce3e534` |
| `ppocrv6_small_det.onnx` (stock detector) | 9,880,512 | `d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e` |
| `ppocrv6_small_rec_manga_real.onnx` (fine-tuned recogniser) | 21,159,418 | `3e898ae24a566d3606f459eaf9ee69b3110e8ea50f200701dc8abb5403467a5d` |
| `ppocrv6_small_dict.txt` (shared dictionary) | 74,947 | `b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d` |

Local sources: `models/ppocrv6_small_det_stock.onnx` (uploaded as
`ppocrv6_small_det.onnx` — the app's `DET_MODEL` filename),
`models/ppocrv6_small_rec_manga_real.onnx`, `models/ppocrv6_small_dict.txt`.

## Release matrix

Two binary choices (detector, recogniser) over five tags; the all-stock and
all-fine-tuned corners were already published.

| release | detector | recogniser | state |
|---|---|---|---|
| `ocr-manga-v2` | stock `d73e0058…` | fine-tuned `3e898ae2…` | **broken manifest** (no `path`) — untouched |
| `ocr-models-v3` | stock `d73e0058…` | stock `5435fd74…` | ok |
| **`ocr-manga-v4`** | **stock `d73e0058…`** | **fine-tuned `3e898ae2…`** | ok — the fixed `ocr-manga-v2` |
| `ocr-models-v4` | fine-tuned `bf5ed74a…` | stock `5435fd74…` | ok |
| `ocr-manga-v3` | fine-tuned `bf5ed74a…` | fine-tuned `3e898ae2…` | ok |

It is also the drop-in recogniser upgrade for `ocr-models-v3`: one weight file
changes (`ppocrv6_small_rec.onnx` → `ppocrv6_small_rec_manga_real.onnx`), the
detector and dictionary bytes are identical.

## Provenance

Identical bytes to already-published assets, cross-checked over HTTP:

- detector = `ocr-models-v3/ppocrv6_small_det.onnx` (byte-identical, `cmp`).
- recogniser = `ocr-manga-v3/ppocrv6_small_rec_manga_real.onnx` (byte-identical, `cmp`).
- dictionary = the same `ppocrv6_small_dict.txt` hash published in all four earlier releases.

Training recipe, dataset split and licensing are unchanged from the recogniser
release; see `MODEL_CARD.md` (base PP-OCRv6_small_rec, synthetic stage 1 → 252,366
manga lines from Manga109-s v2026 + COO, split by book 69/9/9). Results, since the
weights are byte-identical to `ocr-manga-v2`: Manga109 dialogue block CER
28.05 → 25.45 %, block exact 38.42 → 43.63 %, COO onomatopoeia end-to-end CER
91.37 → 75.53 %, SFX lines read character-for-character 246 → 1,120
(stock detector held fixed). See `RESULTS.md` §"Released combinations".

Licences: Apache-2.0 (fine-tuned from Apache-2.0 PP-OCRv6-small weights).
Manga109-s images are not redistributed; COO annotations are CC BY 4.0.

## Manifest schema rationale

The app parses `manifest.json` as (GPL-free `core/ocr`,
`ModelDownloader.kt:48-72`):

```kotlin
data class OcrModelFile(val name: String, val path: String? = null,
                        val sha256: String = "", val size: Long = 0,
                        val abi: String? = null, val spdx: String? = null)
data class OcrModelManifest(val release: String = "", val baseUrl: String = "",
                            val files: List<OcrModelFile> = emptyList())
```

and builds each download URL as (`ModelDownloader.kt:265`):

```kotlin
entries += entry to "$targetBaseUrl/${entry.path ?: "models/${entry.name}"}"
```

So the field is **optional in the type but required in practice**: omit it and the
app guesses `models/<name>`, which is only correct for a release whose assets sit
under a `models/` prefix. These releases upload assets flat at the release root, so
every entry must carry an explicit flat `"path"`, identical to `"name"`:

```json
{
  "release": "ocr-manga-v4",
  "baseUrl": "https://github.com/komi-kun/komi-kun-ocr-models/releases/download/ocr-manga-v4",
  "files": [
    {
      "name": "ppocrv6_small_det.onnx",
      "path": "ppocrv6_small_det.onnx",
      "sha256": "d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e",
      "size": 9880512,
      "spdx": "Apache-2.0"
    },
    {
      "name": "ppocrv6_small_rec_manga_real.onnx",
      "path": "ppocrv6_small_rec_manga_real.onnx",
      "sha256": "3e898ae24a566d3606f459eaf9ee69b3110e8ea50f200701dc8abb5403467a5d",
      "size": 21159418,
      "spdx": "Apache-2.0"
    },
    {
      "name": "ppocrv6_small_dict.txt",
      "path": "ppocrv6_small_dict.txt",
      "sha256": "b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d",
      "size": 74947,
      "spdx": "Apache-2.0"
    }
  ]
}
```

Note also that `release` must equal the tag the app pins, and each `sha256` must be
present — `downloadModels` fails fast otherwise (`ModelDownloader.kt:256-265`):
`"Manifest is missing $name"` / `"Manifest is missing a SHA-256 for $name"`.

## Verification commands (reproduce)

```bash
# 1. local files match the expected hashes/sizes
cd /home/luca/development/ppocrv6-manga-ft/models
sha256sum ppocrv6_small_det_stock.onnx ppocrv6_small_rec_manga_real.onnx ppocrv6_small_dict.txt
#   d73e0058…  ppocrv6_small_det_stock.onnx        (9,880,512)
#   3e898ae2…  ppocrv6_small_rec_manga_real.onnx   (21,159,418)
#   b5f2bfe2…  ppocrv6_small_dict.txt              (74,947)

# 2. cross-check against already-published releases (plain curl, no auth cache)
B=https://github.com/komi-kun/komi-kun-ocr-models/releases/download
curl -sSL -o det.onnx "$B/ocr-models-v3/ppocrv6_small_det.onnx"
curl -sSL -o rec.onnx "$B/ocr-manga-v3/ppocrv6_small_rec_manga_real.onnx"
cmp det.onnx ppocrv6_small_det_stock.onnx && cmp rec.onnx ppocrv6_small_rec_manga_real.onnx
#   both byte-identical

# 3. never edit the broken release
gh release view ocr-manga-v2 -R komi-kun/komi-kun-ocr-models --json publishedAt
#   publishedAt unchanged: 2026-10-23T21:26:49Z
```

HTTP result for the published release — every asset downloaded and hashed, and
every manifest `path` resolved:

```bash
B=https://github.com/komi-kun/komi-kun-ocr-models/releases/download/ocr-manga-v4
for f in manifest.json ppocrv6_small_det.onnx ppocrv6_small_rec_manga_real.onnx ppocrv6_small_dict.txt
do curl -sSL -o "$f" "$B/$f" -w "$f http=%{http_code} bytes=%{size_download}\n"; done
sha256sum *
curl -sIL -o /dev/null -w '%{http_code} %{size_download} %{url_effective}\n' "$B/ppocrv6_small_det.onnx"
```

```
manifest.json                       http=200 bytes=846         sha256=f179e2b7… (round-trip identical to the uploaded bytes)
ppocrv6_small_det.onnx              http=200 bytes=9880512     sha256=d73e0058…  OK vs manifest
ppocrv6_small_rec_manga_real.onnx   http=200 bytes=21159418    sha256=3e898ae2…  OK vs manifest
ppocrv6_small_dict.txt              http=200 bytes=74947       sha256=b5f2bfe2…  OK vs manifest
```

`curl -sIL` on each manifest `path` field:

| URL | status | Content-Length |
|---|---|---|
| `…/ocr-manga-v4/ppocrv6_small_det.onnx` | 200 | 9880512 |
| `…/ocr-manga-v4/ppocrv6_small_rec_manga_real.onnx` | 200 | 21159418 |
| `…/ocr-manga-v4/ppocrv6_small_dict.txt` | 200 | 74947 |

For contrast, the v2 failure mode (`models/` fallback) still 404s:

```
curl -sIL "…/ocr-manga-v2/models/ppocrv6_small_det.onnx"  ->  HTTP 404
```
