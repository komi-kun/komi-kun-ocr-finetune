# Offline run — no network needed

Everything the pipeline touches is already on disk:

| Needed | Status |
|---|---|
| Manga109-s v2026 (87 books, 8,519 pages) | ✅ `data/raw/Manga109-s` (6.4 GB) |
| COO annotations | ✅ `data/raw/COO-public` + `annotations_COO/` in the release |
| 272,853 dialogue line crops | ✅ `data/real_lines` (1.3 GB) |
| 60,000 synthetic lines + eval sets | ✅ `data/synth`, `data/synth_eval`, `data/general_eval` |
| Stage-1 checkpoint | ✅ `output/rec_manga/best_accuracy.pdparams` |
| `manga-ocr` teacher (HF) | ✅ cached; verified with `HF_HUB_OFFLINE=1` |
| Python envs (paddle 3.4 / 3.3.1 / teacher) | ✅ `.venv`, `.venv-export`, `.venv-teacher` |
| PaddleOCR repo, fonts, det/dict models | ✅ local |
| Disk | 991 GB free |

The script sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
`HF_DATASETS_OFFLINE=1`, so nothing will try to reach the internet. If a step
ever fails on a missing cache entry it aborts immediately instead of hanging.

Total time ≈ **2.5–3 h** (25 min labeling, 10 min SFX, 1.5 h training, 2 min
export, ~30 min eval).

---

## 1. Open a detached session

```bash
cd ~/development/ppocrv6-manga-ft
tmux new -s ocr
```

Inside tmux you can disconnect/reconnect freely; the job keeps running.

## 2. Check preflight (5 s)

```bash
bash scripts/run_offline.sh --check
```

Expected: `preflight OK`, 272853 line crops, GPU RTX 5070 Ti, and the list of
steps that will run.

## 3. Start the full run (keeps the PC awake)

```bash
systemd-inhibit --what=idle:sleep --why="OCR fine-tune" \
  bash scripts/run_offline.sh
```

Detach with **Ctrl-b then d**. Reconnect with `tmux attach -t ocr`.
If you prefer not to keep tmux open, this also works:

```bash
nohup bash scripts/run_offline.sh >/dev/null 2>&1 &
```

## 4. Watch progress

```bash
bash scripts/status_offline.sh      # one-screen summary
tail -f output/run_offline.log      # live log
```

Markers in the log: `1/6 …` through `6/6 …`, then `DONE`.

---

## What each step produces

| Step | Output |
|---|---|
| 1 dialogue labels | `data/real/labels_dialogue.tsv` (~200k lines) |
| 2 COO SFX | `data/real/labels_sfx.tsv` (~15k crops) |
| 3 merge | `data/real/labels.tsv`, `data/manifests/{train,val}.txt` |
| 4 train | `output/rec_real/best_accuracy.pdparams` + `TRAINING_DONE` |
| 5 export | `models/ppocrv6_small_rec_manga_real.onnx` (contract-checked) |
| 6 eval | `output/eval/*_blocks.txt`, `output/eval/*_lines.txt` |

---

## If it stops halfway

Just run step 3 again — finished steps are skipped:

* `data/real/labels_dialogue.tsv` exists → labeling skipped
* `data/real/labels_sfx.tsv` exists → SFX skipped
* `output/rec_real/TRAINING_DONE` exists → training skipped
* the ONNX exists → export skipped

Training is checkpointed every 2 epochs, but a run interrupted **before**
`TRAINING_DONE` starts over. To force a clean retrain:
`rm -rf output/rec_real`.

If the desktop feels sluggish, it is the CPU-side steps (1, 2, 6); they run with
`nice -n 10` and ONNX Runtime is capped at 2 threads/session. Training step 4 is
on the GPU.

## After it finishes — send me these

```bash
cat output/eval/ppocrv6_small_rec_blocks.txt        | grep "block CER"
cat output/eval/ppocrv6_small_rec_manga_blocks.txt  | grep "block CER"
cat output/eval/ppocrv6_small_rec_manga_real_blocks.txt | grep "block CER"
sha256sum models/ppocrv6_small_rec_manga_real.onnx; stat -c %s models/ppocrv6_small_rec_manga_real.onnx
wc -l data/real/labels.tsv
```

Then we update `RESULTS.md` / `MODEL_CARD.md` (including the required
"Manga109-s was used" attribution + citation) and the app-side integration.
