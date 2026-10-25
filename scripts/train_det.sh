#!/usr/bin/env bash
# Fine-tune PP-OCRv6_small_det on the manga detection dataset.
#
# Reads configs/manga_det.yml, applies a *minimal* set of run overrides, writes
# the derived config to output/det_configs/<tag>.yml (so every run is
# reproducible from the file next to its log) and starts PaddleOCR training.
#
# Examples
#   # 2-book smoke test: 20 ARMS pages to train on, 8 JijiBabaFight pages to validate
#   mkdir -p output/det_tiny
#   grep 'images/ARMS/' data/det/train.txt | head -20 > output/det_tiny/train_tiny.txt
#   grep 'images/JijiBabaFight/' data/det/val.txt | head -8 > output/det_tiny/val_tiny.txt
#   bash scripts/train_det.sh --tag tiny --epochs 2 --train-list output/det_tiny/train_tiny.txt \
#        --val-list output/det_tiny/val_tiny.txt --save-dir output/det_tiny --batch-size 8
#   bash scripts/train_det.sh --tag fliplr05    --epochs 5  --fliplr 0.5 --save-dir output/det_fliplr05
#   bash scripts/train_det.sh --tag fliplr00    --epochs 5  --fliplr 0.0 --save-dir output/det_fliplr00
#   bash scripts/train_det.sh --tag full        --epochs 40 --save-dir output/det_manga
#   bash scripts/train_det.sh --tag full        --epochs 40 --save-dir output/det_manga \
#        --resume output/det_manga/latest            # resume from a checkpoint
#
# Notes
#   * Run from anywhere; the script cd's to the repo root because the label
#     files use repo-relative image paths (data_dir: ./).
#   * `-o` can only override dict keys in PaddleOCR, not list elements, which is
#     why the derived config is written instead of passing every knob on the CLI.
#   * Long runs get `nice -n 10` by default (NICE=0 to disable).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_CONFIG="$ROOT/configs/manga_det.yml"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

tag=""
epochs=""
fliplr=""
batch_size=""
workers=""
lr=""
seed=""
train_list=""
val_list=""
save_dir=""
save_epoch_step=""
no_copy_paste=""
pretrained="__unset__"
resume=""

usage() {
  sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) tag="$2"; shift 2 ;;
    --epochs) epochs="$2"; shift 2 ;;
    --fliplr) fliplr="$2"; shift 2 ;;
    --batch-size) batch_size="$2"; shift 2 ;;
    --workers) workers="$2"; shift 2 ;;
    --lr) lr="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --train-list) train_list="$2"; shift 2 ;;
    --val-list) val_list="$2"; shift 2 ;;
    --save-dir) save_dir="$2"; shift 2 ;;
    --save-epoch-step) save_epoch_step="$2"; shift 2 ;;
    --no-copy-paste) no_copy_paste=1; shift ;;
    --pretrained) pretrained="$2"; shift 2 ;;
    --resume) resume="$2"; shift 2 ;;
    -h|--help) usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 1 ;;
  esac
done

[[ -n "$tag" ]] || { echo "--tag is required" >&2; usage 1; }
[[ -n "$epochs" ]] || { echo "--epochs is required" >&2; usage 1; }
save_dir="${save_dir:-$ROOT/output/det_$tag}"

cfg_dir="$ROOT/output/det_configs"
log_dir="$ROOT/output/logs"
mkdir -p "$cfg_dir" "$log_dir"
cfg="$cfg_dir/$tag.yml"
log="$log_dir/$tag.log"

"$PYTHON" - "$BASE_CONFIG" "$cfg" "$epochs" "$fliplr" "$batch_size" "$workers" \
  "$lr" "$seed" "$train_list" "$val_list" "$save_dir" "$save_epoch_step" \
  "$no_copy_paste" "$pretrained" "$resume" <<'PY'
import sys
from pathlib import Path

import yaml

(base, out, epochs, fliplr, batch, workers, lr, seed,
 train_list, val_list, save_dir, save_epoch_step, no_copy_paste,
 pretrained, resume) = sys.argv[1:]

cfg = yaml.safe_load(Path(base).read_text())

cfg["Global"]["epoch_num"] = int(epochs)
for i, tr in enumerate(cfg["Train"]["dataset"]["transforms"]):
    if isinstance(tr, dict) and "MakeBorderMap" in tr:
        tr["MakeBorderMap"]["total_epoch"] = int(epochs)
    if isinstance(tr, dict) and "MakeShrinkMap" in tr:
        tr["MakeShrinkMap"]["total_epoch"] = int(epochs)

if fliplr:
    (iaa,) = [t["IaaAugment"] for t in cfg["Train"]["dataset"]["transforms"]
              if isinstance(t, dict) and "IaaAugment" in t]
    (flip,) = [a for a in iaa["augmenter_args"] if a["type"] == "Fliplr"]
    flip["args"]["p"] = float(fliplr)
if batch:
    cfg["Train"]["loader"]["batch_size_per_card"] = int(batch)
if workers:
    cfg["Train"]["loader"]["num_workers"] = int(workers)
if lr:
    cfg["Optimizer"]["lr"]["learning_rate"] = float(lr)
if seed:
    cfg["Global"]["seed"] = int(seed)
if train_list:
    cfg["Train"]["dataset"]["label_file_list"] = [str(Path(train_list).resolve())]
if val_list:
    cfg["Eval"]["dataset"]["label_file_list"] = [str(Path(val_list).resolve())]
if pretrained != "__unset__":
    # an empty --pretrained '' means "train from scratch"
    cfg["Global"]["pretrained_model"] = pretrained or None
cfg["Global"]["save_model_dir"] = str(Path(save_dir).resolve())
if save_epoch_step:
    cfg["Global"]["save_epoch_step"] = int(save_epoch_step)
if no_copy_paste:
    # PaddleOCR's CopyPaste ext-data sampler only accepts pages whose every
    # polygon has exactly 4 points (ppocr/data/simple_dataset.py), so with
    # arbitrary-point labels it re-decodes dozens of pages per sample.
    cfg["Train"]["dataset"]["transforms"] = [
        t for t in cfg["Train"]["dataset"]["transforms"]
        if not (isinstance(t, dict) and "CopyPaste" in t)
    ]
if resume:
    cfg["Global"]["checkpoints"] = str(Path(resume).resolve())

Path(out).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
print(f"config -> {out}")
PY

cd "$ROOT"
echo "training: config=$cfg log=$log"
echo "  PYTHONPATH=\"$ROOT/PaddleOCR:$ROOT\" $PYTHON PaddleOCR/tools/train.py -c $cfg"
NICE_CMD=()
if [[ "${NICE:-10}" != "0" ]]; then
  NICE_CMD=(nice -n "${NICE:-10}")
fi
PYTHONPATH="$ROOT/PaddleOCR:$ROOT" "${NICE_CMD[@]}" "$PYTHON" \
  "$ROOT/PaddleOCR/tools/train.py" -c "$cfg" 2>&1 | tee "$log"
