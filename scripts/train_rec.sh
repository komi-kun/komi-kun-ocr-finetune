#!/usr/bin/env bash
# Fine-tune PP-OCRv6_small_rec on the manga corpus.
#
#   scripts/train_rec.sh              # normal run
#   scripts/train_rec.sh --smoke      # tiny sanity run (no eval)
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
PY=$ROOT/.venv/bin/python

ARGS=(-c "$ROOT/configs/manga_ft_rec.yml")
if [[ "${1:-}" == "--smoke" ]]; then
  ARGS+=(-o Global.epoch_num=1 Global.print_batch_step=1 Global.eval_batch_step=[0,5] \
            Global.save_epoch_step=100 Train.loader.batch_size_per_card=8 \
            Train.loader.num_workers=2 Eval.loader.batch_size_per_card=8 \
            Global.save_model_dir="$ROOT/output/smoke")
fi

cd "$ROOT"
PYTHONPATH="$ROOT/PaddleOCR:$ROOT" "$PY" PaddleOCR/tools/train.py "${ARGS[@]}"
