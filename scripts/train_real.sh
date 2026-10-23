#!/usr/bin/env bash
# Stage 2: adapt the synthetic fine-tune on real Manga109-s + COO lines.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
PY=$ROOT/.venv/bin/python
ARGS=(-c "$ROOT/configs/manga_real_rec.yml")
if [[ "${1:-}" == "--smoke" ]]; then
  ARGS+=(-o Global.epoch_num=1 Global.print_batch_step=1 Global.eval_batch_step=[0,5] \
            Global.save_epoch_step=100 Train.loader.batch_size_per_card=8 \
            Train.loader.num_workers=2 Eval.loader.batch_size_per_card=8 \
            Global.save_model_dir="$ROOT/output/smoke_real")
fi
cd "$ROOT"
PYTHONPATH="$ROOT/PaddleOCR:$ROOT" "$PY" PaddleOCR/tools/train.py "${ARGS[@]}"
