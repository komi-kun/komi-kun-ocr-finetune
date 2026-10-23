#!/usr/bin/env bash
# Export a trained recogniser checkpoint to ONNX and verify the runtime contract.
#
#   scripts/export_onnx.sh [checkpoint] [config] [output.onnx]
#
# Paddle 3.4 (training venv) cannot be converted by paddle2onnx 2.1.0, so the
# export runs in .venv-export (Paddle 3.3.1 CPU + paddle2onnx).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
CKPT=${1:-output/rec_manga/best_accuracy}
CONFIG=${2:-configs/manga_ft_rec.yml}
OUT=${3:-models/ppocrv6_small_rec_manga.onnx}
EXP=$ROOT/.venv-export/bin/python
INFER_DIR=$ROOT/output/inference_$(basename "${OUT%.onnx}")

rm -rf "$INFER_DIR"
PYTHONPATH="$ROOT/PaddleOCR:$ROOT" "$EXP" PaddleOCR/tools/export_model.py \
  -c "$ROOT/$CONFIG" \
  -o "Global.pretrained_model=$ROOT/$CKPT" \
     "Global.save_inference_dir=$INFER_DIR"

"$ROOT/.venv-export/bin/paddle2onnx" \
  --model_dir "$INFER_DIR" \
  --model_filename inference.json \
  --params_filename inference.pdiparams \
  --save_file "$OUT" \
  --opset_version 11 \
  --enable_onnx_checker True

"$ROOT/.venv/bin/python" "$ROOT/scripts/verify_onnx.py" \
  --model "$OUT" \
  --reference "$ROOT/models/ppocrv6_small_rec.onnx"
