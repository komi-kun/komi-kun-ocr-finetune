#!/usr/bin/env bash
# Export a trained manga detector checkpoint to ONNX (opset 14, single map output).
#
#   scripts/export_det_onnx.sh [checkpoint] [config] [output.onnx]
#
# Paddle 3.4 (training venv) cannot be converted by paddle2onnx 2.1.0, so the
# export runs in .venv-export (Paddle 3.3.1 CPU + paddle2onnx 2.1.0); the PIR
# graph additionally needs scripts/pir_scale_f32.py before conversion.  Finally
# scripts/verify_det_onnx.py checks the runtime contract (one [N,1,H,W] map,
# dynamic H/W, opset 14) and numerical parity against Paddle inference.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
CKPT=${1:-output/det_manga/best_accuracy}
CONFIG=${2:-configs/manga_det.yml}
OUT=${3:-models/ppocrv6_small_det_manga.onnx}
EXP=$ROOT/.venv-export/bin/python
TAG=$(basename "${OUT%.onnx}")
INFER_DIR=$ROOT/output/${TAG}_inference
PADDLE_REF=$ROOT/output/${TAG}_paddle

rm -rf "$INFER_DIR" "$PADDLE_REF"
PYTHONPATH="$ROOT/PaddleOCR:$ROOT" "$EXP" PaddleOCR/tools/export_model.py \
  -c "$ROOT/$CONFIG" \
  -o "Global.pretrained_model=$ROOT/$CKPT" \
     "Global.save_inference_dir=$INFER_DIR"

# keep the untouched export for the Paddle<->ONNX parity check
cp -r "$INFER_DIR" "$PADDLE_REF"

"$ROOT/.venv/bin/python" "$ROOT/scripts/pir_scale_f32.py" "$INFER_DIR/inference.json"

"$ROOT/.venv-export/bin/paddle2onnx" \
  --model_dir "$INFER_DIR" \
  --model_filename inference.json \
  --params_filename inference.pdiparams \
  --save_file "$ROOT/$OUT" \
  --opset_version 14 \
  --enable_onnx_checker True

"$ROOT/.venv/bin/python" "$ROOT/scripts/finalize_det_onnx.py" "$ROOT/$OUT"

"$EXP" "$ROOT/scripts/verify_det_onnx.py" \
  --model "$ROOT/$OUT" \
  --paddle-dir "$PADDLE_REF"

ls -l "$ROOT/$OUT"
sha256sum "$ROOT/$OUT"
