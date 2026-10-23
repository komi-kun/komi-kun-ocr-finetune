#!/usr/bin/env bash
# Offline end-to-end run: teacher-label real lines -> SFX crops -> train stage 2
# -> export -> evaluate. Safe to re-run: finished steps are skipped.
#
# Run detached so a dropped SSH/mobile connection cannot kill it:
#   cd ~/development/ppocrv6-manga-ft
#   tmux new -s ocr
#   systemd-inhibit --what=idle:sleep --why="OCR fine-tune" bash scripts/run_offline.sh
#   # Ctrl-b then d to detach; `tmux attach -t ocr` to come back
#
# Progress: tail -f output/run_offline.log   (or: bash scripts/status_offline.sh)
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$PWD
mkdir -p output output/eval

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false

LOG=$ROOT/output/run_offline.log
exec > >(tee -a "$LOG") 2>&1

# single instance
exec 9>/tmp/ppocrv6-offline.lock
flock -n 9 || { echo "another run_offline.sh is already active"; exit 1; }

step() { printf '\n===== %s  %s =====\n' "$(date '+%F %H:%M:%S')" "$*"; }
die()  { echo "FATAL: $*"; exit 1; }

# ---------------------------------------------------------------- preflight
step "preflight"
M109=$(find "$ROOT/data/raw/Manga109-s" -type d -name images -exec test -d '{}/../annotations' ';' -print -quit)
[[ -n "${M109:-}" ]] || die "Manga109-s images/annotations not found"
[[ -f $ROOT/data/real_lines/lines.tsv ]] || die "data/real_lines/lines.tsv missing"
[[ -f $ROOT/data/synth/labels.tsv ]] || die "data/synth/labels.tsv missing"
[[ -f $ROOT/output/rec_manga/best_accuracy.pdparams ]] || die "stage-1 checkpoint missing"
for f in models/ppocrv6_small_det.onnx models/ppocrv6_small_dict.txt models/ppocrv6_small_rec.onnx; do
  [[ -f $ROOT/$f ]] || die "$f missing"
done
for py in .venv/bin/python .venv-teacher/bin/python .venv-export/bin/python; do
  [[ -x $ROOT/$py ]] || die "$py missing"
done
echo "Manga109-s : $M109"
echo "line crops : $(wc -l < data/real_lines/lines.tsv)"
echo "synth      : $(wc -l < data/synth/labels.tsv)"
echo "free disk  : $(df -h "$ROOT" | awk 'NR==2{print $4}')"
echo "GPU        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"

if [[ "${1:-}" == "--check" ]]; then
  echo
  echo "preflight OK. this run would do:"
  [[ -s data/real/labels_dialogue.tsv ]] || echo "  1) teacher-label dialogue (25 min)"
  [[ -s data/real/labels_sfx.tsv ]]      || echo "  2) COO SFX crops (10 min)"
  echo "  3) merge + manifests"
  [[ -f output/rec_real/TRAINING_DONE ]] || echo "  4) train stage 2 (1.5 h)"
  [[ -f models/ppocrv6_small_rec_manga_real.onnx ]] || echo "  5) export ONNX"
  echo "  6) evaluate base / stage-1 / stage-2"
  exit 0
fi

# ------------------------------------------------- 1. teacher-label dialogue
if [[ -s $ROOT/data/real/labels_dialogue.tsv ]]; then
  step "1/6 dialogue labels already present ($(wc -l < data/real/labels_dialogue.tsv)) - skipped"
else
  step "1/6 teacher-label $(wc -l < data/real_lines/lines.tsv) line crops (GPU, ~25 min)"
  nice -n 10 "$ROOT/.venv-teacher/bin/python" scripts/label_lines.py \
    --lines data/real_lines/lines.tsv --out data/real --max-cer 0.30
  mv data/real/labels.tsv data/real/labels_dialogue.tsv
  echo "kept $(wc -l < data/real/labels_dialogue.tsv)"
fi

# ------------------------------------------------------------ 2. COO SFX
if [[ -s $ROOT/data/real/labels_sfx.tsv ]]; then
  step "2/6 SFX labels already present ($(wc -l < data/real/labels_sfx.tsv)) - skipped"
else
  step "2/6 extract straight COO onomatopoeia crops (CPU, ~10 min)"
  nice -n 10 "$ROOT/.venv/bin/python" scripts/extract_coo_sfx.py --out data/real
fi

# ------------------------------------------------------ 3. merge + manifests
step "3/6 merge labels + build PaddleOCR manifests"
"$ROOT/.venv/bin/python" scripts/merge_real_labels.py
"$ROOT/.venv/bin/python" scripts/build_manifest.py --synth-ratio 0.5

# ------------------------------------------------------------------ 4. train
if [[ -f $ROOT/output/rec_real/TRAINING_DONE ]]; then
  step "4/6 stage-2 training already completed - skipped"
else
  step "4/6 stage-2 fine-tune (GPU, ~1.5 h)"
  bash scripts/train_real.sh
  touch "$ROOT/output/rec_real/TRAINING_DONE"
fi

# ----------------------------------------------------------------- 5. export
if [[ -f $ROOT/models/ppocrv6_small_rec_manga_real.onnx ]]; then
  step "5/6 ONNX already exported - skipped"
else
  step "5/6 export ONNX + contract check (CPU, ~2 min)"
  bash scripts/export_onnx.sh output/rec_real/best_accuracy \
    configs/manga_real_rec.yml models/ppocrv6_small_rec_manga_real.onnx
fi

# ------------------------------------------------------------------- 6. eval
step "6/6 evaluate on held-out books (CPU, ~25 min for all four)"
for m in ppocrv6_small_rec ppocrv6_small_rec_manga ppocrv6_small_rec_manga_real; do
  [[ -f $ROOT/models/$m.onnx ]] || continue
  echo "--- $m ---"
  "$ROOT/.venv/bin/python" scripts/eval_blocks.py \
    --model "models/$m.onnx" --split test --show 5 \
    > "output/eval/${m}_blocks.txt" 2>&1 || echo "eval failed for $m (see log)"
  grep -E "block CER" "output/eval/${m}_blocks.txt" || true
  if [[ -f $ROOT/data/real/labels.tsv ]]; then
    "$ROOT/.venv/bin/python" scripts/eval_rec.py \
      --model "models/$m.onnx" \
      --labels <(awk -F'\t' '$4=="test"{print $1"\t"$2"\t"$3}' data/real/labels.tsv) \
      > "output/eval/${m}_lines.txt" 2>&1 || true
    grep -E "^all |^dialogue |^onomatopoeia " "output/eval/${m}_lines.txt" || true
  fi
done

step "DONE"
touch "$ROOT/output/OFFLINE_DONE"
echo "results in output/eval/*.txt, model in models/ppocrv6_small_rec_manga_real.onnx"
