#!/usr/bin/env bash
# Quick progress view for scripts/run_offline.sh
set -uo pipefail
cd "$(dirname "$0")/.."

echo "== steps seen in log =="
grep -a "=====" output/run_offline.log 2>/dev/null | tail -12 || echo "(no log yet)"

echo
echo "== artifacts =="
printf '%-42s %s\n' "dialogue labels" "$([ -s data/real/labels_dialogue.tsv ] && wc -l < data/real/labels_dialogue.tsv || echo -)"
printf '%-42s %s\n' "SFX labels"      "$([ -s data/real/labels_sfx.tsv ] && wc -l < data/real/labels_sfx.tsv || echo -)"
printf '%-42s %s\n' "merged labels"   "$([ -s data/real/labels.tsv ] && wc -l < data/real/labels.tsv || echo -)"
printf '%-42s %s\n' "train manifest"  "$([ -s data/manifests/train.txt ] && wc -l < data/manifests/train.txt || echo -)"
printf '%-42s %s\n' "val manifest"    "$([ -s data/manifests/val.txt ] && wc -l < data/manifests/val.txt || echo -)"
printf '%-42s %s\n' "stage-2 checkpoint" "$([ -f output/rec_real/best_accuracy.pdparams ] && echo yes || echo no)"
printf '%-42s %s\n' "stage-2 ONNX"    "$([ -f models/ppocrv6_small_rec_manga_real.onnx ] && ls -lh models/ppocrv6_small_rec_manga_real.onnx | awk '{print $5}' || echo no)"
printf '%-42s %s\n' "final marker"    "$([ -f output/OFFLINE_DONE ] && echo DONE || echo running/not started)"

echo
echo "== last checkpoint =="
grep -aE "global_step|best metric|kept [0-9]+ /|block CER|epoch: \[" output/run_offline.log 2>/dev/null | tail -5

echo
echo "== live processes =="
ps -eo pcpu,etime,args | grep -E "label_lines|extract_coo_sfx|tools/train|eval_blocks|eval_rec" | grep -v grep | head
echo
echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1)"
