#!/usr/bin/env bash
# E5: SAM ViT-B/L/H + baselines on the val renders, one shard per GPU, then the summary.
set -uo pipefail
cd "$(dirname "$0")"
PY=/media/fulei/fuleivla/robomme_benchmark/.venv/bin/python
LOG=../logs
rm -rf ../results/sam_segmentation_raw
for i in 0 1 2 3 4 5 6 7; do
  (
    for m in vit_b vit_l vit_h; do
      CUDA_VISIBLE_DEVICES=$i $PY exp_sam_segmentation.py run --model $m --shard $i --of 8
    done
  ) > $LOG/sam_seg_gpu$i.log 2>&1 &
done
wait
$PY exp_sam_segmentation.py summarize > $LOG/sam_seg_summary.log 2>&1
echo "E5 finished $(date)" >> $LOG/sam_seg_summary.log
