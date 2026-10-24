#!/usr/bin/env bash
# _dd4-003.sh — Nsight Compute counters on the two attention kernels at 60K (0.8B, herbert-cuda-bf16 011bfdf, the rtx4080's
# 4080, after dd3): why sdpa_split (decode) holds ~145 GB/s at split 8, and what bounds sdpa_flash (prefill, 62 % of
# the prefill at 60K). herbert-cuda-chat --profile (launches one by one, no graph), --warmup 0, 60K prompt.
#   1. sdpa_split, split 8 and 128: the first 2 decode launches after the prefill (--set full).
#   2. sdpa_flash: 2 launches of the last prefill block (the deepest keys), --set full.
# Diagnostic only: times under ncu are not speeds. Every report kept (.ncu-rep and its text details). ALLDONE-dd4.
set -u
O=${O:-$HOME/c003/decode-depth/dd4}; mkdir -p $O
B=~/herbert-cuda-bf16/target/release/herbert-cuda-chat; M=${MODELS_DIR:-$HOME/models}/Qwen3.5-0.8B; LP=~/c003/long-prompts
NCU=/opt/nvidia/nsight-compute/2025.1.1/ncu
{ date -Is; uptime; cat ~/herbert-cuda-bf16/COMMIT; $NCU --version; nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate,memory.used --format=csv
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; } > $O/env.txt 2>&1
for s in 8 128; do
  echo "== $(date -Is) sdpa_split s$s"
  $NCU --set full --kernel-name regex:'^sdpa_split' --launch-skip 0 --launch-count 2 --clock-control none -f -o $O/sdpa_split-s$s \
    $B $M --ctx 61440 --no-system --no-think --warmup 0 --max-tokens 4 --profile --tune sdpa.split=$s --ask-file $LP/pp60000.txt > $O/sdpa_split-s$s.log 2>&1 || echo "FAIL ncu sdpa_split s$s"
  $NCU --import $O/sdpa_split-s$s.ncu-rep --page details > $O/sdpa_split-s$s.details.txt 2>&1
done
# The prompt is 60 000 tokens: blocks of 4 096 rows -> 15 blocks, 6 attention layers each: 90 launches.
echo "== $(date -Is) sdpa_flash, last block"
$NCU --set full --kernel-name regex:'^sdpa_flash' --launch-skip 84 --launch-count 2 --clock-control none -f -o $O/sdpa_flash-last \
  $B $M --ctx 61440 --no-system --no-think --warmup 0 --max-tokens 2 --profile --ask-file $LP/pp60000.txt > $O/sdpa_flash-last.log 2>&1 || echo "FAIL ncu sdpa_flash"
$NCU --import $O/sdpa_flash-last.ncu-rep --page details > $O/sdpa_flash-last.details.txt 2>&1
date -Is > $O/end.txt
echo ALLDONE-dd4
