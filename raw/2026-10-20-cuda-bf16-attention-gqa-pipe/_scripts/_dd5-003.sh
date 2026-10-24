#!/usr/bin/env bash
# _dd5-003.sh — the new decode attention (sdpa.gqa=1: one pass per GQA group, K and V read once for the group,
# online softmax per warp) against the current one, Qwen3.5-0.8B, the RTX 4080 beside the resident GPU service.
# Build: herbert-cuda-bf16 working tree (011bfdf + decode::sdpa_gqa, engine/tuning switch, default unchanged) in
# ~/herbert-cuda-bf16-gqa (built before this script; its sha256 in env.txt).
#   A. Tokens: bench/compare.sh p1..p4 against the CPU oracle logs of runs/plan-final-oracle/0.8B (copied, not rerun),
#      variants: default (old kernel, split 8), old s128, gqa s8, s16, s32, s64, s128.
#   B. Speed, graph on, alternated N = 3, order reversed every other round: old s8, old s128, gqa s16, s32, s64, s128,
#      on p3 (759, thinking, 256 generated, default ctx) and at 10K and 60K (--ctx 61440, 128 generated).
#   C. --profile at 60K: old s128, gqa s32, s64, s128.
# Every launch waits until the GPU holds nothing but the resident service (by its PID). ALLDONE-dd5.
set -u
O=${O:-$HOME/c003/decode-depth/dd5}; mkdir -p $O/b $O/c
NB=~/herbert-cuda-bf16-gqa/target/release/herbert-cuda-chat; M=${MODELS_DIR:-$HOME/models}/Qwen3.5-0.8B; LP=~/c003/long-prompts; PR=~/herbert-cuda-bf16-gqa/bench/prompts
case $(nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader,nounits | tr -d ' ') in 2805,11201) ;; *) echo "clocks not at 2805/11201, stop"; exit 1;; esac
SVC=$(systemctl --user show -p MainPID --value resident-gpu.service)
{ date -Is; uptime; (cd ~/herbert-cuda-bf16-gqa && cat COMMIT 2>/dev/null; git -C ~/herbert-cuda-bf16 log -1 --format='%h %s'); sha256sum $NB; echo "service pid $SVC"
  nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate,memory.used --format=csv; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; } > $O/env.txt 2>&1
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu,utilization.gpu,memory.used,clocks_event_reasons.active --format=csv -l 1 > $O/clocks.csv 2>&1 &
smi=$!; trap 'kill $smi 2>/dev/null' EXIT
free() { local n=0
  while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -vx "$SVC" | grep -q .; do
    echo "$(date -Is) busy: $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';')" >> $O/waits.txt
    n=$((n + 1)); [ $n -gt 40 ] && { echo "GPU never free, stop"; exit 1; }; sleep 30
  done
  echo "$(date -Is) $1 · $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')" >> $O/launches.txt
}
echo "== $(date -Is) A tokens"
mkdir -p $O/compare; cp ~/herbert-cuda-bf16/runs/plan-final-oracle/0.8B/p[1-4]-*-cpu.log $O/compare/
free compare
( cd ~/herbert-cuda-bf16-gqa/bench && PROMPTS="prompts/p1-short.txt prompts/p2-explain.txt prompts/p3-long.txt prompts/p4-xlong.txt" \
  VARIANTS="old:@$NB;old128:@$NB --tune sdpa.split=128;gqa8:@$NB --tune sdpa.gqa=1 --tune sdpa.split=8;gqa16:@$NB --tune sdpa.gqa=1 --tune sdpa.split=16;gqa32:@$NB --tune sdpa.gqa=1 --tune sdpa.split=32;gqa64:@$NB --tune sdpa.gqa=1 --tune sdpa.split=64;gqa128:@$NB --tune sdpa.gqa=1 --tune sdpa.split=128" \
  bash compare.sh $M $O/compare 0 ) > $O/compare.out 2>&1 || echo "FAIL compare"
V=("old8|--tune sdpa.split=8" "old128|--tune sdpa.split=128" "gqa16|--tune sdpa.gqa=1 --tune sdpa.split=16" "gqa32|--tune sdpa.gqa=1 --tune sdpa.split=32" "gqa64|--tune sdpa.gqa=1 --tune sdpa.split=64" "gqa128|--tune sdpa.gqa=1 --tune sdpa.split=128")
for r in 1 2 3; do
  order=("${V[@]}"); [ $((r % 2)) = 0 ] && { order=(); for ((i=${#V[@]}-1; i>=0; i--)); do order+=("${V[$i]}"); done; }
  echo "== $(date -Is) B round $r · $(uptime)"
  for v in "${order[@]}"; do n=${v%%|*}; a=${v#*|}
    free $O/b/r$r-$n-p3.log
    # shellcheck disable=SC2086
    $NB $M --no-system --max-tokens 256 --warmup 1 $a --ask "$(cat $PR/p3-long.txt)" > $O/b/r$r-$n-p3.log 2>&1 || echo "FAIL r$r $n p3"
    for p in 10000 60000; do
      free $O/b/r$r-$n-pp$p.log
      # shellcheck disable=SC2086
      $NB $M --ctx 61440 --no-system --no-think --max-tokens 128 --warmup 1 $a --ask-file $LP/pp$p.txt > $O/b/r$r-$n-pp$p.log 2>&1 || echo "FAIL r$r $n pp$p"
    done
  done
done
echo "== $(date -Is) C profile at 60K"
for v in "old128|--tune sdpa.split=128" "gqa32|--tune sdpa.gqa=1 --tune sdpa.split=32" "gqa64|--tune sdpa.gqa=1 --tune sdpa.split=64" "gqa128|--tune sdpa.gqa=1 --tune sdpa.split=128"; do
  n=${v%%|*}; a=${v#*|}; free $O/c/profile-$n-pp60000.log
  # shellcheck disable=SC2086
  $NB $M --ctx 61440 --no-system --no-think --max-tokens 64 --warmup 1 --profile $a --ask-file $LP/pp60000.txt > $O/c/profile-$n-pp60000.log 2>&1 || echo "FAIL profile $n"
done
date -Is > $O/end.txt
echo ALLDONE-dd5
