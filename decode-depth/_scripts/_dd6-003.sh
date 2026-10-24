#!/usr/bin/env bash
# _dd6-003.sh — sdpa.gqa=1 on the 4B (dd5 was the 0.8B): resident-gpu.service stopped for the run, restarted by a trap.
# Build ~/herbert-cuda-bf16-gqa (bc8e601). Every launch waits for an empty GPU.
#   A. Tokens: compare.sh p1..p4 against runs/plan-final-oracle/4B (copied): old (split 8), old16, gqa32, gqa64.
#   B. Speed, graph on, alternated N = 3: old8, old16, gqa32, gqa64, gqa128 on p3, 10K, 60K.
#   C. --profile at 60K: old16, gqa64.
set -u
O=${O:-$HOME/c003/decode-depth/dd6}; mkdir -p $O/b $O/c
NB=~/herbert-cuda-bf16-gqa/target/release/herbert-cuda-chat; M=${MODELS_DIR:-$HOME/models}/Qwen3.5-4B; LP=~/c003/long-prompts; PR=~/herbert-cuda-bf16-gqa/bench/prompts
case $(nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader,nounits | tr -d ' ') in 2805,11201) ;; *) echo "clocks not at 2805/11201, stop"; exit 1;; esac
echo "== $(date -Is) stopping resident-gpu"; systemctl --user stop resident-gpu.service
restart() { systemctl --user start resident-gpu.service; sleep 5; echo "== $(date -Is) resident-gpu: $(systemctl --user is-active resident-gpu.service)"; }
trap 'kill $smi 2>/dev/null; restart' EXIT
{ date -Is; uptime; cat ~/herbert-cuda-bf16-gqa/COMMIT; sha256sum $NB; nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate,memory.used --format=csv
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; } > $O/env.txt 2>&1
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu,utilization.gpu,memory.used,clocks_event_reasons.active --format=csv -l 1 > $O/clocks.csv 2>&1 &
smi=$!
free() { local n=0
  while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do
    echo "$(date -Is) busy: $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';')" >> $O/waits.txt
    n=$((n + 1)); [ $n -gt 40 ] && { echo "GPU never free, stop"; exit 1; }; sleep 30
  done
  echo "$(date -Is) $1" >> $O/launches.txt
}
echo "== $(date -Is) A tokens"
mkdir -p $O/compare; cp ~/herbert-cuda-bf16/runs/plan-final-oracle/4B/p[1-4]-*-cpu.log $O/compare/
free compare
( cd ~/herbert-cuda-bf16-gqa/bench && PROMPTS="prompts/p1-short.txt prompts/p2-explain.txt prompts/p3-long.txt prompts/p4-xlong.txt" \
  VARIANTS="old:@$NB;old16:@$NB --tune sdpa.split=16;gqa32:@$NB --tune sdpa.gqa=1 --tune sdpa.split=32;gqa64:@$NB --tune sdpa.gqa=1 --tune sdpa.split=64" \
  bash compare.sh $M $O/compare 0 ) > $O/compare.out 2>&1 || echo "FAIL compare"
V=("old8|--tune sdpa.split=8" "old16|--tune sdpa.split=16" "gqa32|--tune sdpa.gqa=1 --tune sdpa.split=32" "gqa64|--tune sdpa.gqa=1 --tune sdpa.split=64" "gqa128|--tune sdpa.gqa=1 --tune sdpa.split=128")
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
for v in "old16|--tune sdpa.split=16" "gqa64|--tune sdpa.gqa=1 --tune sdpa.split=64"; do
  n=${v%%|*}; a=${v#*|}; free $O/c/profile-$n-pp60000.log
  # shellcheck disable=SC2086
  $NB $M --ctx 61440 --no-system --no-think --max-tokens 64 --warmup 1 --profile $a --ask-file $LP/pp60000.txt > $O/c/profile-$n-pp60000.log 2>&1 || echo "FAIL profile $n"
done
date -Is > $O/end.txt
echo ALLDONE-dd6
