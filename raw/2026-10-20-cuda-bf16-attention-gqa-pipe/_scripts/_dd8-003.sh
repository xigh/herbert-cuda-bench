#!/usr/bin/env bash
# _dd8-003.sh — the confirmation: herbert-cuda-bf16 73b5a70 (default PTX identical to 011bfdf's; sdpa.gqa and
# sdpa.pipe behind their switches), built in ~/herbert-cuda-bf16-fix, 0.8B then 4B, the RTX 4080.
# resident-gpu.service stopped for the whole run (the 4B does not fit beside it), restarted by a trap. Every launch
# waits for an empty GPU.
#   A. Tokens: compare.sh p1..p4, pp128..pp4000 against runs/plan-final-oracle/<model> (copied): default, both
#      (sdpa.gqa=1 sdpa.split=64 sdpa.pipe=1).
#   B. Alternated N = 3, order reversed every other round: default, gqa (sdpa.gqa=1 sdpa.split=64), pipe
#      (sdpa.pipe=1), both; on p3 (759, thinking, 256 generated), pp4000 (4 001, 16 generated) and at 10K, 20K,
#      40K, 60K (--ctx 61440, 128 generated): prefill and decode from the same [turn] line.
#   C. --profile at 60K: default, both.
set -u
O=${O:-$HOME/c003/decode-depth/dd8}; mkdir -p $O
NB=~/herbert-cuda-bf16-fix/target/release/herbert-cuda-chat; LP=~/c003/long-prompts; PR=~/herbert-cuda-bf16-fix/bench/prompts
case $(nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader,nounits | tr -d ' ') in 2805,11201) ;; *) echo "clocks not at 2805/11201, stop"; exit 1;; esac
echo "== $(date -Is) stopping resident-gpu"; systemctl --user stop resident-gpu.service
restart() { systemctl --user start resident-gpu.service; sleep 5; echo "== $(date -Is) resident-gpu: $(systemctl --user is-active resident-gpu.service)"; }
trap 'kill $smi 2>/dev/null; restart' EXIT
{ date -Is; uptime; cat ~/herbert-cuda-bf16-fix/COMMIT; sha256sum $NB; nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate,memory.used --format=csv
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
BOTH="--tune sdpa.gqa=1 --tune sdpa.split=64 --tune sdpa.pipe=1"
V=("default|" "gqa|--tune sdpa.gqa=1 --tune sdpa.split=64" "pipe|--tune sdpa.pipe=1" "both|$BOTH")
for m in 0.8B 4B; do
  M=${MODELS_DIR:-$HOME/models}/Qwen3.5-$m; mkdir -p $O/$m/b $O/$m/c $O/$m/compare
  echo "== $(date -Is) $m A tokens"
  cp ~/herbert-cuda-bf16/runs/plan-final-oracle/$m/*-cpu.log $O/$m/compare/
  free $m-compare
  ( cd ~/herbert-cuda-bf16-fix/bench && PROMPTS="prompts/p1-short.txt prompts/p2-explain.txt prompts/p3-long.txt prompts/p4-xlong.txt prompts/pp128.txt prompts/pp256.txt prompts/pp512.txt prompts/pp1024.txt prompts/pp2048.txt prompts/pp4000.txt" \
    VARIANTS="default:@$NB;both:@$NB $BOTH" bash compare.sh $M $O/$m/compare 0 ) > $O/$m/compare.out 2>&1 || echo "FAIL compare $m"
  for r in 1 2 3; do
    order=("${V[@]}"); [ $((r % 2)) = 0 ] && { order=(); for ((i=${#V[@]}-1; i>=0; i--)); do order+=("${V[$i]}"); done; }
    echo "== $(date -Is) $m B round $r · $(uptime)"
    for v in "${order[@]}"; do n=${v%%|*}; a=${v#*|}
      free $O/$m/b/r$r-$n-p3.log
      # shellcheck disable=SC2086
      $NB $M --no-system --max-tokens 256 --warmup 1 $a --ask "$(cat $PR/p3-long.txt)" > $O/$m/b/r$r-$n-p3.log 2>&1 || echo "FAIL $m r$r $n p3"
      free $O/$m/b/r$r-$n-pp4000.log
      # shellcheck disable=SC2086
      $NB $M --no-system --no-think --max-tokens 16 --warmup 1 $a --ask "$(cat $PR/pp4000.txt)" > $O/$m/b/r$r-$n-pp4000.log 2>&1 || echo "FAIL $m r$r $n pp4000"
      for p in 10000 20000 40000 60000; do
        free $O/$m/b/r$r-$n-pp$p.log
        # shellcheck disable=SC2086
        $NB $M --ctx 61440 --no-system --no-think --max-tokens 128 --warmup 1 $a --ask-file $LP/pp$p.txt > $O/$m/b/r$r-$n-pp$p.log 2>&1 || echo "FAIL $m r$r $n pp$p"
      done
    done
  done
  echo "== $(date -Is) $m C profile at 60K"
  for v in "default|" "both|$BOTH"; do n=${v%%|*}; a=${v#*|}; free $O/$m/c/profile-$n-pp60000.log
    # shellcheck disable=SC2086
    $NB $M --ctx 61440 --no-system --no-think --max-tokens 64 --warmup 1 --profile $a --ask-file $LP/pp60000.txt > $O/$m/c/profile-$n-pp60000.log 2>&1 || echo "FAIL profile $m $n"
  done
done
date -Is > $O/end.txt
echo ALLDONE-dd8
