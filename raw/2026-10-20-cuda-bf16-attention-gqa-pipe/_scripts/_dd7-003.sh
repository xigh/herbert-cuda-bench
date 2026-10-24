#!/usr/bin/env bash
# _dd7-003.sh — sdpa.pipe=1 (the prefill attention's copies of K and V overlap its arithmetic, same results) against
# the default, Qwen3.5-0.8B, the RTX 4080 beside the resident GPU service. Build ~/herbert-cuda-bf16-pipe (5ec3063,
# which also carries sdpa.gqa). Every launch waits until the GPU holds nothing but the resident service (by its PID).
#   A. Tokens: compare.sh p1..p4 and pp128..pp4000 against runs/plan-final-oracle/0.8B (copied): default, pipe.
#   B. Prefill tok/s, alternated N = 3: default, pipe, at 10K, 20K, 40K, 60K (--ctx 61440, 16 generated; ids kept).
#   C. --profile at 60K: default, pipe (prefill blocks, sdpa_flash).
#   D. Nsight Compute: sdpa_flash with pipe, 2 launches of the last prefill block at 60K (--set full).
set -u
O=${O:-$HOME/c003/decode-depth/dd7}; mkdir -p $O/b $O/c
NB=~/herbert-cuda-bf16-pipe/target/release/herbert-cuda-chat; M=${MODELS_DIR:-$HOME/models}/Qwen3.5-0.8B; LP=~/c003/long-prompts
NCU=/opt/nvidia/nsight-compute/2025.1.1/ncu
case $(nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader,nounits | tr -d ' ') in 2805,11201) ;; *) echo "clocks not at 2805/11201, stop"; exit 1;; esac
SVC=$(systemctl --user show -p MainPID --value resident-gpu.service)
{ date -Is; uptime; cat ~/herbert-cuda-bf16-pipe/COMMIT; sha256sum $NB; echo "service pid $SVC"
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
mkdir -p $O/compare; cp ~/herbert-cuda-bf16/runs/plan-final-oracle/0.8B/*-cpu.log $O/compare/
free compare
( cd ~/herbert-cuda-bf16-pipe/bench && PROMPTS="prompts/p1-short.txt prompts/p2-explain.txt prompts/p3-long.txt prompts/p4-xlong.txt prompts/pp128.txt prompts/pp256.txt prompts/pp512.txt prompts/pp1024.txt prompts/pp2048.txt prompts/pp4000.txt" \
  VARIANTS="default:@$NB;pipe:@$NB --tune sdpa.pipe=1" bash compare.sh $M $O/compare 0 ) > $O/compare.out 2>&1 || echo "FAIL compare"
V=("default|" "pipe|--tune sdpa.pipe=1")
for r in 1 2 3; do
  order=("${V[@]}"); [ $((r % 2)) = 0 ] && order=("${V[1]}" "${V[0]}")
  echo "== $(date -Is) B round $r · $(uptime)"
  for v in "${order[@]}"; do n=${v%%|*}; a=${v#*|}
    for p in 10000 20000 40000 60000; do
      free $O/b/r$r-$n-pp$p.log
      # shellcheck disable=SC2086
      $NB $M --ctx 61440 --no-system --no-think --max-tokens 16 --warmup 1 $a --ask-file $LP/pp$p.txt > $O/b/r$r-$n-pp$p.log 2>&1 || echo "FAIL r$r $n pp$p"
    done
  done
done
echo "== $(date -Is) C profile at 60K"
for v in "${V[@]}"; do n=${v%%|*}; a=${v#*|}; free $O/c/profile-$n-pp60000.log
  # shellcheck disable=SC2086
  $NB $M --ctx 61440 --no-system --no-think --max-tokens 16 --warmup 1 --profile $a --ask-file $LP/pp60000.txt > $O/c/profile-$n-pp60000.log 2>&1 || echo "FAIL profile $n"
done
echo "== $(date -Is) D ncu"
free ncu
$NCU --set full --kernel-name regex:'^sdpa_flash' --launch-skip 84 --launch-count 2 --clock-control none -f -o $O/sdpa_flash-pipe-last \
  $NB $M --ctx 61440 --no-system --no-think --warmup 0 --max-tokens 2 --profile --tune sdpa.pipe=1 --ask-file $LP/pp60000.txt > $O/sdpa_flash-pipe-last.log 2>&1 || echo "FAIL ncu"
$NCU --import $O/sdpa_flash-pipe-last.ncu-rep --page details > $O/sdpa_flash-pipe-last.details.txt 2>&1
date -Is > $O/end.txt
echo ALLDONE-dd7
