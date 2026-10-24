#!/usr/bin/env bash
# _ablation-003.sh — herbert-cuda against itself, one CUDA feature off at a time, on top of the candidate
# (gemm.auto=3 gemm.layout=xor prefill.block=auto): prefill at 128…4000 tokens (1 generated) and
# the p3 decode (759 + 255, thinking on), both models, N rounds, variants interleaved (order reversed
# every other round), warm (--warmup 1). Clocks every second. No other engine.
set -u
O=${O:-$HOME/ablation-003}; N=${N:-5}; mkdir -p $O/runs
B=$HOME/herbert-cuda-mma/target/release/herbert-cuda-chat
P=$HOME/herbert-cuda-mma/bench/prompts
M=${MODELS_DIR:-$HOME/models}
C="--tune gemm.auto=3 --tune gemm.layout=xor --tune prefill.block=auto"
V=("base|$C" "noldsm|$C --tune gemm.ldmatrix=0" "sync|$C --tune gemm.copy=sync" "noldsm-sync|$C --tune gemm.ldmatrix=0 --tune gemm.copy=sync" \
   "pad|--tune gemm.auto=3 --tune prefill.block=auto" "graph0|$C --tune graph=0" "default|")
{ date -Is; nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate --format=csv; uptime; sha256sum $B; printf '%s\n' "${V[@]}"; } > $O/env.txt 2>&1
case $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits) in 2805) ;; *) echo "SM not at 2805 MHz, stop"; exit 1;; esac
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu,clocks_event_reasons.active --format=csv -l 1 > $O/clocks.csv 2>&1 &
smi=$!
for r in $(seq 1 $N); do
  echo "== $(date -Is) round $r · $(uptime)"
  order=("${V[@]}"); if [ $((r % 2)) = 0 ]; then order=(); for ((i=${#V[@]}-1; i>=0; i--)); do order+=("${V[$i]}"); done; fi
  for m in 0.8B 4B; do
    for v in "${order[@]}"; do
      name=${v%%|*}; args=${v#*|}
      for p in 128 256 512 1024 2048 4000; do
        # shellcheck disable=SC2086
        $B $M/Qwen3.5-$m --device 0 --no-system --no-think --max-tokens 1 --warmup 1 --ctx 4608 $args --ask "$(cat $P/pp$p.txt)" > $O/runs/r$r-$m-$name-pp$p.log 2>&1 || echo "FAIL r$r $m $name pp$p"
      done
      # shellcheck disable=SC2086
      $B $M/Qwen3.5-$m --device 0 --no-system --max-tokens 256 --warmup 1 $args --ask "$(cat $P/p3-long.txt)" > $O/runs/r$r-$m-$name-p3.log 2>&1 || echo "FAIL r$r $m $name p3"
    done
  done
done
kill $smi
date -Is > $O/end.txt
echo ALLDONE-ablation-003
