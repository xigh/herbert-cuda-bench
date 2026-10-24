#!/usr/bin/env bash
# _wmma-softmax-003.sh — the last two switches of the ablation: the GEMM in wmma (the driver's fragment
# layout, as Vulkan's cooperative matrices) and the attention softmax through shared memory (as
# vk/fatc.rs). 1) GEMM alone: mma against wmma, cp.async and register copies, padded layout, every
# projection shape, M 128…2048, 3 rounds, each output checked bit for bit against mma/ldmatrix/cp.async.
# 2) Tokens against the CPU oracle (p1–p4) for each new variant. 3) The engine: prefill 128…4000 and
# the p3 decode, N rounds interleaved. Clocks every second.
set -u
O=${O:-$HOME/wmma-003}; N=${N:-5}; mkdir -p $O/gemm $O/runs
R=$HOME/herbert-cuda-mma; B=$R/target/release/herbert-cuda-chat; PR=$R/target/release/herbert-cuda-probe
M=${MODELS_DIR:-$HOME/models}; P=$R/bench/prompts
{ date -Is; nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate --format=csv; uptime; sha256sum $B $PR; } > $O/env.txt 2>&1
case $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits) in 2805) ;; *) echo "SM not at 2805 MHz, stop"; exit 1;; esac
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu,clocks_event_reasons.active --format=csv -l 1 > $O/clocks.csv 2>&1 &
smi=$!
echo "== $(date -Is) gemm"
for m in 0.8B 4B; do
  for s in dn.proj128:bf16 dn.out:res attn.qkv:bf16 attn.o:res mlp.gate_up:bf16 mlp.down:res; do
    shape=${s%%:*}; out=${s#*:}
    $PR --only gemm --model $m --shape $shape --gemm-out $out --m 128,512,1024,2048 \
      --gemm-tiles 64x64x32_w2x2_s3,64x128x32_w2x2_s2 --gemm-layout pad --gemm-mma both --gemm-copy both \
      --rotate-weights --rounds 3 > $O/gemm/$m-$shape.log 2>&1; echo "gemm $m $shape exit $?"
  done
done
C="--tune gemm.auto=3 --tune gemm.layout=xor --tune prefill.block=auto"
D="--tune gemm.auto=3 --tune prefill.block=auto"
V=("base|$C" "softmax-shared|$C --tune sdpa.softmax=shared" "pad|$D" "wmma|$D --tune gemm.wmma=1" \
   "vklike|$D --tune gemm.wmma=1 --tune gemm.copy=sync --tune sdpa.softmax=shared --tune graph=0")
printf '%s\n' "${V[@]}" >> $O/env.txt
echo "== $(date -Is) oracle"
for m in 0.8B 4B; do
  mkdir -p $O/cmp-$m; cp $HOME/verif-003/cmp-$m/*-cpu.log $O/cmp-$m/
  VV=""; for v in "${V[@]}"; do VV="$VV;${v%%|*}:${v#*|}"; done
  PROMPTS="prompts/p1-short.txt prompts/p2-explain.txt prompts/p3-long.txt prompts/p4-xlong.txt" VARIANTS="${VV#;}" \
    bash $R/bench/compare.sh $M/Qwen3.5-$m $O/cmp-$m 0 > $O/cmp-$m.out 2>&1; echo "compare $m exit $?"
done
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
echo ALLDONE-wmma-003
