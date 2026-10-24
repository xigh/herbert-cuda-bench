#!/usr/bin/env bash
# _p1-vllm.sh OUT — vLLM settings sweep (protocol.md, phase 1): 0.31.0 default, --enforce-eager,
# --max-num-batched-tokens 2048 and 16384, --attention-backend FLASH_ATTN / FLASHINFER / TRITON_ATTN; the
# reference 0.28.0 default. vllm bench latency, batch 1, 2 warm-up iterations, 5 measured; input 128, 512,
# 2048, 4000 with 1 output token, and 759 with 1 and 255 (prefill = len / L1, decode = 254 / (L255 - L1)).
# Phase 4: DTYPE=float16 (default bfloat16); BASE= options added to every setting (phase 4: --max-num-seqs 8
# -cc.max_cudagraph_capture_size=512, the 4B needing the first, phase 2 keeping the second); REF=0 leaves 0.28.0 out.
set -u
DT=${DTYPE:-bfloat16}; BASE=${BASE:-}; REF=${REF:-1}
O=$1; mkdir -p $O
M=${MODELS_DIR:-$HOME/models}; NEW=~/venvs/bench-vllm-latest/bin/vllm; OLD=~/venvs/bench-vllm-cu/bin/vllm
V=("default|$NEW|" "eager|$NEW|--enforce-eager" "mnbt2048|$NEW|--max-num-batched-tokens 2048" "mnbt16384|$NEW|--max-num-batched-tokens 16384"
   "flashattn|$NEW|--attention-backend FLASH_ATTN" "flashinfer|$NEW|--attention-backend FLASHINFER" "triton|$NEW|--attention-backend TRITON_ATTN" "ref0.28.0|$OLD|")
for m in 0.8B 4B; do
  for v in "${V[@]}"; do
    name=${v%%|*}; rest=${v#*|}; bin=${rest%%|*}; args="$BASE ${rest#*|}"
    [ "$REF" = 0 ] && [ "$name" = ref0.28.0 ] && continue
    echo "== $(date -Is) vllm $m $name · $(uptime)"
    for io in 128:1 512:1 2048:1 4000:1 759:1 759:255; do
      i=${io%%:*}; o=${io#*:}
      # shellcheck disable=SC2086
      VLLM_USE_FLASHINFER_SAMPLER=0 $bin bench latency --model $M/Qwen3.5-$m --dtype $DT --input-len $i --output-len $o --batch-size 1 \
        --num-iters-warmup 2 --num-iters 5 --max-model-len 4608 --gpu-memory-utilization 0.85 $args \
        --output-json $O/$m-$name-in$i-out$o.json > $O/$m-$name-in$i-out$o.log 2>&1 || echo "FAIL vllm $m $name in$i out$o"
    done
  done
done
