#!/usr/bin/env bash
# _after-p1-003.sh — queued behind phase 1 (waits for ALLDONE-phase1):
# the whole vLLM sweep again with --max-num-seqs 8 (as #002) and CUDA_HOME at the 12.8 toolkit. Herbert-cuda's
# f16 experiment, first queued here, runs on rtx4080-b in parallel (_f16-rtx4080-b-003.sh).
set -u
until grep -q ALLDONE-phase1 ~/c003/phase1.out; do sleep 60; done
O=${O:-$HOME/c003/after-p1}; mkdir -p $O/vllm
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu --format=csv -l 1 > $O/clocks.csv 2>&1 &
smi=$!
M=${MODELS_DIR:-$HOME/models}
echo "== $(date -Is) vllm again, --max-num-seqs 8"
# Phase 1 ran vllm bench latency without --max-num-seqs: 0.31.0 refused the 4B with CUDA graphs ("max_num_seqs
# (256) exceeds available Mamba cache blocks (127)"), and FLASHINFER failed to compile on /usr/bin/nvcc 12.0.
# The whole vLLM sweep again, with --max-num-seqs 8 as in #002 (V1), CUDA_HOME at the 12.8 toolkit for all.
NEW=~/venvs/bench-vllm-latest/bin/vllm; OLD=~/venvs/bench-vllm-cu/bin/vllm
VV=("default|$NEW|" "eager|$NEW|--enforce-eager" "mnbt2048|$NEW|--max-num-batched-tokens 2048" "mnbt16384|$NEW|--max-num-batched-tokens 16384"
   "flashattn|$NEW|--attention-backend FLASH_ATTN" "flashinfer|$NEW|--attention-backend FLASHINFER" "triton|$NEW|--attention-backend TRITON_ATTN" "ref0.28.0|$OLD|")
for m in 0.8B 4B; do for v in "${VV[@]}"; do name=${v%%|*}; rest=${v#*|}; bin=${rest%%|*}; args=${rest#*|}
  echo "== $(date -Is) vllm $m $name · $(uptime)"
  for io in 128:1 512:1 2048:1 4000:1 759:1 759:255; do i=${io%%:*}; o=${io#*:}
    # shellcheck disable=SC2086
    CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 $bin bench latency --model $M/Qwen3.5-$m --dtype bfloat16 \
      --input-len $i --output-len $o --batch-size 1 --num-iters-warmup 2 --num-iters 5 --max-model-len 4608 --max-num-seqs 8 --gpu-memory-utilization 0.85 $args \
      --output-json $O/vllm/$m-$name-in$i-out$o.json > $O/vllm/$m-$name-in$i-out$o.log 2>&1 || echo "FAIL vllm $m $name in$i out$o"
  done
done; done
# The f16 experiment moved to rtx4080-b (_f16-rtx4080-b-003.sh), run in parallel with phase 1.
kill $smi
echo ALLDONE-after-p1
