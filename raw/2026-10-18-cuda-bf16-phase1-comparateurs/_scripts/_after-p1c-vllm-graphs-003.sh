#!/usr/bin/env bash
# _after-p1c-vllm-graphs-003.sh — queued behind the SGLang rerun (waits for ALLDONE-after-p1b).
# The vLLM rerun (--max-num-seqs 8) captured CUDA graphs only up to 16 tokens (cudagraph_capture_sizes
# [1, 2, 4, 8, 16], max_cudagraph_capture_size 16); phase 1's 0.8B (max_num_seqs 256) captured up to 512. The 0.8B
# prefill at 128 and 512 tokens fell from 10 853 / 41 796 tok/s (phase 1) to 8 934 / 35 748 (rerun). Here: the
# capture size given explicitly with --max-num-seqs 8, and the 0.8B again without --max-num-seqs in the same run.
set -u
until grep -q ALLDONE-after-p1b ~/c003/after-p1b.out; do sleep 60; done
O=${O:-$HOME/c003/after-p1c}; mkdir -p $O/vllm
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu --format=csv -l 1 > $O/clocks.csv 2>&1 &
smi=$!
M=${MODELS_DIR:-$HOME/models}; NEW=~/venvs/bench-vllm-latest/bin/vllm
VV=("cg512|--max-num-seqs 8 -cc.max_cudagraph_capture_size=512" "cg4096|--max-num-seqs 8 -cc.max_cudagraph_capture_size=4096"
   "flashinfer-cg512|--max-num-seqs 8 -cc.max_cudagraph_capture_size=512 --attention-backend FLASHINFER"
   "triton-cg512|--max-num-seqs 8 -cc.max_cudagraph_capture_size=512 --attention-backend TRITON_ATTN"
   "default|--max-num-seqs 8" "mns256|")
for m in 0.8B 4B; do for v in "${VV[@]}"; do name=${v%%|*}; args=${v#*|}
  # 4B: max_num_seqs 256 exceeds the Mamba cache blocks (127), refused in phase 1.
  [ $m = 4B ] && [ $name = mns256 ] && continue
  echo "== $(date -Is) vllm $m $name · $(uptime)"
  for io in 128:1 512:1 2048:1 4000:1 759:1 759:255; do i=${io%%:*}; o=${io#*:}
    # shellcheck disable=SC2086
    CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 $NEW bench latency --model $M/Qwen3.5-$m --dtype bfloat16 \
      --input-len $i --output-len $o --batch-size 1 --num-iters-warmup 2 --num-iters 5 --max-model-len 4608 --gpu-memory-utilization 0.85 $args \
      --output-json $O/vllm/$m-$name-in$i-out$o.json > $O/vllm/$m-$name-in$i-out$o.log 2>&1 || echo "FAIL vllm $m $name in$i out$o"
  done
done; done
kill $smi
echo ALLDONE-after-p1c
