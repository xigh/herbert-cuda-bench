#!/usr/bin/env bash
# _feas-art.sh — feasibility of the article comparison (added on 2026-10-24): TensorRT-LLM 1.4.0rc1 in bf16 on the three models (never measured), and every engine on the 2B (new to
# #003). One pass each, pp512 and the p3 decode. resident-gpu stopped (trap); clocks checked. ALLDONE-feas-art.
set -u
O=~/c003/art/feas; mkdir -p $O
M=${MODELS_DIR:-$HOME/models}; G=${BENCH_DIR:-$HOME/bench-qwen35}; L=$HOME/llama.cpp-b11718/build-cuda/bin; HC=~/herbert-cuda-bf16-art/target/release/herbert-cuda-chat
PR=~/herbert-cuda-bf16-art/bench/prompts; VB=~/venvs/bench-vllm-latest/bin/vllm; SPY=~/venvs/bench-sglang-latest/bin/python
TPY=~/venvs/bench-trtllm/bin/python; TL=~/c003/q4q8/_trtllm_latency.py; C13=~/venvs/bench-trtllm/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
until grep -q "^exit" ~/c003/art/convert-2b.log 2>/dev/null; do sleep 10; done
clocks() { local i; for i in $(seq 1 30); do [ "$(nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader,nounits | tr -d ' ')" = "2805,11201" ] && return 0; sleep 1; done; return 1; }
clocks || { echo "clocks not at 2805/11201, stop"; exit 1; }
echo "== $(date -Is) stopping resident-gpu"; systemctl --user stop resident-gpu
trap 'echo "== $(date -Is) starting resident-gpu"; systemctl --user start resident-gpu; sleep 20; echo "== $(date -Is) resident-gpu: $(systemctl --user is-active resident-gpu)"; echo ALLDONE-feas-art' EXIT
sleep 5
run() { local n=$1; shift; clocks || echo "!! clocks before $n"; echo "== $(date -Is) $n"; { echo "$*"; "$@"; echo "exit $?"; } > $O/$n.log 2>&1
  grep -h "^\[turn\|^\[warmup\|t/s\|^exit\|Avg latency\|prefill_throughput\|median_decode\|Error\|error" $O/$n.log | grep -v "^| model\|^| ---" | tail -3; }
for m in 0.8B 2B 4B; do
  for io in 512:1 759:1 759:255; do i=${io%%:*}; o=${io#*:}
    CUDA_HOME=$C13 PATH=$C13/bin:$PATH LD_LIBRARY_PATH=$HOME/venvs/bench-trtllm/lib TLLM_WORKER_USE_SINGLE_PROCESS=1 \
      run trtllm-$m-in$i-out$o $TPY $TL --model $M/Qwen3.5-$m --input-len $i --output-len $o --max-seq-len 4608 --max-num-tokens 4608 --output-json $O/trtllm-$m-in$i-out$o.json
  done
done
m=2B
run hc-$m-pp512 $HC $M/Qwen3.5-$m --no-system --no-think --max-tokens 1 --warmup 1 --ask-file $PR/pp512.txt
run hc-$m-p3 $HC $M/Qwen3.5-$m --no-system --max-tokens 256 --warmup 1 --ask-file $PR/p3-long.txt
run llama-$m-pp $L/llama-bench -m $G/Qwen3.5-$m-BF16.gguf -dev CUDA0 -ngl 99 -p 512 -n 0 -r 1
run llama-$m-tg $L/llama-bench -m $G/Qwen3.5-$m-BF16.gguf -dev CUDA0 -ngl 99 -p 0 -n 255 -d 759 -r 1
for io in 512:1 759:255; do i=${io%%:*}; o=${io#*:}
  run vllm-$m-in$i-out$o $VB bench latency --model $M/Qwen3.5-$m --dtype bfloat16 --input-len $i --output-len $o --batch-size 1 --num-iters-warmup 2 --num-iters 3 \
    --max-model-len 4608 --max-num-seqs 8 -cc.max_cudagraph_capture_size=512 --gpu-memory-utilization 0.85 --output-json $O/vllm-$m-in$i-out$o.json
done
run sglang-$m $SPY -m sglang.benchmark.one_batch --model-path $M/Qwen3.5-$m --dtype bfloat16 --context-length 4608 --mem-fraction-static 0.8 --batch-size 1 \
  --input-len 512 --output-len 1 --run-name feas-$m-pp --result-filename $O/sglang-results.jsonl
