#!/usr/bin/env bash
# _feas-long2.sh — second feasibility pass at 40 000 tokens for the two that failed in _feas-long.sh on the 4B:
# TensorRT-LLM (out of memory in the warm-up forward with max_num_tokens 40960, kv fraction 0.5): kv fraction 0.2, and
# chunked prefill with max_num_tokens 8192; SGLang (mf 0.7 + mrr1: KV pool 34 186 < 40 000; mf 0.6: no mamba slot):
# mf 0.72 … 0.78 with --max-running-requests 1. TRT chunked 8192 also on the 0.8B (to see it beside the one-chunk
# setting). resident-gpu.service stopped (trap). ALLDONE-feas-long2.
set -u
O=${O:-$HOME/c003/art/feas-long2}; mkdir -p $O
M=${MODELS_DIR:-$HOME/models}; SPY=~/venvs/bench-sglang-latest/bin/python
TPY=~/venvs/bench-trtllm/bin/python; TL=~/c003/art/scripts/_trtllm_latency.py; C13=~/venvs/bench-trtllm/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
CTX=40960; I=40000
echo "== $(date -Is) stopping resident-gpu"; systemctl --user stop resident-gpu
trap 'echo "== $(date -Is) starting resident-gpu"; systemctl --user start resident-gpu; sleep 20; echo "== $(date -Is) resident-gpu: $(systemctl --user is-active resident-gpu)"; echo ALLDONE-feas-long2' EXIT
sleep 5
trt() { # m name args...
  local m=$1 n=$2; shift 2
  echo "== $(date -Is) trtllm $m $n"
  CUDA_HOME=$C13 PATH=$C13/bin:$PATH LD_LIBRARY_PATH=$HOME/venvs/bench-trtllm/lib TLLM_WORKER_USE_SINGLE_PROCESS=1 \
    $TPY $TL --model $M/Qwen3.5-$m --input-len $I --output-len 1 --warmup 1 --iters 1 --max-seq-len $CTX "$@" \
    --output-json $O/trtllm-$m-$n.json > $O/trtllm-$m-$n.log 2>&1 || echo "FAIL trtllm $m $n"
}
trt 4B mnt40960-kv02 --max-num-tokens $CTX --kv-fraction 0.2
trt 4B chunk8192 --max-num-tokens 8192 --chunked-prefill
trt 0.8B chunk8192 --max-num-tokens 8192 --chunked-prefill
trt 0.8B mnt40960 --max-num-tokens $CTX
trt 2B mnt40960 --max-num-tokens $CTX
for mf in 0.72 0.74 0.76 0.78; do
  echo "== $(date -Is) sglang 4B mf$mf-mrr1"
  $SPY -m sglang.benchmark.one_batch --model-path $M/Qwen3.5-4B --dtype bfloat16 --context-length $CTX --mem-fraction-static $mf --batch-size 1 \
    --input-len $I --output-len 1 --run-name 4B-mf$mf-mrr1 --result-filename $O/sglang.jsonl --max-running-requests 1 > $O/sglang-4B-mf$mf-mrr1.log 2>&1 || echo "FAIL sglang 4B mf$mf"
  grep -q skipping $O/sglang-4B-mf$mf-mrr1.log && echo "SKIP sglang 4B mf$mf"
done
