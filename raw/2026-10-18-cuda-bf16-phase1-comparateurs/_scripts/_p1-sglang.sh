#!/usr/bin/env bash
# _p1-sglang.sh OUT — SGLang 0.5.22 settings sweep (protocol.md, phase 1): sglang.benchmark.one_batch (offline,
# batch 1): default, --disable-cuda-graph, --attention-backend flashinfer / triton, --chunked-prefill-size
# 2048 and -1, --enable-torch-compile; input 128, 512, 2048, 4000 with 1 output token, and 759 with 255.
# CUDA_HOME at the 12.8 toolkit (flashinfer compiles with it; /usr/bin/nvcc 12.0 fails).
set -u
O=$1; mkdir -p $O
M=${MODELS_DIR:-$HOME/models}; PY=~/venvs/bench-sglang-latest/bin/python
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
V=("default|" "nograph|--disable-cuda-graph" "flashinfer|--attention-backend flashinfer" "triton|--attention-backend triton"
   "chunk2048|--chunked-prefill-size 2048" "nochunk|--chunked-prefill-size -1" "compile|--enable-torch-compile")
for m in 0.8B 4B; do
  for v in "${V[@]}"; do
    name=${v%%|*}; args=${v#*|}
    echo "== $(date -Is) sglang $m $name · $(uptime)"
    # shellcheck disable=SC2086
    $PY -m sglang.benchmark.one_batch --model-path $M/Qwen3.5-$m --dtype bfloat16 --context-length 4608 --mem-fraction-static 0.8 \
      --batch-size 1 --input-len 128 512 2048 4000 --output-len 1 --run-name $m-$name-pp --result-filename $O/results.jsonl $args \
      > $O/$m-$name-pp.log 2>&1 || echo "FAIL sglang $m $name pp"
    # shellcheck disable=SC2086
    $PY -m sglang.benchmark.one_batch --model-path $M/Qwen3.5-$m --dtype bfloat16 --context-length 4608 --mem-fraction-static 0.8 \
      --batch-size 1 --input-len 759 --output-len 255 --run-name $m-$name-tg --result-filename $O/results.jsonl $args \
      > $O/$m-$name-tg.log 2>&1 || echo "FAIL sglang $m $name tg"
  done
done
