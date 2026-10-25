#!/usr/bin/env bash
# _feas-long3.sh — SGLang 4B at 40 000 tokens, third pass: mf 0.72 skips (KV pool 39 436), 0.74 … 0.78 run out of memory
# in the forward (158–314 MiB short, 252 MiB in CUDA graphs). Tries: mf 0.73; mf 0.74 and 0.76 without CUDA graphs;
# mf 0.74 with expandable segments. --max-running-requests 1 everywhere. resident-gpu stopped (trap). ALLDONE-feas-long3.
set -u
O=${O:-$HOME/c003/art/feas-long3}; mkdir -p $O
M=${MODELS_DIR:-$HOME/models}; SPY=~/venvs/bench-sglang-latest/bin/python
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
CTX=40960; I=40000
echo "== $(date -Is) stopping resident-gpu"; systemctl --user stop resident-gpu
trap 'echo "== $(date -Is) starting resident-gpu"; systemctl --user start resident-gpu; sleep 20; echo "== $(date -Is) resident-gpu: $(systemctl --user is-active resident-gpu)"; echo ALLDONE-feas-long3' EXIT
sleep 5
sg() { # name mf env args...
  local n=$1 mf=$2 e=$3; shift 3
  echo "== $(date -Is) sglang 4B $n"
  env $e $SPY -m sglang.benchmark.one_batch --model-path $M/Qwen3.5-4B --dtype bfloat16 --context-length $CTX --mem-fraction-static $mf --batch-size 1 \
    --input-len $I --output-len 1 --run-name 4B-$n --result-filename $O/sglang.jsonl --max-running-requests 1 "$@" > $O/sglang-4B-$n.log 2>&1 || echo "FAIL sglang 4B $n"
  grep -q skipping $O/sglang-4B-$n.log && echo "SKIP sglang 4B $n"
}
sg mf073-mrr1 0.73 X=1
sg mf074-mrr1-nograph 0.74 X=1 --disable-cuda-graph
sg mf076-mrr1-nograph 0.76 X=1 --disable-cuda-graph
sg mf074-mrr1-expseg 0.74 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
