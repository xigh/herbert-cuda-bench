#!/usr/bin/env bash
# _after-p1b-sglang-003.sh — queued behind the vLLM rerun (waits for ALLDONE-after-p1): SGLang's sweep again with
# each length three times in the same process (sglang.benchmark.one_batch measures each shape once; phase 1 gave
# one sample per shape, and the 0.8B at 4000 tokens looked like a one-off cost). Same variants as phase 1;
# then transformers' 4B settings that phase 1 lost, one process each.
set -u
until grep -q ALLDONE-after-p1 ~/c003/after-p1.out; do sleep 60; done
O=${O:-$HOME/c003/after-p1/sglang-rep}; mkdir -p $O
M=${MODELS_DIR:-$HOME/models}; PY=~/venvs/bench-sglang-latest/bin/python
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu --format=csv -l 1 > $O/clocks.csv 2>&1 &
smi=$!
V=("default|" "nograph|--disable-cuda-graph" "flashinfer|--attention-backend flashinfer" "triton|--attention-backend triton"
   "chunk2048|--chunked-prefill-size 2048" "nochunk|--chunked-prefill-size -1" "compile|--enable-torch-compile")
# The compile variant ran out of memory on the 4B at --mem-fraction-static 0.8 in phase 1: 0.7 for it.
for m in 0.8B 4B; do for v in "${V[@]}"; do name=${v%%|*}; args=${v#*|}
  echo "== $(date -Is) sglang-rep $m $name · $(uptime)"
  mf=0.8; [ $name = compile ] && mf=0.7
  # shellcheck disable=SC2086
  $PY -m sglang.benchmark.one_batch --model-path $M/Qwen3.5-$m --dtype bfloat16 --context-length 4608 --mem-fraction-static $mf --batch-size 1 \
    --input-len 128 512 2048 4000 128 512 2048 4000 128 512 2048 4000 --output-len 1 --run-name $m-$name-pp --result-filename $O/results.jsonl $args > $O/$m-$name-pp.log 2>&1 || echo "FAIL sglang-rep $m $name pp"
  # shellcheck disable=SC2086
  $PY -m sglang.benchmark.one_batch --model-path $M/Qwen3.5-$m --dtype bfloat16 --context-length 4608 --mem-fraction-static $mf --batch-size 1 \
    --input-len 759 759 759 --output-len 255 --run-name $m-$name-tg --result-filename $O/results.jsonl $args > $O/$m-$name-tg.log 2>&1 || echo "FAIL sglang-rep $m $name tg"
done; done
# transformers, the 4B settings that ran out of memory in phase 1 (the previous setting's model not freed in the
# same process): one process per setting.
for st in sdpa-static-compile eager-dynamic eager-static eager-static-compile; do
  echo "== $(date -Is) transformers 4B $st"
  ~/venvs/bench-transformers-latest/bin/python ~/c003/phase1-scripts/_p1_transformers.py $HOME/c003/after-p1/transformers-4B.jsonl 4B $st >> $HOME/c003/after-p1/transformers-4B.log 2>&1 || echo "FAIL transformers 4B $st"
done
kill $smi
echo ALLDONE-after-p1b
