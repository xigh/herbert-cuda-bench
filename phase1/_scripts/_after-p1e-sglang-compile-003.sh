#!/usr/bin/env bash
# _after-p1e-sglang-compile-003.sh — queued behind the token check (waits for ALLDONE-after-p1d): SGLang's
# --enable-torch-compile failed in phase 1 and in the rerun (inductor's Triton kernels: "'NoneType' object has no
# attribute '_unflatten_ir'", then out of memory capturing the CUDA graphs: 3.82 GiB asked at 0.7). Once more with
# the graphs bounded to batch 8 (--cuda-graph-max-bs-decode, --torch-compile-max-bs) and the static fraction at
# 0.6 and 0.5; a failure is kept as such.
set -u
until grep -q ALLDONE-after-p1d ~/c003/after-p1d.out; do sleep 60; done
O=${O:-$HOME/c003/after-p1e}; mkdir -p $O
M=${MODELS_DIR:-$HOME/models}; PY=~/venvs/bench-sglang-latest/bin/python
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu --format=csv -l 1 > $O/clocks.csv 2>&1 &
smi=$!
for m in 0.8B 4B; do for mf in 0.6 0.5; do name=compile-bs8-mf$mf
  echo "== $(date -Is) sglang $m $name · $(uptime)"
  args="--enable-torch-compile --torch-compile-max-bs 8 --cuda-graph-max-bs-decode 8"
  # shellcheck disable=SC2086
  $PY -m sglang.benchmark.one_batch --model-path $M/Qwen3.5-$m --dtype bfloat16 --context-length 4608 --mem-fraction-static $mf --batch-size 1 \
    --input-len 128 512 2048 4000 128 512 2048 4000 128 512 2048 4000 --output-len 1 --run-name $m-$name-pp --result-filename $O/results.jsonl $args > $O/$m-$name-pp.log 2>&1 || echo "FAIL sglang $m $name pp"
  # shellcheck disable=SC2086
  $PY -m sglang.benchmark.one_batch --model-path $M/Qwen3.5-$m --dtype bfloat16 --context-length 4608 --mem-fraction-static $mf --batch-size 1 \
    --input-len 759 759 759 --output-len 255 --run-name $m-$name-tg --result-filename $O/results.jsonl $args > $O/$m-$name-tg.log 2>&1 || echo "FAIL sglang $m $name tg"
done; done
kill $smi
echo ALLDONE-after-p1e
