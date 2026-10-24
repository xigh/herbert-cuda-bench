#!/usr/bin/env bash
# _p1.sh — phase 1 of the #003 campaign (protocol.md): the comparators' settings sweeps, one after the other,
# clocks logged every second throughout, the CudaNoStablePerfLimit profile checked first.
set -u
O=${O:-$HOME/c003/phase1}; mkdir -p $O
D=$(dirname "$0")
{ date -Is; uptime; nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate --format=csv; cat ~/.nv/nvidia-application-profiles-rc
  $HOME/llama.cpp-b11718/build-cuda/bin/llama-server --version; $HOME/llama.cpp-08ffba21e/build-cuda/bin/llama-server --version
  for v in vllm-latest vllm-cu sglang-latest transformers-latest; do echo "-- $v"; ~/venvs/bench-$v/bin/python -c "import importlib.metadata as m; print({p: m.version(p) for p in ['torch','transformers','vllm','sglang','flashinfer-python','flash-linear-attention'] if any(d.metadata['Name']==p for d in m.distributions())})"; done
  sha256sum ${BENCH_DIR:-$HOME/bench-qwen35}/Qwen3.5-*-BF16.gguf; } > $O/env.txt 2>&1
case $(nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader,nounits | tr -d ' ') in 2805,11201) ;; *) echo "clocks not at 2805/11201, stop"; exit 1;; esac
[ -f ~/.nv/nvidia-application-profiles-rc ] || { echo "P0 profile missing, stop"; exit 1; }
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu,clocks_event_reasons.active --format=csv -l 1 > $O/clocks.csv 2>&1 &
smi=$!
bash $D/_p1-llama.sh $O/llama
bash $D/_p1-vllm.sh $O/vllm
bash $D/_p1-sglang.sh $O/sglang
echo "== $(date -Is) transformers"
~/venvs/bench-transformers-latest/bin/python $D/_p1_transformers.py $O/transformers.jsonl > $O/transformers.log 2>&1 || echo "FAIL transformers"
kill $smi
date -Is > $O/end.txt
echo ALLDONE-phase1
