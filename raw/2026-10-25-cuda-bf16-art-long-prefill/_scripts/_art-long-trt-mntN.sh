#!/usr/bin/env bash
# _art-long-trt-mntN.sh — supplement to _art-long-003.sh: TensorRT-LLM with max_num_tokens 40960 warms up a 40 960-token
# forward at start, out of memory on the 4B at every length (256 MiB short). Here the prompt still goes in one chunk but
# max_num_tokens = the input length (max_seq_len 40960 as before), on the three models, same method (2 warm-ups,
# 5 iterations, output 1). Same clocks check, resident-gpu stopped (trap). Outputs in the campaign's trtllm/ as
# <model>-mntN-in<N>-out1. ALLDONE-art-long-trt.
set -u
O=${O:-$HOME/c003/art/campaign-long}; M=${MODELS_DIR:-$HOME/models}
TPY=~/venvs/bench-trtllm/bin/python; TL=~/c003/art/scripts/_trtllm_latency.py; C13=~/venvs/bench-trtllm/lib/python3.12/site-packages/nvidia/cu13
MODELS=(0.8B 2B 4B); SIZES=(2000 5000 10000 20000 40000); CTX=40960
clocks() { local i; for i in $(seq 1 30); do [ "$(nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader,nounits | tr -d ' ')" = "2805,11201" ] && return 0; sleep 1; done; return 1; }
clocks || { echo "clocks not at 2805/11201, stop"; exit 1; }
chk() { clocks || echo "!! clocks not at 2805/11201 before $1"; }
echo "== $(date -Is) stopping resident-gpu"; systemctl --user stop resident-gpu
trap 'echo "== $(date -Is) starting resident-gpu"; systemctl --user start resident-gpu; sleep 20; echo "== $(date -Is) resident-gpu: $(systemctl --user is-active resident-gpu)"; echo ALLDONE-art-long-trt' EXIT
sleep 5
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu,memory.used --format=csv -l 1 > $O/clocks-trt-mntN.csv 2>&1 &
smi=$!
for m in "${MODELS[@]}"; do
  echo "== $(date -Is) trtllm $m mntN · $(uptime)"
  for i in "${SIZES[@]}"; do chk "trtllm $m mntN $i"
    CUDA_HOME=$C13 PATH=$C13/bin:$PATH LD_LIBRARY_PATH=$HOME/venvs/bench-trtllm/lib TLLM_WORKER_USE_SINGLE_PROCESS=1 \
      $TPY $TL --model $M/Qwen3.5-$m --input-len $i --output-len 1 --warmup 2 --iters 5 --max-seq-len $CTX --max-num-tokens $i \
      --output-json $O/trtllm/$m-mntN-in$i-out1.json > $O/trtllm/$m-mntN-in$i-out1.log 2>&1 || echo "FAIL trtllm $m mntN in$i"
  done
done
kill $smi
date -Is > $O/end-trt-mntN.txt
