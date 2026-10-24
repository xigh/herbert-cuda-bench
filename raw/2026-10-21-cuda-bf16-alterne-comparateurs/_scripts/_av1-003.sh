#!/usr/bin/env bash
# _av1-003.sh — herbert-cuda-bf16 73b5a70, default and with the two attention switches (sdpa.gqa=1 sdpa.split=64
# sdpa.pipe=1), against the comparators of phase 5b in one run, alternated unit by unit: N = 5 rounds,
# order reversed every other round, the RTX 4080 with the GPU empty (the resident GPU service stopped for the run and
# restarted at the end). Same shapes and settings as 5b (_p5b-003.sh, each comparator's best setting there):
#   hc-def, hc-sw: chat --ask-file ppN, --ctx 61440, no thinking, 128 tokens, --warmup 1 (prefill and decode at depth);
#   llama-cuda b11718 F16: pp -ub 1024 -b 4096 -fa on, tg -p 0 -n 128 -d N default, -r 1;
#   vllm 0.31.0 --dtype float16 --attention-backend FLASHINFER: bench latency out 1 and 128, 2 warmups, 5 iterations;
#   sglang 0.5.22 --dtype float16 one_batch: 0.8B mf06 at every length, 4B mf07-mrr1 at 10K and 20K (40K and 60K skip
#   there, mf06 fails every 4B length).
# Lengths 10K, 20K, 40K, 60K. Every output kept in full under $O; ALLDONE-av1 at the end.
set -u
O=${O:-$HOME/c003/alt-vs/av1}; mkdir -p $O/chat $O/llama $O/vllm $O/sglang
LP=~/c003/long-prompts; HB=~/herbert-cuda-bf16-fix/target/release/herbert-cuda-chat
M=${MODELS_DIR:-$HOME/models}; G=${BENCH_DIR:-$HOME/bench-qwen35}; L=$HOME/llama.cpp-b11718
VB=~/venvs/bench-vllm-latest/bin/vllm; SPY=~/venvs/bench-sglang-latest/bin/python
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 DTYPE=float16 SGLANG_MAMBA_CONV_DTYPE=float16
PROF=~/.nv/nvidia-application-profiles-rc
LENS="10000 20000 40000 60000"; CTX=61440
SW="--tune sdpa.gqa=1 --tune sdpa.split=64 --tune sdpa.pipe=1"
case $(nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader,nounits | tr -d ' ') in 2805,11201) ;; *) echo "clocks not at 2805/11201, stop"; exit 1;; esac
[ -f $PROF ] || { echo "P0 profile missing, stop"; exit 1; }
envs() {
  { date -Is; uptime; cat /proc/driver/nvidia/version
    nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate,persistence_mode,memory.used --format=csv
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    echo "-- herbert-cuda-bf16 $(cat ~/herbert-cuda-bf16-fix/COMMIT)"; sha256sum $HB
    $L/build-cuda/bin/llama-bench --version 2>&1 | head -2; sha256sum $G/Qwen3.5-*-F16.gguf $LP/*.txt
    for v in vllm sglang; do echo "-- $v-latest"; ~/venvs/bench-$v-latest/bin/python -c "import importlib.metadata as m; print({p: m.version(p) for p in ['torch','transformers','vllm','sglang','flashinfer-python','flash-linear-attention'] if p in {d.metadata['Name'] for d in m.distributions()}})"; done
    echo "-- profile"; cat $PROF; } > $O/env-$1.txt 2>&1
}
echo "== $(date -Is) stopping resident-gpu"
systemctl --user stop resident-gpu
trap 'kill $smi 2>/dev/null; echo "== $(date -Is) starting resident-gpu"; systemctl --user start resident-gpu; sleep 20; echo "== $(date -Is) resident-gpu: $(systemctl --user is-active resident-gpu)"' EXIT
sleep 5
envs start
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu,memory.used,clocks_event_reasons.active --format=csv -l 1 > $O/clocks.csv 2>&1 &
smi=$!
free() { local n=0
  while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do
    echo "$(date -Is) busy: $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';')" >> $O/waits.txt
    n=$((n + 1)); [ $n -gt 60 ] && { echo "GPU never free, stop"; exit 1; }; sleep 30
  done
  echo "$(date -Is) $1" >> $O/launches.txt
}
herbert() { # r m name args...
  local r=$1 m=$2 n=$3; shift 3
  for p in $LENS; do
    free $O/chat/r$r-$m-$n-pp$p.log
    $HB $M/Qwen3.5-$m "$@" --ctx $CTX --no-system --no-think --max-tokens 128 --warmup 1 --ask-file $LP/pp$p.txt > $O/chat/r$r-$m-$n-pp$p.log 2>&1 || echo "FAIL r$r $m $n pp$p"
  done
}
llama() { # r m
  local r=$1 m=$2 g=$G/Qwen3.5-$2-F16.gguf
  free $O/llama/r$r-$m-pp.json
  $L/build-cuda/bin/llama-bench -m $g -dev CUDA0 -ngl 99 -p ${LENS// /,} -n 0 -r 1 -ub 1024 -b 4096 -fa on -o json > $O/llama/r$r-$m-pp.json 2> $O/llama/r$r-$m-pp.err || echo "FAIL r$r $m llama pp"
  free $O/llama/r$r-$m-tg.json
  $L/build-cuda/bin/llama-bench -m $g -dev CUDA0 -ngl 99 -p 0 -n 128 -d ${LENS// /,} -r 1 -o json > $O/llama/r$r-$m-tg.json 2> $O/llama/r$r-$m-tg.err || echo "FAIL r$r $m llama tg"
}
vllm() { # r m
  local r=$1 m=$2 i o
  for i in $LENS; do for o in 1 128; do
    free $O/vllm/r$r-$m-in$i-out$o.json
    $VB bench latency --model $M/Qwen3.5-$m --dtype float16 --input-len $i --output-len $o --batch-size 1 --num-iters-warmup 2 --num-iters 5 \
      --max-model-len $CTX --max-num-seqs 8 -cc.max_cudagraph_capture_size=512 --gpu-memory-utilization 0.85 --attention-backend FLASHINFER \
      --output-json $O/vllm/r$r-$m-in$i-out$o.json > $O/vllm/r$r-$m-in$i-out$o.log 2>&1 || echo "FAIL r$r vllm $m in$i out$o"
  done; done
}
sglang() { # r m
  local r=$1 m=$2 i mf=0.6 sa="" st=mf06 lens=$LENS
  [ $m = 4B ] && { mf=0.7; sa="--max-running-requests 1"; st=mf07-mrr1; lens="10000 20000"; }
  for i in $lens; do
    free $O/sglang/r$r-$m-$st-pp$i.log
    # shellcheck disable=SC2086
    $SPY -m sglang.benchmark.one_batch --model-path $M/Qwen3.5-$m --dtype float16 --context-length $CTX --mem-fraction-static $mf --batch-size 1 \
      --input-len $i $i $i --output-len 1 --run-name r$r-$m-$st-pp --result-filename $O/sglang/results.jsonl $sa > $O/sglang/r$r-$m-$st-pp$i.log 2>&1 || echo "FAIL r$r sglang $m pp$i"
    grep -q skipping $O/sglang/r$r-$m-$st-pp$i.log && echo "SKIP r$r sglang $m pp$i"
    free $O/sglang/r$r-$m-$st-tg$i.log
    # shellcheck disable=SC2086
    $SPY -m sglang.benchmark.one_batch --model-path $M/Qwen3.5-$m --dtype float16 --context-length $CTX --mem-fraction-static $mf --batch-size 1 \
      --input-len $i $i $i --output-len 128 --run-name r$r-$m-$st-tg --result-filename $O/sglang/results.jsonl $sa > $O/sglang/r$r-$m-$st-tg$i.log 2>&1 || echo "FAIL r$r sglang $m tg$i"
    grep -q skipping $O/sglang/r$r-$m-$st-tg$i.log && echo "SKIP r$r sglang $m tg$i"
  done
}
UNITS=(hc-def hc-sw llama-cuda vllm sglang)
unit() { # r m unit
  case $3 in
    hc-def) herbert $1 $2 hc-def ;;
    # shellcheck disable=SC2086
    hc-sw) herbert $1 $2 hc-sw $SW ;;
    llama-cuda) llama $1 $2 ;;
    vllm) vllm $1 $2 ;;
    sglang) sglang $1 $2 ;;
  esac
}
for r in 1 2 3 4 5; do
  order=("${UNITS[@]}"); [ $((r % 2)) = 0 ] && { order=(); for ((i=${#UNITS[@]}-1; i>=0; i--)); do order+=("${UNITS[$i]}"); done; }
  for m in 0.8B 4B; do
    for u in "${order[@]}"; do echo "== $(date -Is) round $r $m $u · $(uptime)"; unit $r $m $u; done
  done
done
envs end
date -Is > $O/end.txt
echo ALLDONE-av1
