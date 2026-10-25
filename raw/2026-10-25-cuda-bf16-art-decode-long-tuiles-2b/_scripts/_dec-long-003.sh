#!/usr/bin/env bash
# _dec-long-003.sh — decode after a long prompt for the article (added on 2026-10-25). Methods of the 128-to-4,000 measure/the long-prefill measure, prompts of the long-prefill measure (~/c003/art/prompts-long), the RTX 4080 locked at
# 2 805 / 11 201 MHz (checked before every unit), P0 profile, resident-gpu.service stopped (trap). bf16 everywhere.
#   herbert-cuda 31b86e1 defaults: --ctx 40960 --no-system --no-think --max-tokens 128 --warmup 1 --ask-file ppN; the
#     [turn 1] line (decode = the 127 tokens after the first).
#   llama.cpp b11718 CUDA BF16: llama-bench -p 0 -n 128 -d 10000,20000,40000 -r 1, default and -fa on.
#   herbert-cuda and llama.cpp alternated round by round, N rounds, order reversed every other round.
#   vLLM 0.31.0: bench latency input N, output 1 and 128, decode = 127 / (L128 - L1), 2 warm-ups, 5 iterations,
#     max-model-len 40960, max-num-seqs 8, capture 512; default attention and FLASHINFER.
#   TensorRT-LLM 1.4.0rc1 (_trtllm_latency.py): the same two measures, max_seq_len 40960, chunked prefill by 8192 and
#     one chunk with max_num_tokens = N (the long-prefill measure: max_num_tokens 40960 does not start on the 4B).
#   SGLang 0.5.22 one_batch: input N ×3 output 128, median_decode_throughput (median of the 3), mf 0.7 mrr 1 (the long-prefill measure: the
#     4B does not run 40 000).
# Every output kept in full under $O. ALLDONE-dec-long.
set -u
N=${N:-5}; O=${O:-$HOME/c003/art/dec-long}; mkdir -p $O/chat $O/llama $O/vllm $O/sglang $O/trtllm
B=~/herbert-cuda-bf16-art; HC=$B/target/release/herbert-cuda-chat; LP=~/c003/art/prompts-long
M=${MODELS_DIR:-$HOME/models}; G=${BENCH_DIR:-$HOME/bench-qwen35}; L=$HOME/llama.cpp-b11718/build-cuda/bin
VB=~/venvs/bench-vllm-latest/bin/vllm; SPY=~/venvs/bench-sglang-latest/bin/python
TPY=~/venvs/bench-trtllm/bin/python; TL=~/c003/art/scripts/_trtllm_latency.py; C13=~/venvs/bench-trtllm/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
MODELS=(0.8B 2B 4B); SIZES=(10000 20000 40000); CTX=40960; OUTS=(1 128)
VL=("default|" "flashinfer|--attention-backend FLASHINFER")
clocks() { local i; for i in $(seq 1 30); do [ "$(nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader,nounits | tr -d ' ')" = "2805,11201" ] && return 0; sleep 1; done; return 1; }
clocks || { echo "clocks not at 2805/11201, stop"; exit 1; }
{ date -Is; cat /proc/driver/nvidia/version; nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate,power.limit --format=csv
  echo "-- herbert-cuda $(cat $B/COMMIT)"; sha256sum $HC; $L/llama-bench --version 2>&1 | head -2
  for v in vllm sglang trtllm; do echo "-- $v"; ~/venvs/bench-$v*/bin/python -c "import importlib.metadata as m; print({p: m.version(p) for p in ['torch','transformers','vllm','sglang','tensorrt_llm','flashinfer-python'] if p in {d.metadata['Name'] for d in m.distributions()}})" 2>/dev/null; done
  echo "-- profile"; cat ~/.nv/nvidia-application-profiles-rc; sha256sum $G/Qwen3.5-*-BF16.gguf $LP/*.txt; } > $O/env.txt 2>&1
echo "== $(date -Is) stopping resident-gpu"; systemctl --user stop resident-gpu
trap 'echo "== $(date -Is) starting resident-gpu"; systemctl --user start resident-gpu; sleep 20; echo "== $(date -Is) resident-gpu: $(systemctl --user is-active resident-gpu)"; echo ALLDONE-dec-long' EXIT
sleep 5
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu,memory.used --format=csv -l 1 > $O/clocks.csv 2>&1 &
smi=$!
chk() { clocks || echo "!! clocks not at 2805/11201 before $1"; }

herbert() { # r m
  local r=$1 m=$2
  for p in "${SIZES[@]}"; do chk "hc $m pp$p"
    $HC $M/Qwen3.5-$m --ctx $CTX --no-system --no-think --max-tokens 128 --warmup 1 --ask-file $LP/pp$p.txt > $O/chat/r$r-$m-pp$p.log 2>&1 || echo "FAIL r$r $m hc pp$p"
  done
}
llama() { # r m
  local r=$1 m=$2 g=$G/Qwen3.5-$2-BF16.gguf
  for s in "default|" "faon|-fa on"; do sn=${s%%|*}; sa=${s#*|}; chk "llama $m tg $sn"
    # shellcheck disable=SC2086
    $L/llama-bench -m $g -dev CUDA0 -ngl 99 -p 0 -n 128 -d $(IFS=,; echo "${SIZES[*]}") -r 1 $sa -o json > $O/llama/r$r-$m-tg-$sn.json 2> $O/llama/r$r-$m-tg-$sn.err || echo "FAIL r$r $m llama tg $sn"
  done
}
for r in $(seq 1 $N); do
  for m in "${MODELS[@]}"; do
    echo "== $(date -Is) round $r $m · $(uptime)"
    if [ $((r % 2)) = 1 ]; then herbert $r $m; llama $r $m; else llama $r $m; herbert $r $m; fi
  done
done
TR=("chunk8192|--max-num-tokens 8192 --chunked-prefill" "mntN|")
for m in "${MODELS[@]}"; do
  for v in "${VL[@]}"; do vn=${v%%|*}; va=${v#*|}
    echo "== $(date -Is) vllm $m $vn · $(uptime)"
    for i in "${SIZES[@]}"; do for o in "${OUTS[@]}"; do chk "vllm $m $vn $i $o"
      # shellcheck disable=SC2086
      $VB bench latency --model $M/Qwen3.5-$m --dtype bfloat16 --input-len $i --output-len $o --batch-size 1 --num-iters-warmup 2 --num-iters 5 \
        --max-model-len $CTX --max-num-seqs 8 -cc.max_cudagraph_capture_size=512 --gpu-memory-utilization 0.85 $va \
        --output-json $O/vllm/$m-$vn-in$i-out$o.json > $O/vllm/$m-$vn-in$i-out$o.log 2>&1 || echo "FAIL vllm $m $vn in$i out$o"
    done; done
  done
  echo "== $(date -Is) sglang $m · $(uptime)"
  for i in "${SIZES[@]}"; do chk "sglang $m $i"
    $SPY -m sglang.benchmark.one_batch --model-path $M/Qwen3.5-$m --dtype bfloat16 --context-length $CTX --mem-fraction-static 0.7 --batch-size 1 \
      --input-len $i $i $i --output-len 128 --run-name $m-tg --result-filename $O/sglang/results.jsonl --max-running-requests 1 > $O/sglang/$m-tg$i.log 2>&1 || echo "FAIL sglang $m tg$i"
    grep -q skipping $O/sglang/$m-tg$i.log && echo "SKIP sglang $m tg$i"
  done
  for t in "${TR[@]}"; do tn=${t%%|*}; ta=${t#*|}
    echo "== $(date -Is) trtllm $m $tn · $(uptime)"
    for i in "${SIZES[@]}"; do for o in "${OUTS[@]}"; do chk "trtllm $m $tn $i $o"
      [ $tn = mntN ] && ta="--max-num-tokens $i"
      # shellcheck disable=SC2086
      CUDA_HOME=$C13 PATH=$C13/bin:$PATH LD_LIBRARY_PATH=$HOME/venvs/bench-trtllm/lib TLLM_WORKER_USE_SINGLE_PROCESS=1 \
        $TPY $TL --model $M/Qwen3.5-$m --input-len $i --output-len $o --warmup 2 --iters 5 --max-seq-len $CTX $ta \
        --output-json $O/trtllm/$m-$tn-in$i-out$o.json > $O/trtllm/$m-$tn-in$i-out$o.log 2>&1 || echo "FAIL trtllm $m $tn in$i out$o"
    done; done
  done
done
kill $smi
date -Is > $O/end.txt
