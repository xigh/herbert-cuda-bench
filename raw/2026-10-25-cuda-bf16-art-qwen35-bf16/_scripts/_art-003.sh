#!/usr/bin/env bash
# _art-003.sh — for the #003 article (added on 2026-10-24), on the RTX 4080
# locked at 2 805 / 11 201 MHz (checked before every unit), every CUDA engine in P0 (CudaNoStablePerfLimit profile),
# resident-gpu.service stopped (trap). bf16 weights, activations and KV everywhere.
#   herbert-cuda: herbert-cuda-bf16 31b86e1 (~/herbert-cuda-bf16-art), its defaults (sdpa.gqa=1 sdpa.split=64
#     sdpa.pipe=1 gemm.auto=3 gemm.layout=xor prefill.block=auto); prefill on pp128 … pp4000 (no thinking, 1 token,
#     --warmup 1), decode on p3 (thinking, 256 tokens, --warmup 1), the [turn 1] line.
#   llama.cpp b11718 CUDA, BF16 GGUF: llama-bench -p 128,256,512,1024,2048,4000 -n 0 with the default and with
#     -ub 1024 -b 4096 -fa on (#003 phase 1's best prefill settings), then -p 0 -n 255 -d 759, default and -fa on.
#   herbert-cuda and llama.cpp alternated round by round, N rounds, the order reversed every other round.
#   vLLM 0.31.0: bench latency, input 128 … 4000 output 1, and 759 output 1 and 255 (decode = 254 / (L255 − L1)),
#     2 warm-ups, 5 iterations, max-model-len 4608, max-num-seqs 8, capture 512 (#003's setting).
#   SGLang 0.5.22: one_batch, the six sizes ×3 output 1, then 759 → 255 ×3 (median decode).
#   TensorRT-LLM 1.4.0rc1 (PyTorch backend, _trtllm_latency.py, as vLLM's method): the same inputs, 2 warm-ups,
#     5 iterations, max_seq_len 4608, CUDA graph batch 1; TLLM_WORKER_USE_SINGLE_PROCESS=1, nvcc 13.0 for DeepGEMM.
#   vLLM, SGLang and TensorRT-LLM: one pass per model (their own repetitions), after the alternated rounds.
# Every output kept in full under $O. ALLDONE-art.
set -u
N=${N:-5}; O=${O:-$HOME/c003/art/campaign}; mkdir -p $O/chat $O/llama $O/vllm $O/sglang $O/trtllm
B=~/herbert-cuda-bf16-art; HC=$B/target/release/herbert-cuda-chat; PR=$B/bench/prompts
M=${MODELS_DIR:-$HOME/models}; G=${BENCH_DIR:-$HOME/bench-qwen35}; L=$HOME/llama.cpp-b11718/build-cuda/bin
VB=~/venvs/bench-vllm-latest/bin/vllm; SPY=~/venvs/bench-sglang-latest/bin/python
TPY=~/venvs/bench-trtllm/bin/python; TL=~/c003/q4q8/_trtllm_latency.py; C13=~/venvs/bench-trtllm/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
MODELS=(0.8B 2B 4B); SIZES=(128 256 512 1024 2048 4000)
clocks() { local i; for i in $(seq 1 30); do [ "$(nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader,nounits | tr -d ' ')" = "2805,11201" ] && return 0; sleep 1; done; return 1; }
clocks || { echo "clocks not at 2805/11201, stop"; exit 1; }
{ date -Is; cat /proc/driver/nvidia/version; nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate --format=csv
  echo "-- herbert-cuda $(cat $B/COMMIT)"; sha256sum $HC; $L/llama-bench --version 2>&1 | head -2
  for v in vllm sglang trtllm; do echo "-- $v"; ~/venvs/bench-$v*/bin/python -c "import importlib.metadata as m; print({p: m.version(p) for p in ['torch','transformers','vllm','sglang','tensorrt_llm','flashinfer-python'] if p in {d.metadata['Name'] for d in m.distributions()}})" 2>/dev/null; done
  echo "-- profile"; cat ~/.nv/nvidia-application-profiles-rc; sha256sum $G/Qwen3.5-*-BF16.gguf; } > $O/env.txt 2>&1
echo "== $(date -Is) stopping resident-gpu"; systemctl --user stop resident-gpu
trap 'echo "== $(date -Is) starting resident-gpu"; systemctl --user start resident-gpu; sleep 20; echo "== $(date -Is) resident-gpu: $(systemctl --user is-active resident-gpu)"; echo ALLDONE-art' EXIT
sleep 5
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu,memory.used --format=csv -l 1 > $O/clocks.csv 2>&1 &
smi=$!
chk() { clocks || echo "!! clocks not at 2805/11201 before $1"; }

herbert() { # r m
  local r=$1 m=$2
  for p in "${SIZES[@]}"; do chk "hc $m pp$p"
    $HC $M/Qwen3.5-$m --no-system --no-think --max-tokens 1 --warmup 1 --ask-file $PR/pp$p.txt > $O/chat/r$r-$m-pp$p.log 2>&1 || echo "FAIL r$r $m hc pp$p"
  done
  chk "hc $m p3"
  $HC $M/Qwen3.5-$m --no-system --max-tokens 256 --warmup 1 --ask-file $PR/p3-long.txt > $O/chat/r$r-$m-p3.log 2>&1 || echo "FAIL r$r $m hc p3"
}
llama() { # r m
  local r=$1 m=$2 g=$G/Qwen3.5-$2-BF16.gguf
  for s in "default|" "ub1024-b4096-faon|-ub 1024 -b 4096 -fa on"; do sn=${s%%|*}; sa=${s#*|}; chk "llama $m pp $sn"
    # shellcheck disable=SC2086
    $L/llama-bench -m $g -dev CUDA0 -ngl 99 -p 128,256,512,1024,2048,4000 -n 0 -r 1 $sa -o json > $O/llama/r$r-$m-pp-$sn.json 2> $O/llama/r$r-$m-pp-$sn.err || echo "FAIL r$r $m llama pp $sn"
  done
  for s in "default|" "faon|-fa on"; do sn=${s%%|*}; sa=${s#*|}; chk "llama $m tg $sn"
    # shellcheck disable=SC2086
    $L/llama-bench -m $g -dev CUDA0 -ngl 99 -p 0 -n 255 -d 759 -r 1 $sa -o json > $O/llama/r$r-$m-tg-$sn.json 2> $O/llama/r$r-$m-tg-$sn.err || echo "FAIL r$r $m llama tg $sn"
  done
}
for r in $(seq 1 $N); do
  for m in "${MODELS[@]}"; do
    echo "== $(date -Is) round $r $m · $(uptime)"
    if [ $((r % 2)) = 1 ]; then herbert $r $m; llama $r $m; else llama $r $m; herbert $r $m; fi
  done
done
IO=(); for p in "${SIZES[@]}"; do IO+=("$p:1"); done; IO+=("759:1" "759:255")
for m in "${MODELS[@]}"; do
  echo "== $(date -Is) vllm $m · $(uptime)"
  for io in "${IO[@]}"; do i=${io%%:*}; o=${io#*:}; chk "vllm $m $io"
    $VB bench latency --model $M/Qwen3.5-$m --dtype bfloat16 --input-len $i --output-len $o --batch-size 1 --num-iters-warmup 2 --num-iters 5 \
      --max-model-len 4608 --max-num-seqs 8 -cc.max_cudagraph_capture_size=512 --gpu-memory-utilization 0.85 \
      --output-json $O/vllm/$m-in$i-out$o.json > $O/vllm/$m-in$i-out$o.log 2>&1 || echo "FAIL vllm $m in$i out$o"
  done
  echo "== $(date -Is) sglang $m · $(uptime)"; chk "sglang $m"
  $SPY -m sglang.benchmark.one_batch --model-path $M/Qwen3.5-$m --dtype bfloat16 --context-length 4608 --mem-fraction-static 0.8 --batch-size 1 \
    --input-len "${SIZES[@]}" "${SIZES[@]}" "${SIZES[@]}" --output-len 1 --run-name $m-pp --result-filename $O/sglang/results.jsonl > $O/sglang/$m-pp.log 2>&1 || echo "FAIL sglang $m pp"
  $SPY -m sglang.benchmark.one_batch --model-path $M/Qwen3.5-$m --dtype bfloat16 --context-length 4608 --mem-fraction-static 0.8 --batch-size 1 \
    --input-len 759 759 759 --output-len 255 --run-name $m-tg --result-filename $O/sglang/results.jsonl > $O/sglang/$m-tg.log 2>&1 || echo "FAIL sglang $m tg"
  echo "== $(date -Is) trtllm $m · $(uptime)"
  for io in "${IO[@]}"; do i=${io%%:*}; o=${io#*:}; chk "trtllm $m $io"
    CUDA_HOME=$C13 PATH=$C13/bin:$PATH LD_LIBRARY_PATH=$HOME/venvs/bench-trtllm/lib TLLM_WORKER_USE_SINGLE_PROCESS=1 \
      $TPY $TL --model $M/Qwen3.5-$m --input-len $i --output-len $o --warmup 2 --iters 5 --max-seq-len 4608 --max-num-tokens 4608 \
      --output-json $O/trtllm/$m-in$i-out$o.json > $O/trtllm/$m-in$i-out$o.log 2>&1 || echo "FAIL trtllm $m in$i out$o"
  done
done
kill $smi
date -Is > $O/end.txt
