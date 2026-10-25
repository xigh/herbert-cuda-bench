#!/usr/bin/env bash
# _feas-long.sh — feasibility of the article's long prefill (pp2000 … pp40000, added on 2026-10-25) at the largest
# length, 40 000 tokens, before the campaign: herbert-cuda (--ctx 40960), llama-bench (both settings), vLLM (default
# and FLASHINFER, phase 5a's best at long), TensorRT-LLM (max_seq_len and max_num_tokens 40960) on the 4B; SGLang on
# the three models with three memory settings (phase 5b: 4B in f16 did not fit at 40K). One iteration each.
# resident-gpu.service stopped (trap). Every output kept in full under $O. ALLDONE-feas-long.
set -u
O=${O:-$HOME/c003/art/feas-long}; mkdir -p $O
B=~/herbert-cuda-bf16-art; HC=$B/target/release/herbert-cuda-chat; LP=~/c003/art/prompts-long
M=${MODELS_DIR:-$HOME/models}; G=${BENCH_DIR:-$HOME/bench-qwen35}; L=$HOME/llama.cpp-b11718/build-cuda/bin
VB=~/venvs/bench-vllm-latest/bin/vllm; SPY=~/venvs/bench-sglang-latest/bin/python
TPY=~/venvs/bench-trtllm/bin/python; TL=~/c003/q4q8/_trtllm_latency.py; C13=~/venvs/bench-trtllm/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
CTX=40960; I=40000
echo "== $(date -Is) stopping resident-gpu"; systemctl --user stop resident-gpu
trap 'echo "== $(date -Is) starting resident-gpu"; systemctl --user start resident-gpu; sleep 20; echo "== $(date -Is) resident-gpu: $(systemctl --user is-active resident-gpu)"; echo ALLDONE-feas-long' EXIT
sleep 5
echo "== $(date -Is) herbert 4B"
$HC $M/Qwen3.5-4B --ctx $CTX --no-system --no-think --max-tokens 1 --warmup 1 --ask-file $LP/pp$I.txt > $O/hc-4B.log 2>&1 || echo "FAIL hc 4B"
for s in "default|" "ub1024-b4096-faon|-ub 1024 -b 4096 -fa on"; do sn=${s%%|*}; sa=${s#*|}
  echo "== $(date -Is) llama 4B $sn"
  # shellcheck disable=SC2086
  $L/llama-bench -m $G/Qwen3.5-4B-BF16.gguf -dev CUDA0 -ngl 99 -p $I -n 0 -r 1 $sa -o json > $O/llama-4B-$sn.json 2> $O/llama-4B-$sn.err || echo "FAIL llama 4B $sn"
done
for s in "default|" "flashinfer|--attention-backend FLASHINFER"; do sn=${s%%|*}; sa=${s#*|}
  echo "== $(date -Is) vllm 4B $sn"
  # shellcheck disable=SC2086
  $VB bench latency --model $M/Qwen3.5-4B --dtype bfloat16 --input-len $I --output-len 1 --batch-size 1 --num-iters-warmup 1 --num-iters 1 \
    --max-model-len $CTX --max-num-seqs 8 -cc.max_cudagraph_capture_size=512 --gpu-memory-utilization 0.85 $sa \
    --output-json $O/vllm-4B-$sn.json > $O/vllm-4B-$sn.log 2>&1 || echo "FAIL vllm 4B $sn"
done
echo "== $(date -Is) trtllm 4B"
CUDA_HOME=$C13 PATH=$C13/bin:$PATH LD_LIBRARY_PATH=$HOME/venvs/bench-trtllm/lib TLLM_WORKER_USE_SINGLE_PROCESS=1 \
  $TPY $TL --model $M/Qwen3.5-4B --input-len $I --output-len 1 --warmup 1 --iters 1 --max-seq-len $CTX --max-num-tokens $CTX \
  --output-json $O/trtllm-4B.json > $O/trtllm-4B.log 2>&1 || echo "FAIL trtllm 4B"
for m in 0.8B 2B 4B; do
  for s in "mf08-mrr1-mc2|0.8|--max-running-requests 1 --max-mamba-cache-size 2" "mf07-mrr1|0.7|--max-running-requests 1" "mf06|0.6|"; do
    sn=${s%%|*}; rest=${s#*|}; mf=${rest%%|*}; sa=${rest#*|}
    echo "== $(date -Is) sglang $m $sn"
    # shellcheck disable=SC2086
    $SPY -m sglang.benchmark.one_batch --model-path $M/Qwen3.5-$m --dtype bfloat16 --context-length $CTX --mem-fraction-static $mf --batch-size 1 \
      --input-len $I --output-len 1 --run-name $m-$sn --result-filename $O/sglang.jsonl $sa > $O/sglang-$m-$sn.log 2>&1 || echo "FAIL sglang $m $sn"
    grep -q skipping $O/sglang-$m-$sn.log && echo "SKIP sglang $m $sn"
  done
done
