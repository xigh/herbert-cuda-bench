#!/usr/bin/env bash
# _p2c-http-skipblank-003.sh — after phase 2: the common HTTP client again with --skip-blank (whitespace-only deltas
# are not the first token). transformers serve sends a reasoning_content "\n" before the model runs, which made its
# phase 2 TTFT and decode void (http-check/). transformers' two servers, and as a control two servers whose phase 2
# numbers looked sound (hc-cand, vllm-cg512): the option should not move them.
set -u
O=${O:-$HOME/c003/phase2c}; mkdir -p $O
S=~/c003/phase2-scripts; B=~/herbert-cuda-mma; HCS=$B/target/release/herbert-cuda-serve; M=${MODELS_DIR:-$HOME/models}
VB=~/venvs/bench-vllm-latest/bin/vllm; TF=~/venvs/bench-transformers-latest/bin
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
CAND="--tune gemm.auto=3 --tune gemm.layout=xor --tune prefill.block=auto"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu --format=csv -l 1 > $O/clocks.csv 2>&1 &
smi=$!
wait_ready() { for _ in $(seq 1 $2); do curl -sf -o /dev/null "$1" && return 0; sleep 1; done; return 1; }
stop() { kill -TERM -- -$1 2>/dev/null; for _ in $(seq 1 60); do kill -0 $1 2>/dev/null || break; sleep 1; done; kill -KILL -- -$1 2>/dev/null; sleep 3; }
for m in 0.8B 4B; do
  for e in transformers-sdpa transformers-sdpa-compile hc-cand vllm-cg512; do
    echo "== $(date -Is) $m $e · $(uptime)"
    U=http://localhost:8123; H=$U/; api=messages; url=$U/v1/messages; mn=Qwen3.5-$m
    case $e in
      # shellcheck disable=SC2086
      hc-cand) setsid $HCS $M/Qwen3.5-$m --ctx 4608 --port 8123 --name $mn $CAND > $O/server-$m-$e.log 2>&1 & ;;
      vllm-cg512) setsid $VB serve $M/Qwen3.5-$m --dtype bfloat16 --max-model-len 4608 --max-num-seqs 8 -cc.max_cudagraph_capture_size=512 --gpu-memory-utilization 0.85 \
          --no-enable-prefix-caching --served-model-name $mn --port 8123 > $O/server-$m-$e.log 2>&1 &
        H=$U/health ;;
      transformers-*) ta=""; [ $e = transformers-sdpa-compile ] && ta="--compile"
        # shellcheck disable=SC2086
        setsid $TF/transformers serve --host localhost --port 8123 --dtype bfloat16 --device cuda:0 --attn-implementation sdpa $ta > $O/server-$m-$e.log 2>&1 &
        H=$U/v1/models; api=openai; url=$U/v1/chat/completions; mn=$M/Qwen3.5-$m ;;
    esac
    P=$!
    if wait_ready $H 900; then
      python3 $S/_http_client.py --skip-blank --url $url --api $api --engine $e --model $mn --out $O/http-$m-$e.json > $O/client-$m-$e.log 2>&1 || echo "FAIL http $m $e"
    else
      echo "FAIL http $m $e: server not ready"
    fi
    stop $P
  done
done
kill $smi
echo ALLDONE-phase2c
