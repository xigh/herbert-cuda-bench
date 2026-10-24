#!/usr/bin/env bash
# _p5b-003.sh — phase 5, step 5b (protocol.md and its 16:10 amendment): herbert against the comparators at 10K, 20K,
# 40K and 60K tokens on the RTX 4080, the two instruments of phase 2 without a change of method.
#   A. each engine's bench: herbert-cuda-bf16 011bfdf, herbert-cuda-f16 0c7d88d (default and --tune gemm.f16=1) by
#      their chat (--ask-file, --ctx 61440, no thinking, 128 tokens: prefill and decode at depth on the [turn] line,
#      --warmup 1), herbert-vk-f16 cc51363 at 10K and 20K (--ask, the prompt fits in one argument there; host.poll=1
#      as a second unit, as in 4b) and llama.cpp b11718 F16 CUDA and Vulkan (their two 5a prefill settings; decode
#      -p 0 -n 128 -d N, default and -fa on as in 4a) alternated round by round (N = 5, order reversed every other
#      round); vLLM (out 1 and 128, decode = 127 / (L128 - L1)), SGLang (one_batch, out 1 three samples, then out 128,
#      one process per length) and transformers (1 and 128 tokens, same definition as vLLM) at their two 5a settings;
#   B. the common HTTP client (_http_client.py --skip-blank --prompts prompts-long.json: five passages per length,
#      three rounds, 128 tokens), one server after the other, every one at a context of 61 440 and without prefix cache.
# Every output kept in full under $O; ALLDONE-phase5b at the end.
set -u
O=${O:-$HOME/c003/phase5b}; mkdir -p $O/chat $O/llama $O/vllm $O/sglang $O/transformers $O/http
S=~/c003/phase5-scripts; S2=~/c003/phase2-scripts; LP=~/c003/long-prompts; T5=~/c003/phase5a/tokens
HB=~/herbert-cuda-bf16/target/release; HF=~/herbert-cuda-f16/target/release; HVF=~/herbert-vk-f16-cc51363/target/release
M=${MODELS_DIR:-$HOME/models}; G=${BENCH_DIR:-$HOME/bench-qwen35}; L=$HOME/llama.cpp-b11718
VB=~/venvs/bench-vllm-latest/bin/vllm; SPY=~/venvs/bench-sglang-latest/bin/python; TF=~/venvs/bench-transformers-latest/bin
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 DTYPE=float16 SGLANG_MAMBA_CONV_DTYPE=float16
PROF=~/.nv/nvidia-application-profiles-rc
LENS="10000 20000 40000 60000"; CTX=61440
VLLM_BASE="--max-num-seqs 8 -cc.max_cudagraph_capture_size=512"
# The two settings per comparator of step 5a (phase5/_p5a_summary.py).
VL=("flashinfer|--attention-backend FLASHINFER" "mnbt2048|--max-num-batched-tokens 2048")
# SGLang: no setting runs the 4B at 40K or 60K on 16 GB (out of memory at 0.75 and up, KV pool under 40K at 0.72
# and down); mf06 is the only one at 60K for the 0.8B and fails every length in 4B.
SG=("mf06|0.6|" "mf07-mrr1|0.7|--max-running-requests 1")
TFS=(sdpa-dynamic sdpa-static-compile)
envs() {
  { date -Is; uptime; cat /proc/driver/nvidia/version
    nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate,persistence_mode --format=csv
    echo "-- herbert-cuda-bf16 $(cat ~/herbert-cuda-bf16/COMMIT)"; sha256sum $HB/herbert-cuda-chat $HB/herbert-cuda-serve
    echo "-- herbert-cuda-f16 $(cat ~/herbert-cuda-f16/COMMIT)"; sha256sum $HF/herbert-cuda-chat $HF/herbert-cuda-serve
    echo "-- herbert-vk-f16 $(cat ~/herbert-vk-f16-cc51363/COMMIT)"; sha256sum $HVF/herbert-vk-chat $HVF/herbert-vk-serve
    $L/build-cuda/bin/llama-bench --version 2>&1 | head -2; sha256sum $G/Qwen3.5-*-F16.gguf $LP/*.txt $LP/prompts-long.json
    for v in vllm sglang transformers; do echo "-- $v-latest"; ~/venvs/bench-$v-latest/bin/python -c "import importlib.metadata as m; print({p: m.version(p) for p in ['torch','transformers','vllm','sglang','flashinfer-python','flash-linear-attention'] if p in {d.metadata['Name'] for d in m.distributions()}})"; done
    echo "-- profile"; cat $PROF; } > $O/env-$1.txt 2>&1
}
case $(nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader,nounits | tr -d ' ') in 2805,11201) ;; *) echo "clocks not at 2805/11201, stop"; exit 1;; esac
[ -f $PROF ] || { echo "P0 profile missing, stop"; exit 1; }
envs start
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu,memory.used,clocks_event_reasons.active --format=csv -l 1 > $O/clocks.csv 2>&1 &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT

# ---- A. each engine's bench -------------------------------------------------------------------------------------
herbert() { # r m name bin args... — prefill and decode at depth, 10K..60K, no thinking, 128 tokens
  local r=$1 m=$2 n=$3 bin=$4; shift 4
  for p in $LENS; do
    $bin $M/Qwen3.5-$m "$@" --ctx $CTX --no-system --no-think --max-tokens 128 --warmup 1 --ask-file $LP/pp$p.txt > $O/chat/r$r-$m-$n-pp$p.log 2>&1 || echo "FAIL r$r $m $n pp$p"
  done
}
hvf() { # r m name args... — herbert-vk-f16 has no --ask-file at cc51363: 10K and 20K only (one argument under 128 KiB)
  local r=$1 m=$2 n=$3; shift 3
  for p in 10000 20000; do
    $HVF/herbert-vk-chat $M/Qwen3.5-$m --device nvidia "$@" --ctx $CTX --no-system --no-think --max-tokens 128 --warmup 1 --ask "$(cat $LP/pp$p.txt)" > $O/chat/r$r-$m-$n-pp$p.log 2>&1 || echo "FAIL r$r $m $n pp$p"
  done
}
llama() { # r m backend dev — the two 5a prefill settings, decode default and -fa on
  local r=$1 m=$2 be=$3 dev=$4 g=$G/Qwen3.5-$2-F16.gguf pp tg
  if [ $be = cuda ]; then pp=("ub1024-b4096-faon|-ub 1024 -b 4096 -fa on" "ub1024-b2048-faon|-ub 1024 -b 2048 -fa on"); else pp=("ub1024-b4096-faon|-ub 1024 -b 4096 -fa on" "ub512-b4096-faon|-ub 512 -b 4096 -fa on"); fi
  tg=("default|" "faon|-fa on")
  for s in "${pp[@]}"; do sn=${s%%|*}; sa=${s#*|}
    # shellcheck disable=SC2086
    $L/build-$be/bin/llama-bench -m $g -dev $dev -ngl 99 -p ${LENS// /,} -n 0 -r 1 $sa -o json > $O/llama/r$r-$m-$be-pp-$sn.json 2> $O/llama/r$r-$m-$be-pp-$sn.err || echo "FAIL r$r $m llama-$be pp $sn"
  done
  for s in "${tg[@]}"; do sn=${s%%|*}; sa=${s#*|}
    # shellcheck disable=SC2086
    $L/build-$be/bin/llama-bench -m $g -dev $dev -ngl 99 -p 0 -n 128 -d ${LENS// /,} -r 1 $sa -o json > $O/llama/r$r-$m-$be-tg-$sn.json 2> $O/llama/r$r-$m-$be-tg-$sn.err || echo "FAIL r$r $m llama-$be tg $sn"
  done
}
UNITS=(hc-bf16 hc-f16 hc-f16-acc16 hvf hvf-poll llama-cuda llama-vk)
unit() { # r m unit
  case $3 in
    hc-bf16) herbert $1 $2 hc-bf16 $HB/herbert-cuda-chat ;;
    hc-f16) herbert $1 $2 hc-f16 $HF/herbert-cuda-chat ;;
    hc-f16-acc16) herbert $1 $2 hc-f16-acc16 $HF/herbert-cuda-chat --tune gemm.f16=1 ;;
    hvf) hvf $1 $2 hvf ;;
    hvf-poll) hvf $1 $2 hvf-poll --tune host.poll=1 ;;
    llama-cuda) llama $1 $2 cuda CUDA0 ;;
    llama-vk) llama $1 $2 vk Vulkan1 ;;
  esac
}
for r in 1 2 3 4 5; do
  order=("${UNITS[@]}"); [ $((r % 2)) = 0 ] && { order=(); for ((i=${#UNITS[@]}-1; i>=0; i--)); do order+=("${UNITS[$i]}"); done; }
  for m in 0.8B 4B; do
    echo "== $(date -Is) A round $r $m · $(uptime)"
    for u in "${order[@]}"; do unit $r $m $u; done
  done
done
for m in 0.8B 4B; do
  for v in "${VL[@]}"; do name=${v%%|*}; args=${v#*|}
    echo "== $(date -Is) A vllm $m $name · $(uptime)"
    for i in $LENS; do for o in 1 128; do
      # shellcheck disable=SC2086
      $VB bench latency --model $M/Qwen3.5-$m --dtype float16 --input-len $i --output-len $o --batch-size 1 --num-iters-warmup 2 --num-iters 5 \
        --max-model-len $CTX $VLLM_BASE --gpu-memory-utilization 0.85 $args \
        --output-json $O/vllm/$m-$name-in$i-out$o.json > $O/vllm/$m-$name-in$i-out$o.log 2>&1 || echo "FAIL vllm $m $name in$i out$o"
    done; done
  done
  for v in "${SG[@]}"; do name=${v%%|*}; rest=${v#*|}; mf=${rest%%|*}; args=${rest#*|}
    echo "== $(date -Is) A sglang $m $name · $(uptime)"
    for i in $LENS; do
      # shellcheck disable=SC2086
      $SPY -m sglang.benchmark.one_batch --model-path $M/Qwen3.5-$m --dtype float16 --context-length $CTX --mem-fraction-static $mf --batch-size 1 \
        --input-len $i $i $i --output-len 1 --run-name $m-$name-pp --result-filename $O/sglang/results.jsonl $args > $O/sglang/$m-$name-pp$i.log 2>&1 || echo "FAIL sglang $m $name pp$i"
      grep -q skipping $O/sglang/$m-$name-pp$i.log && echo "SKIP sglang $m $name pp$i"
      # shellcheck disable=SC2086
      $SPY -m sglang.benchmark.one_batch --model-path $M/Qwen3.5-$m --dtype float16 --context-length $CTX --mem-fraction-static $mf --batch-size 1 \
        --input-len $i $i $i --output-len 128 --run-name $m-$name-tg --result-filename $O/sglang/results.jsonl $args > $O/sglang/$m-$name-tg$i.log 2>&1 || echo "FAIL sglang $m $name tg$i"
      grep -q skipping $O/sglang/$m-$name-tg$i.log && echo "SKIP sglang $m $name tg$i"
    done
  done
  for st in "${TFS[@]}"; do
    echo "== $(date -Is) A transformers $m $st · $(uptime)"
    CASES_JSON=$T5/cases-$m.json $TF/python $S/_p5_transformers.py $O/transformers/transformers.jsonl $m $st \
      10000:1,10000:128,20000:1,20000:128,40000:1,40000:128,60000:1,60000:128 >> $O/transformers/transformers.log 2>&1 || echo "FAIL transformers $m $st"
  done
done
envs mid-a

# ---- B. the common HTTP client ----------------------------------------------------------------------------------
wait_ready() { for _ in $(seq 1 $2); do curl -sf -o /dev/null "$1" && return 0; sleep 1; done; return 1; }
stop() { kill -TERM -- -$1 2>/dev/null; for _ in $(seq 1 60); do kill -0 $1 2>/dev/null || break; sleep 1; done; kill -KILL -- -$1 2>/dev/null; sleep 3; }
SERVERS=(hc-bf16 hc-f16 hc-f16-acc16 hvf llama-cuda llama-vk)
for v in "${VL[@]}"; do SERVERS+=("vllm-${v%%|*}"); done
for v in "${SG[@]}"; do SERVERS+=("sglang-${v%%|*}"); done
SERVERS+=(transformers-sdpa transformers-sdpa-compile)
for m in 0.8B 4B; do
  for e in "${SERVERS[@]}"; do
    echo "== $(date -Is) B $m $e · $(uptime)"
    U=http://localhost:8121; H=$U/; api=messages; url=$U/v1/messages; mn=Qwen3.5-$m; lb="-fa on -ub 1024 -b 4096"
    # llama-server: the 5a prefill setting best at 60 000 tokens (CUDA both models, Vulkan 0.8B; Vulkan 4B: -ub 512)
    [ $e = llama-vk ] && [ $m = 4B ] && lb="-fa on -ub 512 -b 4096"
    case $e in
      hc-bf16) setsid $HB/herbert-cuda-serve $M/Qwen3.5-$m --ctx $CTX --port 8121 --name $mn > $O/http/server-$m-$e.log 2>&1 & ;;
      hc-f16) setsid $HF/herbert-cuda-serve $M/Qwen3.5-$m --ctx $CTX --port 8121 --name $mn > $O/http/server-$m-$e.log 2>&1 & ;;
      hc-f16-acc16) setsid $HF/herbert-cuda-serve $M/Qwen3.5-$m --ctx $CTX --port 8121 --name $mn --tune gemm.f16=1 > $O/http/server-$m-$e.log 2>&1 & ;;
      hvf) setsid $HVF/herbert-vk-serve $M/Qwen3.5-$m --device nvidia --ctx $CTX --port 8121 --name $mn > $O/http/server-$m-$e.log 2>&1 & ;;
      # shellcheck disable=SC2086
      llama-cuda|llama-vk) be=${e#llama-}; dev=CUDA0; [ $be = vk ] && dev=Vulkan1
        setsid $L/build-$be/bin/llama-server -m $G/Qwen3.5-$m-F16.gguf -dev $dev -ngl 99 -c $CTX -np 1 --no-cache-prompt -a $mn --port 8121 $lb > $O/http/server-$m-$e.log 2>&1 &
        H=$U/health ;;
      vllm-*) va=""; for v in "${VL[@]}"; do [ $e = "vllm-${v%%|*}" ] && va=${v#*|}; done
        # shellcheck disable=SC2086
        setsid $VB serve $M/Qwen3.5-$m --dtype float16 --max-model-len $CTX $VLLM_BASE --gpu-memory-utilization 0.85 \
          --no-enable-prefix-caching --served-model-name $mn --port 8121 $va > $O/http/server-$m-$e.log 2>&1 &
        H=$U/health ;;
      sglang-*) sa=""; smf=0.8; for v in "${SG[@]}"; do [ $e = "sglang-${v%%|*}" ] && { rest=${v#*|}; smf=${rest%%|*}; sa=${rest#*|}; }; done
        # shellcheck disable=SC2086
        setsid $SPY -m sglang.launch_server --model-path $M/Qwen3.5-$m --dtype float16 --context-length $CTX --mem-fraction-static $smf --disable-radix-cache \
          --served-model-name $mn --host localhost --port 8121 $sa > $O/http/server-$m-$e.log 2>&1 &
        H=$U/health; api=openai; url=$U/v1/chat/completions ;;
      transformers-*) ta=""; [ $e = transformers-sdpa-compile ] && ta="--compile"
        # shellcheck disable=SC2086
        setsid $TF/transformers serve --host localhost --port 8121 --dtype float16 --device cuda:0 --attn-implementation sdpa $ta > $O/http/server-$m-$e.log 2>&1 &
        H=$U/v1/models; api=openai; url=$U/v1/chat/completions; mn=$M/Qwen3.5-$m ;;
    esac
    P=$!
    if wait_ready $H 900; then
      nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate --format=csv,noheader > $O/http/pstate-$m-$e.txt
      python3 $S2/_http_client.py --skip-blank --prompts $LP/prompts-long.json --url $url --api $api --engine $e --model $mn --out $O/http/http-$m-$e.json > $O/http/client-$m-$e.log 2>&1 || echo "FAIL http $m $e"
      nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate --format=csv,noheader >> $O/http/pstate-$m-$e.txt
    else
      echo "FAIL http $m $e: server not ready"
    fi
    stop $P
  done
done
envs end
date -Is > $O/end.txt
echo ALLDONE-phase5b
