#!/usr/bin/env bash
# _p2-003.sh — phase 2 of the #003 campaign (protocol.md, "Phase 2"): herbert-cuda (bare and the candidate)
# and herbert-vk-bf16 against the comparators at their two best phase 1 settings, on the RTX 4080, every CUDA engine in P0.
#   A. each engine's bench: herbert-cuda, herbert-vk and llama.cpp alternated round by round (N = 5, order reversed
#      every other round); vLLM, SGLang and transformers one pass per setting and model (internal repetitions);
#   B. the common HTTP client (_http_client.py, V2 of #002), one server after the other;
#   C. a control without the P0 profile (the file renamed, put back by a trap).
# Every output kept in full under $O; ALLDONE-phase2 at the end.
set -u
O=${O:-$HOME/c003/phase2}; mkdir -p $O/chat $O/llama $O/vllm $O/sglang $O/transformers $O/http $O/control
S=~/c003/phase2-scripts; B=~/herbert-cuda-mma; HC=$B/target/release/herbert-cuda-chat; HCS=$B/target/release/herbert-cuda-serve
HV=~/herbert-vk-bf16-v1/target/release/herbert-vk-chat; HVS=~/herbert-vk-bf16-serve/target/release/herbert-vk-serve
PR=$B/bench/prompts; M=${MODELS_DIR:-$HOME/models}; G=${BENCH_DIR:-$HOME/bench-qwen35}; L=$HOME/llama.cpp-b11718
VB=~/venvs/bench-vllm-latest/bin/vllm; SPY=~/venvs/bench-sglang-latest/bin/python; TF=~/venvs/bench-transformers-latest/bin
CL="python3 $S/_http_client.py"
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
CAND="--tune gemm.auto=3 --tune gemm.layout=xor --tune prefill.block=auto"
PROF=~/.nv/nvidia-application-profiles-rc
envs() {
  { date -Is; uptime; cat /proc/driver/nvidia/version
    nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate,persistence_mode --format=csv
    echo "-- herbert-cuda $(cat $B/COMMIT)"; sha256sum $HC $HCS
    echo "-- herbert-vk $(cat ~/herbert-vk-bf16-v1/COMMIT) / serve $(cat ~/herbert-vk-bf16-serve/COMMIT)"; sha256sum $HV $HVS
    $L/build-cuda/bin/llama-bench --version 2>&1 | head -2
    for v in vllm sglang transformers; do echo "-- $v-latest"; ~/venvs/bench-$v-latest/bin/python -c "import importlib.metadata as m; print({p: m.version(p) for p in ['torch','transformers','vllm','sglang','flashinfer-python','torchvision'] if p in {d.metadata['Name'] for d in m.distributions()}})"; done
    echo "-- profile"; cat $PROF; } > $O/env-$1.txt 2>&1
}
envs start
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu,memory.used --format=csv -l 1 > $O/clocks.csv 2>&1 &
smi=$!

# ---- A. each engine's bench -------------------------------------------------------------------------------------
herbert() { # r m name bin args... — prefill on pp128..pp4000 (no thinking, 1 token), decode on p3 (thinking, 256)
  local r=$1 m=$2 n=$3 bin=$4; shift 4
  if [ "$n" != hv-poll ]; then for p in 128 512 2048 4000; do
    $bin $M/Qwen3.5-$m "$@" --no-system --no-think --max-tokens 1 --warmup 1 --ask "$(cat $PR/pp$p.txt)" > $O/chat/r$r-$m-$n-pp$p.log 2>&1 || echo "FAIL r$r $m $n pp$p"
  done; fi
  $bin $M/Qwen3.5-$m "$@" --no-system --max-tokens 256 --warmup 1 --ask "$(cat $PR/p3-long.txt)" > $O/chat/r$r-$m-$n-p3.log 2>&1 || echo "FAIL r$r $m $n p3"
}
llama() { # r m backend dev — the two best prefill settings and the two best decode settings of phase 1
  local r=$1 m=$2 be=$3 dev=$4 g=$G/Qwen3.5-$2-BF16.gguf pp tg
  if [ $be = cuda ]; then pp=("default|" "ub1024-b4096-faon|-ub 1024 -b 4096 -fa on"); else pp=("ub2048-b4096-faon|-ub 2048 -b 4096 -fa on" "ub256-b4096-faon|-ub 256 -b 4096 -fa on"); fi
  tg=("default|" "faon|-fa on")
  for s in "${pp[@]}"; do sn=${s%%|*}; sa=${s#*|}
    # shellcheck disable=SC2086
    $L/build-$be/bin/llama-bench -m $g -dev $dev -ngl 99 -p 128,512,2048,4000 -n 0 -r 1 $sa -o json > $O/llama/r$r-$m-$be-pp-$sn.json 2> $O/llama/r$r-$m-$be-pp-$sn.err || echo "FAIL r$r $m llama-$be pp $sn"
  done
  for s in "${tg[@]}"; do sn=${s%%|*}; sa=${s#*|}
    # shellcheck disable=SC2086
    $L/build-$be/bin/llama-bench -m $g -dev $dev -ngl 99 -p 0 -n 255 -d 759 -r 1 $sa -o json > $O/llama/r$r-$m-$be-tg-$sn.json 2> $O/llama/r$r-$m-$be-tg-$sn.err || echo "FAIL r$r $m llama-$be tg $sn"
  done
}
UNITS=(hc-nu hc-cand hv hv-poll llama-cuda llama-vk)
unit() { # r m unit
  case $3 in
    hc-nu) herbert $1 $2 hc-nu $HC ;;
    hc-cand) herbert $1 $2 hc-cand $HC $CAND ;;
    hv) herbert $1 $2 hv $HV --device nvidia ;;
    hv-poll) herbert $1 $2 hv-poll $HV --device nvidia --tune host.poll=1 ;;
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
  for v in "cg512|" "triton-cg512|--attention-backend TRITON_ATTN"; do name=${v%%|*}; args=${v#*|}
    echo "== $(date -Is) A vllm $m $name · $(uptime)"
    for io in 128:1 512:1 2048:1 4000:1 759:1 759:255; do i=${io%%:*}; o=${io#*:}
      # shellcheck disable=SC2086
      $VB bench latency --model $M/Qwen3.5-$m --dtype bfloat16 --input-len $i --output-len $o --batch-size 1 --num-iters-warmup 2 --num-iters 5 \
        --max-model-len 4608 --max-num-seqs 8 -cc.max_cudagraph_capture_size=512 --gpu-memory-utilization 0.85 $args \
        --output-json $O/vllm/$m-$name-in$i-out$o.json > $O/vllm/$m-$name-in$i-out$o.log 2>&1 || echo "FAIL vllm $m $name in$i out$o"
    done
  done
  for v in "default|" "triton|--attention-backend triton"; do name=${v%%|*}; args=${v#*|}
    echo "== $(date -Is) A sglang $m $name · $(uptime)"
    # shellcheck disable=SC2086
    $SPY -m sglang.benchmark.one_batch --model-path $M/Qwen3.5-$m --dtype bfloat16 --context-length 4608 --mem-fraction-static 0.8 --batch-size 1 \
      --input-len 128 512 2048 4000 128 512 2048 4000 128 512 2048 4000 --output-len 1 --run-name $m-$name-pp --result-filename $O/sglang/results.jsonl $args > $O/sglang/$m-$name-pp.log 2>&1 || echo "FAIL sglang $m $name pp"
    # shellcheck disable=SC2086
    $SPY -m sglang.benchmark.one_batch --model-path $M/Qwen3.5-$m --dtype bfloat16 --context-length 4608 --mem-fraction-static 0.8 --batch-size 1 \
      --input-len 759 759 759 --output-len 255 --run-name $m-$name-tg --result-filename $O/sglang/results.jsonl $args > $O/sglang/$m-$name-tg.log 2>&1 || echo "FAIL sglang $m $name tg"
  done
  for st in sdpa-static-compile eager-static-compile; do
    echo "== $(date -Is) A transformers $m $st · $(uptime)"
    $TF/python $S/_p1_transformers.py $O/transformers/transformers.jsonl $m $st >> $O/transformers/transformers.log 2>&1 || echo "FAIL transformers $m $st"
  done
done
envs mid-a

# ---- B. the common HTTP client ----------------------------------------------------------------------------------
wait_ready() { for _ in $(seq 1 $2); do curl -sf -o /dev/null "$1" && return 0; sleep 1; done; return 1; }
stop() { kill -TERM -- -$1 2>/dev/null; for _ in $(seq 1 60); do kill -0 $1 2>/dev/null || break; sleep 1; done; kill -KILL -- -$1 2>/dev/null; sleep 3; }
SERVERS=(hc-nu hc-cand hv llama-cuda llama-vk vllm-cg512 vllm-triton-cg512 sglang-default sglang-triton transformers-sdpa transformers-sdpa-compile)
for m in 0.8B 4B; do
  for e in "${SERVERS[@]}"; do
    echo "== $(date -Is) B $m $e · $(uptime)"
    U=http://localhost:8121; H=$U/; api=messages; url=$U/v1/messages; mn=Qwen3.5-$m; lb=""
    [ $e = llama-cuda ] && { [ $m = 0.8B ] && lb="-ub 1024 -b 4096 -fa on"; }
    [ $e = llama-vk ] && lb="-ub 2048 -b 4096 -fa on"
    case $e in
      hc-nu) setsid $HCS $M/Qwen3.5-$m --ctx 4608 --port 8121 --name $mn > $O/http/server-$m-$e.log 2>&1 & ;;
      # shellcheck disable=SC2086
      hc-cand) setsid $HCS $M/Qwen3.5-$m --ctx 4608 --port 8121 --name $mn $CAND > $O/http/server-$m-$e.log 2>&1 & ;;
      hv) setsid $HVS $M/Qwen3.5-$m --device nvidia --ctx 4608 --port 8121 --name $mn > $O/http/server-$m-$e.log 2>&1 & ;;
      # shellcheck disable=SC2086
      llama-cuda|llama-vk) be=${e#llama-}; dev=CUDA0; [ $be = vk ] && dev=Vulkan1
        setsid $L/build-$be/bin/llama-server -m $G/Qwen3.5-$m-BF16.gguf -dev $dev -ngl 99 -c 4608 -np 1 --no-cache-prompt -a $mn --port 8121 $lb > $O/http/server-$m-$e.log 2>&1 &
        H=$U/health ;;
      vllm-*) va=""; [ $e = vllm-triton-cg512 ] && va="--attention-backend TRITON_ATTN"
        # shellcheck disable=SC2086
        setsid $VB serve $M/Qwen3.5-$m --dtype bfloat16 --max-model-len 4608 --max-num-seqs 8 -cc.max_cudagraph_capture_size=512 --gpu-memory-utilization 0.85 \
          --no-enable-prefix-caching --served-model-name $mn --port 8121 $va > $O/http/server-$m-$e.log 2>&1 &
        H=$U/health ;;
      sglang-*) sa=""; [ $e = sglang-triton ] && sa="--attention-backend triton"
        # shellcheck disable=SC2086
        setsid $SPY -m sglang.launch_server --model-path $M/Qwen3.5-$m --dtype bfloat16 --context-length 4608 --mem-fraction-static 0.8 --disable-radix-cache \
          --served-model-name $mn --host localhost --port 8121 $sa > $O/http/server-$m-$e.log 2>&1 &
        H=$U/health; api=openai; url=$U/v1/chat/completions ;;
      transformers-*) ta=""; [ $e = transformers-sdpa-compile ] && ta="--compile"
        # shellcheck disable=SC2086
        setsid $TF/transformers serve --host localhost --port 8121 --dtype bfloat16 --device cuda:0 --attn-implementation sdpa $ta > $O/http/server-$m-$e.log 2>&1 &
        H=$U/v1/models; api=openai; url=$U/v1/chat/completions; mn=$M/Qwen3.5-$m ;;
    esac
    P=$!
    if wait_ready $H 900; then
      nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate --format=csv,noheader > $O/http/pstate-$m-$e.txt
      $CL --url $url --api $api --engine $e --model $mn --out $O/http/http-$m-$e.json > $O/http/client-$m-$e.log 2>&1 || echo "FAIL http $m $e"
      nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate --format=csv,noheader >> $O/http/pstate-$m-$e.txt
    else
      echo "FAIL http $m $e: server not ready"
    fi
    stop $P
  done
done
envs mid-b

# ---- C. control without the P0 profile ---------------------------------------------------------------------------
# herbert() writes under $O/chat: O=$O/control for its calls puts those logs in $O/control/chat.
mkdir -p $O/control/chat
mv $PROF $PROF.off-phase2
trap 'mv -f $PROF.off-phase2 $PROF 2>/dev/null' EXIT
echo "== $(date -Is) C profile removed · $(ls ~/.nv)"
for r in 1 2 3; do
  for m in 0.8B 4B; do
    echo "== $(date -Is) C round $r $m · $(uptime)"
    for u in hc-nu hc-cand llama-cuda; do
      case $u in
        hc-nu) O=$O/control herbert $r $m hc-nu $HC ;;
        hc-cand) O=$O/control herbert $r $m hc-cand $HC $CAND ;;
        llama-cuda) g=$G/Qwen3.5-$m-BF16.gguf
          $L/build-cuda/bin/llama-bench -m $g -dev CUDA0 -ngl 99 -p 128,512,2048,4000 -n 0 -r 1 -o json > $O/control/r$r-$m-cuda-pp-default.json 2> $O/control/r$r-$m-cuda-pp-default.err || echo "FAIL C r$r $m llama pp"
          $L/build-cuda/bin/llama-bench -m $g -dev CUDA0 -ngl 99 -p 0 -n 255 -d 759 -r 1 -o json > $O/control/r$r-$m-cuda-tg-default.json 2> $O/control/r$r-$m-cuda-tg-default.err || echo "FAIL C r$r $m llama tg" ;;
      esac
    done
  done
done
for m in 0.8B 4B; do
  echo "== $(date -Is) C vllm $m cg512 · $(uptime)"
  for io in 128:1 512:1 2048:1 4000:1 759:1 759:255; do i=${io%%:*}; o=${io#*:}
    $VB bench latency --model $M/Qwen3.5-$m --dtype bfloat16 --input-len $i --output-len $o --batch-size 1 --num-iters-warmup 2 --num-iters 5 \
      --max-model-len 4608 --max-num-seqs 8 -cc.max_cudagraph_capture_size=512 --gpu-memory-utilization 0.85 \
      --output-json $O/control/vllm-$m-cg512-in$i-out$o.json > $O/control/vllm-$m-cg512-in$i-out$o.log 2>&1 || echo "FAIL C vllm $m in$i out$o"
  done
  echo "== $(date -Is) C sglang $m default · $(uptime)"
  $SPY -m sglang.benchmark.one_batch --model-path $M/Qwen3.5-$m --dtype bfloat16 --context-length 4608 --mem-fraction-static 0.8 --batch-size 1 \
    --input-len 128 512 2048 4000 128 512 2048 4000 128 512 2048 4000 --output-len 1 --run-name $m-default-pp --result-filename $O/control/sglang-results.jsonl > $O/control/sglang-$m-default-pp.log 2>&1 || echo "FAIL C sglang $m pp"
  $SPY -m sglang.benchmark.one_batch --model-path $M/Qwen3.5-$m --dtype bfloat16 --context-length 4608 --mem-fraction-static 0.8 --batch-size 1 \
    --input-len 759 759 759 --output-len 255 --run-name $m-default-tg --result-filename $O/control/sglang-results.jsonl > $O/control/sglang-$m-default-tg.log 2>&1 || echo "FAIL C sglang $m tg"
done
mv -f $PROF.off-phase2 $PROF; trap - EXIT
echo "== $(date -Is) C profile restored · $(ls ~/.nv)"
envs end
kill $smi
echo ALLDONE-phase2
