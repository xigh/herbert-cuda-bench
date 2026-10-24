#!/usr/bin/env bash
# _p5a-003.sh — phase 5, step 5a (protocol.md): the long prefill on the RTX 4080, before the bench.
#   1) Tokens: every engine generates 64 greedy tokens after each long prompt (10 000-60 000 tokens), from the same
#      prompt ids (herbert-cuda-bench --prepare, no thinking); herbert-cuda in its three forms by herbert-cuda-bench,
#      herbert-vk-f16 by its chat at 10K and 20K (the prompt fits in one argument there), the comparators in f16 at
#      their 4a token settings. Compared engine against engine afterwards (no CPU reference past 8 192 positions).
#   2) The comparators' settings sweep at the long lengths, prefill only, phase 1's grid: llama.cpp b11718 F16 CUDA
#      and Vulkan (-ub x -b x -fa, 3 rounds alternated), vLLM 0.31.0 (every setting at --max-num-seqs 8
#      -cc.max_cudagraph_capture_size=512), SGLang 0.5.22 (one process per length, 3 samples, plus
#      --mem-fraction-static 0.7 and 0.6), transformers 5.20.1 (one process per setting). float16 throughout.
# A setting that runs out of memory at a length is a FAIL line for that length. ALLDONE-phase5a at the end.
set -u
O=${O:-$HOME/c003/phase5a}; mkdir -p $O
S=~/c003/phase5-scripts; S4=~/c003/phase4-scripts; LP=~/c003/long-prompts; M=${MODELS_DIR:-$HOME/models}; G=${BENCH_DIR:-$HOME/bench-qwen35}; L=$HOME/llama.cpp-b11718
HB=~/herbert-cuda-bf16/target/release; HF=~/herbert-cuda-f16/target/release; HVF=~/herbert-vk-f16-cc51363/target/release
VB=~/venvs/bench-vllm-latest/bin/vllm; SPY=~/venvs/bench-sglang-latest/bin/python; TF=~/venvs/bench-transformers-latest/bin
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 DTYPE=float16 SGLANG_MAMBA_CONV_DTYPE=float16
LENS="10000 20000 40000 60000"; CTX=61440
{ date -Is; uptime; nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate --format=csv; cat ~/.nv/nvidia-application-profiles-rc
  echo "-- herbert-cuda-bf16 $(cat ~/herbert-cuda-bf16/COMMIT)"; sha256sum $HB/herbert-cuda-chat $HB/herbert-cuda-bench $HB/herbert-cuda-serve
  echo "-- herbert-cuda-f16 $(cat ~/herbert-cuda-f16/COMMIT)"; sha256sum $HF/herbert-cuda-chat $HF/herbert-cuda-bench $HF/herbert-cuda-serve
  echo "-- herbert-vk-f16 $(cat ~/herbert-vk-f16-cc51363/COMMIT)"; sha256sum $HVF/herbert-vk-chat $HVF/herbert-vk-serve
  $L/build-cuda/bin/llama-server --version
  for v in vllm sglang transformers; do echo "-- $v-latest"; ~/venvs/bench-$v-latest/bin/python -c "import importlib.metadata as m; print({p: m.version(p) for p in ['torch','transformers','vllm','sglang','flashinfer-python','flash-linear-attention'] if any(d.metadata['Name']==p for d in m.distributions())})"; done
  sha256sum $G/Qwen3.5-*-F16.gguf $LP/*.txt; } > $O/env.txt 2>&1
case $(nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader,nounits | tr -d ' ') in 2805,11201) ;; *) echo "clocks not at 2805/11201, stop"; exit 1;; esac
[ -f ~/.nv/nvidia-application-profiles-rc ] || { echo "P0 profile missing, stop"; exit 1; }
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu,memory.used,clocks_event_reasons.active --format=csv -l 1 > $O/clocks.csv 2>&1 &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT

# 1) tokens
T=$O/tokens; mkdir -p $T
for m in 0.8B 4B; do
  $HB/herbert-cuda-bench $M/Qwen3.5-$m --prepare --cases $T/cases-$m.json --prompts $LP --labels 10000,20000,40000,60000 > $T/cases-$m.log 2>&1 || echo "FAIL cases $m"
  for f in "hc-bf16|$HB|" "hc-f16|$HF|" "hc-f16-acc16|$HF|gemm.f16=1"; do n=${f%%|*}; rest=${f#*|}; dir=${rest%%|*}; tune=${rest#*|}
    echo "== $(date -Is) tokens $n $m · $(uptime)"
    # shellcheck disable=SC2046
    $dir/herbert-cuda-bench $M/Qwen3.5-$m --cases $T/cases-$m.json --outputs 64 --rounds 1 --warmup 0 --ctx $CTX $( [ -n "$tune" ] && echo --tune $tune ) --out $T/$n-$m.json > $T/$n-$m.log 2>&1 || echo "FAIL tokens $n $m"
  done
  for n in 10000 20000; do echo "== $(date -Is) tokens hvf $m pp$n · $(uptime)"
    $HVF/herbert-vk-chat $M/Qwen3.5-$m --device nvidia --ctx $CTX --no-system --no-think --max-tokens 64 --ask "$(cat $LP/pp$n.txt)" > $T/hvf-$m-pp$n.log 2>&1 || echo "FAIL tokens hvf $m pp$n"
  done
done
VL=("default|{\"max_model_len\":$CTX,\"max_num_seqs\":8,\"gpu_memory_utilization\":0.85}"
    "triton|{\"max_model_len\":$CTX,\"max_num_seqs\":8,\"gpu_memory_utilization\":0.85,\"attention_backend\":\"TRITON_ATTN\"}"
    "flashinfer|{\"max_model_len\":$CTX,\"max_num_seqs\":8,\"gpu_memory_utilization\":0.85,\"attention_backend\":\"FLASHINFER\"}"
    "eager|{\"max_model_len\":$CTX,\"max_num_seqs\":8,\"gpu_memory_utilization\":0.85,\"enforce_eager\":true}")
# SGLang at 0.8 ran out of memory at 60K in the start check: 0.6 for the tokens.
SG=("default|{\"context_length\":$CTX,\"mem_fraction_static\":0.6}"
    "triton|{\"context_length\":$CTX,\"mem_fraction_static\":0.6,\"attention_backend\":\"triton\"}"
    "flashinfer|{\"context_length\":$CTX,\"mem_fraction_static\":0.6,\"attention_backend\":\"flashinfer\"}"
    "chunk2048|{\"context_length\":$CTX,\"mem_fraction_static\":0.6,\"chunked_prefill_size\":2048}")
TFS=("sdpa-dynamic|{\"attn\":\"sdpa\"}" "eager-dynamic|{\"attn\":\"eager\"}" "sdpa-static|{\"attn\":\"sdpa\",\"cache\":\"static\"}")
LC=("cuda-default|CUDA0|" "cuda-faon-ub1024|CUDA0|-fa on -ub 1024 -b 4096" "cuda-faoff|CUDA0|-fa off"
    "vk-default|Vulkan1|" "vk-faon-ub2048|Vulkan1|-fa on -ub 2048 -b 4096" "vk-faoff|Vulkan1|-fa off")
export MAXNEW=64
for m in 0.8B 4B; do
  for v in "${VL[@]}"; do n=${v%%|*}; k=${v#*|}; echo "== $(date -Is) tokens vllm $m $n · $(uptime)"
    ~/venvs/bench-vllm-latest/bin/python $S4/_p1_tokens.py vllm $m $n "$k" $T/cases-$m.json $T/tokens.jsonl > $T/vllm-$m-$n.log 2>&1 || echo "FAIL tokens vllm $m $n"; done
  for v in "${SG[@]}"; do n=${v%%|*}; k=${v#*|}; echo "== $(date -Is) tokens sglang $m $n · $(uptime)"
    ~/venvs/bench-sglang-latest/bin/python $S4/_p1_tokens.py sglang $m $n "$k" $T/cases-$m.json $T/tokens.jsonl > $T/sglang-$m-$n.log 2>&1 || echo "FAIL tokens sglang $m $n"; done
  for v in "${TFS[@]}"; do n=${v%%|*}; k=${v#*|}; echo "== $(date -Is) tokens transformers $m $n · $(uptime)"
    ~/venvs/bench-transformers-latest/bin/python $S4/_p1_tokens.py transformers $m $n "$k" $T/cases-$m.json $T/tokens.jsonl > $T/transformers-$m-$n.log 2>&1 || echo "FAIL tokens transformers $m $n"; done
  for v in "${LC[@]}"; do n=${v%%|*}; rest=${v#*|}; dev=${rest%%|*}; args=${rest#*|}; echo "== $(date -Is) tokens llama $m $n · $(uptime)"
    # shellcheck disable=SC2086
    $L/build-${n%%-*}/bin/llama-server -m $G/Qwen3.5-$m-F16.gguf -dev $dev -ngl 99 -c $CTX --parallel 1 --host localhost --port 18731 $args > $T/llama-server-$m-$n.log 2>&1 &
    srv=$!
    for _ in $(seq 120); do curl -sf http://localhost:18731/health > /dev/null && break; sleep 1; done
    ~/venvs/bench-vllm-latest/bin/python $S4/_p1_tokens.py llama $m $n http://localhost:18731 $T/cases-$m.json $T/tokens.jsonl > $T/llama-$m-$n.log 2>&1 || echo "FAIL tokens llama $m $n"
    kill $srv; wait $srv 2>/dev/null
  done
done
unset MAXNEW
echo "== $(date -Is) tokens done"

# 2) settings sweep, prefill only
LL=$O/llama; mkdir -p $LL
SET=("default|"); for ub in 256 512 1024 2048; do for b in 2048 4096; do for fa in on off; do SET+=("ub$ub-b$b-fa$fa|-ub $ub -b $b -fa $fa"); done; done; done
for r in 1 2 3; do
  echo "== $(date -Is) llama round $r · $(uptime)"
  order=("${SET[@]}"); [ $((r % 2)) = 0 ] && { order=(); for ((i=${#SET[@]}-1; i>=0; i--)); do order+=("${SET[$i]}"); done; }
  for m in 0.8B 4B; do for be in cuda:CUDA0 vk:Vulkan1; do name=${be%%:*}; dev=${be#*:}
    for s in "${order[@]}"; do sn=${s%%|*}; sa=${s#*|}
      # shellcheck disable=SC2086
      $L/build-$name/bin/llama-bench -m $G/Qwen3.5-$m-F16.gguf -dev $dev -ngl 99 -p ${LENS// /,} -n 0 -r 1 $sa -o json > $LL/r$r-$m-$name-pp-$sn.json 2> $LL/r$r-$m-$name-pp-$sn.err || echo "FAIL r$r $m llama-$name pp $sn"
    done
  done; done
done
VV=$O/vllm; mkdir -p $VV
V=("default|" "eager|--enforce-eager" "mnbt2048|--max-num-batched-tokens 2048" "mnbt16384|--max-num-batched-tokens 16384"
   "flashattn|--attention-backend FLASH_ATTN" "flashinfer|--attention-backend FLASHINFER" "triton|--attention-backend TRITON_ATTN")
for m in 0.8B 4B; do for v in "${V[@]}"; do name=${v%%|*}; args=${v#*|}
  echo "== $(date -Is) vllm $m $name · $(uptime)"
  for i in $LENS; do
    # shellcheck disable=SC2086
    $VB bench latency --model $M/Qwen3.5-$m --dtype float16 --input-len $i --output-len 1 --batch-size 1 --num-iters-warmup 2 --num-iters 5 \
      --max-model-len $CTX --max-num-seqs 8 -cc.max_cudagraph_capture_size=512 --gpu-memory-utilization 0.85 $args \
      --output-json $VV/$m-$name-in$i-out1.json > $VV/$m-$name-in$i-out1.log 2>&1 || echo "FAIL vllm $m $name in$i"
  done
done; done
SS=$O/sglang-rep; mkdir -p $SS
SV=("default|0.8|" "nograph|0.8|--disable-cuda-graph" "flashinfer|0.8|--attention-backend flashinfer" "triton|0.8|--attention-backend triton"
    "chunk2048|0.8|--chunked-prefill-size 2048" "nochunk|0.8|--chunked-prefill-size -1" "mf07|0.7|" "mf06|0.6|")
for m in 0.8B 4B; do for v in "${SV[@]}"; do name=${v%%|*}; rest=${v#*|}; mf=${rest%%|*}; args=${rest#*|}
  echo "== $(date -Is) sglang $m $name · $(uptime)"
  for i in $LENS; do
    # shellcheck disable=SC2086
    $SPY -m sglang.benchmark.one_batch --model-path $M/Qwen3.5-$m --dtype float16 --context-length $CTX --mem-fraction-static $mf --batch-size 1 \
      --input-len $i $i $i --output-len 1 --run-name $m-$name-pp --result-filename $SS/results.jsonl $args > $SS/$m-$name-pp$i.log 2>&1 || echo "FAIL sglang $m $name pp$i"
  done
done; done
for m in 0.8B 4B; do for st in sdpa-dynamic sdpa-static sdpa-static-compile eager-dynamic eager-static eager-static-compile; do
  echo "== $(date -Is) transformers $m $st · $(uptime)"
  $TF/python $S/_p5_transformers.py $O/transformers.jsonl $m $st 10000:1,20000:1,40000:1,60000:1 >> $O/transformers.log 2>&1 || echo "FAIL transformers $m $st"
done; done
date -Is > $O/end.txt
echo ALLDONE-phase5a
