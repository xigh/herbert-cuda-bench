#!/usr/bin/env bash
# _after-p1d-tokens-003.sh — queued behind the vLLM graph runs (waits for ALLDONE-after-p1c): phase 1's
# correctness check (protocol.md). Each comparator, at the settings that change its kernels (attention backend,
# flash attention, micro-batch, compile), generates greedy from the oracle's prompt ids on p1-p4; the ids are then
# compared with the CPU oracle's (herbert-jit-bf16 v0.1.0 sources, ~/verif-003/oracle-v010-*).
set -u
until grep -q ALLDONE-after-p1c ~/c003/after-p1c.out; do sleep 60; done
O=${O:-$HOME/c003/after-p1d}; mkdir -p $O
S=~/c003/phase1-scripts; B=~/herbert-cuda-mma; M=${MODELS_DIR:-$HOME/models}; G=${BENCH_DIR:-$HOME/bench-qwen35}; L=$HOME/llama.cpp-b11718
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu --format=csv -l 1 > $O/clocks.csv 2>&1 &
smi=$!
for m in 0.8B 4B; do
  $B/target/release/herbert-cuda-bench $M/Qwen3.5-$m --prepare --cases $O/cases-$m.json --prompts $B/bench/prompts --labels p1-short,p2-explain,p3-long,p4-xlong > $O/cases-$m.log 2>&1 || echo "FAIL cases $m"
  mkdir -p $O/oracle-$m; cp ~/verif-003/oracle-v010-$m/p*-cpu.log $O/oracle-$m/
done
VL=("default|{\"max_model_len\":4608,\"max_num_seqs\":8,\"gpu_memory_utilization\":0.85}"
    "triton|{\"max_model_len\":4608,\"max_num_seqs\":8,\"gpu_memory_utilization\":0.85,\"attention_backend\":\"TRITON_ATTN\"}"
    "flashinfer|{\"max_model_len\":4608,\"max_num_seqs\":8,\"gpu_memory_utilization\":0.85,\"attention_backend\":\"FLASHINFER\"}"
    "eager|{\"max_model_len\":4608,\"max_num_seqs\":8,\"gpu_memory_utilization\":0.85,\"enforce_eager\":true}")
SG=("default|{\"context_length\":4608,\"mem_fraction_static\":0.8}"
    "triton|{\"context_length\":4608,\"mem_fraction_static\":0.8,\"attention_backend\":\"triton\"}"
    "flashinfer|{\"context_length\":4608,\"mem_fraction_static\":0.8,\"attention_backend\":\"flashinfer\"}"
    "chunk2048|{\"context_length\":4608,\"mem_fraction_static\":0.8,\"chunked_prefill_size\":2048}")
TF=("sdpa-dynamic|{\"attn\":\"sdpa\"}" "eager-dynamic|{\"attn\":\"eager\"}" "sdpa-static|{\"attn\":\"sdpa\",\"cache\":\"static\"}")
LC=("cuda-default|CUDA0|" "cuda-faon-ub1024|CUDA0|-fa on -ub 1024 -b 4096" "cuda-faoff|CUDA0|-fa off"
    "vk-default|Vulkan1|" "vk-faon-ub2048|Vulkan1|-fa on -ub 2048 -b 4096" "vk-faoff|Vulkan1|-fa off")
for m in 0.8B 4B; do
  for v in "${VL[@]}"; do n=${v%%|*}; k=${v#*|}; echo "== $(date -Is) vllm $m $n · $(uptime)"
    ~/venvs/bench-vllm-latest/bin/python $S/_p1_tokens.py vllm $m $n "$k" $O/cases-$m.json $O/tokens.jsonl > $O/vllm-$m-$n.log 2>&1 || echo "FAIL vllm $m $n"; done
  for v in "${SG[@]}"; do n=${v%%|*}; k=${v#*|}; echo "== $(date -Is) sglang $m $n · $(uptime)"
    ~/venvs/bench-sglang-latest/bin/python $S/_p1_tokens.py sglang $m $n "$k" $O/cases-$m.json $O/tokens.jsonl > $O/sglang-$m-$n.log 2>&1 || echo "FAIL sglang $m $n"; done
  for v in "${TF[@]}"; do n=${v%%|*}; k=${v#*|}; echo "== $(date -Is) transformers $m $n · $(uptime)"
    ~/venvs/bench-transformers-latest/bin/python $S/_p1_tokens.py transformers $m $n "$k" $O/cases-$m.json $O/tokens.jsonl > $O/transformers-$m-$n.log 2>&1 || echo "FAIL transformers $m $n"; done
  for v in "${LC[@]}"; do n=${v%%|*}; rest=${v#*|}; dev=${rest%%|*}; args=${rest#*|}; echo "== $(date -Is) llama $m $n · $(uptime)"
    # shellcheck disable=SC2086
    $L/build-${n%%-*}/bin/llama-server -m $G/Qwen3.5-$m-BF16.gguf -dev $dev -ngl 99 -c 4608 --parallel 1 --host localhost --port 18731 $args > $O/llama-server-$m-$n.log 2>&1 &
    srv=$!
    for _ in $(seq 120); do curl -sf http://localhost:18731/health > /dev/null && break; sleep 1; done
    ~/venvs/bench-vllm-latest/bin/python $S/_p1_tokens.py llama $m $n http://localhost:18731 $O/cases-$m.json $O/tokens.jsonl > $O/llama-$m-$n.log 2>&1 || echo "FAIL llama $m $n"
    kill $srv; wait $srv 2>/dev/null
  done
done
kill $smi
echo ALLDONE-after-p1d
