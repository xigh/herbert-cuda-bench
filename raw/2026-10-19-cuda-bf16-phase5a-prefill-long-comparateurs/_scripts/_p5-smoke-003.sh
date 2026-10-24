#!/usr/bin/env bash
# _p5-smoke-003.sh — before the long prefill protocol: does each engine run at 10 000-60 000 prompt tokens on the
# 4080 (start, no out-of-memory, sane output)? One pass each, 4B (the larger), f16 for the comparators, then the 0.8B
# for whatever failed. Not a measurement: the timings these runs print are not compared or reported before
# the predictions on the long prefill are committed.
set -u
O=${O:-$HOME/c003/phase5-smoke}; mkdir -p $O
M=${MODELS_DIR:-$HOME/models}; G=${BENCH_DIR:-$HOME/bench-qwen35}; L=$HOME/llama.cpp-b11718; LP=~/c003/long-prompts
VB=~/venvs/bench-vllm-latest/bin/vllm; SPY=~/venvs/bench-sglang-latest/bin/python; TF=~/venvs/bench-transformers-latest/bin
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 DTYPE=float16 SGLANG_MAMBA_CONV_DTYPE=float16
HVF=~/herbert-vk-f16-cc51363/target/release
for m in 4B 0.8B; do
  echo "== $(date -Is) vk chat $m"
  for n in 10000 20000; do $HVF/herbert-vk-chat $M/Qwen3.5-$m --device nvidia --ctx 61440 --no-system --no-think --max-tokens 16 --ask "$(cat $LP/pp$n.txt)" > $O/hvf-$m-pp$n.log 2>&1 || echo "FAIL hvf chat $m pp$n"; done
  echo "== $(date -Is) vk serve $m"
  setsid $HVF/herbert-vk-serve $M/Qwen3.5-$m --device nvidia --ctx 61440 --port 8131 --name Qwen3.5-$m > $O/hvf-serve-$m.log 2>&1 &
  P=$!
  for _ in $(seq 120); do curl -sf -o /dev/null http://localhost:8131/ && break; sleep 1; done
  python3 - $LP/pp60000.txt Qwen3.5-$m > $O/hvf-serve-$m-pp60000.json 2>&1 <<'PY' || echo "FAIL hvf serve $m"
import sys, json, urllib.request
body = {"model": sys.argv[2], "max_tokens": 16, "messages": [{"role": "user", "content": open(sys.argv[1]).read()}]}
r = urllib.request.urlopen(urllib.request.Request("http://localhost:8131/v1/messages", json.dumps(body).encode(), {"content-type": "application/json"}), timeout=600)
print(r.read().decode())
PY
  kill -TERM -- -$P 2>/dev/null; sleep 3; kill -KILL -- -$P 2>/dev/null
  echo "== $(date -Is) llama $m"
  for be in cuda vk; do dev=CUDA0; [ $be = vk ] && dev=Vulkan1
    $L/build-$be/bin/llama-bench -m $G/Qwen3.5-$m-F16.gguf -dev $dev -ngl 99 -p 60000 -n 0 -r 1 -fa on -ub 1024 -b 2048 -o json > $O/llama-$be-$m.json 2> $O/llama-$be-$m.err || echo "FAIL llama $be $m"
  done
  echo "== $(date -Is) vllm $m"
  $VB bench latency --model $M/Qwen3.5-$m --dtype float16 --input-len 60000 --output-len 1 --batch-size 1 --num-iters-warmup 1 --num-iters 1 \
    --max-model-len 61440 --max-num-seqs 8 -cc.max_cudagraph_capture_size=512 --gpu-memory-utilization 0.85 --output-json $O/vllm-$m.json > $O/vllm-$m.log 2>&1 || echo "FAIL vllm $m"
  echo "== $(date -Is) sglang $m"
  $SPY -m sglang.benchmark.one_batch --model-path $M/Qwen3.5-$m --dtype float16 --context-length 61440 --mem-fraction-static 0.8 --batch-size 1 \
    --input-len 60000 --output-len 1 --run-name $m-smoke --result-filename $O/sglang.jsonl > $O/sglang-$m.log 2>&1 || echo "FAIL sglang $m"
  echo "== $(date -Is) transformers $m"
  for st in sdpa-dynamic sdpa-static-compile; do
    $TF/python - $m $st > $O/transformers-$m-$st.log 2>&1 <<'PY' || echo "FAIL transformers $m $st"
import sys, time, torch
from transformers import AutoModelForCausalLM
m, st = sys.argv[1], sys.argv[2]
model = AutoModelForCausalLM.from_pretrained(f"${MODELS_DIR:-$HOME/models}/Qwen3.5-{m}", dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
kw = {"max_new_tokens": 1, "do_sample": False}
if "static" in st: kw["cache_implementation"] = "static"
if st.endswith("compile"): model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)
ids = torch.randint(1000, 100000, (1, 60000), device="cuda")
with torch.no_grad(): model.generate(ids, attention_mask=torch.ones_like(ids), **kw)
print("ok, peak GiB", torch.cuda.max_memory_allocated() / 2**30)
PY
  done
done
echo ALLDONE-p5-smoke
