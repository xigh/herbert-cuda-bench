#!/usr/bin/env bash
# _p2b-llama-b11719-003.sh — queued behind phase 2 (waits for ALLDONE-phase2). llama.cpp b11719 was released at
# 10:59 today, after phase 1 had started on b11718 (10:38); the predictions ask for the latest stable release at the
# time of the campaign. b11719 changes one line of the CUDA flash attention tile kernel (ggml-cuda/fattn-tile.cuh,
# 5aeac37a3); Vulkan is untouched. Built after phase 2 (a build loads every core), with b11718's configuration
# (Release, CUDA arch 89, GGML_NATIVE); then b11719 against b11718, CUDA, alternated round by round (N = 5), with
# phase 2's two prefill and two decode settings; then b11719's tokens against the oracle (p1-p4, as after-p1d).
set -u
until grep -q ALLDONE-phase2 ~/c003/phase2.out; do sleep 60; done
O=${O:-$HOME/c003/phase2b}; mkdir -p $O/llama $O/tokens
N=$HOME/llama.cpp-b11719; G=${BENCH_DIR:-$HOME/bench-qwen35}; M=${MODELS_DIR:-$HOME/models}
echo "== $(date -Is) build b11719 · $(uptime)"
if [ ! -x $N/build-cuda/bin/llama-bench ]; then
  { git clone $HOME/llama.cpp-b11718 $N && cd $N && git fetch https://github.com/ggml-org/llama.cpp tag b11719 && git checkout b11719 && git log -1 --format="%h %ci %s" \
    && cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_NATIVE=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    && cmake --build build-cuda -j 12 --target llama-bench llama-server; } > $O/build.log 2>&1 || { echo "FAIL build b11719"; echo ALLDONE-phase2b; exit 1; }
fi
$N/build-cuda/bin/llama-bench --version > $O/version.txt 2>&1
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu --format=csv -l 1 > $O/clocks.csv 2>&1 &
smi=$!
sleep 60   # let the build's heat settle
PP=("default|" "ub1024-b4096-faon|-ub 1024 -b 4096 -fa on"); TG=("default|" "faon|-fa on")
for r in 1 2 3 4 5; do
  vs=(b11718 b11719); [ $((r % 2)) = 0 ] && vs=(b11719 b11718)
  for m in 0.8B 4B; do
    echo "== $(date -Is) round $r $m · $(uptime)"
    g=$G/Qwen3.5-$m-BF16.gguf
    for v in "${vs[@]}"; do d=$HOME/llama.cpp-$v/build-cuda/bin
      for s in "${PP[@]}"; do sn=${s%%|*}; sa=${s#*|}
        # shellcheck disable=SC2086
        $d/llama-bench -m $g -dev CUDA0 -ngl 99 -p 128,512,2048,4000 -n 0 -r 1 $sa -o json > $O/llama/r$r-$m-$v-pp-$sn.json 2> $O/llama/r$r-$m-$v-pp-$sn.err || echo "FAIL r$r $m $v pp $sn"
      done
      for s in "${TG[@]}"; do sn=${s%%|*}; sa=${s#*|}
        # shellcheck disable=SC2086
        $d/llama-bench -m $g -dev CUDA0 -ngl 99 -p 0 -n 255 -d 759 -r 1 $sa -o json > $O/llama/r$r-$m-$v-tg-$sn.json 2> $O/llama/r$r-$m-$v-tg-$sn.err || echo "FAIL r$r $m $v tg $sn"
      done
    done
  done
done
# Tokens against the oracle, as after-p1d (same cases, same oracle logs).
cp -r ~/c003/after-p1d/cases-*.json ~/c003/after-p1d/oracle-* $O/tokens/
for m in 0.8B 4B; do
  for v in "cuda-default|" "cuda-faon-ub1024|-fa on -ub 1024 -b 4096"; do n=${v%%|*}; args=${v#*|}
    echo "== $(date -Is) tokens $m $n · $(uptime)"
    # shellcheck disable=SC2086
    $N/build-cuda/bin/llama-server -m $G/Qwen3.5-$m-BF16.gguf -dev CUDA0 -ngl 99 -c 4608 --parallel 1 --host localhost --port 18732 $args > $O/tokens/llama-server-$m-$n.log 2>&1 &
    srv=$!
    for _ in $(seq 120); do curl -sf http://localhost:18732/health > /dev/null && break; sleep 1; done
    ~/venvs/bench-vllm-latest/bin/python ~/c003/phase1-scripts/_p1_tokens.py llama $m b11719-$n http://localhost:18732 $O/tokens/cases-$m.json $O/tokens/tokens.jsonl > $O/tokens/llama-$m-$n.log 2>&1 || echo "FAIL tokens $m $n"
    kill $srv; wait $srv 2>/dev/null
  done
done
kill $smi
echo ALLDONE-phase2b
