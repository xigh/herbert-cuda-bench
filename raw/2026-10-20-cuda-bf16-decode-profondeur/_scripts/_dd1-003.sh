#!/usr/bin/env bash
# _dd1-003.sh — why herbert-cuda's decode falls with depth (phase 5: 0.8B 298.6 -> 133.4 tok/s from 10K to 60K, vLLM
# 343.7 -> 270.4). herbert-cuda-bf16 011bfdf, 0.8B, the RTX 4080 (clocks checked), the resident inference services left running
# (one on the GPU, one on the CPU; their activity is logged).
#   A. --profile (launches one by one, events) at 10K, 20K, 40K, 60K: per-kernel GPU time of the decode step.
#   B. sdpa.split (blocks per head of the decode attention, default 8) in {8, 16, 32, 64, 128, 256} at 10K and 60K:
#      decode tok/s from the [turn] line (graph on), alternated, N = 3, order reversed every other round; the [ids]
#      of every run kept (the split changes the order of the sums).
#   C. --profile at 60K for every split: the sdpa_split and sdpa_combine times.
# Every output kept in full under $O; ALLDONE-dd1 at the end.
set -u
O=${O:-$HOME/c003/decode-depth/dd1}; mkdir -p $O/a $O/b $O/c
B=~/herbert-cuda-bf16/target/release/herbert-cuda-chat; M=${MODELS_DIR:-$HOME/models}/Qwen3.5-0.8B; LP=~/c003/long-prompts
case $(nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader,nounits | tr -d ' ') in 2805,11201) ;; *) echo "clocks not at 2805/11201, stop"; exit 1;; esac
{ date -Is; uptime; cat ~/herbert-cuda-bf16/COMMIT; sha256sum $B; nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate,memory.used --format=csv
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; systemctl --user list-units --type=service --state=running | grep -i herbert; } > $O/env.txt 2>&1
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu,utilization.gpu,memory.used,clocks_event_reasons.active --format=csv -l 1 > $O/clocks.csv 2>&1 &
smi=$!; trap 'kill $smi 2>/dev/null' EXIT
run() { local out=$1 p=$2; shift 2
  $B $M --ctx 61440 --no-system --no-think --warmup 1 --ask-file $LP/pp$p.txt "$@" > $out 2>&1 || echo "FAIL $out"
}
echo "== $(date -Is) A profile"
for p in 10000 20000 40000 60000; do run $O/a/profile-pp$p.log $p --max-tokens 64 --profile; done
SPLITS=(8 16 32 64 128 256)
for r in 1 2 3; do
  order=("${SPLITS[@]}"); [ $((r % 2)) = 0 ] && { order=(); for ((i=${#SPLITS[@]}-1; i>=0; i--)); do order+=("${SPLITS[$i]}"); done; }
  echo "== $(date -Is) B round $r · $(uptime)"
  for s in "${order[@]}"; do for p in 10000 60000; do run $O/b/r$r-s$s-pp$p.log $p --max-tokens 128 --tune sdpa.split=$s; done; done
done
echo "== $(date -Is) C profile per split at 60K"
for s in "${SPLITS[@]}"; do run $O/c/profile-s$s-pp60000.log 60000 --max-tokens 64 --profile --tune sdpa.split=$s; done
date -Is > $O/end.txt
echo ALLDONE-dd1
