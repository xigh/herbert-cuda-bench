#!/usr/bin/env bash
# _dd3-003.sh — the decode at depth on the 4B (dd1/dd2b were 0.8B only: the 4B does not fit beside the resident GPU
# service). resident-gpu.service is stopped for the run and restarted by a trap,
# even on failure. Every launch waits for an empty GPU (no compute process at all, checked by PID list) and logs it.
#   A. --profile at 10K, 20K, 40K, 60K (sdpa.split=8): decode step per kernel, and the prefill blocks (sdpa_flash).
#   B. sdpa.split in {8, 16, 32, 128} on p3 (759, thinking, 256 generated, default ctx) and at 10K, 20K, 40K, 60K
#      (--ctx 61440, 128 generated), graph on, alternated, N = 3, order reversed every other round.
#   C. --profile at 60K for split 16, 32, 128.
# herbert-cuda-bf16 011bfdf, Qwen3.5-4B, the RTX 4080. ALLDONE-dd3 at the end.
set -u
O=${O:-$HOME/c003/decode-depth/dd3}; mkdir -p $O/a $O/b $O/c
B=~/herbert-cuda-bf16/target/release/herbert-cuda-chat; M=${MODELS_DIR:-$HOME/models}/Qwen3.5-4B; LP=~/c003/long-prompts; PR=~/herbert-cuda-bf16/bench/prompts
case $(nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader,nounits | tr -d ' ') in 2805,11201) ;; *) echo "clocks not at 2805/11201, stop"; exit 1;; esac
echo "== $(date -Is) stopping resident-gpu"; systemctl --user stop resident-gpu.service
restart() { systemctl --user start resident-gpu.service; sleep 5; echo "== $(date -Is) resident-gpu: $(systemctl --user is-active resident-gpu.service)"; }
trap 'kill $smi 2>/dev/null; restart' EXIT
{ date -Is; uptime; cat ~/herbert-cuda-bf16/COMMIT; sha256sum $B; nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate,memory.used --format=csv
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; systemctl --user list-units --type=service --all | grep -i herbert; } > $O/env.txt 2>&1
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu,utilization.gpu,memory.used,clocks_event_reasons.active --format=csv -l 1 > $O/clocks.csv 2>&1 &
smi=$!
free() { # wait (at most 20 min per launch) while any compute process holds the GPU; log it
  local n=0
  while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do
    echo "$(date -Is) busy: $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';')" >> $O/waits.txt
    n=$((n + 1)); [ $n -gt 40 ] && { echo "GPU never free, stop"; exit 1; }; sleep 30
  done
  echo "$(date -Is) $1" >> $O/launches.txt
}
run() { local out=$1 p=$2; shift 2; free $out
  $B $M --ctx 61440 --no-system --no-think --warmup 1 --ask-file $LP/pp$p.txt "$@" > $out 2>&1 || echo "FAIL $out"
}
echo "== $(date -Is) A profile"
for p in 10000 20000 40000 60000; do run $O/a/profile-pp$p.log $p --max-tokens 64 --profile; done
SPLITS=(8 16 32 128)
for r in 1 2 3; do
  order=("${SPLITS[@]}"); [ $((r % 2)) = 0 ] && { order=(); for ((i=${#SPLITS[@]}-1; i>=0; i--)); do order+=("${SPLITS[$i]}"); done; }
  echo "== $(date -Is) B round $r · $(uptime)"
  for s in "${order[@]}"; do
    free $O/b/r$r-s$s-p3.log
    $B $M --no-system --max-tokens 256 --warmup 1 --tune sdpa.split=$s --ask "$(cat $PR/p3-long.txt)" > $O/b/r$r-s$s-p3.log 2>&1 || echo "FAIL r$r s$s p3"
    for p in 10000 20000 40000 60000; do run $O/b/r$r-s$s-pp$p.log $p --max-tokens 128 --tune sdpa.split=$s; done
  done
done
echo "== $(date -Is) C profile per split at 60K"
for s in 16 32 128; do run $O/c/profile-s$s-pp60000.log 60000 --max-tokens 64 --profile --tune sdpa.split=$s; done
date -Is > $O/end.txt
echo ALLDONE-dd3
