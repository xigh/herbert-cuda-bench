#!/usr/bin/env bash
# _dd2-003.sh — after dd1 (sdpa.split 8 -> 32..128: 0.8B decode at 60K 133.5 -> 246.8..251.8 tok/s): does a larger
# split cost at short depth? herbert-cuda-bf16 011bfdf, 0.8B, the RTX 4080, resident inference services left running.
# sdpa.split in {8, 16, 32, 64, 128} on p3 (759 tokens, thinking, 256 generated: the decode shape of phase 2 and #002,
# default ctx) and at 20K and 40K (--ctx 61440, 128 generated), alternated, N = 3, order reversed every other round.
set -u
O=${O:-$HOME/c003/decode-depth/dd2}; mkdir -p $O/b
B=~/herbert-cuda-bf16/target/release/herbert-cuda-chat; M=${MODELS_DIR:-$HOME/models}/Qwen3.5-0.8B; LP=~/c003/long-prompts; PR=~/herbert-cuda-bf16/bench/prompts
case $(nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader,nounits | tr -d ' ') in 2805,11201) ;; *) echo "clocks not at 2805/11201, stop"; exit 1;; esac
{ date -Is; uptime; cat ~/herbert-cuda-bf16/COMMIT; sha256sum $B; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; } > $O/env.txt 2>&1
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu,utilization.gpu,memory.used,clocks_event_reasons.active --format=csv -l 1 > $O/clocks.csv 2>&1 &
smi=$!; trap 'kill $smi 2>/dev/null' EXIT
SPLITS=(8 16 32 64 128)
for r in 1 2 3; do
  order=("${SPLITS[@]}"); [ $((r % 2)) = 0 ] && { order=(); for ((i=${#SPLITS[@]}-1; i>=0; i--)); do order+=("${SPLITS[$i]}"); done; }
  echo "== $(date -Is) round $r · $(uptime)"
  for s in "${order[@]}"; do
    $B $M --no-system --max-tokens 256 --warmup 1 --tune sdpa.split=$s --ask "$(cat $PR/p3-long.txt)" > $O/b/r$r-s$s-p3.log 2>&1 || echo "FAIL r$r s$s p3"
    for p in 20000 40000; do
      $B $M --ctx 61440 --no-system --no-think --max-tokens 128 --warmup 1 --tune sdpa.split=$s --ask-file $LP/pp$p.txt > $O/b/r$r-s$s-pp$p.log 2>&1 || echo "FAIL r$r s$s pp$p"
    done
  done
done
date -Is > $O/end.txt
echo ALLDONE-dd2
