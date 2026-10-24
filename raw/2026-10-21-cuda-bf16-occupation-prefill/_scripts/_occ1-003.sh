#!/usr/bin/env bash
# _occ1-003.sh — the prefill attention at 32 query rows per block (sdpa.rows=32: 8 warps sharing each K and V tile,
# 101 376 bytes of shared memory, the most a block may have on sm_89) against 16 (the default, 4 warps, 67.6 KB:
# one block per SM), with and without sdpa.pipe=1. Build ~/herbert-cuda-bf16-occ (working tree on top of
# 696b75d; its COMMIT file and the diff in env.txt). resident-gpu.service stopped for the run, restarted by a trap.
#   A. Tokens: compare.sh p1..p4 and pp128..pp4000 against runs/plan-final-oracle/{0.8B,4B} (copied, not rerun):
#      default, rows32, pipe, pipe+rows32.
#   B. Prefill tok/s, alternated N = 3 (order reversed every other round): the four variants at pp4000 (--ask), 10K,
#      20K, 40K and 60K (--ask-file, --ctx 61440), no thinking, 16 generated, --warmup 1; 0.8B then 4B.
#   C. --profile at 60K: the four variants, both models.
#   D. Nsight Compute (--set full): sdpa_flash with rows32 and pipe+rows32, 2 launches of the last prefill block at
#      60K, both models (default and pipe: dd4 and dd7).
# Every launch waits for an empty GPU. ALLDONE-occ1.
set -u
O=${O:-$HOME/c003/occupancy/occ1}; mkdir -p $O/b $O/c $O/ncu
NB=~/herbert-cuda-bf16-occ/target/release/herbert-cuda-chat; LP=~/c003/long-prompts; PR=~/herbert-cuda-bf16-occ/bench/prompts
NCU=/opt/nvidia/nsight-compute/2025.1.1/ncu
case $(nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader,nounits | tr -d ' ') in 2805,11201) ;; *) echo "clocks not at 2805/11201, stop"; exit 1;; esac
echo "== $(date -Is) stopping resident-gpu"
systemctl --user stop resident-gpu
trap 'kill $smi 2>/dev/null; echo "== $(date -Is) starting resident-gpu"; systemctl --user start resident-gpu; sleep 20; echo "== $(date -Is) resident-gpu: $(systemctl --user is-active resident-gpu)"' EXIT
sleep 5
{ date -Is; uptime; cat ~/herbert-cuda-bf16-occ/COMMIT; sha256sum $NB; cat ~/herbert-cuda-bf16-occ/DIFF 2>/dev/null
  nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate,memory.used --format=csv; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; } > $O/env.txt 2>&1
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu,utilization.gpu,memory.used,clocks_event_reasons.active --format=csv -l 1 > $O/clocks.csv 2>&1 &
smi=$!
free() { local n=0
  while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do
    echo "$(date -Is) busy: $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';')" >> $O/waits.txt
    n=$((n + 1)); [ $n -gt 60 ] && { echo "GPU never free, stop"; exit 1; }; sleep 30
  done
  echo "$(date -Is) $1" >> $O/launches.txt
}
V=("default|" "rows32|--tune sdpa.rows=32" "pipe|--tune sdpa.pipe=1" "pipe-rows32|--tune sdpa.pipe=1 --tune sdpa.rows=32")
for m in 0.8B 4B; do
  M=${MODELS_DIR:-$HOME/models}/Qwen3.5-$m
  echo "== $(date -Is) A tokens $m"
  mkdir -p $O/compare-$m; cp ~/herbert-cuda-bf16/runs/plan-final-oracle/$m/*-cpu.log $O/compare-$m/
  ps=""; for f in $O/compare-$m/*-cpu.log; do b=$(basename $f -cpu.log); ps="$ps prompts/$b.txt"; done
  free compare-$m
  ( cd ~/herbert-cuda-bf16-occ/bench && PROMPTS="${ps# }" \
    VARIANTS="default:@$NB;rows32:@$NB --tune sdpa.rows=32;pipe:@$NB --tune sdpa.pipe=1;pipe-rows32:@$NB --tune sdpa.pipe=1 --tune sdpa.rows=32" \
    bash compare.sh $M $O/compare-$m 0 ) > $O/compare-$m.out 2>&1 || echo "FAIL compare $m"
done
for m in 0.8B 4B; do
  M=${MODELS_DIR:-$HOME/models}/Qwen3.5-$m
  for r in 1 2 3; do
    order=("${V[@]}"); [ $((r % 2)) = 0 ] && { order=(); for ((i=${#V[@]}-1; i>=0; i--)); do order+=("${V[$i]}"); done; }
    echo "== $(date -Is) B $m round $r · $(uptime)"
    for v in "${order[@]}"; do n=${v%%|*}; a=${v#*|}
      free $O/b/r$r-$m-$n-pp4000.log
      # shellcheck disable=SC2086
      $NB $M --no-system --no-think --max-tokens 16 --warmup 1 $a --ask "$(cat $PR/pp4000.txt)" > $O/b/r$r-$m-$n-pp4000.log 2>&1 || echo "FAIL r$r $m $n pp4000"
      for p in 10000 20000 40000 60000; do
        free $O/b/r$r-$m-$n-pp$p.log
        # shellcheck disable=SC2086
        $NB $M --ctx 61440 --no-system --no-think --max-tokens 16 --warmup 1 $a --ask-file $LP/pp$p.txt > $O/b/r$r-$m-$n-pp$p.log 2>&1 || echo "FAIL r$r $m $n pp$p"
      done
    done
  done
  echo "== $(date -Is) C $m profile at 60K"
  for v in "${V[@]}"; do n=${v%%|*}; a=${v#*|}; free $O/c/profile-$m-$n-pp60000.log
    # shellcheck disable=SC2086
    $NB $M --ctx 61440 --no-system --no-think --max-tokens 16 --warmup 1 --profile $a --ask-file $LP/pp60000.txt > $O/c/profile-$m-$n-pp60000.log 2>&1 || echo "FAIL profile $m $n"
  done
done
echo "== $(date -Is) D ncu"
for m in 0.8B 4B; do
  M=${MODELS_DIR:-$HOME/models}/Qwen3.5-$m; skip=84; [ $m = 4B ] && skip=232
  for v in "rows32|--tune sdpa.rows=32" "pipe-rows32|--tune sdpa.pipe=1 --tune sdpa.rows=32"; do n=${v%%|*}; a=${v#*|}
    free ncu-$m-$n
    # shellcheck disable=SC2086
    $NCU --set full --kernel-name regex:'^sdpa_flash' --launch-skip $skip --launch-count 2 --clock-control none -f -o $O/ncu/sdpa_flash-$m-$n-last \
      $NB $M --ctx 61440 --no-system --no-think --warmup 0 --max-tokens 2 --profile $a --ask-file $LP/pp60000.txt > $O/ncu/sdpa_flash-$m-$n-last.log 2>&1 || echo "FAIL ncu $m $n"
    $NCU --import $O/ncu/sdpa_flash-$m-$n-last.ncu-rep --page details > $O/ncu/sdpa_flash-$m-$n-last.details.txt 2>&1
  done
done
date -Is > $O/end.txt
echo ALLDONE-occ1
