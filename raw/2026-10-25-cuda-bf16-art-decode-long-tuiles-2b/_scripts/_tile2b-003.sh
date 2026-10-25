#!/usr/bin/env bash
# _tile2b-003.sh — the 2B's prefill against hand-set tiles (added on 2026-10-25). herbert-cuda-bf16 31b86e1: gemm.auto=3 only has
# measured choices for hidden 2560 (4B) and 1024 (0.8B) on this GPU (src/cuda/engine.rs:249); the 2B (hidden 2048)
# falls back to policy 1 (wide tile 64×128, K 32, three stages); prefill.block=auto gives 2048 rows to the 4B, 4096 to
# the 0.8B and 1024 to the 2B (engine.rs:318). Variants, the two factors apart:
#   def            the default (gemm.auto=3 gemm.layout=xor prefill.block=auto = 1024 on the 2B)
#   def-b2048/4096 the default with the 4B's and the 0.8B's block size
#   t64x128        gemm.auto=0, 64×128 K32 three stages (policy 1's wide tile, every height), block 1024
#   w2x2s2         gemm.auto=0, 64×128 warps 2×2 K32 two stages (the wide tile auto 3 gives the 0.8B and the 4B), block 1024
#   w2x2s2-b2048/4096
#   t128-k64s2     gemm.auto=0, 128×128 warps 4×4 K64 two stages (the 4B's tile at M=256 padded), block 1024
# --ctx 40960 (as the long-prefill measure), no thinking, 1 token, --warmup 1, the [turn 1] line; pp2000 and pp10000 of the long-prefill measure. Variants
# alternated, N rounds, order reversed every other round. Same clocks check, resident-gpu stopped (trap). ALLDONE-tile2b.
set -u
N=${N:-5}; O=${O:-$HOME/c003/art/tile2b}; mkdir -p $O/chat
B=~/herbert-cuda-bf16-art; HC=$B/target/release/herbert-cuda-chat; LP=~/c003/art/prompts-long; M=${MODELS_DIR:-$HOME/models}
SIZES=(2000 10000); CTX=40960
M0="--tune gemm.auto=0 --tune gemm.layout=xor"
VARIANTS=("def|" "def-b2048|--tune prefill.block=2048" "def-b4096|--tune prefill.block=4096"
  "t64x128|$M0 --tune gemm.tile=64x128 --tune gemm.bk=32 --tune gemm.stages=3 --tune prefill.block=1024"
  "w2x2s2|$M0 --tune gemm.tile=64x128w2x2 --tune gemm.bk=32 --tune gemm.stages=2 --tune prefill.block=1024"
  "w2x2s2-b2048|$M0 --tune gemm.tile=64x128w2x2 --tune gemm.bk=32 --tune gemm.stages=2 --tune prefill.block=2048"
  "w2x2s2-b4096|$M0 --tune gemm.tile=64x128w2x2 --tune gemm.bk=32 --tune gemm.stages=2 --tune prefill.block=4096"
  "t128-k64s2|$M0 --tune gemm.tile=128w4x4 --tune gemm.bk=64 --tune gemm.stages=2 --tune prefill.block=1024")
clocks() { local i; for i in $(seq 1 30); do [ "$(nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader,nounits | tr -d ' ')" = "2805,11201" ] && return 0; sleep 1; done; return 1; }
clocks || { echo "clocks not at 2805/11201, stop"; exit 1; }
chk() { clocks || echo "!! clocks not at 2805/11201 before $1"; }
{ date -Is; nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate,power.limit --format=csv; echo "-- herbert-cuda $(cat $B/COMMIT)"; sha256sum $HC $LP/pp2000.txt $LP/pp10000.txt
  for v in "${VARIANTS[@]}"; do echo "variant ${v%%|*}: ${v#*|}"; done; } > $O/env.txt 2>&1
echo "== $(date -Is) stopping resident-gpu"; systemctl --user stop resident-gpu
trap 'echo "== $(date -Is) starting resident-gpu"; systemctl --user start resident-gpu; sleep 20; echo "== $(date -Is) resident-gpu: $(systemctl --user is-active resident-gpu)"; echo ALLDONE-tile2b' EXIT
sleep 5
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu,memory.used --format=csv -l 1 > $O/clocks.csv 2>&1 &
smi=$!
for r in $(seq 1 $N); do
  order=("${VARIANTS[@]}"); [ $((r % 2)) = 0 ] && { order=(); for ((i=${#VARIANTS[@]}-1; i>=0; i--)); do order+=("${VARIANTS[$i]}"); done; }
  echo "== $(date -Is) round $r · $(uptime)"
  for v in "${order[@]}"; do vn=${v%%|*}; va=${v#*|}
    for p in "${SIZES[@]}"; do chk "$vn pp$p"
      # shellcheck disable=SC2086
      $HC $M/Qwen3.5-2B $va --ctx $CTX --no-system --no-think --max-tokens 1 --warmup 1 --ask-file $LP/pp$p.txt > $O/chat/r$r-$vn-pp$p.log 2>&1 || echo "FAIL r$r $vn pp$p"
    done
  done
done
kill $smi
date -Is > $O/end.txt
