#!/usr/bin/env bash
# _chain-occ1.sh — waits for the end of _av1-003.sh (ALLDONE-av1), builds ~/herbert-cuda-bf16-occ (the GPU and CPU
# free of the comparator run by then), then runs _occ1-003.sh. ALLDONE-chain-occ1.
set -u
until grep -q ALLDONE-av1 ~/c003/alt-vs/av1.out 2>/dev/null || ! ps -eo args | grep -q "[_]av1-003.sh"; do sleep 60; done
sleep 40
echo "== $(date -Is) build"
. ~/.cargo/env; cargo --version; cd ~/herbert-cuda-bf16-occ && find src -name '*.rs' -exec touch {} + && cargo build --release > ~/c003/occupancy/build-occ.log 2>&1 || { echo "FAIL build"; exit 1; }
echo "== $(date -Is) built $(cat ~/herbert-cuda-bf16-occ/COMMIT)"
bash ~/c003/_occ1-003.sh
echo ALLDONE-chain-occ1
