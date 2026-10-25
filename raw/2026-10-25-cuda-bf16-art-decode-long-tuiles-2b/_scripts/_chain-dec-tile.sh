#!/usr/bin/env bash
# _chain-dec-tile.sh — added on 2026-10-25: test 1 (decode after a long prompt) then test 2 (2B tiles).
mkdir -p ~/c003/art/dec-long ~/c003/art/tile2b
bash ~/c003/art/scripts/_dec-long-003.sh > ~/c003/art/dec-long/run.out 2>&1
bash ~/c003/art/scripts/_tile2b-003.sh > ~/c003/art/tile2b/run.out 2>&1
echo ALLDONE-chain-dec-tile
