#!/usr/bin/env bash
# _chain-dd8.sh — build 73b5a70 in ~/herbert-cuda-bf16-fix, check its emitted PTX against the 011bfdf build (all of
# the base's modules must be identical), then dd8. ALLDONE-chain8.
set -u
D=~/c003/decode-depth
export PATH="$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH"
rm -rf ~/herbert-cuda-bf16-fix && git clone -q $D/fix.bundle ~/herbert-cuda-bf16-fix 2>/dev/null; cd ~/herbert-cuda-bf16-fix && git checkout -q main \
  && echo "$(git log -1 --format='%h (%cI) %s' | cut -c1-120)" > COMMIT && cargo build --release --locked > $D/build-fix.log 2>&1 \
  || { echo "BUILD FAILED"; tail -20 $D/build-fix.log; exit 1; }
echo "== $(date -Is) built $(cat COMMIT)"
rm -rf $D/ptx-fix && ./target/release/herbert-cuda-probe --emit $D/ptx-fix > $D/ptx-fix.log 2>&1
n=0; d=0; for f in $D/ptx-base/*.ptx; do n=$((n + 1)); cmp -s "$f" "$D/ptx-fix/$(basename $f)" || { d=$((d + 1)); echo "PTX DIFFERS $(basename $f)"; }; done
echo "base modules $n, differing $d; new: $(ls $D/ptx-fix | grep -v -F -x -f <(ls $D/ptx-base) | tr '\n' ' ')"
[ $d = 0 ] || { echo "default PTX changed, stop"; exit 1; }
bash $D/_dd8-003.sh > $D/dd8.out 2>&1; echo "== $(date -Is) dd8: $(tail -1 $D/dd8.out)"
echo ALLDONE-chain8
