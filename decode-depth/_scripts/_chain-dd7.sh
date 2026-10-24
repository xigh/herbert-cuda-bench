#!/usr/bin/env bash
# _chain-dd7.sh — after dd6 (resident service restarted): build 5ec3063 in ~/herbert-cuda-bf16-pipe, ptxas on its sdpa_flash PTX
# (default and pipe), then dd7. ALLDONE-chain7.
set -u
D=~/c003/decode-depth
until grep -q "resident-gpu: " $D/dd6.out 2>/dev/null; do sleep 30; done
echo "== $(date -Is) dd6 ended: $(tail -2 $D/dd6.out | tr '\n' ' ')"
sleep 20
export PATH="$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH"
rm -rf ~/herbert-cuda-bf16-pipe && git clone -q $D/pipe.bundle ~/herbert-cuda-bf16-pipe 2>/dev/null; cd ~/herbert-cuda-bf16-pipe && git checkout -q main \
  && echo "$(git log -1 --format='%h (%cI) %s' | cut -c1-120)" > COMMIT && cargo build --release --locked > $D/build-pipe.log 2>&1 \
  || { echo "BUILD FAILED"; tail -20 $D/build-pipe.log; exit 1; }
echo "== $(date -Is) built $(cat COMMIT)"
rm -rf $D/ptx-pipe && ./target/release/herbert-cuda-probe --emit $D/ptx-pipe > $D/ptx-pipe.log 2>&1
for f in $D/ptx-pipe/*sdpa_flash*.ptx; do /usr/local/cuda-12.8/bin/ptxas -arch=sm_89 -v "$f" -o /dev/null >> $D/ptxas-pipe.log 2>&1 && echo "ptxas ok $(basename $f)" || echo "ptxas FAIL $(basename $f)"; done
# The default PTX of sdpa_flash must be the same text as in the 011bfdf build.
mkdir -p $D/ptx-base && ~/herbert-cuda-bf16/target/release/herbert-cuda-probe --emit $D/ptx-base > $D/ptx-base.log 2>&1
for f in $D/ptx-base/sdpa_flash*.ptx; do cmp -s "$f" "$D/ptx-pipe/$(basename $f)" && echo "same PTX $(basename $f)" || echo "PTX DIFFERS $(basename $f)"; done
bash $D/_dd7-003.sh > $D/dd7.out 2>&1; echo "== $(date -Is) dd7: $(tail -1 $D/dd7.out)"
echo ALLDONE-chain7
