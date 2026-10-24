#!/usr/bin/env bash
# _chain-dd.sh — after dd3 (4B, resident service restarted by its trap): dd4 (ncu), then the build of bc8e601 (sdpa.gqa)
# in ~/herbert-cuda-bf16-gqa with ptxas on its PTX, then dd5. ALLDONE-chain at the end.
set -u
D=~/c003/decode-depth
until grep -q "resident-gpu: " $D/dd3.out 2>/dev/null; do sleep 30; done
echo "== $(date -Is) dd3 ended: $(tail -2 $D/dd3.out | tr '\n' ' ')"
sleep 20
bash $D/_dd4-003.sh > $D/dd4.out 2>&1; echo "== $(date -Is) dd4: $(tail -1 $D/dd4.out)"
export PATH="$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH"
rm -rf ~/herbert-cuda-bf16-gqa && git clone -q $D/gqa.bundle ~/herbert-cuda-bf16-gqa && cd ~/herbert-cuda-bf16-gqa && git checkout -q main \
  && echo "$(git log -1 --format='%h (%cI) %s' | cut -c1-120)" > COMMIT && cargo build --release --locked > $D/build-gqa.log 2>&1 \
  || { echo "BUILD FAILED"; tail -20 $D/build-gqa.log; exit 1; }
echo "== $(date -Is) built $(cat COMMIT)"
rm -rf $D/ptx-gqa && ./target/release/herbert-cuda-probe --emit $D/ptx-gqa > $D/ptx-gqa.log 2>&1
for f in $D/ptx-gqa/*sdpa_gqa*.ptx; do /usr/local/cuda-12.8/bin/ptxas -arch=sm_89 -v "$f" -o /dev/null >> $D/ptxas-gqa.log 2>&1 && echo "ptxas ok $(basename $f)" || echo "ptxas FAIL $(basename $f)"; done
bash $D/_dd5-003.sh > $D/dd5.out 2>&1; echo "== $(date -Is) dd5: $(tail -1 $D/dd5.out)"
echo ALLDONE-chain
