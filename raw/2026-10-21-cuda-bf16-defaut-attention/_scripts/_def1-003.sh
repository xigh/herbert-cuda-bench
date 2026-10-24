#!/usr/bin/env bash
# _def1-003.sh — the attention defaults changed on 2026-10-21: sdpa.gqa=1, sdpa.split=64,
# sdpa.pipe=1. Build ~/herbert-cuda-bf16-def (working tree on top of dac152c; COMMIT and DIFF files) against the
# build before the change, ~/herbert-cuda-bf16-occ (8db5b92's tree). resident-gpu.service stopped, restarted by a trap.
#   0. PTX: herbert-cuda-probe --emit of both builds, every module of the old one compared with the new one's.
#   A. Tokens: compare.sh p1..p4 and pp128..pp4000 against runs/plan-final-oracle/{0.8B,4B} (copied, not rerun):
#      new (new default), old-sw (old build with the three options), new-old (new build with
#      sdpa.gqa=0 sdpa.split=8 sdpa.pipe=0), old (old default).
#   B. Long prompts 10K and 60K (--ask-file, --ctx 61440), no thinking, 64 generated, --warmup 0, the same four.
# Token checks only, nothing timed for a comparison. ALLDONE-def1.
set -u
O=${O:-$HOME/c003/defaults/def1}; mkdir -p $O/ptx $O/long
NEW=~/herbert-cuda-bf16-def/target/release; OLD=~/herbert-cuda-bf16-occ/target/release; LP=~/c003/long-prompts
echo "== $(date -Is) build"
. ~/.cargo/env; cargo --version; ( cd ~/herbert-cuda-bf16-def && find src -name '*.rs' -exec touch {} + && cargo build --release ) > $O/build.log 2>&1 || { echo "FAIL build"; exit 1; }
echo "== $(date -Is) built $(cat ~/herbert-cuda-bf16-def/COMMIT)"
$OLD/herbert-cuda-probe --emit $O/ptx/old > $O/ptx/emit-old.log 2>&1 || echo "FAIL emit old"
$NEW/herbert-cuda-probe --emit $O/ptx/new > $O/ptx/emit-new.log 2>&1 || echo "FAIL emit new"
{ n=0; d=0; for f in $O/ptx/old/*.ptx; do n=$((n + 1)); cmp -s $f $O/ptx/new/$(basename $f) || { d=$((d + 1)); echo "differs: $(basename $f)"; }; done
  echo "old modules $n, differing $d"; echo "new only: $(cd $O/ptx/new && for f in *.ptx; do [ -f ../old/$f ] || printf '%s ' $f; done)"; } > $O/ptx/compare.txt
cat $O/ptx/compare.txt
case $(nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader,nounits | tr -d ' ') in 2805,11201) ;; *) echo "clocks not at 2805/11201 (token checks only, going on)";; esac
echo "== $(date -Is) stopping resident-gpu"
systemctl --user stop resident-gpu
trap 'echo "== $(date -Is) starting resident-gpu"; systemctl --user start resident-gpu; sleep 20; echo "== $(date -Is) resident-gpu: $(systemctl --user is-active resident-gpu)"' EXIT
sleep 5
{ date -Is; uptime; cat ~/herbert-cuda-bf16-def/COMMIT; cat ~/herbert-cuda-bf16-occ/COMMIT; sha256sum $NEW/herbert-cuda-chat $OLD/herbert-cuda-chat
  cat ~/herbert-cuda-bf16-def/DIFF; nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate,memory.used --format=csv
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; } > $O/env.txt 2>&1
SW="--tune sdpa.gqa=1 --tune sdpa.split=64 --tune sdpa.pipe=1"; BACK="--tune sdpa.gqa=0 --tune sdpa.split=8 --tune sdpa.pipe=0"
V=("new|$NEW/herbert-cuda-chat|" "old-sw|$OLD/herbert-cuda-chat|$SW" "new-old|$NEW/herbert-cuda-chat|$BACK" "old|$OLD/herbert-cuda-chat|")
for m in 0.8B 4B; do
  M=${MODELS_DIR:-$HOME/models}/Qwen3.5-$m
  echo "== $(date -Is) A tokens $m"
  mkdir -p $O/compare-$m; cp ~/herbert-cuda-bf16/runs/plan-final-oracle/$m/*-cpu.log $O/compare-$m/
  ps=""; for f in $O/compare-$m/*-cpu.log; do b=$(basename $f -cpu.log); ps="$ps prompts/$b.txt"; done
  vs=""; for v in "${V[@]}"; do IFS='|' read -r n b a <<< "$v"; vs="$vs;$n:@$b $a"; done
  ( cd ~/herbert-cuda-bf16-def/bench && PROMPTS="${ps# }" VARIANTS="${vs#;}" bash compare.sh $M $O/compare-$m 0 ) > $O/compare-$m.out 2>&1 || echo "FAIL compare $m"
  for p in 10000 60000; do
    echo "== $(date -Is) B $m pp$p"
    for v in "${V[@]}"; do IFS='|' read -r n b a <<< "$v"
      # shellcheck disable=SC2086
      $b $M --ctx 61440 --no-system --no-think --max-tokens 64 --warmup 0 $a --ask-file $LP/pp$p.txt > $O/long/$m-$n-pp$p.log 2>&1 || echo "FAIL $m $n pp$p"
    done
  done
done
date -Is > $O/end.txt
echo ALLDONE-def1
