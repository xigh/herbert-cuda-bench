#!/usr/bin/env bash
# _p1-llama.sh OUT — llama.cpp settings sweep (protocol.md, phase 1): b11718 CUDA (CUDA0) and Vulkan (Vulkan1),
# -ub 256…2048 × -b 2048/4096 × -fa on/off, plus the default (-ub 512 -b 2048 -fa auto); prefill at 128, 512,
# 2048 and 4000 tokens (-n 0), decode 255 at depth 759 (-fa on/off/auto); one process per setting, 3 rounds,
# settings in reversed order every other round; the reference 08ffba21e (CUDA, default) in the same rounds.
set -u
O=$1; mkdir -p $O
G=${BENCH_DIR:-$HOME/bench-qwen35}; L=$HOME/llama.cpp-b11718; R=$HOME/llama.cpp-08ffba21e
S=("default|" ); for ub in 256 512 1024 2048; do for b in 2048 4096; do for fa in on off; do S+=("ub$ub-b$b-fa$fa|-ub $ub -b $b -fa $fa"); done; done; done
D=("default|" "faon|-fa on" "faoff|-fa off")
for r in 1 2 3; do
  echo "== $(date -Is) llama round $r · $(uptime)"
  for m in 0.8B 4B; do
    g=$G/Qwen3.5-$m-BF16.gguf
    for be in cuda:CUDA0:$L/build-cuda vk:Vulkan1:$L/build-vk ref08ffba21e:CUDA0:$R/build-cuda; do
      name=${be%%:*}; rest=${be#*:}; dev=${rest%%:*}; dir=${rest#*:}
      set1=("${S[@]}"); set2=("${D[@]}")
      [ $name = ref08ffba21e ] && { set1=("default|"); set2=("default|"); }
      order=("${set1[@]}"); [ $((r % 2)) = 0 ] && { order=(); for ((i=${#set1[@]}-1; i>=0; i--)); do order+=("${set1[$i]}"); done; }
      for s in "${order[@]}"; do sn=${s%%|*}; sa=${s#*|}
        # shellcheck disable=SC2086
        $dir/bin/llama-bench -m $g -dev $dev -ngl 99 -p 128,512,2048,4000 -n 0 -r 1 $sa -o json > $O/r$r-$m-$name-pp-$sn.json 2> $O/r$r-$m-$name-pp-$sn.err || echo "FAIL r$r $m $name pp $sn"
      done
      for s in "${set2[@]}"; do sn=${s%%|*}; sa=${s#*|}
        # shellcheck disable=SC2086
        $dir/bin/llama-bench -m $g -dev $dev -ngl 99 -p 0 -n 255 -d 759 -r 1 $sa -o json > $O/r$r-$m-$name-tg-$sn.json 2> $O/r$r-$m-$name-tg-$sn.err || echo "FAIL r$r $m $name tg $sn"
      done
    done
  done
done
