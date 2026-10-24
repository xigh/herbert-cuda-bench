#!/usr/bin/env bash
# _p5a2-003.sh — phase 5, step 5a, second pass (after _p5a-003.sh):
#   1) transformers at 60 000 tokens again: the raw text of pp60000.txt is 59 988 tokens, the 5a run failed on its
#      own length check (AssertionError), not on memory; ids now from the pp60000 case of herbert-cuda-bench --prepare.
#   2) SGLang where no setting ran: 4B at 40K/60K (0.8: out of memory from 20K; 0.7: KV pool under 40K tokens,
#      "skipping … max batch size limit"; 0.6: nothing), 0.8B at 60K only at 0.6. Fractions between 0.7 and 0.8,
#      and --max-running-requests 1 (the mamba state cache is sized for several requests), 3 samples per length.
# Same output tree as 5a (transformers.jsonl appended, sglang-rep/results.jsonl appended). ALLDONE-phase5a2.
set -u
O=${O:-$HOME/c003/phase5a}
S=~/c003/phase5-scripts; M=${MODELS_DIR:-$HOME/models}; SPY=~/venvs/bench-sglang-latest/bin/python; TF=~/venvs/bench-transformers-latest/bin
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH DTYPE=float16 SGLANG_MAMBA_CONV_DTYPE=float16
LENS="10000 20000 40000 60000"; CTX=61440
case $(nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader,nounits | tr -d ' ') in 2805,11201) ;; *) echo "clocks not at 2805/11201, stop"; exit 1;; esac
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu,memory.used,clocks_event_reasons.active --format=csv -l 1 > $O/clocks-a2.csv 2>&1 &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
for m in 0.8B 4B; do for st in sdpa-dynamic sdpa-static sdpa-static-compile eager-dynamic eager-static eager-static-compile; do
  echo "== $(date -Is) transformers $m $st 60000 · $(uptime)"
  CASES_JSON=$O/tokens/cases-$m.json $TF/python $S/_p5_transformers.py $O/transformers.jsonl $m $st 60000:1 >> $O/transformers-a2.log 2>&1 || echo "FAIL transformers $m $st"
done; done
SS=$O/sglang-rep
SV=("mf075|0.75|" "mf072|0.72|" "mf078|0.78|" "mf08-mrr1|0.8|--max-running-requests 1" "mf07-mrr1|0.7|--max-running-requests 1")
for m in 0.8B 4B; do for v in "${SV[@]}"; do name=${v%%|*}; rest=${v#*|}; mf=${rest%%|*}; args=${rest#*|}
  echo "== $(date -Is) sglang $m $name · $(uptime)"
  for i in $LENS; do
    # shellcheck disable=SC2086
    $SPY -m sglang.benchmark.one_batch --model-path $M/Qwen3.5-$m --dtype float16 --context-length $CTX --mem-fraction-static $mf --batch-size 1 \
      --input-len $i $i $i --output-len 1 --run-name $m-$name-pp --result-filename $SS/results.jsonl $args > $SS/$m-$name-pp$i.log 2>&1 || echo "FAIL sglang $m $name pp$i"
    grep -q "skipping" $SS/$m-$name-pp$i.log && echo "SKIP sglang $m $name pp$i (KV pool too small)"
  done
done; done
echo ALLDONE-phase5a2
