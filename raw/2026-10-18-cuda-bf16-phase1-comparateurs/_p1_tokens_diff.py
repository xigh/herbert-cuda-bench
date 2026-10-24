"""_p1_tokens_diff.py DIR — first divergence of each comparator's greedy ids (DIR/tokens.jsonl, _p1_tokens.py)
against the CPU oracle (DIR/oracle-<model>/<prompt>-cpu.log, the [ids] line). The comparators' end-of-turn ids
(<|im_end|>, <|endoftext|>) end the comparison: the ids are cut at the first one (the oracle stops there and does
not print it; transformers, with no generation_config.json, stops only at <|endoftext|> and runs on past <|im_end|>)."""
import sys, os, json
D = sys.argv[1]
EOS = {248044, 248046}
def oracle(m, p):
    for l in open(os.path.join(D, f"oracle-{m}", f"{p}-cpu.log"), errors="replace"):
        if l.startswith("[ids]"): return [int(x) for x in l.split()[1:]]
for l in open(os.path.join(D, "tokens.jsonl")):
    d = json.loads(l)
    tag = f"{d['engine']:<12} {d['model']:<4} {d['setting']:<16}"
    if "error" in d: print(f"{tag} FAILED {d['error'][:160]}"); continue
    ref = oracle(d["model"], d["prompt"]); got = d["ids"][:next((i for i, t in enumerate(d["ids"]) if t in EOS), len(d["ids"]))]
    n = min(len(ref), len(got)); first = next((i for i in range(n) if ref[i] != got[i]), None)
    if first is None and len(ref) == len(got): v = f"identical ({len(got)} tokens)"
    elif first is None: v = f"identical prefix of {n}, lengths {len(ref)} (oracle) vs {len(got)}"
    else: v = f"first divergence at token {first} of {len(ref)}"
    print(f"{tag} {d['prompt']:<11} {v}")
