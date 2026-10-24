"""_p5_tokens_diff.py TOKENS — the 64 greedy tokens after each long prompt (step 5a), engine against engine (no CPU
reference past 8 192 positions). Sources in TOKENS: herbert-cuda-bench's JSON ({hc-bf16,hc-f16,hc-f16-acc16}-{model}
.json, output_ids per case), herbert-vk-f16's chat logs (hvf-{model}-pp{N}.log, the [ids] line) and tokens.jsonl
(vLLM, SGLang, transformers, llama.cpp, one line per engine, setting and prompt). Per model and prompt: the classes
of identical sequences (over the common length), their members, and the first divergence of each class against the
class of hc-bf16 (index of the first differing token, 0-based), then between every two classes.
herbert-vk-f16's chat prints 63 generated ids where the others give 64: compared over the common length."""
import sys, os, re, glob, json, collections
T = sys.argv[1]
seqs = collections.defaultdict(dict)   # (model, prompt) -> {name: ids}
for f in glob.glob(os.path.join(T, "hc-*.json")):
    n, m = re.match(r"(hc-\S+)-(0\.8B|4B)\.json", os.path.basename(f)).groups()
    for s in json.load(open(f))["samples"]: seqs[(m, s["case"])][n] = s["output_ids"]
for f in glob.glob(os.path.join(T, "hvf-*-pp*.log")):
    m, p = re.match(r"hvf-(\S+)-(pp\d+)\.log", os.path.basename(f)).groups()
    ids = [l.split()[1:] for l in open(f, errors="replace") if l.startswith("[ids]")]
    if ids: seqs[(m, p)]["hvf"] = list(map(int, ids[-1]))
for l in open(os.path.join(T, "tokens.jsonl")):
    d = json.loads(l)
    if "ids" in d: seqs[(d["model"], d["prompt"])][f"{d['engine']}:{d['setting']}"] = d["ids"]
def first_diff(a, b):
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), None), n
for (m, p) in sorted(seqs, key=lambda k: (k[0], int(k[1][2:]))):
    d = seqs[(m, p)]
    print(f"== {m} {p}: {len(d)} sequences, lengths {sorted({len(v) for v in d.values()})}")
    classes = []   # [ids, [names]]
    for n in sorted(d, key=lambda n: (not n.startswith("hc-bf16"), n)):
        for c in classes:
            i, k = first_diff(c[0], d[n])
            if i is None: c[1].append(n); break
        else: classes.append([d[n], [n]])
    ref = classes[0][0]
    for j, (ids, names) in enumerate(classes):
        i, k = first_diff(ref, ids)
        tag = "=" if i is None else f"first divergence at {i}"
        print(f"  class {j} ({tag} against class 0): {' '.join(names)}")
    if len(classes) > 2:
        print("  between classes: " + "  ".join(f"{a}/{b} {first_diff(classes[a][0], classes[b][0])[0]}"
                                               for a in range(1, len(classes)) for b in range(a + 1, len(classes))))
