"""_av1_summary.py AV1 — herbert-cuda-bf16 73b5a70 (default; sdpa.gqa=1 sdpa.split=64 sdpa.pipe=1) against the
comparators of phase 5b in one run, alternated (alt-vs/_scripts/_av1-003.sh). Definitions of _p5b_summary.py:
herbert from its chat [turn] line (prefill; decode: the 127 generated after the first); llama.cpp CUDA by
llama-bench avg_ts; vLLM bench latency, prefill N / L(out 1), decode 127 / (L(out 128) - median L(out 1)), here per
round; SGLang one_batch, prefill_throughput of the out-1 runs, decode median throughput of the out-128 runs past
the first of each process. Per unit and shape: median over the rounds of each round's median (n rounds, spread);
then hc-def and hc-sw / comparator, as the ratio of medians and as the median of the per-round ratios. tok/s."""
import sys, os, re, glob, json, statistics, collections
P = sys.argv[1]; LENS = [10000, 20000, 40000, 60000]
SH = [f"pp{i}" for i in LENS] + [f"tg128@{i}" for i in LENS]
med = statistics.median
def spread(v): return (max(v) - min(v)) / med(v) * 100 if len(v) > 1 else 0.0
R = collections.defaultdict(lambda: collections.defaultdict(list))  # (unit, m, shape) -> round -> samples
for f in glob.glob(os.path.join(P, "chat", "r*-*.log")):
    mt = re.match(r"r(\d+)-(\S+?)-(hc-def|hc-sw)-pp(\d+)\.log", os.path.basename(f))
    if not mt: continue
    r, m, n, k = mt.groups()
    for l in open(f, errors="replace"):
        t = re.search(r"\[turn \d+\] (\d+) prompt tokens in [\d.]+ s \(([\d.]+) tok/s\) · (\d+) generated in [\d.]+ s \(([\d.]+) tok/s\)", l)
        if t: R[(n, m, f"pp{k}")][int(r)].append(float(t.group(2))); R[(n, m, f"tg128@{k}")][int(r)].append(float(t.group(4))); break
for f in glob.glob(os.path.join(P, "llama", "r*-*.json")):
    r, m, kind = re.match(r"r(\d+)-(\S+?)-(pp|tg)\.json", os.path.basename(f)).groups()
    try: d = json.load(open(f))
    except Exception: continue
    for t in d: R[("llama-cuda", m, f"pp{t['n_prompt']}" if kind == "pp" else f"tg{t['n_gen']}@{t['n_depth']}")][int(r)].append(t["avg_ts"])
lat = {}
for f in glob.glob(os.path.join(P, "vllm", "r*-*.json")):
    r, m, i, o = re.match(r"r(\d+)-(\S+?)-in(\d+)-out(\d+)\.json", os.path.basename(f)).groups()
    try: lat[(int(r), m, int(i), int(o))] = json.load(open(f))["latencies"]
    except Exception: pass
for (r, m, i, o), v in lat.items():
    if o == 1: R[("vllm", m, f"pp{i}")][r] += [i / l for l in v]
    if o == 128 and (r, m, i, 1) in lat:
        l1 = med(lat[(r, m, i, 1)]); R[("vllm", m, f"tg128@{i}")][r] += [127 / (l - l1) for l in v]
rj = os.path.join(P, "sglang", "results.jsonl")
for l in (open(rj) if os.path.isfile(rj) else []):
    d = json.loads(l); mt = re.match(r"r(\d+)-(\S+?)-(\S+)-(pp|tg)$", d["run_name"])
    if mt and mt.group(4) == "pp": R[("sglang", mt.group(2), f"pp{d['input_len']}")][int(mt.group(1))].append(d["prefill_throughput"])
for f in glob.glob(os.path.join(P, "sglang", "r*-tg*.log")):
    r, m, st, i = re.match(r"r(\d+)-(\S+?)-(\S+)-tg(\d+)\.log", os.path.basename(f)).groups()
    found = [float(mm.group(1)) for l in open(f, errors="replace")
             for mm in [re.search(r"Decode\.\s+median latency: [\d.]+ s, median throughput:\s+([\d.]+)", l)] if mm]
    R[("sglang", m, f"tg128@{i}")][int(r)] += found[1:]
per_round = {k: {r: med(v) for r, v in rr.items() if v} for k, rr in R.items()}
def cell(k):
    v = list(per_round.get(k, {}).values())
    return (med(v), len(v), spread(v)) if v else None
UNITS = ["hc-def", "hc-sw", "llama-cuda", "vllm", "sglang"]
print("# Per unit (median over rounds of each round's median tok/s; n rounds, spread)")
for m in ["0.8B", "4B"]:
    print(f"\n## {m}\n| unit | " + " | ".join(SH) + " |\n|---|" + "---|" * len(SH))
    for u in UNITS:
        cells = [cell((u, m, s)) for s in SH]
        if any(cells): print(f"| {u} | " + " | ".join(f"{c[0]:.1f} (n={c[1]}, {c[2]:.1f} %)" if c else "" for c in cells) + " |")
print("\n# hc-sw / hc-def (ratio of medians · median of per-round ratios)")
for m in ["0.8B", "4B"]:
    cells = []
    for s in SH:
        a, b = per_round.get(("hc-sw", m, s), {}), per_round.get(("hc-def", m, s), {})
        rr = [a[r] / b[r] for r in a if r in b]
        cells.append(f"×{med(a.values()) / med(b.values()):.3f} · ×{med(rr):.3f}" if rr else "")
    print(f"| {m} | " + " | ".join(cells) + " |")
print("\n# herbert / comparator (ratio of medians · median of per-round ratios, min..max of per-round ratios)")
for h in ["hc-def", "hc-sw"]:
    for m in ["0.8B", "4B"]:
        print(f"\n## {h} {m}\n| against | " + " | ".join(SH) + " |\n|---|" + "---|" * len(SH))
        for e in ["llama-cuda", "vllm", "sglang"]:
            cells = []
            for s in SH:
                a, b = per_round.get((h, m, s), {}), per_round.get((e, m, s), {})
                rr = [a[r] / b[r] for r in a if r in b]
                cells.append(f"×{med(a.values()) / med(b.values()):.3f} · ×{med(rr):.3f} ({min(rr):.3f}..{max(rr):.3f})" if rr else "")
            if any(cells): print(f"| {e} | " + " | ".join(cells) + " |")
cl = os.path.join(P, "clocks.csv")
if os.path.isfile(cl):
    rows = [[x.strip() for x in l.split(",")] for l in open(cl) if l[:4].isdigit()]
    c = collections.Counter(tuple(r[1:4]) for r in rows)
    print("\nclocks (sm, mem, pstate) over the whole run:", ", ".join(f"{k}: {n}" for k, n in c.most_common(8)))
    busy = [r for r in rows if r[4].split()[0].replace(".", "").isdigit() and float(r[4].split()[0]) > 100]
    if busy:
        sm = [int(r[1].split()[0]) for r in busy]
        print(f"samples over 100 W: {len(busy)}, SM under 2805 on {sum(s < 2805 for s in sm)} (min {min(sm)}), reasons:",
              ", ".join(f"{k}: {n}" for k, n in collections.Counter(r[7] for r in busy if int(r[1].split()[0]) < 2805).most_common(4)))
w = os.path.join(P, "waits.txt")
print("\nwaits for a busy GPU:", sum(1 for _ in open(w)) if os.path.isfile(w) else 0)
