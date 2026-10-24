"""_p5b_summary.py PHASE5B — step 5b of phase 5 (protocol.md): herbert against the comparators at 10K..60K tokens on
the RTX 4080, derived from phase4/_p4b_summary.py with the shapes of phase 5: prefill ppN and decode of 128 tokens at
depth N (tg128@N). Per model and shape:
A. each engine's bench — herbert-cuda-bf16, herbert-cuda-f16 (default, gemm.f16=1), herbert-vk-f16 (default and
   host.poll=1, 10K and 20K) from their chat [turn] line (decode: the 127 generated after the first); llama.cpp CUDA
   and Vulkan (llama-bench, pp at the two 5a settings, tg -p 0 -n 128 -d N default and -fa on); vLLM (bench latency,
   prefill N / L(out 1), decode 127 / (L(out 128) - median L(out 1))); SGLang (one_batch, prefill_throughput of the
   out-1 runs, decode median throughput of the out-128 runs past the first of each process); transformers (same
   definition as vLLM); the best of each comparator's settings shape by shape, with n and spread;
then each herbert-cuda form divided by each engine;
B. the common HTTP client (--skip-blank), prefill (input tokens / TTFT) and decode medians per prompt and server.
tok/s throughout."""
import sys, os, re, glob, json, statistics, collections
P = sys.argv[1]; LENS = [10000, 20000, 40000, 60000]
SH = [f"pp{i}" for i in LENS] + [f"tg128@{i}" for i in LENS]
med = statistics.median
def spread(v): return (max(v) - min(v)) / med(v) * 100 if len(v) > 1 else 0.0
A = collections.defaultdict(list)
HC = ["hc-bf16", "hc-f16", "hc-f16-acc16", "hvf", "hvf-poll"]
for f in glob.glob(os.path.join(P, "chat", "r*-*.log")):
    mt = re.match(r"r(\d+)-(\S+?)-(hc-bf16|hc-f16-acc16|hc-f16|hvf-poll|hvf)-pp(\d+)\.log", os.path.basename(f))
    if not mt: continue
    _, m, n, k = mt.groups()
    for l in open(f, errors="replace"):
        t = re.search(r"\[turn \d+\] (\d+) prompt tokens in [\d.]+ s \(([\d.]+) tok/s\) · (\d+) generated in [\d.]+ s \(([\d.]+) tok/s\)", l)
        if t: A[(n, m, f"pp{k}", "-")].append(float(t.group(2))); A[(n, m, f"tg128@{k}", "-")].append(float(t.group(4))); break
for f in glob.glob(os.path.join(P, "llama", "r*-*.json")):
    mt = re.match(r"r(\d+)-(\S+?)-(cuda|vk)-(pp|tg)-(\S+)\.json", os.path.basename(f))
    if not mt: continue
    _, m, be, kind, st = mt.groups()
    try: d = json.load(open(f))
    except Exception: continue
    for t in d: A[(f"llama-{be}", m, f"pp{t['n_prompt']}" if kind == "pp" else f"tg{t['n_gen']}@{t['n_depth']}", st)].append(t["avg_ts"])
lat = {}
for f in glob.glob(os.path.join(P, "vllm", "*.json")):
    m, st, i, o = re.match(r"(\S+?)-(\S+)-in(\d+)-out(\d+)\.json", os.path.basename(f)).groups()
    try: lat[(m, st, int(i), int(o))] = json.load(open(f))["latencies"]
    except Exception: pass
for (m, st, i, o), v in lat.items():
    if o == 1: A[("vllm", m, f"pp{i}", st)] += [i / l for l in v]
    if o == 128 and (m, st, i, 1) in lat:
        l1 = med(lat[(m, st, i, 1)]); A[("vllm", m, f"tg128@{i}", st)] += [127 / (l - l1) for l in v]
rj = os.path.join(P, "sglang", "results.jsonl")
for l in (open(rj) if os.path.isfile(rj) else []):
    d = json.loads(l); rn = d["run_name"]; m = rn.split("-")[0]; kind = rn.rsplit("-", 1)[1]; st = rn[len(m) + 1:-len(kind) - 1]
    if kind == "pp": A[("sglang", m, f"pp{d['input_len']}", st)].append(d["prefill_throughput"])
for f in glob.glob(os.path.join(P, "sglang", "*-tg*.log")):
    m, st, i = re.match(r"(\S+?)-(\S+)-tg(\d+)\.log", os.path.basename(f)).groups()
    found = [float(mm.group(1)) for l in open(f, errors="replace")
             for mm in [re.search(r"Decode\.\s+median latency: [\d.]+ s, median throughput:\s+([\d.]+)", l)] if mm]
    A[("sglang", m, f"tg128@{i}", st)] += found[1:]
tl = collections.defaultdict(list); terr = []
tj = os.path.join(P, "transformers", "transformers.jsonl")
for l in (open(tj) if os.path.isfile(tj) else []):
    d = json.loads(l)
    if "error" in d: terr.append(f"{d['model']} {d['setting']} {d.get('input')}:{d.get('output')} {d['error'][:100]}"); continue
    tl[(d["model"], d["setting"], d["input"], d["output"])].append(d["seconds"])
for (m, st, i, o), v in tl.items():
    if o == 1: A[("transformers", m, f"pp{i}", st)] += [i / s for s in v]
    if o == 128 and (m, st, i, 1) in tl:
        l1 = med(tl[(m, st, i, 1)]); A[("transformers", m, f"tg128@{i}", st)] += [127 / (s - l1) for s in v]
best = {}
for (e, m, s, st), v in A.items():
    if not v: continue
    c = (med(v), st, len(v), spread(v))
    if (e, m, s) not in best or c[0] > best[(e, m, s)][0]: best[(e, m, s)] = c
COMP = ["llama-cuda", "llama-vk", "vllm", "sglang", "transformers"]
print("# A. each engine's bench (median tok/s; comparators: best of their settings; n, spread)")
for m in ["0.8B", "4B"]:
    print(f"\n## {m}\n| engine | " + " | ".join(SH) + " |\n|---|" + "---|" * len(SH))
    for e in HC + COMP:
        cells = [best.get((e, m, s)) for s in SH]
        if any(cells): print(f"| {e} | " + " | ".join(f"{c[0]:.1f} ({c[1]}, n={c[2]}, {c[3]:.1f} %)" if c else "" for c in cells) + " |")
    print("\nall settings:")
    for (e, mm, s, st), v in sorted(A.items()):
        if mm == m and v and e not in HC: print(f"  {e} {s} {st}: {med(v):.1f} (n={len(v)}, {spread(v):.1f} %)")
def ratio(a, b): return f"×{a[0] / b[0]:.3f}" if a and b else ""
print("\n# Ratios: herbert-cuda (each form) / engine, shape by shape (medians of A)")
for h in ["hc-bf16", "hc-f16", "hc-f16-acc16"]:
    for m in ["0.8B", "4B"]:
        print(f"\n## {h} {m}\n| against | " + " | ".join(SH) + " |\n|---|" + "---|" * len(SH))
        for e in HC + COMP:
            if e == h: continue
            cells = [ratio(best.get((h, m, s)), best.get((e, m, s))) for s in SH]
            if any(cells): print(f"| {e} | " + " | ".join(cells) + " |")
for t in terr: print("transformers FAILED", t)
cl = os.path.join(P, "clocks.csv")
if os.path.isfile(cl):
    rows = [[x.strip() for x in l.split(",")] for l in open(cl) if l[:4].isdigit()]
    c = collections.Counter(tuple(r[1:4]) for r in rows)
    print("\nclocks (sm, mem, pstate) over the whole step:", ", ".join(f"{k}: {n}" for k, n in c.most_common(8)))
    busy = [r for r in rows if r[4].split()[0].replace(".", "").isdigit() and float(r[4].split()[0]) > 100]
    if busy:
        sm = [int(r[1].split()[0]) for r in busy]
        print(f"samples over 100 W: {len(busy)}, SM under 2805 on {sum(s < 2805 for s in sm)} (min {min(sm)}), reasons:",
              ", ".join(f"{k}: {n}" for k, n in collections.Counter(r[7] for r in busy if int(r[1].split()[0]) < 2805).most_common(4)))
for f in sorted(glob.glob(os.path.join(P, "http", "pstate-*.txt"))):
    states = {l.strip().split(", ")[-1] for l in open(f) if l.strip()}
    if states != {"P0"}: print(f"  {os.path.basename(f)}: {sorted(states)}")
http = {}
for f in sorted(glob.glob(os.path.join(P, "http", "http-*.json"))):
    m, e = re.match(r"http-(\S+?)-(\S+)\.json", os.path.basename(f)).groups()
    try: rs = json.load(open(f))
    except Exception: continue
    for r in rs: http[(e, m, r["prompt_key"])] = r
keys = sorted({k[2] for k in http})
print("\n# B. common HTTP client --skip-blank (median prefill tok/s = input tokens / TTFT (input tokens); decode tok/s; n)")
for m in ["0.8B", "4B"]:
    print(f"\n## {m}\n| server | " + " | ".join(f"p{k} prefill · decode" for k in keys) + " |\n|---|" + "---|" * len(keys))
    for e in sorted({k[0] for k in http if k[1] == m}):
        cells = []
        for k in keys:
            r = http.get((e, m, k))
            cells.append(f"{r['prefill_tps_median']:.0f} ({'/'.join(map(str, r['input_tokens']))}) · {r['decode_tps_median']:.1f} (n={r['samples']})"
                         if r and r.get("prefill_tps_median") else "")
        print(f"| {e} | " + " | ".join(cells) + " |")
    for h in ["hc-bf16", "hc-f16-acc16"]:
        print(f"\n| {m} {h} / (prefill · decode) | " + " | ".join(f"p{k}" for k in keys) + " |\n|---|" + "---|" * len(keys))
        for e in sorted({k[0] for k in http if k[1] == m}):
            if e == h: continue
            cells = []
            for k in keys:
                a, b = http.get((h, m, k)), http.get((e, m, k))
                ok = a and b and a.get("prefill_tps_median") and b.get("prefill_tps_median")
                cells.append(f"×{a['prefill_tps_median'] / b['prefill_tps_median']:.3f} · ×{a['decode_tps_median'] / b['decode_tps_median']:.3f}" if ok else "")
            if any(cells): print(f"| {e} | " + " | ".join(cells) + " |")
