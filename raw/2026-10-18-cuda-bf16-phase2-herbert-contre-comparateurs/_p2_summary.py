"""_p2_summary.py PHASE2 [PHASE2B [PHASE2C]] — phase 2 of the #003 campaign (protocol.md, "Phase 2"). Per model and shape:
A. each engine's bench — the median of every setting, its spread and n; for the comparators the best of their two
   settings (shape by shape, as phase 1), for herbert-cuda its two forms (bare, candidate) and herbert-vk (default;
   host.poll=1 at decode);
B. the common HTTP client — prefill (input tokens / TTFT) and decode medians per prompt length and server;
C. the control without the P0 profile — the same engines with the profile against without;
then herbert-cuda (bare and candidate) divided by each engine, shape by shape, for the predictions P1-P9.
PHASE2B (optional): llama.cpp b11719 against b11718 (CUDA), alternated. PHASE2C (optional): the HTTP client again
with --skip-blank (transformers serve sends a blank reasoning delta before the model runs; its phase 2 HTTP numbers
are void), transformers' two servers plus two controls. tok/s throughout."""
import sys, os, re, glob, json, statistics, collections
P = sys.argv[1]; P2B = sys.argv[2] if len(sys.argv) > 2 else None; P2C = sys.argv[3] if len(sys.argv) > 3 else None
SH = ["pp128", "pp512", "pp2048", "pp4000", "tg255@759"]
med = statistics.median
def spread(v): return (max(v) - min(v)) / med(v) * 100 if len(v) > 1 else 0.0

def chat(dirn):
    """herbert chat logs r<r>-<m>-<name>-<pp…|p3>.log: [turn] prompt tok/s for pp, generated tok/s for p3."""
    rows = collections.defaultdict(list)
    for f in glob.glob(os.path.join(dirn, "r*-*.log")):
        mt = re.match(r"r(\d+)-(\S+?)-(hc-nu|hc-cand|hv-poll|hv)-(pp\d+|p3)\.log", os.path.basename(f))
        if not mt: continue
        _, m, n, k = mt.groups()
        for l in open(f, errors="replace"):
            t = re.search(r"\[turn \d+\] (\d+) prompt tokens in [\d.]+ s \(([\d.]+) tok/s\) · (\d+) generated in [\d.]+ s \(([\d.]+) tok/s\)", l)
            if t:
                if k == "p3": rows[(n, m, "tg255@759")].append(float(t.group(4)))
                else: rows[(n, m, k)].append(float(t.group(2)))
                break
    return rows

def llama(dirn, pat=r"r(\d+)-(\S+?)-(cuda|vk|b11718|b11719)-(pp|tg)-(\S+)\.json"):
    rows = collections.defaultdict(list)
    for f in glob.glob(os.path.join(dirn, "*.json")):
        mt = re.match(pat, os.path.basename(f))
        if not mt: continue
        _, m, be, kind, st = mt.groups()
        try: d = json.load(open(f))
        except Exception: continue
        for t in d:
            rows[(f"llama-{be}", m, f"pp{t['n_prompt']}" if kind == "pp" else "tg255@759", st)].append(t["avg_ts"])
    return rows

def vllm(dirn, prefix=""):
    rows = collections.defaultdict(list); lat = {}
    for f in glob.glob(os.path.join(dirn, prefix + "*.json")):
        mt = re.match(re.escape(prefix) + r"(\S+?)-(\S+)-in(\d+)-out(\d+)\.json", os.path.basename(f))
        if not mt: continue
        m, st, i, o = mt.groups()
        try: lat[(m, st, int(i), int(o))] = json.load(open(f))["latencies"]
        except Exception: pass
    for (m, st, i, o), v in lat.items():
        if o == 1 and i != 759:
            for l in v: rows[("vllm", m, f"pp{i}", st)].append(i / l)
        if o == 255 and (m, st, 759, 1) in lat:
            l1 = med(lat[(m, st, 759, 1)])
            for l in v: rows[("vllm", m, "tg255@759", st)].append(254 / (l - l1))
    return rows

def sglang(results, logs):
    rows = collections.defaultdict(list)
    if os.path.isfile(results):
        for l in open(results):
            d = json.loads(l); rn = d["run_name"]; m = rn.split("-")[0]; kind = rn.rsplit("-", 1)[1]; st = rn[len(m) + 1:-len(kind) - 1]
            if kind == "pp": rows[("sglang", m, f"pp{d['input_len']}", st)].append(d["prefill_throughput"])
    for f in glob.glob(logs):
        b = os.path.basename(f); b = b[len("sglang-"):] if b.startswith("sglang-") else b
        m, st = b[:-len("-tg.log")].split("-", 1)
        found = [float(mm.group(1)) for l in open(f, errors="replace")
                 for mm in [re.search(r"Decode\.\s+median latency: [\d.]+ s, median throughput:\s+([\d.]+)", l)] if mm]
        for v in found[1:]: rows[("sglang", m, "tg255@759", st)].append(v)
    return rows

def transformers(fn):
    rows = collections.defaultdict(list)
    if not os.path.isfile(fn): return rows
    lat = collections.defaultdict(list)
    for l in open(fn):
        d = json.loads(l)
        if "error" in d: print(f"transformers {d['model']} {d['setting']}: FAILED {d['error'][:160]}"); continue
        lat[(d["model"], d["setting"], d["input"], d["output"])].append(d["seconds"])
    for (m, st, i, o), v in lat.items():
        if o == 1 and i != 759:
            for s in v: rows[("transformers", m, f"pp{i}", st)].append(i / s)
        if o == 255 and (m, st, 759, 1) in lat:
            l1 = med(lat[(m, st, 759, 1)])
            for s in v: rows[("transformers", m, "tg255@759", st)].append(254 / (s - l1))
    return rows

# ---- A
A = collections.defaultdict(list)
for k, v in chat(os.path.join(P, "chat")).items(): A[(k[0], k[1], k[2], "-")] += v
for src in (llama(os.path.join(P, "llama")), vllm(os.path.join(P, "vllm")),
            sglang(os.path.join(P, "sglang", "results.jsonl"), os.path.join(P, "sglang", "*-tg.log")),
            transformers(os.path.join(P, "transformers", "transformers.jsonl"))):
    for k, v in src.items(): A[k] += v
best = {}   # (engine, m, shape) -> (median, setting, n, spread)
for (e, m, s, st), v in A.items():
    if not v: continue
    c = (med(v), st, len(v), spread(v))
    if (e, m, s) not in best or c[0] > best[(e, m, s)][0]: best[(e, m, s)] = c
ENG = ["hc-nu", "hc-cand", "hv", "hv-poll", "llama-cuda", "llama-vk", "vllm", "sglang", "transformers"]
print("# A. each engine's bench (median tok/s; comparators: best of their two settings; n, spread)")
for m in ["0.8B", "4B"]:
    print(f"\n## {m}\n| engine | " + " | ".join(SH) + " |\n|---|" + "---|" * len(SH))
    for e in ENG:
        cells = [best.get((e, m, s)) for s in SH]
        if not any(cells): continue
        print(f"| {e} | " + " | ".join(f"{c[0]:.1f} ({c[1]}, n={c[2]}, {c[3]:.1f} %)" if c else "" for c in cells) + " |")
    print("\nall settings:")
    for (e, mm, s, st), v in sorted(A.items()):
        if mm == m and v and e not in ("hc-nu", "hc-cand", "hv", "hv-poll"): print(f"  {e} {s} {st}: {med(v):.1f} (n={len(v)}, {spread(v):.1f} %)")

print("\n# Ratios: herbert-cuda / engine, shape by shape (medians of A)")
for h in ["hc-nu", "hc-cand"]:
    for m in ["0.8B", "4B"]:
        print(f"\n## {h} {m}\n| against | " + " | ".join(SH) + " |\n|---|" + "---|" * len(SH))
        for e in ENG:
            if e == h: continue
            cells = []
            for s in SH:
                a = best.get((h, m, s)); b = best.get((e, m, s))
                cells.append(f"×{a[0] / b[0]:.3f}" if a and b else "")
            if any(cells): print(f"| {e} | " + " | ".join(cells) + " |")

# ---- B
print("\n# B. common HTTP client (median prefill tok/s = input tokens / TTFT; decode tok/s; per prompt key)")
def http_table(dirn, title):
  http = {}
  for f in sorted(glob.glob(os.path.join(dirn, "http-*.json"))):
    mt = re.match(r"http-(\S+?)-(\S+)\.json", os.path.basename(f)); m, e = mt.groups()
    for r in json.load(open(f)): http[(e, m, r["prompt_key"])] = r
  keys = sorted({k[2] for k in http})
  print(title)
  for m in ["0.8B", "4B"]:
    print(f"\n## {m}\n| server | " + " | ".join(f"p{k} prefill · decode" for k in keys) + " |\n|---|" + "---|" * len(keys))
    for e in sorted({k[0] for k in http if k[1] == m}):
        cells = []
        for k in keys:
            r = http.get((e, m, k))
            cells.append(f"{r['prefill_tps_median']:.0f} ({r['input_tokens']}) · {r['decode_tps_median']:.1f}" if r and r["prefill_tps_median"] else "")
        print(f"| {e} | " + " | ".join(cells) + " |")
http = {}
for f in sorted(glob.glob(os.path.join(P, "http", "http-*.json"))):
    mt = re.match(r"http-(\S+?)-(\S+)\.json", os.path.basename(f)); m, e = mt.groups()
    for r in json.load(open(f)): http[(e, m, r["prompt_key"])] = r
for f in sorted(glob.glob(os.path.join(P, "http", "pstate-*.txt"))):
    states = {l.strip().split(", ")[-1] for l in open(f) if l.strip()}
    if states != {"P0"}: print(f"  {os.path.basename(f)}: {sorted(states)}")
keys = sorted({k[2] for k in http})
for m in ["0.8B", "4B"]:
    print(f"\n## {m}\n| server | " + " | ".join(f"p{k} prefill · decode" for k in keys) + " |\n|---|" + "---|" * len(keys))
    for e in sorted({k[0] for k in http if k[1] == m}):
        cells = []
        for k in keys:
            r = http.get((e, m, k))
            cells.append(f"{r['prefill_tps_median']:.0f} ({r['input_tokens']}) · {r['decode_tps_median']:.1f}" if r and r["prefill_tps_median"] else "")
        print(f"| {e} | " + " | ".join(cells) + " |")

if P2C: http_table(P2C, "\n# B'. HTTP client again with --skip-blank (transformers: these replace B; hc-cand and vllm-cg512: controls)")

# ---- C
print("\n# C. without the P0 profile (control) against with it (A)")
C = collections.defaultdict(list)
for k, v in chat(os.path.join(P, "control", "chat")).items(): C[(k[0], k[1], k[2], "-")] += v
for k, v in llama(os.path.join(P, "control")).items(): C[(k[0], k[1], k[2], k[3])] += v
for k, v in vllm(os.path.join(P, "control"), "vllm-").items(): C[k] += v
for k, v in sglang(os.path.join(P, "control", "sglang-results.jsonl"), os.path.join(P, "control", "sglang-*-tg.log")).items(): C[k] += v
for (e, m, s, st), v in sorted(C.items()):
    a = A.get((e, m, s, st))
    if v and a: print(f"  {e} {m} {s} {st}: without {med(v):.1f} (n={len(v)}) · with {med(a):.1f} (n={len(a)}) · ×{med(v) / med(a):.3f}")
cl = os.path.join(P, "clocks.csv")
if os.path.isfile(cl):
    c = collections.Counter(tuple(x.strip() for x in l.split(",")[1:4]) for l in open(cl) if l[:4].isdigit())
    print("\nclocks (sm, mem, pstate) over the whole phase:", ", ".join(f"{k}: {n}" for k, n in c.most_common(6)))

# ---- 2B
if P2B:
    print("\n# llama.cpp b11719 against b11718 (CUDA, alternated)")
    L = llama(os.path.join(P2B, "llama"))
    for m in ["0.8B", "4B"]:
        for s in SH:
            for st in sorted({k[3] for k in L if k[1] == m and k[2] == s}):
                a = L.get(("llama-b11718", m, s, st)); b = L.get(("llama-b11719", m, s, st))
                if a and b: print(f"  {m} {s} {st}: b11718 {med(a):.1f} · b11719 {med(b):.1f} (n={len(b)}) · ×{med(b) / med(a):.3f}")
