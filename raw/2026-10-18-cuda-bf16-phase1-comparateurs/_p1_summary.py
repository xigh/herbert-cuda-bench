"""_p1_summary.py PHASE1 [AFTER [AFTER_C [AFTER_E]]] — phase 1 of the #003 campaign: per engine, model and shape, the median of
every setting, its spread and the best setting, the default beside it. tok/s throughout: prefill = tokens /
time to the first token, decode = generated tokens per second at depth 759 (as each engine's bench defines
them, protocol.md). AFTER (optional) holds the vLLM rerun (--max-num-seqs 8) and the SGLang rerun
(three samples per shape), which replace the phase 1 runs of those two engines when present, and transformers'
4B settings run again one process each (transformers-4B.jsonl). AFTER_C (optional) holds vLLM with explicit CUDA
graph capture sizes, reported apart as vllm-graphs. AFTER_E (optional) holds SGLang's torch.compile attempt, added
to the SGLang rerun. transformers' default is sdpa-dynamic (attn_implementation and cache left alone)."""
import sys, os, re, glob, json, statistics, collections
P = sys.argv[1]; A = sys.argv[2] if len(sys.argv) > 2 else None; C = sys.argv[3] if len(sys.argv) > 3 else None
E = sys.argv[4] if len(sys.argv) > 4 else None
rows = collections.defaultdict(list)   # (engine, model, shape, setting) -> [tok/s]
def add(e, m, s, st, v): rows[(e, m, s, st)].append(v)
# llama.cpp
for f in glob.glob(os.path.join(P, "llama", "r*-*.json")):
    mt = re.match(r"r(\d)-(\S+?)-(cuda|vk|ref08ffba21e)-(pp|tg)-(\S+)\.json", os.path.basename(f))
    if not mt: continue
    _, m, be, kind, st = mt.groups()
    try: d = json.load(open(f))
    except Exception: continue
    for t in d:
        shape = f"pp{t['n_prompt']}" if kind == "pp" else "tg255@759"
        add(f"llama-{be}", m, shape, st, t["avg_ts"])
# vLLM: prefill = len / L1, decode = 254 / (L255 - L1)
def vllm(dirn, tag):
    lat = {}
    for f in glob.glob(os.path.join(dirn, "*.json")):
        mt = re.match(r"(\S+?)-(\S+)-in(\d+)-out(\d+)\.json", os.path.basename(f))
        if not mt: continue
        m, st, i, o = mt.groups()
        try: lat[(m, st, int(i), int(o))] = json.load(open(f))
        except Exception: pass
    for (m, st, i, o), d in lat.items():
        if o == 1 and i != 759:
            for l in d["latencies"]: add(tag, m, f"pp{i}", st, i / l)
        if o == 255 and (m, st, 759, 1) in lat:
            l1 = statistics.median(lat[(m, st, 759, 1)]["latencies"])
            for l in d["latencies"]: add(tag, m, "tg255@759", st, 254 / (l - l1))
vllm(os.path.join(P, "vllm"), "vllm-phase1")
if A and os.path.isdir(os.path.join(A, "vllm")): vllm(os.path.join(A, "vllm"), "vllm")
if C and os.path.isdir(os.path.join(C, "vllm")): vllm(os.path.join(C, "vllm"), "vllm-graphs")
# SGLang: one_batch JSONL (prefill_throughput; decode from the log's median)
def sglang(dirn, tag):
    fn = os.path.join(dirn, "results.jsonl")
    if not os.path.isfile(fn): return
    for l in open(fn):
        d = json.loads(l); m, st, kind = d["run_name"].split("-", 2)[0], d["run_name"].split("-")[1], d["run_name"].rsplit("-", 1)[1]
        st = d["run_name"][len(m) + 1:-len(kind) - 1]
        if kind == "pp": add(tag, m, f"pp{d['input_len']}", st, d["prefill_throughput"])
    for f in glob.glob(os.path.join(dirn, "*-tg.log")):
        m, st = os.path.basename(f)[:-7].split("-", 1)
        # The first run of a process is its warm-up: one_batch prints it like the measured ones.
        found = [float(mm.group(1)) for l in open(f, errors="replace")
                 for mm in [re.search(r"Decode\.\s+median latency: [\d.]+ s, median throughput:\s+([\d.]+)", l)] if mm]
        for v in found[1:]: add(tag, m, "tg255@759", st, v)
sglang(os.path.join(P, "sglang"), "sglang-phase1")
if A: sglang(os.path.join(A, "sglang-rep"), "sglang")
if E: sglang(E, "sglang")
# transformers
tfs = [f for f in [os.path.join(P, "transformers.jsonl"), A and os.path.join(A, "transformers-4B.jsonl")] if f and os.path.isfile(f)]
if tfs:
    lat = collections.defaultdict(list); errs = {}
    for l in (l for f in tfs for l in open(f)):
        d = json.loads(l)
        if "error" in d: errs[(d["model"], d["setting"])] = d["error"][:160]; continue
        errs.pop((d["model"], d["setting"]), None)
        lat[(d["model"], d["setting"], d["input"], d["output"])].append(d["seconds"])
    for (m, st, i, o), v in lat.items():
        if o == 1 and i != 759:
            for s in v: add("transformers", m, f"pp{i}", st, i / s)
        if o == 255 and (m, st, 759, 1) in lat:
            l1 = statistics.median(lat[(m, st, 759, 1)])
            for s in v: add("transformers", m, "tg255@759", st, 254 / (s - l1))
    for k, e in errs.items(): print(f"transformers {k[0]} {k[1]}: FAILED {e}")
shapes = ["pp128", "pp512", "pp2048", "pp4000", "tg255@759"]
engines = sorted({k[0] for k in rows})
for e in engines:
    for m in ["0.8B", "4B"]:
        print(f"== {e} {m}")
        for s in shapes:
            cells = {st: v for (ee, mm, ss, st), v in rows.items() if (ee, mm, ss) == (e, m, s) and v}
            if not cells: continue
            med = {st: statistics.median(v) for st, v in cells.items()}
            best = max(med, key=med.get)
            dflt = med.get("sdpa-dynamic" if e == "transformers" else "default")
            sp = lambda v: (max(v) - min(v)) / statistics.median(v) * 100 if len(v) > 1 else 0.0
            line = f"  {s:<10} best {best:<18} {med[best]:10.1f} tok/s (n={len(cells[best])}, spread {sp(cells[best]):.1f} %)"
            if dflt: line += f" · default {dflt:10.1f} (×{med[best] / dflt:.3f})"
            print(line)
        print("  all settings: " + " ".join(f"{st}" for st in sorted({k[3] for k in rows if k[0] == e and k[1] == m})))
# One table: per engine and model, the best median over every setting (vllm and vllm-graphs merged: both are
# vLLM 0.31.0; ref0.28.0 apart), the setting in brackets, then the default.
print("\n## best of all settings (default below)")
merge = lambda e, st: ("vllm-0.28.0", st) if st == "ref0.28.0" else (("vllm", st) if e.startswith("vllm") else (e, st))
best = collections.defaultdict(dict)
for (e, m, s, st), v in rows.items():
    if not v or e.endswith("-phase1"): continue
    e2, st2 = merge(e, st); med = statistics.median(v)
    cur = best[(e2, m)].get(s)
    if cur is None or med > cur[0]: best[(e2, m)][s] = (med, st2 if e != "vllm-graphs" else st2 + "*")
print("| engine | model | " + " | ".join(shapes) + " |")
print("|---|---|" + "---|" * len(shapes))
for (e2, m) in sorted(best):
    print(f"| {e2} | {m} | " + " | ".join(f"{best[(e2, m)][s][0]:.1f} ({best[(e2, m)][s][1]})" if s in best[(e2, m)] else "" for s in shapes) + " |")
    d = {s: statistics.median(v) for (e, mm, s, st), v in rows.items() if v and mm == m and merge(e, st)[0] == e2 and not e.endswith("-phase1") and e != "vllm-graphs"
         and st == ("sdpa-dynamic" if e2 == "transformers" else ("ref0.28.0" if e2 == "vllm-0.28.0" else "default"))}
    print(f"| {e2} | {m} default | " + " | ".join(f"{d[s]:.1f}" if s in d else "" for s in shapes) + " |")
print("(* : vLLM run with explicit CUDA graph capture sizes, after-p1c)")
