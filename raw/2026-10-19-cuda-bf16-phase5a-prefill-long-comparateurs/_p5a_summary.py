"""_p5a_summary.py PHASE5A — step 5a of phase 5: the comparators' settings sweep at 10K/20K/40K/60K (prefill only).
Per engine, model and length: the median tok/s of every setting, the settings that failed (out of memory) at that
length, the four best; then the pairs of settings by _p4a_pairs.py's criterion (geometric mean over both models and
the four lengths of max(pair) / best of all settings, and the worst cell), cells where no setting ran excluded, a
pair missing a cell that some setting covers ranked apart (its covered cells only, marked). Ties (a setting best
in every cell makes every pair holding it 1.0000) broken by the same mean taken over min(pair) / best: the partner
closest to the best as well. PHASE5A holds llama/
(r{round}-{model}-{cuda|vk}-pp-{setting}.json), vllm/, sglang-rep/ and transformers.jsonl, and phase5a.out beside it.
Step 5c (another machine, not in this release): the same on its sweep/ (r{round}-{model}-{cuda|vulkan}-pp-{setting}.json, 0.8B and 2B), llama.cpp only."""
import sys, os, re, glob, json, statistics, collections, itertools, math
P = sys.argv[1]
rows = collections.defaultdict(list)   # (engine, model, length, setting) -> [tok/s]
ran = collections.defaultdict(set)     # (engine, model, setting) -> lengths attempted
LENS = [10000, 20000, 40000, 60000]
def add(e, m, i, st, v): rows[(e, m, i, st)].append(v)
for f in glob.glob(os.path.join(P, "llama", "r*-*-pp-*.json")) + glob.glob(os.path.join(P, "sweep", "r*-*-pp-*.json")):
    r, m, be, st = re.match(r"r(\d)-(\S+?)-(cuda|vk|vulkan)-pp-(\S+)\.json", os.path.basename(f)).groups()
    be = "vk" if be == "vulkan" else be
    ran[(f"llama-{be}", m, st)] |= set(LENS)
    try: d = json.load(open(f))
    except Exception: d = []
    for t in d: add(f"llama-{be}", m, t["n_prompt"], st, t["avg_ts"])
for f in glob.glob(os.path.join(P, "vllm", "*-out1.log")):
    m, st, i = re.match(r"(\S+?)-(\S+)-in(\d+)-out1\.log", os.path.basename(f)).groups()
    ran[("vllm", m, st)].add(int(i))
    j = f[:-4] + ".json"
    if os.path.isfile(j):
        for l in json.load(open(j))["latencies"]: add("vllm", m, int(i), st, int(i) / l)
for f in glob.glob(os.path.join(P, "sglang-rep", "*-pp*.log")):
    m, st, i = re.match(r"(\S+?)-(\S+)-pp(\d+)\.log", os.path.basename(f)).groups()
    ran[("sglang", m, st)].add(int(i))
for l in (open(os.path.join(P, "sglang-rep", "results.jsonl")) if os.path.isdir(os.path.join(P, "sglang-rep")) else []):
    d = json.loads(l); m = d["run_name"].split("-", 1)[0]; st = d["run_name"][len(m) + 1:-3]
    add("sglang", m, d["input_len"], st, d["prefill_throughput"])
terr = {}
for l in (open(os.path.join(P, "transformers.jsonl")) if os.path.isfile(os.path.join(P, "transformers.jsonl")) else []):
    d = json.loads(l)
    if "input" in d: ran[("transformers", d["model"], d["setting"])].add(d["input"])
    else: ran[("transformers", d["model"], d["setting"])] |= set(LENS)
    if "error" in d: terr[(d["model"], d["setting"], d.get("input"))] = d["error"][:120]; continue
    terr.pop((d["model"], d["setting"], d["input"]), None)   # a later run (5a2) replaces an earlier failure
    add("transformers", d["model"], d["input"], d["setting"], d["input"] / d["seconds"])
med = {k: statistics.median(v) for k, v in rows.items() if v}
MODELS = sorted({k[1] for k in ran}, key=lambda m: float(m[:-1]))
engines = ["llama-cuda", "llama-vk", "vllm", "sglang", "transformers"]
print("# medians, tok/s (n samples); FAIL = attempted, no measure (out of memory, see logs)\n")
for e in [e for e in engines if any(k[0] == e for k in ran)]:
    sts = sorted({st for (ee, m, st) in ran if ee == e})
    for m in MODELS:
        print(f"== {e} {m}")
        print(f"  {'setting':<22}" + "".join(f"{i:>15}" for i in LENS))
        for st in sts:
            if (e, m, st) not in ran: continue
            cells = []
            for i in LENS:
                k = (e, m, i, st)
                if k in med: cells.append(f"{med[k]:>10.0f} ({len(rows[k])})")
                elif i in ran[(e, m, st)]: cells.append(f"{'FAIL':>15}")
                else: cells.append(f"{'-':>15}")
            print(f"  {st:<22}" + "".join(cells))
        for i in LENS:
            c = {st: v for (ee, mm, ii, st), v in med.items() if (ee, mm, ii) == (e, m, i)}
            print(f"  best {i:<6} " + "  ".join(f"{st} {c[st]:.0f}" for st in sorted(c, key=c.get, reverse=True)[:4]))
    sts = sorted({st for (ee, m, i, st) in med if ee == e})
    cells = [(m, i) for m in MODELS for i in LENS if any((e, m, i, st) in med for st in sts)]
    best = {c: max(med[(e, *c, st)] for st in sts if (e, *c, st) in med) for c in cells}
    full, part = [], []
    for a, b in itertools.combinations(sts, 2):
        r = [max(med.get((e, *c, a), 0), med.get((e, *c, b), 0)) / best[c] for c in cells]
        cov = [x for x in r if x > 0]
        if not cov: continue
        gm = math.exp(sum(map(math.log, cov)) / len(cov))
        lo = [min(med.get((e, *c, a), 0), med.get((e, *c, b), 0)) / best[c] for c in cells]
        gl = math.exp(sum(math.log(x) for x in lo if x > 0) / max(1, sum(x > 0 for x in lo)))
        (full if len(cov) == len(r) else part).append((gm, min(cov), len(r) - len(cov), a, b, gl))
    full.sort(key=lambda t: (-round(t[0], 4), -t[5])); part.sort(key=lambda t: (t[2], -t[0]))
    excl = [(m, i) for m in MODELS for i in LENS if (m, i) not in cells]
    print(f"  cells with no setting: {excl or 'none'}")
    print(f"  pairs covering every cell: " + (" | ".join(f"{a}+{b} gm {gm:.4f} min {mn:.3f} (low {gl:.4f})" for gm, mn, _, a, b, gl in full[:4]) or "none"))
    if not full:
        print(f"  pairs missing cells: " + " | ".join(f"{a}+{b} missing {k} gm {gm:.4f} min {mn:.3f}" for gm, mn, k, a, b, _ in part[:4]))
    print()
for k, v in sorted(terr.items(), key=str): print(f"transformers {k}: {v}")
