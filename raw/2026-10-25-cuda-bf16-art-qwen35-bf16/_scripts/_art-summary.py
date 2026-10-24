#!/usr/bin/env python3
# _art-summary.py CAMPAIGN_DIR — the article tables from _art-003.sh: per model, prefill tok/s at each prompt size and
# decode tok/s on p3 (759 + 255), per engine.
#   herbert-cuda: the [turn 1] line, median [min..max] over the rounds;
#   llama.cpp: llama-bench's avg_ts per setting (default, ub1024-b4096-faon; tg default, faon), median over the rounds;
#   vLLM, TensorRT-LLM: avg latency L of 5 iterations; prefill = input / L(out 1); decode = 254 / (L(759→255) − L(759→1));
#   SGLang: one_batch's prefill_throughput (median of its 3 runs per size) and median_decode_throughput (median of 3).
import glob, json, os, re, statistics, sys

C = sys.argv[1]
MODELS = ["0.8B", "2B", "4B"]
SIZES = [128, 256, 512, 1024, 2048, 4000]
med = lambda xs: statistics.median(xs) if xs else None

def fmt(v, spread=None):
    if v is None:
        return "—"
    s = f"{v:,.0f}".replace(",", " ") if v >= 1000 else f"{v:.1f}"
    if spread:
        lo, hi = spread
        s += f" [{lo:.0f}..{hi:.0f}]" if v >= 1000 else f" [{lo:.1f}..{hi:.1f}]"
    return s

out = {}
for m in MODELS:
    rows = {}
    # herbert-cuda
    pp, tg = {}, []
    for f in glob.glob(f"{C}/chat/r*-{m}-*.log"):
        t = open(f).read()
        mm = re.search(r"^\[turn 1\] (\d+) prompt tokens in [\d.]+ s \(([\d.]+) tok/s\) · (\d+) generated in [\d.]+ s \(([\d.]+) tok/s\)", t, re.M)
        if not mm:
            continue
        k = re.search(r"-(pp\d+|p3)\.log$", f).group(1)
        if k == "p3":
            tg.append(float(mm.group(4)))
        else:
            pp.setdefault(int(k[2:]), []).append(float(mm.group(2)))
    rows["herbert-cuda"] = ({s: pp.get(s, []) for s in SIZES}, tg)
    # llama.cpp
    for sn in ["default", "ub1024-b4096-faon"]:
        pp = {}
        for f in glob.glob(f"{C}/llama/r*-{m}-pp-{sn}.json"):
            for e in json.load(open(f)):
                pp.setdefault(e["n_prompt"], []).append(e["avg_ts"])
        rows[f"llama.cpp {sn}"] = ({s: pp.get(s, []) for s in SIZES}, None)
    for sn in ["default", "faon"]:
        tg = []
        for f in glob.glob(f"{C}/llama/r*-{m}-tg-{sn}.json"):
            for e in json.load(open(f)):
                tg.append(e["avg_ts"])
        key = "llama.cpp default" if sn == "default" else "llama.cpp ub1024-b4096-faon"
        rows[key] = (rows[key][0], tg)
    # vLLM and TensorRT-LLM
    for eng, d in [("vLLM", "vllm"), ("TensorRT-LLM", "trtllm")]:
        lat = lambda i, o: (json.load(open(p))["avg_latency"] if os.path.exists(p := f"{C}/{d}/{m}-in{i}-out{o}.json") else None)
        pp = {s: ([s / lat(s, 1)] if lat(s, 1) else []) for s in SIZES}
        l1, l255 = lat(759, 1), lat(759, 255)
        rows[eng] = (pp, [254 / (l255 - l1)] if l1 and l255 else [])
    # SGLang
    pp, tg = {}, []
    p = f"{C}/sglang/results.jsonl"
    if os.path.exists(p):
        for line in open(p):
            e = json.loads(line)
            if e["run_name"] == f"{m}-pp":
                pp.setdefault(e["input_len"], []).append(e["prefill_throughput"])
            elif e["run_name"] == f"{m}-tg":
                tg.append(e["median_decode_throughput"])
    rows["SGLang"] = ({s: pp.get(s, []) for s in SIZES}, tg)
    out[m] = rows

for m, rows in out.items():
    print(f"\n### Qwen3.5-{m} (bf16)\n")
    print("| Engine | " + " | ".join(f"pp{s}" for s in SIZES) + " | decode p3 |")
    print("|---|" + "---|" * (len(SIZES) + 1))
    for eng, (pp, tg) in rows.items():
        cells = []
        for s in SIZES:
            xs = pp.get(s, [])
            cells.append(fmt(med(xs), (min(xs), max(xs)) if len(xs) > 1 else None))
        tgc = fmt(med(tg), (min(tg), max(tg)) if tg and len(tg) > 1 else None) if tg is not None else "—"
        print(f"| {eng} | " + " | ".join(cells) + f" | {tgc} |")
json.dump({m: {e: {"pp": {str(s): v for s, v in pp.items()}, "tg": tg} for e, (pp, tg) in r.items()} for m, r in out.items()},
          open(f"{C}/summary.json", "w"), indent=1)
