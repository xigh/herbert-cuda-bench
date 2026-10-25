#!/usr/bin/env python3
# _art-long-summary.py CAMPAIGN_DIR — the article's long-prefill tables from _art-long-003.sh: per model, prefill tok/s
# at 2 000 … 40 000 tokens, per engine and setting.
#   herbert-cuda: the [turn 1] line, median [min..max] over the rounds;
#   llama.cpp: llama-bench's avg_ts per setting, median [min..max] over the rounds;
#   vLLM (default, FLASHINFER), TensorRT-LLM (one chunk with max_num_tokens 40960 or N, chunked by 8192): input / avg latency
#     of 5 iterations with 1 token generated, and in brackets input / median latency of the same 5;
#   SGLang: one_batch's prefill_throughput, median [min..max] of its 3 runs per length.
import glob, json, os, re, statistics, sys

C = sys.argv[1]
MODELS = ["0.8B", "2B", "4B"]
SIZES = [2000, 5000, 10000, 20000, 40000]

def fmt(xs):
    if not xs:
        return "—"
    if xs[0] == "lat":  # input / mean latency (the 128-to-4,000 measure's method), then input / median latency of the 5 iterations
        return f"{xs[1]:,.0f} (med {xs[2]:,.0f})".replace(",", " ")
    s = f"{statistics.median(xs):,.0f}".replace(",", " ")
    return s + (f" [{min(xs):.0f}..{max(xs):.0f}]" if len(xs) > 1 else "")

out = {}
for m in MODELS:
    rows = {}
    pp = {}
    for f in glob.glob(f"{C}/chat/r*-{m}-pp*.log"):
        mm = re.search(r"^\[turn 1\] (\d+) prompt tokens in [\d.]+ s \(([\d.]+) tok/s\)", open(f).read(), re.M)
        if mm:
            pp.setdefault(int(re.search(r"-pp(\d+)\.log$", f).group(1)), []).append(float(mm.group(2)))
    rows["herbert-cuda"] = pp
    for sn in ["default", "ub1024-b4096-faon"]:
        pp = {}
        for f in glob.glob(f"{C}/llama/r*-{m}-pp-{sn}.json"):
            try:
                for e in json.load(open(f)):
                    pp.setdefault(e["n_prompt"], []).append(e["avg_ts"])
            except (json.JSONDecodeError, KeyError):
                pass
        rows[f"llama.cpp {sn}"] = pp
    for eng, pat in [("vLLM default", "vllm/{m}-default-in{s}-out1.json"), ("vLLM FLASHINFER", "vllm/{m}-flashinfer-in{s}-out1.json"),
                     ("TensorRT-LLM one chunk, mnt 40960", "trtllm/{m}-mnt40960-in{s}-out1.json"),
                     ("TensorRT-LLM one chunk, mnt N", "trtllm/{m}-mntN-in{s}-out1.json"),
                     ("TensorRT-LLM chunk 8192", "trtllm/{m}-chunk8192-in{s}-out1.json")]:
        pp = {}
        for s in SIZES:
            p = f"{C}/" + pat.format(m=m, s=s)
            if os.path.exists(p):
                d = json.load(open(p))
                pp[s] = ("lat", s / d["avg_latency"], s / statistics.median(d["latencies"]))
        rows[eng] = pp
    pp = {}
    p = f"{C}/sglang/results.jsonl"
    if os.path.exists(p):
        for line in open(p):
            e = json.loads(line)
            if e["run_name"].startswith(f"{m}-"):
                pp.setdefault(e["input_len"], []).append(e["prefill_throughput"])
    rows["SGLang"] = pp
    out[m] = rows

for m, rows in out.items():
    print(f"\n### Qwen3.5-{m} (bf16)\n")
    print("| Engine | " + " | ".join(f"pp{s}" for s in SIZES) + " |")
    print("|---|" + "---|" * len(SIZES))
    for eng, pp in rows.items():
        print(f"| {eng} | " + " | ".join(fmt(pp.get(s, [])) for s in SIZES) + " |")
json.dump({m: {e: {str(s): (list(v[1:]) if v and v[0] == "lat" else v) for s, v in pp.items()} for e, pp in r.items()} for m, r in out.items()},
          open(f"{C}/summary.json", "w"), indent=1)
