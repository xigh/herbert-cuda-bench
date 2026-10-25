#!/usr/bin/env python3
# _dec-long-summary.py DIR — _dec-long-003.sh: per model, decode tok/s of 128 tokens after 10 000, 20 000 and 40 000
# tokens of prompt, per engine and setting.
#   herbert-cuda: the [turn 1] line's generated rate (the 127 tokens after the first), median [min..max] over the rounds,
#     with the generated count (it must be 127);
#   llama.cpp: llama-bench tg128 at depth N (avg_ts), median [min..max] over the rounds;
#   vLLM, TensorRT-LLM: 127 / (L128 - L1), L the mean latency of 5 iterations; in brackets the same with medians;
#   SGLang: one_batch median_decode_throughput, median [min..max] of its 3 runs.
import glob, json, os, re, statistics, sys

C = sys.argv[1]
MODELS = ["0.8B", "2B", "4B"]
SIZES = [10000, 20000, 40000]
med = statistics.median

def fmt(xs):
    if not xs:
        return "—"
    if xs[0] == "lat":
        return f"{xs[1]:.1f} (méd. {xs[2]:.1f})"
    return f"{med(xs):.1f}" + (f" [{min(xs):.1f}..{max(xs):.1f}]" if len(xs) > 1 else "")

out, gen = {}, {}
for m in MODELS:
    rows = {}
    tg = {}
    for f in glob.glob(f"{C}/chat/r*-{m}-pp*.log"):
        mm = re.search(r"^\[turn 1\] (\d+) prompt tokens in [\d.]+ s \(([\d.]+) tok/s\) · (\d+) generated in [\d.]+ s \(([\d.]+) tok/s\)", open(f).read(), re.M)
        if mm:
            s = int(re.search(r"-pp(\d+)\.log$", f).group(1))
            tg.setdefault(s, []).append(float(mm.group(4)))
            gen.setdefault((m, s), set()).add(int(mm.group(3)))
    rows["herbert-cuda"] = tg
    for sn, name in [("default", "llama.cpp défaut"), ("faon", "llama.cpp -fa on")]:
        tg = {}
        for f in glob.glob(f"{C}/llama/r*-{m}-tg-{sn}.json"):
            try:
                for e in json.load(open(f)):
                    tg.setdefault(e["n_depth"], []).append(e["avg_ts"])
            except (json.JSONDecodeError, KeyError):
                pass
        rows[name] = tg
    for name, d, sn in [("vLLM défaut", "vllm", "default"), ("vLLM FLASHINFER", "vllm", "flashinfer"),
                        ("TensorRT-LLM blocs de 8192", "trtllm", "chunk8192"), ("TensorRT-LLM un bloc, mnt N", "trtllm", "mntN")]:
        tg = {}
        for s in SIZES:
            p1, p128 = (f"{C}/{d}/{m}-{sn}-in{s}-out{o}.json" for o in (1, 128))
            if os.path.exists(p1) and os.path.exists(p128):
                a, b = json.load(open(p1)), json.load(open(p128))
                tg[s] = ("lat", 127 / (b["avg_latency"] - a["avg_latency"]), 127 / (med(b["latencies"]) - med(a["latencies"])))
        rows[name] = tg
    tg = {}
    p = f"{C}/sglang/results.jsonl"
    if os.path.exists(p):
        for line in open(p):
            e = json.loads(line)
            if e["run_name"] == f"{m}-tg":
                tg.setdefault(e["input_len"], []).append(e["median_decode_throughput"])
    rows["SGLang"] = tg
    out[m] = rows

for m, rows in out.items():
    print(f"\n### Qwen3.5-{m} (bf16), decode of 128 tokens after N tokens of prompt (tok/s)\n")
    print("| Engine | " + " | ".join(f"after {s}" for s in SIZES) + " |")
    print("|---|" + "---|" * len(SIZES))
    for eng, tg in rows.items():
        print(f"| {eng} | " + " | ".join(fmt(tg.get(s, [])) for s in SIZES) + " |")
print("\nherbert generated counts:", {f"{m} {s}": sorted(v) for (m, s), v in sorted(gen.items())})
json.dump({m: {e: {str(s): (list(v[1:]) if v and v[0] == "lat" else v) for s, v in tg.items()} for e, tg in r.items()} for m, r in out.items()},
          open(f"{C}/summary.json", "w"), indent=1)
