"""_p1_transformers.py OUT — transformers 5.20.1 settings sweep (protocol.md, phase 1): generate, greedy, batch 1,
attn_implementation sdpa / eager, dynamic cache / static cache / static cache + torch.compile; prompts of exactly
128, 512, 2048 and 4000 tokens (the pp fixtures tokenized, cut) with 1 new token, and 759 with 255. Each case:
1 warm-up, then 5 timed runs (CUDA synchronized); prefill = len / L1, decode = 254 / (L255 - L1) as for vLLM.
One JSON line per case and run; a failing setting is written as such. Optional MODEL and SETTING arguments
restrict the run to one model and one setting (one process per setting frees the GPU between settings).
DTYPE=float16 in the environment loads the model in f16 (phase 4); default bfloat16."""
import json, os, sys, time, traceback, statistics, torch
DT = os.environ.get("DTYPE", "bfloat16")
from transformers import AutoTokenizer, AutoModelForCausalLM
out = sys.argv[1]
only_m = sys.argv[2] if len(sys.argv) > 2 else None
only_s = sys.argv[3] if len(sys.argv) > 3 else None
P = os.path.expanduser("~/herbert-cuda-mma/bench/prompts")
def ids_of(tok, n):
    text = open(f"{P}/pp4000.txt").read() + "\n" + open(f"{P}/p4-xlong.txt").read()
    ids = tok(text, return_tensors="pt").input_ids[:, :n]
    assert ids.shape[1] == n, ids.shape
    return ids.cuda()
def timed(model, ids, new, **kw):
    torch.cuda.synchronize(); t = time.perf_counter()
    model.generate(ids, max_new_tokens=new, min_new_tokens=new, do_sample=False, **kw)
    torch.cuda.synchronize(); return time.perf_counter() - t
with open(out, "a") as f:
    for m in ["0.8B", "4B"]:
        if only_m and m != only_m: continue
        path = os.path.join(os.environ.get("MODELS_DIR", os.path.expanduser("~/models")), f"Qwen3.5-{m}")
        tok = AutoTokenizer.from_pretrained(path)
        for attn in ["sdpa", "eager"]:
            for cache in ["dynamic", "static", "static-compile"]:
                name = f"{attn}-{cache}"
                if only_s and name != only_s: continue
                try:
                    model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, DT), device_map="cuda", attn_implementation=attn)
                    kw = {}
                    if cache.startswith("static"):
                        kw["cache_implementation"] = "static"
                    if cache == "static-compile":
                        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)
                    for n, new in [(128, 1), (512, 1), (2048, 1), (4000, 1), (759, 1), (759, 255)]:
                        ids = ids_of(tok, n)
                        timed(model, ids, new, **kw)
                        for r in range(5):
                            s = timed(model, ids, new, **kw)
                            f.write(json.dumps({"model": m, "setting": name, "dtype": DT, "input": n, "output": new, "round": r, "seconds": s}) + "\n"); f.flush()
                        print(m, name, n, new, "done", flush=True)
                    del model; torch.cuda.empty_cache()
                except Exception as e:
                    f.write(json.dumps({"model": m, "setting": name, "error": repr(e), "trace": traceback.format_exc()[-2000:]}) + "\n"); f.flush()
                    print(m, name, "FAIL", repr(e), flush=True)
                    torch.cuda.empty_cache()
