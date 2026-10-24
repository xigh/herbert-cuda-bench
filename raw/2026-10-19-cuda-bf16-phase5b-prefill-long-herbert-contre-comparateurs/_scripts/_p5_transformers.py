"""_p5_transformers.py OUT MODEL SETTING CASES — transformers 5.20.1 at the long lengths (protocol.md, phase 5), one
process per model and setting: generate, greedy, batch 1, as phase1/_p1_transformers.py (attn sdpa / eager, cache
dynamic / static / static + torch.compile), in DTYPE (float16 by default here). CASES = "N:new,…" (5a: 10000:1,…;
5b: also N:128 for the decode at depth). Prompt ids: pp60000.txt of the long prompts tokenized, cut to N. Each
case: 1 warm-up, then 5 timed runs (CUDA synchronized). One JSON line per run; a failing case (out of memory) is
written as such and the next cases still run. The raw text of pp60000.txt is 59 988 tokens (the 12 template tokens
are not in it), so 60000 failed in 5a: with CASES_JSON set (herbert-cuda-bench --prepare's file), the ids are those
of its pp60000 case (template included, exactly 60 000), cut to N."""
import json, os, sys, time, traceback, torch
DT = os.environ.get("DTYPE", "float16")
from transformers import AutoTokenizer, AutoModelForCausalLM
out, m, name, cases = sys.argv[1:5]
path = f"$MODELS_DIR/Qwen3.5-{m}"
tok = AutoTokenizer.from_pretrained(path)
text = open(os.path.expanduser("~/c003/long-prompts/pp60000.txt")).read()
all_ids = tok(text, return_tensors="pt").input_ids
if os.environ.get("CASES_JSON"):
    all_ids = torch.tensor([next(c["ids"] for c in json.load(open(os.environ["CASES_JSON"])) if c["name"] == "pp60000")])
attn, cache = name.split("-", 1)
def timed(model, ids, new, **kw):
    torch.cuda.synchronize(); t = time.perf_counter()
    model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=new, min_new_tokens=new, do_sample=False, **kw)
    torch.cuda.synchronize(); return time.perf_counter() - t
with open(out, "a") as f:
    def write(rec): f.write(json.dumps(rec) + "\n"); f.flush()
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, DT), device_map="cuda", attn_implementation=attn)
    except Exception as e:
        write({"model": m, "setting": name, "error": repr(e), "trace": traceback.format_exc()[-2000:]}); raise
    kw = {"cache_implementation": "static"} if cache.startswith("static") else {}
    if cache == "static-compile":
        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)
    for c in cases.split(","):
        n, new = map(int, c.split(":"))
        try:
            ids = all_ids[:, :n].cuda(); assert ids.shape[1] == n, ids.shape
            timed(model, ids, new, **kw)
            for r in range(5):
                write({"model": m, "setting": name, "dtype": DT, "input": n, "output": new, "round": r, "seconds": timed(model, ids, new, **kw)})
            print(m, name, n, new, "done", flush=True)
        except Exception as e:
            write({"model": m, "setting": name, "input": n, "output": new, "error": repr(e), "trace": traceback.format_exc()[-2000:]})
            print(m, name, n, new, "FAIL", repr(e), flush=True)
            torch.cuda.empty_cache()
