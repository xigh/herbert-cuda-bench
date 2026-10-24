"""_p1_tokens.py ENGINE MODEL SETTING KWARGS CASES OUT.jsonl — phase 1's correctness check (protocol.md): greedy
output ids of one engine, at one setting, on p1-p4, from the same prompt ids as the CPU oracle (herbert-cuda-bench
--prepare: the Qwen3.5 template, no system prompt, thinking on for p3 only). KWARGS is a JSON object passed to
the engine's constructor (vllm.LLM, sglang.Engine, from_pretrained) or, for llama, the URL of a running
llama-server. Up to 200 new tokens (256 for p3), as bench/compare.sh. One JSON line per prompt.
vLLM and SGLang start their engine in child processes (spawn): the work runs under the __main__ guard.
DTYPE=float16 in the environment runs the engines in f16 (phase 4); default bfloat16. MAXNEW=N sets the new tokens
for every prompt (phase 5: 64 after the long prompts)."""
import json, os, sys, time, traceback
DT = os.environ.get("DTYPE", "bfloat16")

def main():
    engine, model, setting, kwargs, cases, out = sys.argv[1:7]
    kw = json.loads(kwargs) if engine != "llama" else {}
    path = os.path.join(os.environ.get("MODELS_DIR", os.path.expanduser("~/models")), f"Qwen3.5-{model}")
    cases = json.load(open(cases))
    def maxnew(name): return int(os.environ["MAXNEW"]) if "MAXNEW" in os.environ else (256 if name == "p3-long" else 200)
    def write(rec):
        with open(out, "a") as f: f.write(json.dumps(rec) + "\n")
    base = {"engine": engine, "model": model, "setting": setting, "kwargs": kwargs, "dtype": DT}
    try:
        if engine == "vllm":
            from vllm import LLM, SamplingParams
            from vllm.inputs import TokensPrompt
            llm = LLM(model=path, dtype=DT, **kw)
            def gen(ids, n):
                o = llm.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(temperature=0.0, max_tokens=n), use_tqdm=False)
                return list(o[0].outputs[0].token_ids)
        elif engine == "sglang":
            import sglang
            llm = sglang.Engine(model_path=path, dtype=DT, **kw)
            def gen(ids, n):
                o = llm.generate(input_ids=ids, sampling_params={"temperature": 0.0, "max_new_tokens": n})
                if "output_ids" not in o: raise RuntimeError(f"no output_ids in {sorted(o)}")
                return list(o["output_ids"])
        elif engine == "transformers":
            import torch
            from transformers import AutoModelForCausalLM
            m = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, DT), device_map="cuda", attn_implementation=kw.get("attn", "sdpa"))
            gkw = {"cache_implementation": "static"} if kw.get("cache") == "static" else {}
            def gen(ids, n):
                t = torch.tensor([ids], device="cuda")
                o = m.generate(t, max_new_tokens=n, do_sample=False, **gkw)
                return o[0, len(ids):].tolist()
        elif engine == "llama":
            import urllib.request
            url = kwargs
            def gen(ids, n):
                body = {"prompt": ids, "n_predict": n, "temperature": 0.0, "top_k": 1, "samplers": ["top_k"], "cache_prompt": False, "return_tokens": True}
                req = urllib.request.Request(url + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
                o = json.load(urllib.request.urlopen(req, timeout=600))
                if "tokens" not in o: raise RuntimeError(f"no tokens in {sorted(o)}")
                return list(o["tokens"])
        else:
            raise SystemExit(f"unknown engine {engine}")
        for c in cases:
            t = time.perf_counter()
            got = gen(c["ids"], maxnew(c["name"]))
            write({**base, "prompt": c["name"], "prompt_len": len(c["ids"]), "ids": got, "seconds": time.perf_counter() - t})
            print(engine, model, setting, c["name"], len(got), "tokens", flush=True)
    except Exception as e:
        write({**base, "error": repr(e), "trace": traceback.format_exc()[-3000:]})
        print(engine, model, setting, "FAIL", repr(e), flush=True)
        raise
    finally:
        if engine == "sglang" and "llm" in locals(): llm.shutdown()

if __name__ == "__main__":
    main()
