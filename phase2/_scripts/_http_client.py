"""The common client of herbert-gpu-bench campaign-v1 (_scripts/_v2_http.py), with one addition:
--api openai speaks /v1/chat/completions (streamed, usage in the last chunk) for the servers without the
Messages API (SGLang); the thinking block is opened as for the others (chat_template_kwargs). The
timings below are defined the same way for both APIs.

One client, the same requests to every engine: the Anthropic Messages API (/v1/messages),
streamed, greedy, one request at a time, no prompt reuse (every server started without prefix
cache; five different passages per length). Times seen from the client:
  TTFT    = first non-empty content_block_delta (thinking or text) - request sent
  prefill = input_tokens (the server's own count, message_start) / TTFT
  decode  = (output_tokens - 1) / (last delta - first delta), output_tokens from message_delta
Prompts: $BENCH_DIR/prompts.json (default ~/bench-qwen35/prompts.json) (the passages of http_bench.py), each sent as the user message.
Usage: _v2_http.py --url URL --engine NAME --model NAME --out FILE.json [--n-gen 128] [--rounds 3]
--skip-blank (added for #003 phase 2): deltas made only of whitespace are not taken as the first token. transformers
serve 5.20.1 sends a reasoning_content "\n" 10 ms after the request, before the model has run (4B, 4 096 tokens:
"\n" at 9.9 ms, first model token at 503 ms, phase2/http-check); without this option its TTFT and decode are void.
"""
import argparse, json, os, statistics, time
import requests

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--engine", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--n-gen", type=int, default=128)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--prompts", default=os.path.join(os.environ.get("BENCH_DIR", os.path.expanduser("~/bench-qwen35")), "prompts.json"))
ap.add_argument("--out", required=True)
ap.add_argument("--api", choices=["messages", "openai"], default="messages")
ap.add_argument("--skip-blank", action="store_true")
args = ap.parse_args()
prompts = json.load(open(args.prompts))

def one_openai(text, n_gen):
    body = {"model": args.model, "max_tokens": n_gen, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": True},
            "messages": [{"role": "user", "content": text}]}
    first = last = None
    inp = out = None
    t0 = time.perf_counter()
    with requests.post(args.url, json=body, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            for c in d.get("choices") or []:
                delta = c.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                if piece and not (args.skip_blank and first is None and not piece.strip()):
                    now = time.perf_counter()
                    first = first or now
                    last = now
            u = d.get("usage")
            if u:
                inp = u.get("prompt_tokens", inp)
                out = u.get("completion_tokens", out)
    return t0, first, last, inp, out

def one(text, n_gen):
    if args.api == "openai":
        return one_openai(text, n_gen)
    body = {"model": args.model, "max_tokens": n_gen, "temperature": 0.0, "stream": True,
            "messages": [{"role": "user", "content": text}]}
    if args.engine.startswith("llama"):
        body["cache_prompt"] = False
    first = last = None
    inp = out = None
    t0 = time.perf_counter()
    with requests.post(args.url, json=body, stream=True, timeout=600, headers={"anthropic-version": "2023-06-01"}) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            d = json.loads(line[5:].strip())
            t = d.get("type")
            if t == "message_start":
                inp = (d.get("message", {}).get("usage") or {}).get("input_tokens")
            elif t == "content_block_delta":
                delta = d.get("delta", {})
                piece = delta.get("text") or delta.get("thinking")
                if piece and not (args.skip_blank and first is None and not piece.strip()):
                    now = time.perf_counter()
                    first = first or now
                    last = now
            elif t == "message_delta":
                out = (d.get("usage") or {}).get("output_tokens")
                inp = inp or (d.get("usage") or {}).get("input_tokens")
            elif t == "error":
                raise RuntimeError(d)
    return t0, first, last, inp, out

rows = []
for n in sorted(prompts, key=int):
    passages = prompts[n]
    one(passages[0], 8)  # warm-up, discarded
    ttft, pf, dec, inps, outs = [], [], [], set(), set()
    for _ in range(args.rounds):
        for p in passages:
            t0, first, last, inp, out = one(p, args.n_gen)
            if first is None or not inp:
                continue
            ttft.append((first - t0) * 1000)
            pf.append(inp / (first - t0))
            inps.add(inp); outs.add(out)
            if out and out > 1 and last > first:
                dec.append((out - 1) / (last - first))
    med = lambda v: statistics.median(v) if v else None
    sp = lambda v: (max(v) - min(v)) / statistics.median(v) * 100 if len(v) > 1 else 0.0
    row = {"skip_blank": args.skip_blank, "engine": args.engine, "model": args.model, "prompt_key": int(n), "samples": len(ttft),
           "input_tokens": sorted(inps), "output_tokens": sorted(x for x in outs if x is not None),
           "ttft_ms_median": med(ttft), "ttft_ms_all": ttft, "prefill_tps_median": med(pf), "prefill_spread_pct": sp(pf),
           "decode_tps_median": med(dec), "decode_spread_pct": sp(dec), "decode_tps_all": dec}
    rows.append(row)
    print(f"{args.engine} {args.model} p{n:>5}: in {row['input_tokens']} · TTFT {row['ttft_ms_median']:.2f} ms · prefill {row['prefill_tps_median']:.0f} tok/s ({row['prefill_spread_pct']:.1f} %) · decode {row['decode_tps_median']:.2f} tok/s ({row['decode_spread_pct']:.1f} %) · out {row['output_tokens']} · n={len(ttft)}", flush=True)
json.dump(rows, open(args.out, "w"), indent=2)
