"""_build_long_http_prompts.py OUT.json — the common HTTP client's prompts for the long prefill (phase 5), in the
format of $BENCH_DIR/prompts.json (default ~/bench-qwen35): per length (10 000, 20 000, 40 000, 60 000 tokens of text, the template
added by each server), five different passages of Moby-Dick from "Call me Ishmael", exact length by the decode /
re-encode round trip of build_prompts.py. Within one length the five passages do not overlap (5 x 60 000 tokens fit
in the book's 303 880), so no prefix cache could serve one from another; the servers run without prefix cache anyway."""
import sys, os, json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.path.join(os.environ.get("MODELS_DIR", os.path.expanduser("~/models")), "Qwen3.5-0.8B"))
raw = open(os.path.join(os.environ.get("BENCH_DIR", os.path.expanduser("~/bench-qwen35")), "moby.txt"), encoding="utf-8").read()
ids = tok(raw[raw.index("Call me Ishmael"):], add_special_tokens=False)["input_ids"]
def exact(src, n):
    cand = list(src[:n])
    for _ in range(16):
        back = tok(tok.decode(cand), add_special_tokens=False)["input_ids"]
        if len(back) == n: return tok.decode(cand)
        cand = src[:len(cand) - (len(back) - n)]
    raise SystemExit(f"no exact round trip for {n}")
out = {}
for n in [10000, 20000, 40000, 60000]:
    stride = (len(ids) - n) // 4
    out[str(n)] = [exact(ids[k * stride:], n) for k in range(5)]
    print(n, "stride", stride, [len(tok(p, add_special_tokens=False)["input_ids"]) for p in out[str(n)]])
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
