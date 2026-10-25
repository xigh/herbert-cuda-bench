"""_build_pp_prompts.py OUTDIR N... — prompts of N tokens for the article's long prefill, built as phase 5's
_build_long_prompts.py (itself campaign-v1's build_prompts.py): Moby-Dick prose from "Call me Ishmael", decoded then
re-encoded until the round trip lands on the exact length, the length counting the 12 tokens of herbert's chat template
with --no-system --no-think (text = N - 12 tokens). phase 5's pp10000 … pp60000 take the first 130 000 tokens of the
book; these start at offset 130 000, one after the other. Tokenizer: Qwen3.5-0.8B's (the 2B's and 4B's are the same)."""
import sys, os
from transformers import AutoTokenizer
O = sys.argv[1]; os.makedirs(O, exist_ok=True)
tok = AutoTokenizer.from_pretrained("$MODELS_DIR/Qwen3.5-0.8B")
raw = open("$BENCH_DIR/moby.txt", encoding="utf-8").read()
ids = tok(raw[raw.index("Call me Ishmael"):], add_special_tokens=False)["input_ids"]
print("corpus tokens:", len(ids))
TPL = 12
def exact(src, n):
    cand = list(src[:n])
    for _ in range(16):
        back = tok(tok.decode(cand), add_special_tokens=False)["input_ids"]
        if len(back) == n: return tok.decode(cand)
        cand = src[:len(cand) - (len(back) - n)]
    raise SystemExit(f"no exact round trip for {n}")
off = 130000
for n in map(int, sys.argv[2:]):
    s = exact(ids[off:], n - TPL)
    open(os.path.join(O, f"pp{n}.txt"), "w", encoding="utf-8").write(s)
    print(f"pp{n}: offset {off}, text {len(tok(s, add_special_tokens=False)['input_ids'])} tokens + {TPL} template")
    off += n
