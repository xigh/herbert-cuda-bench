"""_build_long_prompts.py OUTDIR — prompts of 10 000, 20 000, 40 000 and 60 000 tokens for the long prefill, built as
campaign-v1's build_prompts.py ($BENCH_DIR): Moby-Dick prose from "Call me Ishmael", decoded then re-encoded
until the round trip lands on the exact length. The length counts the chat template of herbert's chat binaries with
--no-system --no-think, 12 tokens (pp128.txt: 116 + 12 = 128, pp4000.txt: 3 989 + 12 = 4 001, phase 4b logs), so the
text itself is N - 12 tokens. Each prompt starts at a different offset of the book. Tokenizer: Qwen3.5-0.8B's
(the 4B's is the same file)."""
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
off = 0
for n in [10000, 20000, 40000, 60000]:
    s = exact(ids[off:], n - TPL)
    open(os.path.join(O, f"pp{n}.txt"), "w", encoding="utf-8").write(s)
    print(f"pp{n}: offset {off}, text {len(tok(s, add_special_tokens=False)['input_ids'])} tokens + {TPL} template")
    off += n
