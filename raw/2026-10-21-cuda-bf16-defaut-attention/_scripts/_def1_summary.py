"""_def1_summary.py DEF1 — the attention defaults' token check (defaults/_scripts/_def1-003.sh): for each prompt, the
first [ids] element where each variant differs from the CPU oracle, then the pairs that must match: new (new default)
against old-sw (old build with sdpa.gqa=1 sdpa.split=64 sdpa.pipe=1), new-old (new build with sdpa.gqa=0
sdpa.split=8 sdpa.pipe=0) against old (old default); the same pairs on the long prompts (10K, 60K, 64 generated);
and the [tuning] line of one log per variant."""
import sys, os, re, glob
P = sys.argv[1]; V = ["new", "old-sw", "new-old", "old"]; PAIRS = [("new", "old-sw"), ("new-old", "old")]
def ids(f):
    if not os.path.isfile(f): return None
    for l in open(f, errors="replace"):
        if l.startswith("[ids]"): return l.split()[1:]
    return None
def first(a, b):
    if a is None or b is None: return "missing"
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y: return f"differs at {i}"
    return "identical" if len(a) == len(b) else f"prefix ({len(a)} vs {len(b)})"
bad = 0
for m in ["0.8B", "4B"]:
    D = os.path.join(P, f"compare-{m}")
    names = sorted(os.path.basename(f)[:-8] for f in glob.glob(os.path.join(D, "*-cpu.log")))
    print(f"\n## {m}, against the oracle (first differing element)\n| prompt | " + " | ".join(V) + " | " + " | ".join(f"{a} = {b}" for a, b in PAIRS) + " |\n|---|" + "---|" * (len(V) + len(PAIRS)))
    for n in names:
        o = ids(os.path.join(D, f"{n}-cpu.log")); r = {v: ids(os.path.join(D, f"{n}-cuda-0-{v}.log")) for v in V}
        pc = [first(r[a], r[b]) for a, b in PAIRS]; bad += sum(c != "identical" for c in pc)
        print(f"| {n} | " + " | ".join(first(r[v], o) for v in V) + " | " + " | ".join(pc) + " |")
    print(f"\n## {m}, long prompts (64 generated)")
    for p in ["pp10000", "pp60000"]:
        r = {v: ids(os.path.join(P, "long", f"{m}-{v}-{p}.log")) for v in V}
        pc = [first(r[a], r[b]) for a, b in PAIRS]; bad += sum(c != "identical" for c in pc)
        print(f"  {p}: " + " · ".join(f"{a} = {b}: {c}" for (a, b), c in zip(PAIRS, pc)) + f" · new against old: {first(r['new'], r['old'])}")
print("\n## [tuning] lines (0.8B p1)")
for v in V:
    f = glob.glob(os.path.join(P, "compare-0.8B", f"p1*-cuda-0-{v}.log"))
    t = [l.strip() for l in open(f[0], errors="replace") if "sdpa" in l and ("tun" in l.lower())][:1] if f else []
    print(f"  {v}: {t[0] if t else 'not found'}")
print(f"\npairs not identical: {bad}")
