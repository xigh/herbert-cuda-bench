"""_dd_summary.py DIR — the decode-depth campaigns under DIR (dd1 … dd6):
A. dd1/a (0.8B) and dd3/a (4B), sdpa.split=8, --profile: per depth, the decode step (GPU median), sdpa_split and
   sdpa_combine, the rest of the step, the K and V bytes read once per step (kv_heads × 256 × K and V × bf16 ×
   attention layers × the median position, prompt + 32) and the bandwidth that implies; the prefill block and its
   sdpa_flash share.
B. Decode tok/s per variant and shape (graph on; median of N, spread, ratio to the first variant), and the classes of
   identical [ids] per shape with the first index that differs from the first variant: dd1/b and dd2b/b (0.8B,
   sdpa.split; dd2 left out: its rounds 2 and 3 ran while other tests held the 4080; in dd2b a run is kept
   only if apps.txt shows the resident service alone on the GPU at its launch), dd3/b (4B), dd5/b (0.8B) and dd6/b
   (4B), the old kernel against sdpa.gqa=1.
C. --profile at 60K per variant (dd1/c, dd3/c, dd5/c, dd6/c).
D. Nsight Compute (dd4): the main counters of each captured launch.
E. Tokens against the CPU oracle, p1..p4 (dd5, dd6: compare.out of bench/compare.sh, kept verbatim)."""
import sys, os, re, glob, statistics, collections
D = sys.argv[1]; med = statistics.median
MODEL = {"dd1": (2, 6), "dd2": (2, 6), "dd2b": (2, 6), "dd3": (4, 8), "dd5": (2, 6), "dd6": (4, 8)}   # kv heads, attention layers
def prof(f):
    out, sect = {"pre": {}, "dec": {}}, None
    for l in open(f, errors="replace"):
        m = re.search(r"\[profile\] (\d+) decode steps: GPU ([\d.]+) µs per step", l)
        if m: sect = "dec"; out["step"] = float(m.group(2)); continue
        m = re.search(r"\[profile\] (\d+) prefill blocks: GPU ([\d.]+) µs per block", l)
        if m: sect = "pre"; out["block"] = float(m.group(2)); out["blocks"] = int(m.group(1)); continue
        m = re.search(r"\[profile\]\s+([\d.]+) µs\s+([\d.]+) %\s+×(\d+)\s+(\S+)", l)
        if m and sect: out[sect][m.group(4)] = (float(m.group(1)), float(m.group(2)))
        m = re.search(r"\[turn 1\] (\d+) prompt", l)
        if m: out["prompt"] = int(m.group(1))
    return out
def pick(d, *prefixes):
    return next((v for k, v in d.items() if k.startswith(prefixes)), (0.0, 0.0))
def kv_bytes(dd, prompt):
    kvh, layers = MODEL[dd]
    return (prompt + 32) * kvh * 256 * 2 * 2 * layers
print("# A. decode step and attention by depth (sdpa.split=8, --profile)")
for dd, name in (("dd1", "0.8B"), ("dd3", "4B")):
    print(f"## {name} ({dd}/a)")
    for f in sorted(glob.glob(os.path.join(D, dd, "a", "profile-pp*.log")), key=lambda f: int(re.findall(r"\d+", f)[-1])):
        p = prof(f); sp = pick(p["dec"], "sdpa_split"); cb = pick(p["dec"], "sdpa_combine"); fl = pick(p["pre"], "sdpa_flash")
        kv = kv_bytes(dd, p["prompt"])
        print(f"  {p['prompt']:>6}: step {p['step']:.1f} µs · sdpa_split {sp[0]:.1f} µs ({sp[1]:.1f} %) · combine {cb[0]:.1f} µs · rest {p['step'] - sp[0] - cb[0]:.1f} µs"
              f" · KV once {kv / 1e6:.0f} MB → {kv / sp[0] / 1e3:.0f} GB/s · prefill {p['blocks']} blocks, {p['block']:.0f} µs per block, sdpa_flash {fl[0]:.0f} µs ({fl[1]:.1f} %)")
skip = set()
ap = os.path.join(D, "dd2b", "apps.txt")
if os.path.isfile(ap):
    S = [8, 16, 32, 64, 128]
    order = [(r, s, sh) for r in (1, 2, 3) for s in (S if r % 2 else S[::-1]) for sh in ("p3", "pp20000", "pp40000")]
    for (r, s, sh), l in zip(order, open(ap)):
        if l.count(";") != 1: skip.add(os.path.join(D, "dd2b", "b", f"r{r}-s{s}-{sh}.log"))
def speeds(files, first):
    tps = collections.defaultdict(list); ids = collections.defaultdict(dict)
    for f in files:
        if f in skip: continue
        r, n, shape = re.match(r"r(\d)-(\w+?)-(p3|pp\d+)\.log", os.path.basename(f)).groups()
        for l in open(f, errors="replace"):
            m = re.search(r"\[turn 1\].*generated in [\d.]+ s \(([\d.]+) tok/s\)", l)
            if m: tps[(shape, n)].append(float(m.group(1)))
            if l.startswith("[ids]"): ids[shape][(n, int(r))] = l.split()[1:]
    names = sorted({k[1] for k in tps}, key=lambda n: (n != first, re.sub(r"\d+", "", n), int(re.sub(r"\D", "", n) or 0)))
    for shape in sorted({k[0] for k in tps}, key=lambda x: (x != "p3", int(re.sub(r"\D", "", x) or 0))):
        b = med(tps[(shape, first)]) if tps.get((shape, first)) else None
        print(f"  {shape}: " + " · ".join(f"{n} {med(v):.1f} (n={len(v)}, {(max(v) - min(v)) / med(v) * 100:.1f} %" + (f", ×{med(v) / b:.3f})" if b else ")")
                                         for n in names for v in [tps.get((shape, n))] if v))
        cls = collections.defaultdict(list)
        for (n, r), v in sorted(ids[shape].items()): cls[tuple(v)].append(f"{n}r{r}")
        ref = next((k for k, v in cls.items() if any(x.startswith(first + "r") for x in v)), None)
        for k, v in cls.items():
            fd = next((i for i in range(min(len(k), len(ref))) if k[i] != ref[i]), None) if ref else "?"
            print(f"    ids {' '.join(v)}: first difference against {first} at {fd}")
print("\n# B. decode tok/s per variant (graph on; median, n, spread, ratio to the first variant)")
print(f"## 0.8B, sdpa.split (dd1/b, dd2b/b; dd2b runs left out: {len(skip)})")
speeds(glob.glob(os.path.join(D, "dd1", "b", "r*.log")) + glob.glob(os.path.join(D, "dd2b", "b", "r*.log")), "s8")
for dd, name, first in (("dd3", "4B, sdpa.split", "s8"), ("dd5", "0.8B, old kernel against sdpa.gqa=1", "old8"), ("dd6", "4B, old kernel against sdpa.gqa=1", "old8")):
    if os.path.isdir(os.path.join(D, dd, "b")):
        print(f"## {name} ({dd}/b)"); speeds(glob.glob(os.path.join(D, dd, "b", "r*.log")), first)
print("\n# C. 60K, --profile per variant")
for dd in ("dd1", "dd3", "dd5", "dd6"):
    for f in sorted(glob.glob(os.path.join(D, dd, "c", "profile-*-pp60000.log"))):
        p = prof(f); v = re.match(r"profile-(\w+?)-pp60000\.log", os.path.basename(f)).group(1)
        at = pick(p["dec"], "sdpa_split", "sdpa_gqa"); cb = pick(p["dec"], "sdpa_combine")
        print(f"  {dd} {v:>7}: step {p['step']:.1f} µs · attention {at[0]:.1f} µs ({kv_bytes(dd, p['prompt']) / at[0] / 1e3:.0f} GB/s if KV read once) · combine {cb[0]:.1f} µs")
print("\n# D. Nsight Compute (dd4, 0.8B, 60K)")
KEYS = ["Duration", "DRAM Throughput", "Memory Throughput", "L2 Cache Throughput", "Compute (SM) Throughput", "Registers Per Thread", "Dynamic Shared Memory Per Block",
        "Static Shared Memory Per Block", "Theoretical Occupancy", "Achieved Occupancy", "Active Warps Per Scheduler", "Eligible Warps Per Scheduler", "No Eligible", "Issue Slots Busy"]
for f in sorted(glob.glob(os.path.join(D, "dd4", "*.details.txt"))):
    head = None
    for l in open(f, errors="replace"):
        if re.match(r"\s{2}(sdpa_\w+)", l) and "Context" in l:
            if head: break
            head = l.strip(); print(f"  {os.path.basename(f)}: {head}")
        elif head and any(l.strip().startswith(k) for k in KEYS):
            print("     " + " ".join(l.split()))
    for l in open(f, errors="replace"):
        if "is the highest-utilized pipeline" in l: print("     " + " ".join(l.split())); break
print("\n# E. tokens against the CPU oracle (compare.out)")
for dd in ("dd5", "dd6"):
    f = os.path.join(D, dd, "compare.out")
    if os.path.isfile(f):
        print(f"## {dd}"); print("".join("  " + l for l in open(f) if not l.rstrip().endswith("exit 0")))

def turn_values(files):
    """(shape, variant) -> {prefill: [...], decode: [...]} and (shape) -> {(variant, round): ids}."""
    v = collections.defaultdict(lambda: {"prefill": [], "decode": []}); ids = collections.defaultdict(dict)
    for f in files:
        r, n, shape = re.match(r"r(\d)-(\w+?)-(p3|pp\d+)\.log", os.path.basename(f)).groups()
        for l in open(f, errors="replace"):
            m = re.search(r"\[turn 1\] (\d+) prompt tokens in [\d.]+ s \(([\d.]+) tok/s\) · (\d+) generated in [\d.]+ s \(([\d.]+) tok/s\)", l)
            if m: v[(shape, n)]["prefill"].append(float(m.group(2))); v[(shape, n)]["decode"].append(float(m.group(4)))
            if l.startswith("[ids]"): ids[shape][(n, int(r))] = l.split()[1:]
    return v, ids
def table(v, ids, first, kinds):
    names = sorted({k[1] for k in v}, key=lambda n: (n != first, n))
    for shape in sorted({k[0] for k in v}, key=lambda x: (x != "p3", int(re.sub(r"\D", "", x) or 0))):
        for kind in kinds:
            b = v.get((shape, first), {}).get(kind)
            cells = []
            for n in names:
                x = v.get((shape, n), {}).get(kind)
                if x: cells.append(f"{n} {med(x):.1f} (n={len(x)}, {(max(x) - min(x)) / med(x) * 100:.1f} %" + (f", ×{med(x) / med(b):.3f})" if b else ")"))
            print(f"  {shape} {kind}: " + " · ".join(cells))
        cls = collections.defaultdict(list)
        for (n, r), s in sorted(ids[shape].items()): cls[tuple(s)].append(f"{n}r{r}")
        ref = next((k for k, s in cls.items() if any(x.startswith(first + "r") for x in s)), None)
        for k, s in cls.items():
            fd = next((i for i in range(min(len(k), len(ref))) if k[i] != ref[i]), None) if ref else "?"
            print(f"    ids {' '.join(s)}: first difference against {first} at {fd}")
if os.path.isdir(os.path.join(D, "dd7", "b")):
    print("\n# F. dd7 (0.8B, a build with one extra instruction in the default sdpa_flash, see the README): default against sdpa.pipe=1, prefill tok/s")
    v, ids = turn_values(glob.glob(os.path.join(D, "dd7", "b", "r*.log"))); table(v, ids, "default", ["prefill"])
    for f in sorted(glob.glob(os.path.join(D, "dd7", "c", "profile-*-pp60000.log"))):
        p = prof(f); fl = pick(p["pre"], "sdpa_flash")
        print(f"  60K --profile {os.path.basename(f)}: {p['blocks']} blocks, {p['block']:.0f} µs per block, sdpa_flash {fl[0]:.0f} µs ({fl[1]:.1f} %)")
    f = os.path.join(D, "dd7", "compare.out")
    if os.path.isfile(f): print("  tokens (compare.out):\n" + "".join("    " + l for l in open(f) if not l.rstrip().endswith("exit 0")))
for m in ("0.8B", "4B"):
    b = os.path.join(D, "dd8", m)
    if os.path.isdir(os.path.join(b, "b")):
        print(f"\n# G. dd8 {m} (73b5a70): default, gqa (sdpa.gqa=1 split 64), pipe (sdpa.pipe=1), both; prefill and decode tok/s")
        v, ids = turn_values(glob.glob(os.path.join(b, "b", "r*.log"))); table(v, ids, "default", ["prefill", "decode"])
        for f in sorted(glob.glob(os.path.join(b, "c", "profile-*-pp60000.log"))):
            p = prof(f); fl = pick(p["pre"], "sdpa_flash"); at = pick(p["dec"], "sdpa_split", "sdpa_gqa")
            print(f"  60K --profile {os.path.basename(f)}: prefill {p['block']:.0f} µs per block, sdpa_flash {fl[0]:.0f} µs ({fl[1]:.1f} %) · decode step {p['step']:.0f} µs, attention {at[0]:.0f} µs ({at[1]:.1f} %)")
        f = os.path.join(b, "compare.out")
        if os.path.isfile(f): print("  tokens (compare.out):\n" + "".join("    " + l for l in open(f) if not l.rstrip().endswith("exit 0")))
