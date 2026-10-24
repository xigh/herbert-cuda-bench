"""_dd_summary.py DIR — the decode-depth campaigns (_dd1-003.sh, _dd2-003.sh) under DIR (dd1/, dd2/):
A. dd1/a: per depth, the decode step (GPU median) and its sdpa_split / sdpa_combine share, with the KV bytes read
   once per step (2 KV heads × 256 × K and V × bf16 × 6 attention layers × the median position, prompt + 32) and the bandwidth that implies;
B. dd1/b and dd2b/b: decode tok/s per sdpa.split and shape (median of N, spread, ratio to split 8), and the classes of
   identical [ids] per shape with the first index that differs from split 8. dd2 is left out (its rounds 2 and 3 ran
   while other tests held the 4080); in dd2b, a run is kept only if apps.txt shows the resident service
   alone on the GPU at its launch (runs in script order: rounds, splits reversed every other round, p3/pp20000/pp40000);
C. dd1/c: at 60K, the step and the two attention kernels per split, from --profile."""
import sys, os, re, glob, statistics, collections
D = sys.argv[1]; med = statistics.median
def prof(f):
    out, sect = {}, None
    for l in open(f, errors="replace"):
        m = re.search(r"\[profile\] (\d+) decode steps: GPU ([\d.]+) µs per step", l)
        if m: sect = "dec"; out["step"] = float(m.group(2)); continue
        if "prefill blocks" in l: sect = "pre"; continue
        m = re.search(r"\[profile\]\s+([\d.]+) µs\s+([\d.]+) %\s+×(\d+)\s+(\S+)", l)
        if m and sect == "dec": out[m.group(4)] = (float(m.group(1)), float(m.group(2)))
        m = re.search(r"\[turn 1\] (\d+) prompt", l)
        if m: out["prompt"] = int(m.group(1))
    return out
print("# A. decode step and attention by depth (dd1/a, sdpa.split=8, --profile)")
for f in sorted(glob.glob(os.path.join(D, "dd1", "a", "profile-pp*.log")), key=lambda f: int(re.findall(r"\d+", f)[-1])):
    p = prof(f); L = p["prompt"] + 32
    sp = next(v for k, v in p.items() if k.startswith("sdpa_split")); cb = next(v for k, v in p.items() if k.startswith("sdpa_combine"))
    kv = L * 2 * 256 * 2 * 2 * 6
    print(f"  {p['prompt']:>6}: step {p['step']:.1f} µs · sdpa_split {sp[0]:.1f} µs ({sp[1]:.1f} %) · combine {cb[0]:.1f} µs · rest {p['step'] - sp[0] - cb[0]:.1f} µs · KV once {kv / 1e6:.0f} MB → {kv / sp[0] / 1e3:.0f} GB/s")
print("\n# B. decode tok/s per sdpa.split (graph on; median, n, spread, ratio to split 8)")
tps = collections.defaultdict(list); ids = collections.defaultdict(dict)
skip = set()
ap = os.path.join(D, "dd2b", "apps.txt")
if os.path.isfile(ap):
    S = [8, 16, 32, 64, 128]
    order = [(r, s, sh) for r in (1, 2, 3) for s in (S if r % 2 else S[::-1]) for sh in ("p3", "pp20000", "pp40000")]
    for (r, s, sh), l in zip(order, open(ap)):
        if l.count(";") != 1: skip.add(os.path.join(D, "dd2b", "b", f"r{r}-s{s}-{sh}.log"))
    print(f"  dd2b runs left out (another compute process on the GPU at launch): {len(skip)}")
for f in glob.glob(os.path.join(D, "dd1", "b", "r*-s*-*.log")) + glob.glob(os.path.join(D, "dd2b", "b", "r*-s*-*.log")):
    if f in skip: continue
    r, s, shape = re.match(r"r(\d)-s(\d+)-(p3|pp\d+)\.log", os.path.basename(f)).groups()
    for l in open(f, errors="replace"):
        m = re.search(r"\[turn 1\].*generated in [\d.]+ s \(([\d.]+) tok/s\)", l)
        if m: tps[(shape, int(s))].append(float(m.group(1)))
        if l.startswith("[ids]"): ids[shape][(int(s), int(r))] = l.split()[1:]
for shape in sorted({k[0] for k in tps}, key=lambda x: (x != "p3", int(re.sub(r"\D", "", x) or 0))):
    b = med(tps[(shape, 8)])
    print(f"  {shape}: " + " · ".join(f"s{s} {med(v):.1f} (n={len(v)}, {(max(v) - min(v)) / med(v) * 100:.1f} %, ×{med(v) / b:.3f})"
                                     for s, v in sorted((k[1], v) for k, v in tps.items() if k[0] == shape)))
    cls = collections.defaultdict(list)
    for (s, r), v in sorted(ids[shape].items()): cls[tuple(v)].append(f"s{s}r{r}")
    ref = next(k for k, v in cls.items() if any(x.startswith("s8r") for x in v))
    for k, v in cls.items():
        fd = next((i for i in range(min(len(k), len(ref))) if k[i] != ref[i]), None)
        print(f"    ids {' '.join(v)}: first difference against split 8 at {fd}")
print("\n# C. 60K, --profile per split (dd1/c)")
for f in sorted(glob.glob(os.path.join(D, "dd1", "c", "profile-s*-pp60000.log")), key=lambda f: int(re.search(r"-s(\d+)-", f).group(1))):
    p = prof(f); s = re.search(r"-s(\d+)-", f).group(1)
    sp = next(v for k, v in p.items() if k.startswith("sdpa_split")); cb = next(v for k, v in p.items() if k.startswith("sdpa_combine"))
    kv = (p["prompt"] + 32) * 2 * 256 * 2 * 2 * 6
    print(f"  s{s:>3}: step {p['step']:.1f} µs · sdpa_split {sp[0]:.1f} µs ({kv / sp[0] / 1e3:.0f} GB/s if KV read once) · combine {cb[0]:.1f} µs")
