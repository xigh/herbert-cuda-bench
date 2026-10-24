"""_occ1_summary.py OCC1 — sdpa.rows=32 and sdpa.pipe=1 against the default (occupancy/_scripts/_occ1-003.sh):
prefill tok/s of the chat [turn] line, median of N = 3 alternated rounds (spread), each variant / default (ratio of
medians, median of per-round ratios); --profile at 60K: the prefill block's GPU time (median) and sdpa_flash's line
(time per block, share); Nsight Compute: occupancy, warps, shared memory, registers and the main throughput lines of
each report's details page."""
import sys, os, re, glob, statistics, collections
P = sys.argv[1]; med = statistics.median
def spread(v): return (max(v) - min(v)) / med(v) * 100 if len(v) > 1 else 0.0
V = ["default", "rows32", "pipe", "pipe-rows32"]; LENS = ["pp4000", "pp10000", "pp20000", "pp40000", "pp60000"]
R = collections.defaultdict(dict)
for f in glob.glob(os.path.join(P, "b", "r*.log")):
    mt = re.match(r"r(\d+)-(0\.8B|4B)-(\S+)-(pp\d+)\.log", os.path.basename(f))
    if not mt: continue
    r, m, v, s = mt.groups()
    for l in open(f, errors="replace"):
        t = re.search(r"\[turn \d+\] (\d+) prompt tokens in [\d.]+ s \(([\d.]+) tok/s\)", l)
        if t: R[(m, v, s)][int(r)] = float(t.group(2)); break
for m in ["0.8B", "4B"]:
    print(f"\n## {m}: prefill tok/s (median of rounds, spread)\n| variant | " + " | ".join(LENS) + " |\n|---|" + "---|" * len(LENS))
    for v in V:
        print(f"| {v} | " + " | ".join(f"{med(R[(m, v, s)].values()):.0f} (n={len(R[(m, v, s)])}, {spread(list(R[(m, v, s)].values())):.1f} %)" if R[(m, v, s)] else "" for s in LENS) + " |")
    print(f"\n| {m} / default (ratio of medians · median of per-round ratios) | " + " | ".join(LENS) + " |\n|---|" + "---|" * len(LENS))
    for v in V[1:]:
        cells = []
        for s in LENS:
            a, b = R[(m, v, s)], R[(m, "default", s)]
            rr = [a[r] / b[r] for r in a if r in b]
            cells.append(f"×{med(a.values()) / med(b.values()):.3f} · ×{med(rr):.3f}" if rr else "")
        print(f"| {v} | " + " | ".join(cells) + " |")
print("\n## --profile at 60K")
for m in ["0.8B", "4B"]:
    for v in V:
        f = os.path.join(P, "c", f"profile-{m}-{v}-pp60000.log")
        if not os.path.isfile(f): continue
        blk = fl = ""
        for l in open(f, errors="replace"):
            if "prefill blocks: GPU" in l and not blk: blk = l.split("[profile]")[1].strip()
            if re.search(r"\[profile\].*sdpa_flash", l) and not fl: fl = re.sub(r"\s+", " ", l.split("[profile]")[1].strip())
        print(f"  {m} {v}: {blk} | {fl}")
print("\n## Nsight Compute (sdpa_flash, last prefill block at 60K)")
keys = ["Theoretical Occupancy", "Achieved Occupancy", "Block Limit Shared Mem", "Block Limit Registers", "Registers Per Thread",
        "Dynamic Shared Memory Per Block", "Duration", "Compute (SM) Throughput", "Memory Throughput", "DRAM Throughput",
        "Eligible Warps Per Scheduler", "No Eligible", "Issued Warp Per Scheduler", "Waves Per SM"]
for f in sorted(glob.glob(os.path.join(P, "ncu", "*.details.txt"))):
    print(f"\n{os.path.basename(f)}")
    seen = collections.Counter()
    for l in open(f, errors="replace"):
        for k in keys:
            if l.strip().startswith(k) and seen[k] < 2:
                seen[k] += 1; print("   ", re.sub(r"\s{2,}", "  ", l.strip()))
