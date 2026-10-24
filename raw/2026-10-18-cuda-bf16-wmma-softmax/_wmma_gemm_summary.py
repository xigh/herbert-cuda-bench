# _wmma_gemm_summary.py DIR — the GEMM of _wmma-softmax-003.sh: medians over the rounds, wmma against
# mma (ldmatrix) at the same tile and copy path, padded layout.
import sys, re, glob, os, statistics, collections
d = sys.argv[1]
pts = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(d, "gemm", "*.log"))):
    for l in open(f):
        m = re.match(r"gemm\s+(\S+)\s+(\S+)\s+M (\d+)\s+\d+×\d+\s+(\S+):\s+([\d.]+) µs", l)
        if not m: continue
        model, shape, M, tag, us = m.groups()
        tile = re.sub(r"_(sync|wmma)", "", tag)
        copy = "sync" if "_sync" in tag else "async"
        mma = "wmma" if "_wmma" in tag else "mma"
        pts[(model, shape, int(M), tile, copy, mma)].append(float(us))
ratios = collections.defaultdict(list)
print("model shape M tile copy | µs mma | µs wmma | wmma / mma")
for k in sorted({k[:5] for k in pts}):
    a = statistics.median(pts[k + ("mma",)]); b = statistics.median(pts[k + ("wmma",)])
    ratios[(k[0], k[3], k[4], k[2])].append(b / a)
    print(f"{k[0]} {k[1]:<12} {k[2]:>5} {k[3]:<20} {k[4]:<5} | {a:9.2f} | {b:9.2f} | ×{b / a:.3f}")
print()
print("geometric mean of wmma / mma over the 6 shapes")
for (model, tile, copy, M), v in sorted(ratios.items()):
    print(f"{model} {tile:<20} {copy:<5} M {M:>5}: ×{statistics.geometric_mean(v):.3f} (min {min(v):.3f} max {max(v):.3f})")
