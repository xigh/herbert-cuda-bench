#!/usr/bin/env python3
# _tile2b-summary.py DIR — _tile2b-003.sh: per variant and length, prefill tok/s of the [turn 1] line, median [min..max]
# over the rounds, the ratio of the medians to `def`, the [tuning] line of the first round and the generated id.
import glob, re, statistics, sys
D = sys.argv[1]
V = ["def", "def-b2048", "def-b4096", "t64x128", "w2x2s2", "w2x2s2-b2048", "w2x2s2-b4096", "t128-k64s2"]
S = [2000, 10000]
val, ids, tun = {}, {}, {}
for f in glob.glob(f"{D}/chat/r*-pp*.log"):
    r, v, p = re.search(r"/r(\d+)-(.+)-pp(\d+)\.log$", f).groups()
    t = open(f).read()
    m = re.search(r"^\[turn 1\] (\d+) prompt tokens in [\d.]+ s \(([\d.]+) tok/s\)", t, re.M)
    if m:
        val.setdefault((v, int(p)), []).append(float(m.group(2)))
    i = re.search(r"^\[ids\] (.*)$", t, re.M)
    if i:
        ids.setdefault((v, int(p)), set()).add(i.group(1).strip()[-40:])
    if r == "1":
        tl = re.search(r"^\[tuning\] (.*)$", t, re.M)
        if tl:
            tun[v] = tl.group(1)
print("| Variant | " + " | ".join(f"pp{s}" for s in S) + " | " + " | ".join(f"× def pp{s}" for s in S) + " |")
print("|---|" + "---|" * (2 * len(S)))
for v in V:
    cells, rat = [], []
    for s in S:
        xs = val.get((v, s), [])
        cells.append(f"{statistics.median(xs):,.0f} [{min(xs):.0f}..{max(xs):.0f}]".replace(",", " ") if xs else "—")
        d = val.get(("def", s), [])
        rat.append(f"{statistics.median(xs) / statistics.median(d):.3f}" if xs and d else "—")
    print(f"| {v} | " + " | ".join(cells) + " | " + " | ".join(rat) + " |")
print()
for v in V:
    print(f"{v}: ids {dict((s, sorted(ids.get((v, s), []))) for s in S)}")
    print(f"  [tuning] {tun.get(v, '—')}")
