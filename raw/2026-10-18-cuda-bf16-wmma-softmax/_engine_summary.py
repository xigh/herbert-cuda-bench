# _engine_summary.py DIR — medians over the rounds of _wmma-softmax-003.sh (variants of _ablation_summary.py): prefill tok/s per length and
# p3 decode tok/s, each variant against the base (the candidate), and whether the output ids
# match the base's in every round.
import sys, re, glob, os, statistics, collections
d = sys.argv[1]
val = collections.defaultdict(list); ids = collections.defaultdict(set)
for f in glob.glob(os.path.join(d, "runs", "*.log")):
    m = re.match(r"r(\d+)-(\S+?)-(\S+)-(pp\d+|p3)\.log", os.path.basename(f))
    r, model, var, form = m.groups()
    for l in open(f, errors="replace"):
        if l.startswith("[ids]"): ids[(model, var, form)].add(l.strip())
        if l.startswith("[turn"):
            p = re.search(r"prompt tokens in [\d.]+ s \(([\d.]+) tok/s\)", l); g = re.search(r"generated in [\d.]+ s \(([\d.]+) tok/s\)", l)
            val[(model, var, form)].append(float(p.group(1)) if form != "p3" else float(g.group(1)))
            if form == "p3": val[(model, var, "p3-prefill")].append(float(p.group(1)))
variants = ["base", "softmax-shared", "pad", "wmma", "vklike"]
forms = ["pp128", "pp256", "pp512", "pp1024", "pp2048", "pp4000", "p3-prefill", "p3"]
for model in ["0.8B", "4B"]:
    print(f"== {model}: median tok/s (prefill per length; p3 = decode at depth 759), N per cell, variant / base")
    print("variant      " + " ".join(f"{f:>16}" for f in forms))
    for v in variants:
        cells = []
        for f in forms:
            x = val.get((model, v, f)); b = val.get((model, "base", f))
            if not x: cells.append(f"{'-':>16}"); continue
            mx = statistics.median(x); mb = statistics.median(b)
            cells.append(f"{mx:9.1f} ×{mx / mb:5.3f}" if v != "base" else f"{mx:9.1f} n={len(x):<3}")
        print(f"{v:<12} " + " ".join(f"{c:>16}" for c in cells))
    print("ids: identical to base in every round?")
    for v in variants:
        same = all(len(ids[(model, v, f)]) == 1 and ids[(model, v, f)] == ids[(model, "base", f)] for f in ["pp128", "pp256", "pp512", "pp1024", "pp2048", "pp4000", "p3"])
        print(f"  {v:<12} {'yes' if same else 'NO: ' + ', '.join(f for f in ['pp128','pp256','pp512','pp1024','pp2048','pp4000','p3'] if ids[(model, v, f)] != ids[(model, 'base', f)])}")
