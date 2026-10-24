"""_glcache_sass.py FRAMES_DIR OUT_DIR — the SASS of every compute pipeline in the decompressed frames of the
NVIDIA shader disk cache (_glcache_frames.py). Container "NVDANVVMNVuc…": u32 at 76 = code size (the "Binary Size"
of VK_KHR_pipeline_executable_properties), u32 at 80 = code offset - 8. Each code block is written raw
(<frame>.sass.bin) and disassembled by nvdisasm -b SM89 (<frame>.sass); a summary line per pipeline gives the size,
the instruction count and the counts of a few opcodes. Printable strings of each frame go to <frame>.strings."""
import os, sys, glob, struct, subprocess, re, collections
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
NV = "/usr/local/cuda-12.8/bin/nvdisasm"
OPS = ["HMMA", "LDSM", "LDGSTS", "LDS", "STS", "LDG", "STG", "LDL", "STL", "BAR", "SHFL", "MUFU", "FFMA", "IMAD", "BRA", "DEPBAR"]
for f in sorted(glob.glob(os.path.join(src, "*.raw"))):
    b = open(f, "rb").read(); base = os.path.join(out, os.path.basename(f)[:-4])
    if b[:4] != b"NVDA": continue
    size, off = struct.unpack_from("<II", b, 76); start = off + 8
    code = b[start:start + size]
    open(base + ".sass.bin", "wb").write(code)
    r = subprocess.run([NV, "-b", "SM89", base + ".sass.bin"], capture_output=True, text=True)
    open(base + ".sass", "w").write(r.stdout + r.stderr)
    strs = re.findall(rb"[\x20-\x7e]{6,}", b)
    open(base + ".strings", "w").write("\n".join(s.decode() for s in strs))
    ins = [l for l in r.stdout.splitlines() if re.match(r"\s+/\*[0-9a-f]{4,}\*/", l)]
    c = collections.Counter()
    for l in ins:
        m = re.search(r"\*/\s+(?:@!?U?P\w+\s+)?([A-Z0-9]+)", l)
        if m:
            op = m.group(1)
            for o in OPS:
                if op == o: c[o] += 1
    print(f"{os.path.basename(f)[:-4]} size {size} instr {len(ins)} rc {r.returncode} " + " ".join(f"{o}={c[o]}" for o in OPS if c[o]))
