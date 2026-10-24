"""_glcache_frames.py CACHE_DIR OUT_DIR — every zstd frame of the NVIDIA shader disk cache decompressed to
OUT_DIR/<bin>-<offset>.raw, with the count of 16-byte-aligned sm_8x EXIT instructions (low word 0x...794d)."""
import os, sys, glob, zstandard
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
d = zstandard.ZstdDecompressor()
EXIT = b"\x4d\x79\x00\x00\x00\x00\x00\x00"
for f in sorted(glob.glob(os.path.join(src, "**", "*.bin"), recursive=True)):
    b = open(f, "rb").read(); i = 0
    while True:
        i = b.find(b"\x28\xb5\x2f\xfd", i)
        if i < 0: break
        try: dec = d.decompressobj().decompress(b[i:])
        except Exception: dec = b""
        if len(dec) > 64:
            pos = [k for k in range(0, len(dec) - 7) if dec[k:k + 8] == EXIT] if EXIT in dec else []
            name = f"{os.path.basename(f)[:-4]}-{i}.raw"
            open(os.path.join(out, name), "wb").write(dec)
            print(name, "bytes", len(dec), "EXIT", len(pos), "aligned16 mod", sorted({p % 16 for p in pos}))
        i += 4
