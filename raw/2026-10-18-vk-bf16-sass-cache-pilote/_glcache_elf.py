"""_glcache_elf.py CACHE_DIR OUT_DIR — every ELF found in the zstd frames of the NVIDIA shader disk cache
(GLCache/**/*.bin), written as OUT_DIR/<bin>-<frame offset>-<n>.elf (size from the ELF header: e_shoff +
e_shnum * e_shentsize, 64-bit little-endian), with an index of the frames."""
import os, sys, glob, struct, zstandard
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
d = zstandard.ZstdDecompressor(); n = 0
for f in sorted(glob.glob(os.path.join(src, "**", "*.bin"), recursive=True)):
    b = open(f, "rb").read(); i = 0
    while True:
        i = b.find(b"\x28\xb5\x2f\xfd", i)
        if i < 0: break
        try: dec = d.decompressobj().decompress(b[i:])
        except Exception: dec = b""
        j = 0
        while True:
            j = dec.find(b"\x7fELF", j)
            if j < 0: break
            h = dec[j:j + 64]
            if len(h) == 64 and h[4] == 2:   # ELFCLASS64
                shoff, = struct.unpack_from("<Q", h, 0x28); shentsize, shnum = struct.unpack_from("<HH", h, 0x3A)
                size = shoff + shentsize * shnum
                e_machine, = struct.unpack_from("<H", h, 0x12)
                name = f"{os.path.basename(f)[:-4]}-{i}-{j}.elf"
                open(os.path.join(out, name), "wb").write(dec[j:j + size]); n += 1
                print(name, "size", size, "e_machine", e_machine)
            j += 4
        i += 4
print("elf files", n)
