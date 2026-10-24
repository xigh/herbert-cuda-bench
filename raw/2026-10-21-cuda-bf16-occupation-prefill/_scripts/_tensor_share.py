"""_tensor_share.py — the mma work sdpa_flash executes in the last prefill block at 60K (prompt 60 000 tokens:
0.8B blocks of 4 096 rows, last block 2 656 rows at position 57 344; 4B blocks of 2 048, last 608 at 59 392), counted
from the kernel: per warp and tile of 32 keys, 64 mma for S (hd/16 steps · 4) and 64 for P·V (bk/16 steps · hd/8), each
m16n8k16 = 2 048 FMA; every warp of a block runs ceil((p0 + rows_end)/32) tiles, masked or not. Divided by the
Nsight Compute duration (clock control off) and by the measured ceiling of mma.sync bf16 -> f32 on the 4080,
108.5 TFLOPS (probe, clocks locked)."""
import math
def flops(m, p0, qrows, g, kvh):
    t = sum(math.ceil((p0 + min(qrows * (qt + 1), m)) / 32) * g * (qrows // 16) * kvh for qt in range(math.ceil(m / qrows)))
    return t * 128 * 2048 * 2
CEIL = 108.5e12
for name, m, p0, q, g, kvh, dur in [("0.8B default (dd4)", 2656, 57344, 16, 4, 2, 21.55e-3), ("0.8B pipe (dd7)", 2656, 57344, 16, 4, 2, 15.62e-3),
                                    ("0.8B rows32", 2656, 57344, 32, 4, 2, 21.15e-3), ("0.8B pipe-rows32", 2656, 57344, 32, 4, 2, 17.70e-3),
                                    ("4B rows32", 608, 59392, 32, 4, 4, 7.26e-3), ("4B pipe-rows32", 608, 59392, 32, 4, 4, 6.08e-3)]:
    f = flops(m, p0, q, g, kvh)
    print(f"{name}: {f / 1e12:.3f} TFLOP, {dur * 1e3:.2f} ms, {f / dur / 1e12:.1f} TFLOPS, {f / dur / CEIL * 100:.0f} % of 108.5")
