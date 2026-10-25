| Variant | pp2000 | pp10000 | × def pp2000 | × def pp10000 |
|---|---|---|---|---|
| def | 25 684 [25635..25754] | 24 524 [24510..24530] | 1.000 | 1.000 |
| def-b2048 | 26 796 [26690..26917] | 25 570 [25554..25584] | 1.043 | 1.043 |
| def-b4096 | 26 755 [26724..26908] | 25 749 [25703..25770] | 1.042 | 1.050 |
| t64x128 | 25 619 [25580..25640] | 24 527 [24502..24568] | 0.997 | 1.000 |
| w2x2s2 | 27 722 [27650..27879] | 26 419 [26353..26429] | 1.079 | 1.077 |
| w2x2s2-b2048 | 29 056 [29017..29291] | 27 706 [27655..27724] | 1.131 | 1.130 |
| w2x2s2-b4096 | 29 064 [28996..29263] | 27 805 [27759..27860] | 1.132 | 1.134 |
| t128-k64s2 | 24 459 [24435..24465] | 23 456 [23437..23470] | 0.952 | 0.956 |

def: ids {2000: ['27775'], 10000: ['1919']}
  [tuning] matvec.rows=2 matvec.warps=4 matvec.depth=8 matvec.cache=nol1 sdpa.split=64 sdpa.gqa=1 act=bf16 graph=1 prefill.block=1024 gemm.tile=64 gemm.bk=32 gemm.stages=3 gemm.auto=3 gemm.carveout=driver gemm.layout=xor gemm.schedule=0 gemm.pipeline=0 gemm.down_short=same gemm.ldmatrix=1 gemm.copy=async gemm.wmma=0 gemm.f16=0 sdpa.bk=32 sdpa.softmax=reg sdpa.pipe=1 sdpa.rows=16 sdpa.alias=0 rec.tc=1 ld.scope=plain sdpa.maxnreg=0 fuse.norm=0 fuse.swiglu=0 fuse.dn=0
def-b2048: ids {2000: ['27775'], 10000: ['1919']}
  [tuning] matvec.rows=2 matvec.warps=4 matvec.depth=8 matvec.cache=nol1 sdpa.split=64 sdpa.gqa=1 act=bf16 graph=1 prefill.block=2048 gemm.tile=64 gemm.bk=32 gemm.stages=3 gemm.auto=3 gemm.carveout=driver gemm.layout=xor gemm.schedule=0 gemm.pipeline=0 gemm.down_short=same gemm.ldmatrix=1 gemm.copy=async gemm.wmma=0 gemm.f16=0 sdpa.bk=32 sdpa.softmax=reg sdpa.pipe=1 sdpa.rows=16 sdpa.alias=0 rec.tc=1 ld.scope=plain sdpa.maxnreg=0 fuse.norm=0 fuse.swiglu=0 fuse.dn=0
def-b4096: ids {2000: ['27775'], 10000: ['1919']}
  [tuning] matvec.rows=2 matvec.warps=4 matvec.depth=8 matvec.cache=nol1 sdpa.split=64 sdpa.gqa=1 act=bf16 graph=1 prefill.block=4096 gemm.tile=64 gemm.bk=32 gemm.stages=3 gemm.auto=3 gemm.carveout=driver gemm.layout=xor gemm.schedule=0 gemm.pipeline=0 gemm.down_short=same gemm.ldmatrix=1 gemm.copy=async gemm.wmma=0 gemm.f16=0 sdpa.bk=32 sdpa.softmax=reg sdpa.pipe=1 sdpa.rows=16 sdpa.alias=0 rec.tc=1 ld.scope=plain sdpa.maxnreg=0 fuse.norm=0 fuse.swiglu=0 fuse.dn=0
t64x128: ids {2000: ['27775'], 10000: ['1919']}
  [tuning] matvec.rows=2 matvec.warps=4 matvec.depth=8 matvec.cache=nol1 sdpa.split=64 sdpa.gqa=1 act=bf16 graph=1 prefill.block=1024 gemm.tile=64x128 gemm.bk=32 gemm.stages=3 gemm.auto=0 gemm.carveout=driver gemm.layout=xor gemm.schedule=0 gemm.pipeline=0 gemm.down_short=same gemm.ldmatrix=1 gemm.copy=async gemm.wmma=0 gemm.f16=0 sdpa.bk=32 sdpa.softmax=reg sdpa.pipe=1 sdpa.rows=16 sdpa.alias=0 rec.tc=1 ld.scope=plain sdpa.maxnreg=0 fuse.norm=0 fuse.swiglu=0 fuse.dn=0
w2x2s2: ids {2000: ['27775'], 10000: ['1919']}
  [tuning] matvec.rows=2 matvec.warps=4 matvec.depth=8 matvec.cache=nol1 sdpa.split=64 sdpa.gqa=1 act=bf16 graph=1 prefill.block=1024 gemm.tile=64x128w2x2 gemm.bk=32 gemm.stages=2 gemm.auto=0 gemm.carveout=driver gemm.layout=xor gemm.schedule=0 gemm.pipeline=0 gemm.down_short=same gemm.ldmatrix=1 gemm.copy=async gemm.wmma=0 gemm.f16=0 sdpa.bk=32 sdpa.softmax=reg sdpa.pipe=1 sdpa.rows=16 sdpa.alias=0 rec.tc=1 ld.scope=plain sdpa.maxnreg=0 fuse.norm=0 fuse.swiglu=0 fuse.dn=0
w2x2s2-b2048: ids {2000: ['27775'], 10000: ['1919']}
  [tuning] matvec.rows=2 matvec.warps=4 matvec.depth=8 matvec.cache=nol1 sdpa.split=64 sdpa.gqa=1 act=bf16 graph=1 prefill.block=2048 gemm.tile=64x128w2x2 gemm.bk=32 gemm.stages=2 gemm.auto=0 gemm.carveout=driver gemm.layout=xor gemm.schedule=0 gemm.pipeline=0 gemm.down_short=same gemm.ldmatrix=1 gemm.copy=async gemm.wmma=0 gemm.f16=0 sdpa.bk=32 sdpa.softmax=reg sdpa.pipe=1 sdpa.rows=16 sdpa.alias=0 rec.tc=1 ld.scope=plain sdpa.maxnreg=0 fuse.norm=0 fuse.swiglu=0 fuse.dn=0
w2x2s2-b4096: ids {2000: ['27775'], 10000: ['1919']}
  [tuning] matvec.rows=2 matvec.warps=4 matvec.depth=8 matvec.cache=nol1 sdpa.split=64 sdpa.gqa=1 act=bf16 graph=1 prefill.block=4096 gemm.tile=64x128w2x2 gemm.bk=32 gemm.stages=2 gemm.auto=0 gemm.carveout=driver gemm.layout=xor gemm.schedule=0 gemm.pipeline=0 gemm.down_short=same gemm.ldmatrix=1 gemm.copy=async gemm.wmma=0 gemm.f16=0 sdpa.bk=32 sdpa.softmax=reg sdpa.pipe=1 sdpa.rows=16 sdpa.alias=0 rec.tc=1 ld.scope=plain sdpa.maxnreg=0 fuse.norm=0 fuse.swiglu=0 fuse.dn=0
t128-k64s2: ids {2000: ['27775'], 10000: ['1919']}
  [tuning] matvec.rows=2 matvec.warps=4 matvec.depth=8 matvec.cache=nol1 sdpa.split=64 sdpa.gqa=1 act=bf16 graph=1 prefill.block=1024 gemm.tile=128w4x4 gemm.bk=64 gemm.stages=2 gemm.auto=0 gemm.carveout=driver gemm.layout=xor gemm.schedule=0 gemm.pipeline=0 gemm.down_short=same gemm.ldmatrix=1 gemm.copy=async gemm.wmma=0 gemm.f16=0 sdpa.bk=32 sdpa.softmax=reg sdpa.pipe=1 sdpa.rows=16 sdpa.alias=0 rec.tc=1 ld.scope=plain sdpa.maxnreg=0 fuse.norm=0 fuse.swiglu=0 fuse.dn=0
