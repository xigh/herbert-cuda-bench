# herbert-cuda-bench

Reproducible GPU benchmarks of [herbert-cuda-bf16](https://github.com/xigh/herbert-cuda-bf16) against llama.cpp (CUDA and Vulkan), vLLM, SGLang and transformers, with [herbert-vk-bf16](https://github.com/xigh/herbert-vk-bf16) and [herbert-vk-f16](https://github.com/xigh/herbert-vk-f16) alongside, on Qwen3.5-0.8B and Qwen3.5-4B, one request at a time: the protocol, the predictions written before measuring, and the exact scripts of every measure. Companion to article #003 (CUDA) on [philippe-anel.fr](https://philippe-anel.fr).

The question behind it: herbert-cuda-bf16 writes its compute kernels at start-up, as PTX, without nvcc, NVRTC or cuBLAS. How does that hold against the reference GPU engines on a consumer card, at their best settings and not only at their defaults, from 128 to 60 000 tokens of prompt?

Author: Philippe Anel. License: MIT, see `LICENSE`.

## Layout

```
protocol.md       what is measured and how, phase by phase, each part written before its measures (French)
predictions.md    the predictions, dictated and written before the comparative measures (French)
phase1/           the comparators' settings sweeps (llama.cpp, vLLM, SGLang, transformers) and their token check
phase2/           herbert-cuda against the comparators at their two best settings, two instruments, a control without P0
phase5/           the long prefill, 10 000 to 60 000 tokens: 5a (settings sweep and tokens), 5b (the bench)
decode-depth/     why herbert-cuda's decode falls with depth, and the two attention switches that came out of it
alt-vs/           herbert-cuda (former and new attention) against the comparators of 5b, alternated in one run
defaults/         token check of the new attention defaults against the former ones
occupancy/        the prefill attention at 32 query rows per block (sdpa.rows=32)
raw/              raw logs of each measure, as produced, machine names and paths replaced
```

In each folder, `_scripts/` holds the drivers exactly as they ran (`*-003.sh`) and the folder itself the summary scripts (`_*_summary.py`, `_*_diff.py`) that rebuild the tables from the drivers' output directories. `protocol.md` and `predictions.md` are in French, like the article; P1 to P9 and P14 to P15 are the predictions read in this release.

## Machine

| | |
|---|---|
| GPU | NVIDIA RTX 4080 16 GB (sm_89, 76 SM) |
| host | Ryzen 9 9900X, Linux |
| driver | NVIDIA 595.99.02 |
| clocks | SM locked at 2 805 MHz, memory at 11 201 MHz; logged every second (`nvidia-smi … -l 1`), and checked before starting by `_p1.sh`, the 5a and 5b drivers and the later scripts (they stop otherwise) |
| power state | P0 for every CUDA engine, through the application profile `CudaNoStablePerfLimit` (`~/.nv/nvidia-application-profiles-rc`): without it the driver holds the card in P2 during any CUDA context. Phase 2 measures a control without the profile. |

The CPU load is logged (`uptime`) at every round. The later scripts (decode-depth, alt-vs, defaults, occupancy) wait for an empty GPU before every launch and log what they waited for; a resident inference service of the machine (`resident-gpu` in the scripts) was stopped for those runs and restarted by a trap, or, where the scripts say so, left running beside the 0.8B.

## Engines

| engine | version | format |
|---|---|---|
| herbert-cuda-bf16 | see below | safetensors, bf16 |
| herbert-vk-bf16 | `2f3ae48` (the build of #002), its server `9738be7` | safetensors, bf16 |
| herbert-vk-f16 | `cc51363` (phase 5) | safetensors, f16 |
| herbert-cuda-f16 | `0c7d88d` (phase 5 only), at its default and with `--tune gemm.f16=1`; the IEEE f16 variant of the engine, [xigh/herbert-cuda-f16](https://github.com/xigh/herbert-cuda-f16) tag `v0.1.0` (its tree is `0c7d88d`) | safetensors, f16 |
| llama.cpp | release `b11718`, CUDA and Vulkan builds (Release, CUDA arch 89, `GGML_NATIVE`); `b11719` compared in phase 2b; `08ffba21e` (the version of #002) as a reference in phase 1 | GGUF BF16 (phases 1, 2), F16 (phase 5), `-ngl 99` |
| vLLM | 0.31.0; 0.28.0 (the version of #002) as a reference in phase 1 | bfloat16 (phases 1, 2), float16 (phase 5) |
| SGLang | 0.5.22, `CUDA_HOME` at the 12.8 toolkit | bfloat16, float16 (phase 5) |
| transformers | 5.20.1, flash-linear-attention installed, causal-conv1d absent | bfloat16, float16 (phase 5) |

In phase 5 the comparators run in float16 (llama.cpp on F16 GGUFs, the others converting the bf16 weights at load), as the protocol says; herbert-cuda-bf16 stays in bf16, and herbert-cuda-f16 runs in the same rounds under its two forms.

**herbert-cuda-bf16 builds.** The public repository is published as a single commit without history: it is the tree of development commit `f0b646a`. The hashes named in the scripts are commits of that development history:

| where | build | relation to the public tree |
|---|---|---|
| phases 1, 2 | `71920b9`, run bare and as the candidate `--tune gemm.auto=3 --tune gemm.layout=xor --tune prefill.block=auto` | the candidate became the default prefill in `638364b` |
| phase 5, decode-depth dd1–dd4 | `011bfdf` (`638364b` plus `--ask-file`) | |
| decode-depth dd5–dd8, alt-vs | `bc8e601` (`sdpa.gqa`), `5ec3063` (`sdpa.pipe`), `73b5a70` (both behind switches; its default PTX is the same text as `011bfdf`'s, all 432 modules) | |
| occupancy | `8db5b92` (`sdpa.rows=32` behind a switch; default PTX unchanged) | |
| defaults, public tree | `f0b646a`: attention defaults `sdpa.gqa=1 sdpa.split=64 sdpa.pipe=1` | with `--tune sdpa.gqa=0 --tune sdpa.split=8 --tune sdpa.pipe=0` it emits the same PTX as `8db5b92`'s default (440 modules) and gives the same `[ids]` (p1–p4, pp128–pp4000, 10K and 60K, 0.8B and 4B) |

So phases 1 to 5 measured the former attention: to replay them with the public tree, add `--tune sdpa.gqa=0 --tune sdpa.split=8 --tune sdpa.pipe=0`.

## Two instruments

- **Each engine's own bench**: `herbert-cuda-chat` and `herbert-vk-chat` (the `[turn]` line), `llama-bench`, `vllm bench latency`, `python -m sglang.benchmark.one_batch`, and `generate` timed in `phase1/_scripts/_p1_transformers.py` (and `phase5/_scripts/_p5_transformers.py`). Herbert and llama.cpp alternated round by round, N = 5, order reversed every other round; vLLM, SGLang and transformers one process per setting and model, with their internal repetitions. Definitions of prefill and decode for each are in `protocol.md` and at the top of each summary script.
- **The common HTTP client** (`phase2/_scripts/_http_client.py`, the client of herbert-gpu-bench campaign-v1): the same streamed, greedy requests to every server (the Anthropic Messages API, or OpenAI chat completions for SGLang and transformers), one at a time, no prefix cache, five different passages per length, so the first token and the decode are timed from the user's side.

Correctness comes before speed: each comparator's greedy output ids at its settings are compared with the CPU oracle ([herbert-jit-bf16](https://github.com/xigh/herbert-jit-bf16) v0.1.0) on p1–p4 (`phase1/_p1_tokens_diff.py`), and engine against engine after the long prompts (`phase5/_p5_tokens_diff.py`), where the oracle (8 192 positions) does not reach.

## Replay

Each driver starts with the variables it reads; the defaults assume this layout, change them to yours:

```
$MODELS_DIR (default ~/models)              Qwen3.5-0.8B, Qwen3.5-4B (Hugging Face checkpoints)
$BENCH_DIR  (default ~/bench-qwen35)        Qwen3.5-{0.8B,4B}-{BF16,F16}.gguf, prompts.json and moby.txt of herbert-gpu-bench campaign-v1
~/llama.cpp-b11718/build-{cuda,vk}          llama.cpp builds (also -b11719, -08ffba21e)
~/venvs/bench-{vllm-latest,vllm-cu,sglang-latest,transformers-latest}   one Python environment per engine
~/herbert-cuda-bf16, ~/herbert-vk-bf16-v1 …  the engines, built with cargo build --release --locked (--features serve for the servers)
~/c003/                                      outputs, copies of the scripts (phase1-scripts, phase2-scripts, phase5-scripts; phase4-scripts in _p5a-003.sh holds the same _p1_tokens.py), long-prompts
```

The long prompts come from `phase5/_scripts/_build_long_prompts.py ~/c003/long-prompts` and `_build_long_http_prompts.py ~/c003/long-prompts/prompts-long.json`. Then, for example:

```sh
bash phase1/_scripts/_p1.sh                       # O=~/c003/phase1 by default
python3 phase1/_p1_summary.py ~/c003/phase1 ~/c003/after-p1 ~/c003/after-p1c ~/c003/after-p1e
bash phase2/_scripts/_p2-003.sh
python3 phase2/_p2_summary.py ~/c003/phase2 ~/c003/phase2b ~/c003/phase2c
bash phase5/_scripts/_p5a-003.sh && bash phase5/_scripts/_p5b-003.sh
python3 phase5/_p5a_summary.py ~/c003/phase5a; python3 phase5/_p5b_summary.py ~/c003/phase5b
```

The `_after-*`, `_chain-*` and `*2b`/`*2c` scripts are the follow-up runs the protocol and the earlier runs called for (reruns, token checks, a newer llama.cpp release, a client fix); each says in its header why it exists and what it waits for.

## Raw logs

`raw/` holds the raw logs of each measure, as produced, machine names and paths replaced: one folder per measure, named by date, each with its own `README.md` (in French) that says what ran, in which order, what failed and what came out, the copy of the scripts it was run with, the summaries (`summary.txt`) and every output file (`*.log`, `*.json`, `clocks.csv`, `env*.txt`, emitted PTX, Nsight Compute details pages).

| folder | measure |
|---|---|
| `2026-10-18-cuda-bf16-p0-sous-cuda` | the card held in P2 under CUDA, and the `CudaNoStablePerfLimit` profile that keeps it in P0 |
| `2026-10-18-cuda-bf16-ablation-moteur` | herbert-cuda against itself, one CUDA feature off at a time |
| `2026-10-18-cuda-bf16-wmma-softmax` | `gemm.wmma=1` and `sdpa.softmax=shared`, the two switches closest to what a Vulkan shader can write |
| `2026-10-18-vk-bf16-sass-cache-pilote` | herbert-vk-bf16's SASS read from the driver's disk cache (text disassembly only) |
| `2026-10-18-cuda-bf16-phase1-comparateurs` | phase 1 |
| `2026-10-18-cuda-bf16-phase2-herbert-contre-comparateurs` | phase 2, 2b and 2c |
| `2026-10-19-cuda-bf16-phase5a-prefill-long-comparateurs` | phase 5, step 5a, with the start check at 60K and the long prompts |
| `2026-10-19-cuda-bf16-phase5b-prefill-long-herbert-contre-comparateurs` | phase 5, step 5b |
| `2026-10-20-cuda-bf16-decode-profondeur` | decode-depth dd1, dd2, dd2b |
| `2026-10-20-cuda-bf16-attention-gqa-pipe` | decode-depth dd3 to dd8: `sdpa.gqa` and `sdpa.pipe` |
| `2026-10-21-cuda-bf16-alterne-comparateurs` | alt-vs |
| `2026-10-21-cuda-bf16-occupation-prefill` | occupancy |
| `2026-10-21-cuda-bf16-defaut-attention` | defaults |

The replacement is mechanical and the same everywhere: the home directory becomes `~`, the models directory becomes `$MODELS_DIR` and the bench data directory `$BENCH_DIR` (so the copies of the scripts inside `raw/` are records, the runnable ones are the top-level copies), the local servers' address `localhost`, the machine `rtx4080` (and a second RTX 4080 `rtx4080-b`), the resident inference services of the machine `resident-gpu` and `resident-cpu`, other work that shared the machine `other-tests`; references to private working notes are removed. No number is touched. The long prompts are Moby-Dick, which explains the whales in the logs.

## What this release leaves out

Published as a single commit without history. Phases 3, 4 and 6 of the protocol and step 5c (another machine) are not in this release, nor the predictions that belong to phase 4; the copies of `protocol.md` and `predictions.md` inside `raw/` are left out (the top-level ones are the public text), as is the f16 script of phase 1 that ran on a second RTX 4080. Also left out of `raw/`: the Nsight Compute reports (`.ncu-rep`, 8 files, 49 MB; their text details pages are kept), the binary files of the driver's shader cache (168 files, 5.4 MB; the extracted SASS is kept as text) and the copies of third-party web pages (forum and GitHub discussions) gathered for the P0 question, whose URLs stay in that folder's README. Machine names, network addresses and local paths were replaced by neutral ones, as above. The measuring logic of the scripts is untouched.
