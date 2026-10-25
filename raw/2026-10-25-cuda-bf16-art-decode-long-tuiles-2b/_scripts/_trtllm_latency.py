#!/usr/bin/env python3
# _trtllm_latency.py — TensorRT-LLM (PyTorch backend, LLM API) counterpart of `vllm bench latency`: one request of
# INPUT random token ids (seed 0, ids below 10 000, as vLLM's dummy prompt), OUTPUT tokens generated greedily with
# ignore_eos, WARMUP untimed runs then ITERS timed ones; end-to-end latency of each, written as JSON like vLLM's
# --output-json (avg_latency, latencies). Prefill = INPUT / latency(output 1); decode = (128 - 1) / (lat128 - lat1).
# Text only, like vLLM's --language-model-only: disable_mm_encoder=True (the vision/audio encoders are not built).
# --chunked-prefill (article long prefill, 2026-10-25): enable_chunked_prefill=True, the prompt goes in max_num_tokens
# chunks (as vLLM's default max_num_batched_tokens 8192 at long context).
import argparse, json, time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input-len", type=int, default=4000)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--max-num-tokens", type=int, default=0, help="0: max-seq-len")
    ap.add_argument("--kv-fraction", type=float, default=0.5)
    ap.add_argument("--no-cuda-graph", action="store_true")
    ap.add_argument("--chunked-prefill", action="store_true")
    ap.add_argument("--with-mm-encoder", action="store_true")
    ap.add_argument("--output-json", required=True)
    a = ap.parse_args()

    import numpy as np
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig
    import tensorrt_llm

    kw = dict(model=a.model, max_seq_len=a.max_seq_len, max_batch_size=1, max_num_tokens=a.max_num_tokens or a.max_seq_len,
              kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=a.kv_fraction, enable_block_reuse=False),
              disable_mm_encoder=not a.with_mm_encoder, enable_chunked_prefill=a.chunked_prefill)
    if not a.no_cuda_graph:
        kw["cuda_graph_config"] = CudaGraphConfig(batch_sizes=[1])
    llm = LLM(**kw)
    ids = np.random.default_rng(0).integers(0, 10000, a.input_len).tolist()
    sp = SamplingParams(max_tokens=a.output_len, ignore_eos=True, temperature=0.0)

    def once():
        t = time.perf_counter()
        out = llm.generate([ids], sampling_params=sp, use_tqdm=False)
        dt = time.perf_counter() - t
        assert len(out[0].outputs[0].token_ids) == a.output_len, len(out[0].outputs[0].token_ids)
        return dt

    for _ in range(a.warmup):
        once()
    lat = [once() for _ in range(a.iters)]
    for x in lat:
        print(f"latency {x:.6f} s")
    print(f"Avg latency: {sum(lat) / len(lat)} seconds")
    with open(a.output_json, "w") as f:
        json.dump({"model": a.model, "trtllm": tensorrt_llm.__version__, "input_len": a.input_len,
                   "output_len": a.output_len, "avg_latency": sum(lat) / len(lat), "latencies": lat}, f, indent=2)
    llm.shutdown()

if __name__ == "__main__":
    main()
