"""Decode-throughput benchmark (nano-vllm-style workload).

Many requests, a fixed encoder length and a fixed output length with EOS
suppressed, so the whole run is decode-dominated and tokens/s is comparable
across engines.

    PYTHONPATH=. python benchmarks/throughput.py --requests 64 --enc-len 512 --max-tokens 256
    PYTHONPATH=. python benchmarks/throughput.py --requests 64 --enc-len 512 --max-tokens 256 --no-cudagraph
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch

from fastencdec import LLM, SamplingParams

TEXT = (
    "The quick brown fox jumps over the lazy dog. Scientists at the university "
    "announced on Monday that they had discovered a new species of butterfly in "
    "the Amazon rainforest. The city council approved a new plan to expand "
    "public transport, including two new tram lines and protected bicycle lanes. "
)


def make_prompt(tokenizer, enc_len):
    ids = tokenizer(TEXT, add_special_tokens=False)["input_ids"]
    while len(ids) < enc_len:
        ids += ids
    return tokenizer.decode(ids[:enc_len])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/bart-large-cnn")
    ap.add_argument("--requests", type=int, default=64)
    ap.add_argument("--enc-len", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=0, help="0 = device default")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--cudagraph",
        dest="use_cudagraph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="decode CUDA graph (default on; CUDA + flash only)",
    )
    ap.add_argument(
        "--compile-model", action="store_true", help="torch.compile decoder"
    )
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--profile", action="store_true", help="torch.profiler breakdown")
    ap.add_argument("--rows", type=int, default=18)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    block_size = args.block_size or (256 if str(args.device).startswith("cuda") else 16)
    n = args.requests
    tokens_per_seq = args.enc_len + args.max_tokens
    num_blocks = n * ((tokens_per_seq + block_size - 1) // block_size) + 64
    llm = LLM(
        args.model,
        num_blocks=num_blocks,
        block_size=block_size,
        max_num_seqs=n,
        dtype=dtype,
        device=args.device,
        use_cudagraph=args.use_cudagraph,
        compile_model=args.compile_model,
    )
    prompt = make_prompt(tokenizer, args.enc_len)
    prompts = [prompt] * n
    params = SamplingParams(
        max_tokens=args.max_tokens,
        num_beams=1,
        temperature=0.0,
        min_length=args.max_tokens + 1,  # EOS suppressed -> exactly max_tokens
        length_penalty=2.0,
        no_repeat_ngram_size=3,
    )

    def run():
        return llm.generate(prompts, sampling_params=params)

    for _ in range(args.warmup):
        run()
    best = float("inf")
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        run()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - start)

    generated = n * args.max_tokens
    print(
        f"model={args.model} device={args.device} dtype={args.dtype} "
        f"requests={n} enc_len={args.enc_len} max_tokens={args.max_tokens} "
        f"block_size={block_size} num_blocks={num_blocks} "
        f"cudagraph={args.use_cudagraph} compile={args.compile_model}"
    )
    print(
        f"best {best * 1e3:8.0f} ms   {generated} tokens   "
        f"{generated / best:8.1f} tok/s"
    )

    if args.profile:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            run()
        torch.cuda.synchronize()
        events = prof.key_averages()
        counts = {e.key: e.count for e in events}
        print(
            "totals: self_cpu_ms={:.0f}  syncs={}  launches={}  graph_launches={}".format(
                sum(e.self_cpu_time_total for e in events) / 1e3,
                counts.get("cudaStreamSynchronize", 0),
                counts.get("cudaLaunchKernel", 0) + counts.get("cuLaunchKernel", 0),
                counts.get("cudaGraphLaunch", 0),
            )
        )
        print(
            prof.key_averages().table(
                sort_by="self_cpu_time_total", row_limit=args.rows
            )
        )
        print(
            prof.key_averages().table(sort_by="device_time_total", row_limit=args.rows)
        )


if __name__ == "__main__":
    main()
