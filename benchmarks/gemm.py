"""bf16 GEMM throughput at the shapes the engine actually runs (BART-large).

The decode step is GEMM-bound, so this is the number that explains most of the
per-step time: the decoder projects one token per sequence (batch 1-32) through
``qkv``/``out_proj``/``fc1``/``fc2``, the encoder prefills many tokens at once,
and a large square GEMM gives a peak-ish reference.  ``torch.nn.functional.linear``
with a bias is used because that is what the model calls (addmm).

    PYTHONPATH=. python benchmarks/gemm.py                       # bf16, 8 threads
    PYTHONPATH=. python benchmarks/gemm.py --threads 4,8         # thread sweep
    PYTHONPATH=. python benchmarks/gemm.py --dtype float32
"""

import argparse
import platform
import sys
import time

import torch
import torch.nn.functional as F

D_MODEL, D_FFN = 1024, 4096  # bart-large-cnn

# (label, M, N, K) -- M = tokens, N = output features, K = input features
SHAPES = []
for batch in (1, 8, 32):
    SHAPES += [
        (f"decode batch={batch}: qkv (fused)", batch, 3 * D_MODEL, D_MODEL),
        (f"decode batch={batch}: out_proj", batch, D_MODEL, D_MODEL),
        (f"decode batch={batch}: fc1", batch, D_FFN, D_MODEL),
        (f"decode batch={batch}: fc2", batch, D_MODEL, D_FFN),
    ]
for tokens in (256, 2048):
    SHAPES += [
        (f"prefill {tokens}: qkv (fused)", tokens, 3 * D_MODEL, D_MODEL),
        (f"prefill {tokens}: fc1", tokens, D_FFN, D_MODEL),
        (f"prefill {tokens}: fc2", tokens, D_MODEL, D_FFN),
    ]
SHAPES.append(("square 4096^3 (peak-ish)", 4096, 4096, 4096))


def bench(m, n, k, dtype, iters: int, warmup: int = 3) -> float:
    """Best-of-``iters`` seconds for one ``[m,k] x [k,n]`` addmm."""
    x = torch.randn(m, k, dtype=dtype)
    weight = torch.randn(n, k, dtype=dtype)
    bias = torch.randn(n, dtype=dtype)
    for _ in range(warmup):
        F.linear(x, weight, bias)
    best = float("inf")
    for _ in range(iters):
        start = time.perf_counter()
        F.linear(x, weight, bias)
        best = min(best, time.perf_counter() - start)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--threads", default="8", help="comma-separated thread counts")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)

    try:
        cpu = next(
            line.split(":", 1)[1].strip()
            for line in open("/proc/cpuinfo")  # noqa: SIM115
            if "model name" in line
        )
    except (OSError, StopIteration):
        cpu = platform.machine()
    print(f"host: {cpu} ({platform.machine()})")
    print(f"torch {torch.__version__}, {args.dtype}, addmm with bias\n")

    for threads in (int(t) for t in str(args.threads).split(",")):
        torch.set_num_threads(threads)
        print(f"== threads={threads} ==")
        print(f"{'shape':<34} {'ms':>9} {'GFLOP/s':>9}")
        for label, m, n, k in SHAPES:
            seconds = bench(m, n, k, dtype, args.iters)
            gflops = 2 * m * n * k / seconds / 1e9
            print(f"{label:<34} {seconds * 1e3:>8.2f} {gflops:>9.0f}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
