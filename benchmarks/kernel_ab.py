"""Drift-controlled A/B of the two paged CPU attention kernels.

Both kernels are timed back to back on the same tensors, in alternating rounds,
so the columns see the same clock state; the printed `mojo/vllm` ratio is the
pair of best times and the median of the per-round ratios. Use this rather than
`attn_backend.py --mode kernel` when you need a *decision* (did the change
help?): the kernel table is best-of-N per column and this laptop's clock moves
+-10% between columns.

    PYTHONPATH=. python benchmarks/kernel_ab.py [--dtype bfloat16] \
        [--threads N] [--workers N]

``--threads 0`` keeps torch's default (physical cores). ``--workers 0`` keeps the
kernel's own rule (`mojo_attn._workers`: torch's pool capped at the physical core
count); any other value pins it, which is how the worker sweeps were run.
"""

import argparse
import statistics
import sys
import time

sys.path.insert(0, "/home/ducviet00/fast-encdec")

import torch

from fastencdec.layers import attention, cpu_attn, mojo_attn

HEADS, DIM, BLOCK = 16, 64, 32


def timed(fn, iters=40, warmup=8):
    for _ in range(warmup):
        fn()
    best = 1e9
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e6


def self_case(batch, ctx, dtype, workers):
    pages = (ctx + BLOCK - 1) // BLOCK
    blocks = torch.arange(batch * pages, dtype=torch.int32).reshape(batch, pages)
    slots = blocks[:, (ctx - 1) // BLOCK].long() * BLOCK + (ctx - 1) % BLOCK
    lens = torch.full((batch,), ctx, dtype=torch.int32)
    shape = (batch * pages, HEADS, BLOCK, DIM)
    q = torch.randn(batch, HEADS, DIM, dtype=dtype)
    k = torch.randn(batch, HEADS, DIM, dtype=dtype)
    v = torch.randn(batch, HEADS, DIM, dtype=dtype)
    kc = torch.randn(shape, dtype=dtype)
    vc = torch.randn(shape, dtype=dtype)
    scale = DIM**-0.5
    if workers:  # 0 keeps the kernel's own physical-core rule
        mojo_attn._OVERRIDE = str(workers)
    return {
        "vllm": lambda: cpu_attn.self_attention(
            q, k, v, kc, vc, slots, blocks, lens, scale
        ),
        "mojo": lambda: mojo_attn.self_attention(
            q, k, v, kc, vc, slots, blocks, lens, scale
        ),
    }


def cross_case(batch, ctx, dtype, workers):
    kv = {
        i: (
            torch.randn(HEADS, ctx, DIM, dtype=dtype),
            torch.randn(HEADS, ctx, DIM, dtype=dtype),
        )
        for i in range(batch)
    }
    ids = list(range(batch))
    staged = {b: attention.stage_cross_kv(kv, ids, BLOCK, b) for b in ("vllm", "mojo")}
    q = torch.randn(batch, HEADS, DIM, dtype=dtype)
    scale = DIM**-0.5
    if workers:  # 0 keeps the kernel's own physical-core rule
        mojo_attn._OVERRIDE = str(workers)
    return {
        "vllm": lambda: cpu_attn.cross_attention(q, *staged["vllm"], scale),
        "mojo": lambda: mojo_attn.cross_attention(q, *staged["mojo"], scale),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)
    if args.threads:
        torch.set_num_threads(args.threads)

    import os
    import platform

    host = os.uname().nodename
    pinned = args.workers
    if pinned:
        mojo_attn._OVERRIDE = str(pinned)
    workers = mojo_attn._workers(1024)
    print(
        f"{host} {platform.machine()} cpus={os.cpu_count()} "
        f"physical={mojo_attn._physical_cores()} "
        f"torch_threads={torch.get_num_threads()} mojo_workers={workers}"
        f"{' (pinned)' if pinned else ''}, {args.dtype}"
    )
    for batch, ctx in [(1, 256), (8, 256), (8, 1024), (32, 1024), (1, 4096)]:
        for name, case in (("self(store)", self_case), ("cross(no store)", cross_case)):
            runs = case(batch, ctx, dtype, args.workers)
            pairs = []
            for rep in range(3):
                if rep % 2 == 0:
                    t_v, t_m = timed(runs["vllm"]), timed(runs["mojo"])
                else:
                    t_m, t_v = timed(runs["mojo"]), timed(runs["vllm"])
                pairs.append((t_v, t_m))
            t_v = min(p[0] for p in pairs)
            t_m = min(p[1] for p in pairs)
            med = statistics.median(m / v for v, m in pairs)
            print(
                f"b={batch:<3} ctx={ctx:<5}{name:<14}"
                f"{t_v:>9.1f}{t_m:>9.1f}{t_m / t_v:>11.2f}"
                f"   (median {med:.2f})"
            )


if __name__ == "__main__":
    main()
