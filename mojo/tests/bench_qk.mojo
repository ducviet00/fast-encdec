"""Micro-benchmark: QK phase, fp32 broadcast-FMA vs AVX512-BF16 dot product.

Both loops compute the same thing (one page of 32 KV positions dotted with one
query head, head_dim 64) and are sized so a page is processed per iteration.

    pixi run mojo run tests/bench_qk.mojo
"""

from std.memory import stack_allocation
from std.math import iota
from std.sys import llvm_intrinsic
from std.time import perf_counter_ns

comptime D = 64
comptime TILE = 32
comptime PAGES = 64


def qk_fp32(
    k_page: Pointer[BFloat16, MutUntrackedOrigin],
    q_scaled: Pointer[Float32, MutUntrackedOrigin],
) -> SIMD[DType.float32, TILE]:
    """logits[j] = sum_d q[d] * k[d][j]; K page is [D][TILE] row-major."""
    var l0 = SIMD[DType.float32, TILE](0.0)
    var l1 = SIMD[DType.float32, TILE](0.0)
    var l2 = SIMD[DType.float32, TILE](0.0)
    var l3 = SIMD[DType.float32, TILE](0.0)
    var d = 0
    while d + 4 <= D:
        l0 += (
            q_scaled.unsafe_load(d)
            * k_page.unsafe_load[width=TILE](d * TILE).cast[DType.float32]()
        )
        l1 += (
            q_scaled.unsafe_load(d + 1)
            * k_page.unsafe_load[width=TILE]((d + 1) * TILE).cast[
                DType.float32
            ]()
        )
        l2 += (
            q_scaled.unsafe_load(d + 2)
            * k_page.unsafe_load[width=TILE]((d + 2) * TILE).cast[
                DType.float32
            ]()
        )
        l3 += (
            q_scaled.unsafe_load(d + 3)
            * k_page.unsafe_load[width=TILE]((d + 3) * TILE).cast[
                DType.float32
            ]()
        )
        d += 4
    return (l0 + l1) + (l2 + l3)


def qk_bfdot(
    k_page: Pointer[BFloat16, MutUntrackedOrigin],
    q_pairs: Pointer[BFloat16, MutUntrackedOrigin],
) -> SIMD[DType.float32, TILE]:
    """Same logits with vdpbf16ps; K page is [D/2][TILE][2] (d pairs interleaved).
    """
    var lo = SIMD[DType.float32, 16](0.0)
    var hi = SIMD[DType.float32, 16](0.0)
    var dp = 0
    while dp < D // 2:
        var a = q_pairs.unsafe_load[width=32](dp * 32)
        lo = llvm_intrinsic[
            "llvm.x86.avx512bf16.dpbf16ps.512",
            SIMD[DType.float32, 16],
            has_side_effect=False,
        ](lo, a, k_page.unsafe_load[width=32](dp * TILE * 2))
        hi = llvm_intrinsic[
            "llvm.x86.avx512bf16.dpbf16ps.512",
            SIMD[DType.float32, 16],
            has_side_effect=False,
        ](hi, a, k_page.unsafe_load[width=32](dp * TILE * 2 + 32))
        dp += 1
    return lo.join(hi)


def main() raises:
    var k_fp32 = stack_allocation[PAGES * D * TILE, BFloat16]()
    var k_bfdot = stack_allocation[PAGES * D * TILE, BFloat16]()
    var q_scaled = stack_allocation[D, DType.float32]()
    var q_pairs = stack_allocation[D * 16, BFloat16]()
    for i in range(PAGES * D * TILE):
        k_fp32.unsafe_store(i, BFloat16(0.01))
        # d-pair-interleaved copy of one page: (d, j) -> (d // 2) * TILE * 2 + j * 2 + d % 2
        var d = (i % (D * TILE)) // TILE
        var j = i % TILE
        var page = i // (D * TILE)
        k_bfdot.unsafe_store(
            page * D * TILE + (d // 2) * TILE * 2 + j * 2 + d % 2,
            BFloat16(0.01),
        )
    for d in range(D):
        q_scaled.unsafe_store(d, Float32(0.02))
        for j in range(16):
            q_pairs.unsafe_store(j * 32 + (d // 2) * 2 + d % 2, BFloat16(0.02))

    var sink = SIMD[DType.float32, TILE](0.0)
    var iters = 200000

    for _ in range(1000):
        sink += qk_fp32(k_fp32, q_scaled)
    var t0 = perf_counter_ns()
    for i in range(iters):
        var page = i % PAGES
        sink += qk_fp32(k_fp32.unsafe_offset(page * D * TILE), q_scaled)
    var t1 = perf_counter_ns()
    print("fp32 broadcast-FMA:", Float64(t1 - t0) / Float64(iters), "ns/tile")

    for _ in range(1000):
        sink += qk_bfdot(k_bfdot, q_pairs)
    t0 = perf_counter_ns()
    for i in range(iters):
        var page = i % PAGES
        sink += qk_bfdot(k_bfdot.unsafe_offset(page * D * TILE), q_pairs)
    t1 = perf_counter_ns()
    print("avx512-bf16 dp:", Float64(t1 - t0) / Float64(iters), "ns/tile")
    print("sink", sink[0])
