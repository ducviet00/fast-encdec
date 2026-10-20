"""Reference benchmark: MAX's own CPU attention vs the fast-encdec kernels.

Times ``nn.attention.cpu.mha.flash_attention`` at the shape the engine's
decoder attention runs (heads=16, head_dim=64, one query token per sequence,
dense non-causal KV), so its latency can be compared with
``benchmarks/attn_backend.py``'s paged vLLM and Mojo columns.

MAX's kernel is not paged: the head dimension is folded into the batch
dimension (BSD layout) and K/V are contiguous, which is the friendliest layout
for it. Its softmax and accumulation run in the input dtype (bf16 here), while
the fast-encdec kernels widen to fp32.

    pixi run mojo run -I src tests/bench_max_attention.mojo
"""

from max.algorithm import parallelize, sync_parallelize
from max.runtime.asyncrt import parallelism_level
from std.memory import alloc, unsafe_memset_zero
from std.sys import simd_width_of
from std.sys.info import align_of
from std.time import perf_counter_ns
from layout import Coord, TileTensor, row_major
from nn.attention.cpu.mha import flash_attention
from std.utils import IndexList
from std.utils.index import Index

comptime HEADS = 16
comptime DEPTH = 64
comptime SCALE = 0.125


def _bench[dtype: DType](batch: Int, ctx: Int, iters: Int) -> Float64:
    """Return microseconds per call for one (batch, ctx) point."""
    var bh = batch * HEADS
    var q_shape = Index(bh, 1, DEPTH)
    var kv_shape = Index(bh, ctx, DEPTH)
    var mask_shape = Index(bh, 1, ctx)

    var q_alloc = alloc[Scalar[dtype]](
        {length = q_shape.flattened_length()}
    ).into_managed()
    var k_alloc = alloc[Scalar[dtype]](
        {length = kv_shape.flattened_length()}
    ).into_managed()
    var v_alloc = alloc[Scalar[dtype]](
        {length = kv_shape.flattened_length()}
    ).into_managed()
    var mask_alloc = alloc[Scalar[dtype]](
        {length = mask_shape.flattened_length()}
    ).into_managed()
    var out_alloc = alloc[Scalar[dtype]](
        {length = q_shape.flattened_length()}
    ).into_managed()

    # Ones for q/k/v so nothing depends on uninitialised memory; a zero mask is
    # the additive identity, i.e. plain non-causal attention.
    var one = Scalar[dtype](1.0)
    var qp = q_alloc.unsafe_ptr()
    var kp = k_alloc.unsafe_ptr()
    var vp = v_alloc.unsafe_ptr()
    for i in range(q_shape.flattened_length()):
        qp.unsafe_store(i, one)
    for i in range(kv_shape.flattened_length()):
        kp.unsafe_store(i, one)
        vp.unsafe_store(i, one)
    unsafe_memset_zero(mask_alloc.unsafe_ptr(), mask_shape.flattened_length())

    var q = (
        TileTensor(q_alloc.unsafe_span(), row_major(len(q_alloc)))
        .reshape(Coord(q_shape))
        .as_imm()
    )
    var k = (
        TileTensor(k_alloc.unsafe_span(), row_major(len(k_alloc)))
        .reshape(Coord(kv_shape))
        .as_imm()
        .as_unsafe_any_origin()
    )
    var v = (
        TileTensor(v_alloc.unsafe_span(), row_major(len(v_alloc)))
        .reshape(Coord(kv_shape))
        .as_imm()
        .as_unsafe_any_origin()
    )
    var mask = (
        TileTensor(mask_alloc.unsafe_span(), row_major(len(mask_alloc)))
        .reshape(Coord(mask_shape))
        .as_imm()
        .as_unsafe_any_origin()
    )
    var output = TileTensor(
        out_alloc.unsafe_span(), row_major(len(out_alloc))
    ).reshape(Coord(q_shape))

    @inline(.always)
    def input_k_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) {imm} -> SIMD[dtype, simd_width]:
        comptime assert _rank == 3
        return k.load[width=simd_width, alignment=align_of[dtype]()](Coord(idx))

    @inline(.always)
    def input_v_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) {imm} -> SIMD[dtype, simd_width]:
        comptime assert _rank == 3
        return v.load[width=simd_width, alignment=align_of[dtype]()](Coord(idx))

    @inline(.always)
    def mask_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) {imm} -> SIMD[dtype, simd_width]:
        comptime assert _rank == 3
        return mask.load[width=simd_width, alignment=align_of[dtype]()](
            Coord(idx)
        )

    @inline(.always)
    def run_once() {imm}:
        flash_attention(
            input_k_fn,
            input_v_fn,
            mask_fn,
            q,
            kv_shape,
            kv_shape,
            mask_shape,
            output,
            scale=SCALE,
        )

    for _ in range(5):
        run_once()
    var best = 1e18
    for _ in range(iters):
        var t0 = perf_counter_ns()
        run_once()
        var dt = Float64(perf_counter_ns() - t0)
        best = min(best, dt)
    return best / 1000.0


def _dispatch_cost(iters: Int) -> Tuple[Float64, Float64]:
    """Return (parallelize, sync_parallelize) overhead in us for 8 workers."""
    var sink = List[Float64](length=64, fill=0.0)
    var sp = sink.unsafe_ptr()

    def body(tid: Int) {imm}:
        sp[tid] = Float64(tid)

    for _ in range(20):
        parallelize(body, 8, 8)
    var best_a = 1e18
    for _ in range(iters):
        var t0 = perf_counter_ns()
        parallelize(body, 8, 8)
        best_a = min(best_a, Float64(perf_counter_ns() - t0))

    def body_r(tid: Int) raises {imm}:
        sp[tid] = Float64(tid)

    for _ in range(20):
        sync_parallelize(body_r, 8)
    var best_b = 1e18
    for _ in range(iters):
        var t0 = perf_counter_ns()
        sync_parallelize(body_r, 8)
        best_b = min(best_b, Float64(perf_counter_ns() - t0))
    return Tuple(Float64(best_a / 1000.0), Float64(best_b / 1000.0))


def main() raises:
    print("parallelism_level:", parallelism_level(None))
    var costs = _dispatch_cost(iters=200)
    print(
        "dispatch overhead (8 workers): parallelize",
        costs[0],
        "us; sync_parallelize",
        costs[1],
        "us",
    )
    print(
        "simd_width_of: bf16",
        simd_width_of[DType.bfloat16](),
        "f32",
        simd_width_of[DType.float32](),
    )
    print("MAX CPU flash_attention (heads=16, head_dim=64, dense KV)")
    print("dtype  batch   ctx   us/call")
    for batch, ctx in [(1, 256), (8, 256), (8, 1024), (1, 4096), (32, 1024)]:
        var bf = _bench[DType.bfloat16](batch, ctx, iters=30)
        var f32 = _bench[DType.float32](batch, ctx, iters=30)
        print("bf16", batch, ctx, bf)
        print("f32 ", batch, ctx, f32)
