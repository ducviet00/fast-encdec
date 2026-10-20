"""Standalone parity test for the Mojo paged-attention kernel.

Builds a tiny paged cache, scatters K/V through the kernel, then checks the
attention output against a scalar reference computed from the same cache.

    pixi run mojo run -I src tests/test_attention.mojo
"""

from std.memory import MutPointer, stack_allocation
from std.math import exp
from std.origin import MutUntrackedOrigin
from attention import BF16_DOT, LANES, NEON_DOT, Job, run

comptime HEAD_DIM = 64
comptime HEADS = 2
comptime MAX_PAGES = 6
comptime BLOCKS = 16
comptime NUM_SEQS = 2
comptime MAX_LEN = 200


@inline(.always)
def _ptr[T: AnyType](address: Int) -> MutPointer[T, MutUntrackedOrigin]:
    return MutPointer[T, MutUntrackedOrigin](unsafe_from_address=address)


def _fill[
    dtype: DType, n: Int
](p: MutPointer[Scalar[dtype], MutUntrackedOrigin], seed: Int) -> Int:
    """Deterministic pseudo-random fill in [-1, 1); returns the next seed."""
    var s = seed
    for i in range(n):
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        p.unsafe_store(i, Scalar[dtype]((Float64(s) / 1073741824.0) - 1.0))
    return s


def _reference[
    dtype: DType, BLOCK: Int
](
    q: Int,
    k_cache: Int,
    v_cache: Int,
    dst: Int,
    block_table: Int,
    lens: Int,
    num_seqs: Int,
    scale: Float64,
):
    """Scalar fp32 reference over the (already populated) paged cache."""
    comptime paired = BF16_DOT and dtype == DType.bfloat16 and BLOCK % LANES == 0
    var qp = _ptr[Scalar[dtype]](q)
    var kp = _ptr[Scalar[dtype]](k_cache)
    var vp = _ptr[Scalar[dtype]](v_cache)
    var op = _ptr[Scalar[dtype]](dst)
    var bt = _ptr[Int32](block_table)
    var lp = _ptr[Int32](lens)
    var logits = stack_allocation[MAX_LEN, DType.float32]()
    for seq in range(num_seqs):
        var len = Int(lp.unsafe_load(seq))
        for head in range(HEADS):
            var qrow = (seq * HEADS + head) * HEAD_DIM
            var row_max = Scalar[DType.float32](-3.4e38)
            for j in range(len):
                var block = Int(bt.unsafe_load(seq * MAX_PAGES + j // BLOCK))
                var off = j % BLOCK
                var kpage = (block * HEADS + head) * (BLOCK * HEAD_DIM)
                var sum = Scalar[DType.float32](0.0)
                for d in range(HEAD_DIM):
                    comptime if paired:
                        var k_idx = (
                            kpage + (d // 2) * BLOCK * 2 + off * 2 + d % 2
                        )
                        sum += Scalar[DType.float32](
                            qp.unsafe_load(qrow + d)
                        ) * Scalar[DType.float32](kp.unsafe_load(k_idx))
                    else:
                        sum += Scalar[DType.float32](
                            qp.unsafe_load(qrow + d)
                        ) * Scalar[DType.float32](
                            kp.unsafe_load(kpage + d * BLOCK + off)
                        )
                logits.unsafe_store(j, sum * Scalar[DType.float32](scale))
                row_max = max(row_max, logits.unsafe_load(j))
            var total = Scalar[DType.float32](0.0)
            for j in range(len):
                var e = exp(logits.unsafe_load(j) - row_max)
                logits.unsafe_store(j, e)
                total += e
            for d in range(HEAD_DIM):
                var acc = Scalar[DType.float32](0.0)
                for j in range(len):
                    var block = Int(
                        bt.unsafe_load(seq * MAX_PAGES + j // BLOCK)
                    )
                    var off = j % BLOCK
                    var vpage = (block * HEADS + head) * (BLOCK * HEAD_DIM)
                    comptime if NEON_DOT and dtype == DType.bfloat16:
                        # position-paired V page (see _store_head)
                        acc += logits.unsafe_load(j) * Scalar[DType.float32](
                            vp.unsafe_load(
                                vpage
                                + (off // 2) * HEAD_DIM * 2
                                + d * 2
                                + off % 2
                            )
                        )
                    else:
                        acc += logits.unsafe_load(j) * Scalar[DType.float32](
                            vp.unsafe_load(vpage + off * HEAD_DIM + d)
                        )
                op.unsafe_store(qrow + d, Scalar[dtype](acc / total))


def _run[dtype: DType, BLOCK: Int, len0: Int, len1: Int]() raises -> Bool:
    comptime N = NUM_SEQS * HEADS * HEAD_DIM
    comptime CACHE = BLOCKS * HEADS * BLOCK * HEAD_DIM
    var q = stack_allocation[N, dtype]()
    var k = stack_allocation[N, dtype]()
    var v = stack_allocation[N, dtype]()
    var k_cache = stack_allocation[CACHE, dtype]()
    var v_cache = stack_allocation[CACHE, dtype]()
    var out = stack_allocation[N, dtype]()
    var ref_out = stack_allocation[N, dtype]()
    var lens = stack_allocation[NUM_SEQS, DType.int32]()
    var slots = stack_allocation[NUM_SEQS, DType.int64]()
    var table = stack_allocation[NUM_SEQS * MAX_PAGES, DType.int32]()

    var seed = 1
    seed = _fill[dtype, N](q, seed)
    seed = _fill[dtype, N](k, seed)
    seed = _fill[dtype, N](v, seed)
    seed = _fill[dtype, CACHE](k_cache, seed)
    seed = _fill[dtype, CACHE](v_cache, seed)
    lens.unsafe_store(0, Int32(len0))
    lens.unsafe_store(1, Int32(len1))
    slots.unsafe_store(0, Int64(len0 - 1))
    slots.unsafe_store(1, Int64(len1 - 1))
    var next_page = 0
    for seq in range(NUM_SEQS):
        var len = len0 if seq == 0 else len1
        var pages = (len + BLOCK - 1) // BLOCK
        for p in range(MAX_PAGES):
            if p < pages:
                table.unsafe_store(seq * MAX_PAGES + p, Int32(next_page))
                next_page += 1
            else:
                table.unsafe_store(seq * MAX_PAGES + p, Int32(0))

    var scale = 1.0 / Float64(HEAD_DIM) ** 0.5
    var job = Job()
    job.q = q.__int__()
    job.k = k.__int__()
    job.v = v.__int__()
    job.k_cache = k_cache.__int__()
    job.v_cache = v_cache.__int__()
    job.dst = out.__int__()
    job.slots = slots.__int__()
    job.block_table = table.__int__()
    job.lens = lens.__int__()
    job.num_seqs = NUM_SEQS
    job.num_heads = HEADS
    job.num_kv_heads = HEADS
    job.head_dim = HEAD_DIM
    job.block_size = BLOCK
    job.bt_stride = MAX_PAGES
    job.q_stride = HEADS * HEAD_DIM
    job.kv_stride = HEADS * HEAD_DIM
    job.dst_stride = HEADS * HEAD_DIM
    job.scale = Float32(scale)
    run(
        job,
        0 if dtype == DType.float32 else 1,
        num_workers=1,
        do_store=True,
        do_attend=True,
    )
    _reference[dtype, BLOCK](
        q.__int__(),
        k_cache.__int__(),
        v_cache.__int__(),
        ref_out.__int__(),
        table.__int__(),
        lens.__int__(),
        NUM_SEQS,
        scale,
    )
    var worst = Float64(0)
    for i in range(N):
        var d = abs(
            Float64(out.unsafe_load(i)) - Float64(ref_out.unsafe_load(i))
        )
        worst = max(worst, d)
    var ok = worst < 2e-3
    print(
        "dtype",
        dtype,
        "block",
        BLOCK,
        "lens",
        len0,
        len1,
        "max|diff|",
        worst,
        "OK" if ok else "MISMATCH",
    )
    return ok


def main() raises:
    var ok = True
    ok = _run[DType.float32, 32, 1, 37]() and ok
    ok = _run[DType.float32, 32, 32, 96]() and ok
    ok = _run[DType.bfloat16, 32, 1, 37]() and ok
    ok = _run[DType.bfloat16, 32, 32, 96]() and ok
    ok = _run[DType.float32, 16, 1, 37]() and ok
    ok = _run[DType.float32, 16, 48, 96]() and ok
    ok = _run[DType.bfloat16, 16, 33, 79]() and ok
    # tile < block_size: several tiles per page, groups straddle pages
    ok = _run[DType.float32, 64, 1, 37]() and ok
    ok = _run[DType.float32, 64, 96, 200]() and ok
    ok = _run[DType.bfloat16, 64, 33, 200]() and ok
    ok = _run[DType.bfloat16, 48, 79, 190]() and ok
    if not ok:
        raise Error("parity mismatch")
