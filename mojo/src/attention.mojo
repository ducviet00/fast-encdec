"""Paged decoder attention for fast-encdec, in Mojo.

A port of the x86 ``VEC`` path of ``csrc/cpu`` (the vendored vLLM CPU
attention kernel) restricted to what this engine actually calls:

* one query token per sequence (the runner feeds exactly one token per step),
  attending to every cached position -- the query is the last position, so the
  causal mask is the cache tail;
* KV cache pages laid out ``[num_blocks, num_kv_heads, block_size * head_dim]``
  with K stored transposed inside the page (``[head_dim, block_size]``,
  row-major) and V row-major (``[block_size, head_dim]``). This is exactly the
  vLLM layout, so both backends share cache tensors and staging code;
* fp32 arithmetic, the cache dtype (bf16/fp32) widened on load -- the same
  numerics class as the C++ ``VEC`` kernel, which also converts to fp32.

Work is parallelized over (sequence, head) pairs: one task owns one head of one
sequence end to end, so no cross-thread reduction or scratch buffer is needed.
"""

from std.math import exp, iota
from std.memory import MutPointer, stack_allocation
from std.origin import MutUntrackedOrigin
from std.sys import llvm_intrinsic, simd_width_of
from std.sys.info import CompilationTarget
from max.algorithm import parallelize

# Compile-time ISA gating, the same shape as MAX's `_MatmulConfig._get_config()`
# (which switches inner loops on `CompilationTarget.has_neon()/has_avx512f()`).
comptime X86_DOT = CompilationTarget.is_x86() and CompilationTarget._has_feature[
    "avx512bf16"
]()
comptime NEON_DOT = CompilationTarget.has_neon() and CompilationTarget._has_feature[
    "bf16"
]()
comptime BF16_DOT = X86_DOT or NEON_DOT
"""Use a bf16 dot-product instruction for Q@K^T (2 MACs per lane, fp32 accumulate).

One source for both ISAs: the accumulator width is ``LANES`` (the f32 lane
count) and one instruction consumes ``2 * LANES`` bf16, i.e. two head dims of
``LANES`` positions, so the K page layout is shared: each (k[2d], k[2d+1]) pair
is stored next to the position it belongs to, ``[head_dim/2, block_size, 2]``.
Products of two bf16 values are exact in fp32, so this is numerically the same
math as widening first -- only the summation order differs.
"""

comptime LANES = simd_width_of[DType.float32]()
"""f32 lanes per register, i.e. the dot-product accumulator width: 16 on AVX-512,
4 on NEON (and 8 on AVX2, which has no dot instruction and takes the fp32 path).
"""

comptime GROUP = 4
"""Pages scored before one online-softmax step (amortizes max/exp/rescale).

The idea is MAX's (`max/kernels/src/nn/attention/cpu/mha.mojo` scores
`qk_block_n = 128` positions per `_online_softmax` call) and vLLM's C++ kernel
sizes the equivalent tile from its L2 budget; four 32-token pages keeps the
four logits vectors plus the accumulator inside the AVX-512 register file while
cutting the softmax bookkeeping 4x at the ~128-token contexts a decode step
spends most of its time at.
"""

comptime NEG_INF = -3.4028234663852886e38
"""Stand-in for -inf: ``exp()`` underflows it to 0 and it never produces NaN."""


struct Job(Copyable, Movable):
    """One attention call, as raw addresses (see ``run``)."""

    var q: Int
    var k: Int
    var v: Int
    var k_cache: Int
    var v_cache: Int
    var dst: Int
    var slots: Int
    var block_table: Int
    var lens: Int
    var num_seqs: Int
    var num_heads: Int
    var num_kv_heads: Int
    var head_dim: Int
    var block_size: Int
    var bt_stride: Int
    # Token strides: q/k/v are views of one fused qkv projection, so a token can
    # sit further apart than num_heads * head_dim (head_dim itself is contiguous).
    var q_stride: Int
    var kv_stride: Int
    var dst_stride: Int
    var scale: Float32

    def __init__(out self):
        self.q = 0
        self.k = 0
        self.v = 0
        self.k_cache = 0
        self.v_cache = 0
        self.dst = 0
        self.slots = 0
        self.block_table = 0
        self.lens = 0
        self.num_seqs = 0
        self.num_heads = 0
        self.num_kv_heads = 0
        self.head_dim = 0
        self.block_size = 0
        self.bt_stride = 0
        self.q_stride = 0
        self.kv_stride = 0
        self.dst_stride = 0
        self.scale = 1.0


@inline(.always)
def _ptr[T: AnyType](address: Int) -> MutPointer[T, MutUntrackedOrigin]:
    return MutPointer[T, MutUntrackedOrigin](unsafe_from_address=address)


@inline(.always)
def _store_head[
    dtype: DType, head_dim: Int, tile: Int
](job: Job, token: Int, head: Int, dst_head: Int, slot: Int):
    """Scatter one head's new K/V into its page.

    V is row-major ``[block_size, head_dim]``; K stores one token as a column of
    ``[head_dim, block_size]``, or, when Q@K^T runs as bf16 dot products, as
    ``[head_dim/2, block_size, 2]`` pairs so a (k[2d], k[2d+1]) pair of one
    position is contiguous.
    """
    var block_size = job.block_size
    var page = (slot // block_size * job.num_kv_heads + dst_head) * (
        block_size * head_dim
    )
    var off = slot % block_size
    var src = token * job.kv_stride + head * head_dim
    var k_src = _ptr[Scalar[dtype]](job.k).unsafe_offset(src)
    var v_src = _ptr[Scalar[dtype]](job.v).unsafe_offset(src)
    comptime if NEON_DOT and dtype == DType.bfloat16 and tile % LANES == 0:
        # bf16-dot P@V: `bfdot` contracts 2*LANES adjacent bf16 values, so the two
        # positions of a pair are stored next to each other per head dim,
        # [block_size/2, head_dim, 2], and one instruction covers LANES dims.
        var v_dst = _ptr[Scalar[dtype]](job.v_cache).unsafe_offset(
            page + (off // 2) * (head_dim * 2) + off % 2
        )
        var d = 0
        while d < head_dim:
            v_dst.unsafe_store(d * 2, v_src.unsafe_load(d))
            d += 1
    else:
        var v_dst = _ptr[Scalar[dtype]](job.v_cache).unsafe_offset(
            page + off * head_dim
        )
        v_dst.unsafe_store[width=head_dim](v_src.unsafe_load[width=head_dim]())
    comptime if BF16_DOT and dtype == DType.bfloat16 and tile % LANES == 0:
        var k_dst = _ptr[Scalar[dtype]](job.k_cache).unsafe_offset(
            page + off * 2
        )
        var dp = 0
        while dp < head_dim // 2:
            k_dst.unsafe_store[width=2](
                dp * block_size * 2, k_src.unsafe_load[width=2](dp * 2)
            )
            dp += 1
    else:
        var k_dst = _ptr[Scalar[dtype]](job.k_cache).unsafe_offset(page + off)
        comptime STEP = 8
        var d = 0
        while d < head_dim:
            comptime for lane in range(STEP):
                k_dst.unsafe_store(
                    (d + lane) * block_size, k_src.unsafe_load(d + lane)
                )
            d += STEP


@inline(.always)
def _bf16_dot[
    dtype: DType
](
    acc: SIMD[DType.float32, LANES],
    a: SIMD[DType.bfloat16, 2 * LANES],
    b: SIMD[dtype, 2 * LANES],
) -> SIMD[DType.float32, LANES]:
    """acc[i] += a[2i]*b[2i] + a[2i+1]*b[2i+1]; fp32 accumulate, exact products.
    """
    comptime if X86_DOT:
        return llvm_intrinsic[
            "llvm.x86.avx512bf16.dpbf16ps.512",
            SIMD[DType.float32, LANES],
            has_side_effect=False,
        ](acc, a, b.cast[DType.bfloat16]())
    elif NEON_DOT:
        return llvm_intrinsic[
            "llvm.aarch64.neon.bfdot",
            SIMD[DType.float32, LANES],
            has_side_effect=False,
        ](acc, a, b.cast[DType.bfloat16]())
    else:
        var out = acc
        for i in range(LANES):
            out[i] += Scalar[DType.float32](a[2 * i]) * Scalar[DType.float32](
                b[2 * i]
            ) + Scalar[DType.float32](a[2 * i + 1]) * Scalar[DType.float32](
                b[2 * i + 1]
            )
        return out


@inline(.always)
def _attention_head[
    dtype: DType, head_dim: Int, tile: Int, group: Int
](job: Job, seq: Int, head: Int, kv_head: Int):
    """Paged attention for one (sequence, query head) with online softmax.

    ``group`` tiles of ``tile`` KV positions are scored before one softmax pass,
    so the online-softmax bookkeeping (max, exp, rescale) is amortized the way
    the C++ kernel amortizes it over its L2-sized tile. ``tile`` must divide
    ``block_size`` (32 on x86, 16 for aarch64): a page holds ``block_size //
    tile`` tiles, so a group can straddle pages (and a page can span groups).
    """
    var block_size = job.block_size
    var page = block_size * head_dim
    var lens = Int(_ptr[Int32](job.lens).unsafe_load(seq))
    var bt = _ptr[Int32](job.block_table).unsafe_offset(seq * job.bt_stride)

    var q_src = _ptr[Scalar[dtype]](job.q).unsafe_offset(
        seq * job.q_stride + head * head_dim
    )
    # The fp32 path widens Q and folds `scale` in (like the C++ kernel); the
    # bf16-dot path keeps Q in bf16 and scales the logits instead, so Q is not
    # rounded through a scale multiply.
    var q = stack_allocation[head_dim, DType.float32]()
    q.unsafe_store[width=head_dim](
        q_src.unsafe_load[width=head_dim]().cast[DType.float32]() * job.scale
    )
    var dst = _ptr[Scalar[dtype]](job.dst).unsafe_offset(
        seq * job.dst_stride + head * head_dim
    )

    var acc = SIMD[DType.float32, head_dim](0.0)
    var row_max = Scalar[DType.float32](NEG_INF)
    var row_sum = Scalar[DType.float32](0.0)
    var group_logits = Array[SIMD[DType.float32, tile], group](
        uninitialized=True
    )

    var block_idx = 0
    var pos = 0
    while pos < lens:
        # Tiles scored before one softmax step. A tile is `tile` consecutive
        # positions inside a page; a group spans pages (and a page spans groups)
        # whenever block_size != tile, so (page, offset) is resolved per tile and
        # shared by Q@K^T and P@V instead of being recomputed from the block table.
        var in_page = pos % block_size
        var tiles = min(group, (lens - pos + tile - 1) // tile)
        var tile_page = Array[Int, group](uninitialized=True)
        var tile_off = Array[Int, group](uninitialized=True)
        comptime for g in range(group):
            if g < tiles:
                var tpos = in_page + g * tile
                tile_page[g] = Int(
                    bt.unsafe_load(block_idx + tpos // block_size)
                )
                tile_off[g] = tpos % block_size

        # QK^T for every tile in the group. The page holds k[d][0:block_size]
        # contiguously, so one row of K is a tile-wide vector; four accumulators
        # keep the FMA chain independent.
        comptime for g in range(group):
            if g < tiles:
                var k_tile = _ptr[Scalar[dtype]](job.k_cache).unsafe_offset(
                    (tile_page[g] * job.num_kv_heads + kv_head) * page
                    + tile_off[g]
                )
                var l0 = SIMD[DType.float32, tile](0.0)
                var l1 = SIMD[DType.float32, tile](0.0)
                var l2 = SIMD[DType.float32, tile](0.0)
                var l3 = SIMD[DType.float32, tile](0.0)
                var d = 0
                while d + 4 <= head_dim:
                    l0 += (
                        q.unsafe_load(d)
                        * k_tile.unsafe_load[width=tile](d * block_size).cast[
                            DType.float32
                        ]()
                    )
                    l1 += (
                        q.unsafe_load(d + 1)
                        * k_tile.unsafe_load[width=tile](
                            (d + 1) * block_size
                        ).cast[DType.float32]()
                    )
                    l2 += (
                        q.unsafe_load(d + 2)
                        * k_tile.unsafe_load[width=tile](
                            (d + 2) * block_size
                        ).cast[DType.float32]()
                    )
                    l3 += (
                        q.unsafe_load(d + 3)
                        * k_tile.unsafe_load[width=tile](
                            (d + 3) * block_size
                        ).cast[DType.float32]()
                    )
                    d += 4
                while d < head_dim:
                    l0 += (
                        q.unsafe_load(d)
                        * k_tile.unsafe_load[width=tile](d * block_size).cast[
                            DType.float32
                        ]()
                    )
                    d += 1
                var logits = (l0 + l1) + (l2 + l3)
                # Only the sequence's last tile can be short.
                var tile_len = min(tile, lens - pos - g * tile)
                if tile_len < tile:
                    var idx = iota[DType.int32, tile](Int32(pos + g * tile))
                    var valid = idx.lt(SIMD[DType.int32, tile](Int32(lens)))
                    logits = valid.select(
                        logits, SIMD[DType.float32, tile](NEG_INF)
                    )
                group_logits[g] = logits

        # One online-softmax step for the whole group.
        var new_max = row_max
        comptime for g in range(group):
            if g < tiles:
                new_max = max(
                    new_max, Scalar[DType.float32](group_logits[g].reduce_max())
                )
        var alpha = exp(row_max - new_max)
        var prob_sum = Scalar[DType.float32](0.0)
        comptime for g in range(group):
            if g < tiles:
                var p = exp(group_logits[g] - new_max)
                prob_sum += Scalar[DType.float32](p.reduce_add())
                group_logits[g] = p
        row_sum = row_sum * alpha + prob_sum
        row_max = new_max
        acc = acc * alpha

        # P@V: acc[d] += sum_j prob[j] * v[j][d]; V is row-major [block_size, head_dim].
        comptime for g in range(group):
            if g < tiles:
                var v_tile = _ptr[Scalar[dtype]](job.v_cache).unsafe_offset(
                    (tile_page[g] * job.num_kv_heads + kv_head) * page
                    + tile_off[g] * head_dim
                )
                var p = group_logits[g]
                var tile_len = min(tile, lens - pos - g * tile)
                if tile_len == tile:
                    comptime for j in range(tile):
                        acc += (
                            p[j]
                            * v_tile.unsafe_load[width=head_dim](
                                j * head_dim
                            ).cast[DType.float32]()
                        )
                else:
                    # Short final tile: probabilities go through memory so the
                    # loop can stop at tile_len.
                    var probs = stack_allocation[tile, DType.float32]()
                    probs.unsafe_store[width=tile](p)
                    for j in range(tile_len):
                        acc += (
                            probs.unsafe_load(j)
                            * v_tile.unsafe_load[width=head_dim](
                                j * head_dim
                            ).cast[DType.float32]()
                        )
        pos += tiles * tile
        block_idx += (in_page + tiles * tile) // block_size

    var inv = Scalar[DType.float32](1.0) / row_sum
    dst.unsafe_store[width=head_dim]((acc * inv).cast[dtype]())


@inline(.always)
def _attention_head_dot[
    dtype: DType, head_dim: Int, tile: Int
](job: Job, seq: Int, head: Int, kv_head: Int):
    """``_attention_head`` with Q@K^T as bf16 dot products (AVX512-BF16 or NEON).

    Same online softmax and P@V; only Q@K^T and the K page layout differ. Q is
    kept in bf16 and ``scale`` is applied to the logits, so Q is never rounded
    through a scaled multiply. A (k[2d], k[2d+1]) pair of one position is
    contiguous in the page, and one instruction covers two head dims of
    ``LANES`` positions, so the page needs ``tile / LANES`` accumulators. A page
    holds ``block_size // tile`` tiles, walked by the same group loop as
    ``_attention_head``.
    """
    comptime NV = tile // LANES
    comptime OPN = 2 * LANES
    var block_size = job.block_size
    var page = block_size * head_dim
    var lens = Int(_ptr[Int32](job.lens).unsafe_load(seq))
    var bt = _ptr[Int32](job.block_table).unsafe_offset(seq * job.bt_stride)

    # Broadcast each (q[2d], q[2d+1]) pair across LANES lanes, once per task.
    var q_pairs = stack_allocation[(head_dim // 2) * OPN, BFloat16]()
    var q_row = _ptr[Scalar[dtype]](job.q).unsafe_offset(
        seq * job.q_stride + head * head_dim
    )
    # One 4-byte load per pair, then broadcast; constant indices keep everything
    # in registers.
    comptime for dp in range(head_dim // 2):
        var q0 = Scalar[DType.bfloat16](q_row.unsafe_load(2 * dp))
        var q1 = Scalar[DType.bfloat16](q_row.unsafe_load(2 * dp + 1))
        var lo = SIMD[DType.bfloat16, LANES](q0)
        var hi = SIMD[DType.bfloat16, LANES](q1)
        q_pairs.unsafe_store[width=OPN](dp * OPN, lo.interleave(hi))

    var dst = _ptr[Scalar[dtype]](job.dst).unsafe_offset(
        seq * job.dst_stride + head * head_dim
    )
    var acc = SIMD[DType.float32, head_dim](0.0)
    var row_max = Scalar[DType.float32](NEG_INF)
    var row_sum = Scalar[DType.float32](0.0)
    var group_logits = Array[SIMD[DType.float32, tile], GROUP](
        uninitialized=True
    )

    var block_idx = 0
    var pos = 0
    while pos < lens:
        # Tiles of `tile` positions; `block_size // tile` of them make one page.
        var in_page = pos % block_size
        var tiles = min(GROUP, (lens - pos + tile - 1) // tile)
        var tile_page = Array[Int, GROUP](uninitialized=True)
        var tile_off = Array[Int, GROUP](uninitialized=True)
        comptime for g in range(GROUP):
            if g < tiles:
                var tpos = in_page + g * tile
                tile_page[g] = Int(
                    bt.unsafe_load(block_idx + tpos // block_size)
                )
                tile_off[g] = tpos % block_size
        comptime for g in range(GROUP):
            if g < tiles:
                var k_tile = _ptr[Scalar[dtype]](job.k_cache).unsafe_offset(
                    (tile_page[g] * job.num_kv_heads + kv_head) * page
                    + tile_off[g] * 2
                )
                # Two accumulator sets (even/odd head-dim pair) for ILP.
                var acc0 = Array[SIMD[DType.float32, LANES], NV](
                    uninitialized=True
                )
                var acc1 = Array[SIMD[DType.float32, LANES], NV](
                    uninitialized=True
                )
                comptime for v in range(NV):
                    acc0[v] = SIMD[DType.float32, LANES](0.0)
                    acc1[v] = SIMD[DType.float32, LANES](0.0)
                var dp = 0
                while dp + 2 <= head_dim // 2:
                    var a0 = q_pairs.unsafe_load[width=OPN](dp * OPN)
                    var a1 = q_pairs.unsafe_load[width=OPN]((dp + 1) * OPN)
                    comptime for v in range(NV):
                        acc0[v] = _bf16_dot[dtype](
                            acc0[v],
                            a0,
                            k_tile.unsafe_load[width=OPN](
                                dp * block_size * 2 + v * OPN
                            ),
                        )
                        acc1[v] = _bf16_dot[dtype](
                            acc1[v],
                            a1,
                            k_tile.unsafe_load[width=OPN](
                                (dp + 1) * block_size * 2 + v * OPN
                            ),
                        )
                    dp += 2
                var logits = SIMD[DType.float32, tile](0.0)
                comptime for v in range(NV):
                    var row = acc0[v] + acc1[v]
                    comptime for i in range(LANES):
                        logits[v * LANES + i] = row[i]
                logits = logits * job.scale
                var tile_len = min(tile, lens - pos - g * tile)
                if tile_len < tile:
                    var idx = iota[DType.int32, tile](Int32(pos + g * tile))
                    var valid = idx.lt(SIMD[DType.int32, tile](Int32(lens)))
                    logits = valid.select(
                        logits, SIMD[DType.float32, tile](NEG_INF)
                    )
                group_logits[g] = logits

        var new_max = row_max
        comptime for g in range(GROUP):
            if g < tiles:
                new_max = max(
                    new_max, Scalar[DType.float32](group_logits[g].reduce_max())
                )
        var alpha = exp(row_max - new_max)
        var prob_sum = Scalar[DType.float32](0.0)
        comptime for g in range(GROUP):
            if g < tiles:
                var p = exp(group_logits[g] - new_max)
                prob_sum += Scalar[DType.float32](p.reduce_add())
                group_logits[g] = p
        row_sum = row_sum * alpha + prob_sum
        row_max = new_max
        acc = acc * alpha

        comptime for g in range(GROUP):
            if g < tiles:
                var p = group_logits[g]
                var tile_len = min(tile, lens - pos - g * tile)
                comptime if NEON_DOT:
                    # One bfdot per LANES head dims, from the position-paired V
                    # page; the probabilities are rounded to bf16, as the C++
                    # kernel's prob_buffer_t does. The tile's contribution lands
                    # in its own accumulator, folded into the fp32 one at the end.
                    var pairs = tile_len // 2
                    var accv = Array[
                        SIMD[DType.float32, LANES], head_dim // LANES
                    ](uninitialized=True)
                    comptime for dg in range(head_dim // LANES):
                        accv[dg] = SIMD[DType.float32, LANES](0.0)
                    var v_pairs = _ptr[Scalar[dtype]](
                        job.v_cache
                    ).unsafe_offset(
                        (tile_page[g] * job.num_kv_heads + kv_head) * page
                        + (tile_off[g] // 2) * (head_dim * 2)
                    )
                    comptime for jp in range(tile // 2):
                        if jp < pairs:
                            var b = SIMD[DType.bfloat16, LANES](
                                Scalar[DType.bfloat16](p[2 * jp])
                            ).interleave(
                                SIMD[DType.bfloat16, LANES](
                                    Scalar[DType.bfloat16](p[2 * jp + 1])
                                )
                            )
                            comptime for dg in range(head_dim // LANES):
                                accv[dg] = _bf16_dot[dtype](
                                    accv[dg],
                                    b,
                                    v_pairs.unsafe_load[width=2 * LANES](
                                        (jp * head_dim + dg * LANES) * 2
                                    ),
                                )
                    # Odd tile length: the last position has no pair, so fp32
                    # with a strided gather -- a paired read would fold in the
                    # neighbour, which the next decode step overwrites.
                    if 2 * pairs < tile_len:
                        var pl = Scalar[DType.float32](p[2 * pairs])
                        comptime for dg in range(head_dim // LANES):
                            var v8 = v_pairs.unsafe_load[width=2 * LANES](
                                pairs * head_dim * 2 + dg * 2 * LANES
                            ).cast[DType.float32]()
                            var even = SIMD[DType.float32, LANES](0.0)
                            comptime for i in range(LANES):
                                even[i] = v8[2 * i]
                            accv[dg] += pl * even
                    var joined = stack_allocation[head_dim, DType.float32]()
                    comptime for dg in range(head_dim // LANES):
                        joined.unsafe_store[width=LANES](dg * LANES, accv[dg])
                    acc += joined.unsafe_load[width=head_dim]()
                else:
                    var v_tile = _ptr[Scalar[dtype]](job.v_cache).unsafe_offset(
                        (tile_page[g] * job.num_kv_heads + kv_head) * page
                        + tile_off[g] * head_dim
                    )
                    if tile_len == tile:
                        comptime for j in range(tile):
                            acc += (
                                p[j]
                                * v_tile.unsafe_load[width=head_dim](
                                    j * head_dim
                                ).cast[DType.float32]()
                            )
                    else:
                        var probs = stack_allocation[tile, DType.float32]()
                        probs.unsafe_store[width=tile](p)
                        for j in range(tile_len):
                            acc += (
                                probs.unsafe_load(j)
                                * v_tile.unsafe_load[width=head_dim](
                                    j * head_dim
                                ).cast[DType.float32]()
                            )
        pos += tiles * tile
        block_idx += (in_page + tiles * tile) // block_size

    var inv = Scalar[DType.float32](1.0) / row_sum
    dst.unsafe_store[width=head_dim]((acc * inv).cast[dtype]())


@inline(.always)
def _task[
    dtype: DType, head_dim: Int, tile: Int, group: Int
](job: Job, task: Int, do_store: Bool, do_attend: Bool):
    """Store (self-attention only) then attend one (sequence, head) pair."""
    var seq = task // job.num_heads
    var head = task - seq * job.num_heads
    # MHA (BART, Florence-2) needs no group mapping; skip the divide.
    var kv_head = head
    if job.num_heads != job.num_kv_heads:
        kv_head = head // (job.num_heads // job.num_kv_heads)
    if do_store:
        var slot = Int(_ptr[Int64](job.slots).unsafe_load(seq))
        _store_head[dtype, head_dim, tile](job, seq, head, kv_head, slot)
    if do_attend:
        if dtype == DType.bfloat16 and BF16_DOT and tile % LANES == 0:
            # bf16 dot products, paired K page layout (see _store_head).
            _attention_head_dot[dtype, head_dim, tile](job, seq, head, kv_head)
        else:
            _attention_head[dtype, head_dim, tile, group](
                job, seq, head, kv_head
            )


def _dispatch[
    dtype: DType, head_dim: Int, tile: Int, group: Int
](job: Job, num_workers: Int, do_store: Bool, do_attend: Bool):
    var items = job.num_seqs * job.num_heads
    if items == 0:
        return

    def body(task: Int) {imm}:
        _task[dtype, head_dim, tile, group](job, task, do_store, do_attend)

    parallelize(body, items, min(num_workers, items))


def _run_dtype[
    dtype: DType
](job: Job, num_workers: Int, do_store: Bool, do_attend: Bool) raises:
    # The tile must divide the page: 32 is the x86 page size, 16 the aarch64 one
    # (and the 16-wide `vec16` unit of the C++ kernel).
    comptime if True:
        if job.block_size % 32 == 0:
            if job.head_dim == 64:
                _dispatch[dtype, 64, 32, GROUP](
                    job, num_workers, do_store, do_attend
                )
            elif job.head_dim == 32:
                _dispatch[dtype, 32, 32, GROUP](
                    job, num_workers, do_store, do_attend
                )
            elif job.head_dim == 128:
                _dispatch[dtype, 128, 32, GROUP](
                    job, num_workers, do_store, do_attend
                )
            else:
                raise Error("fastencdec Mojo attention: unsupported head_dim")
        elif job.block_size % 16 == 0:
            if job.head_dim == 64:
                _dispatch[dtype, 64, 16, GROUP](
                    job, num_workers, do_store, do_attend
                )
            elif job.head_dim == 32:
                _dispatch[dtype, 32, 16, GROUP](
                    job, num_workers, do_store, do_attend
                )
            elif job.head_dim == 128:
                _dispatch[dtype, 128, 16, GROUP](
                    job, num_workers, do_store, do_attend
                )
            else:
                raise Error("fastencdec Mojo attention: unsupported head_dim")
        else:
            raise Error(
                "fastencdec Mojo attention: block_size must be a multiple of 16"
            )


def run(
    job: Job,
    dtype_code: Int,
    num_workers: Int,
    do_store: Bool,
    do_attend: Bool,
) raises:
    """Dispatch ``job`` on its dtype (0 = fp32, 1 = bf16) and head_dim (32/64/128).
    """
    if dtype_code == 0:
        _run_dtype[DType.float32](job, num_workers, do_store, do_attend)
    elif dtype_code == 1:
        _run_dtype[DType.bfloat16](job, num_workers, do_store, do_attend)
    else:
        raise Error("fastencdec Mojo attention: unsupported dtype")
