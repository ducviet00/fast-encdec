"""Loader and thin wrapper for the Mojo paged-attention kernel.

The kernel is built from ``mojo/src`` into ``mojo/build/fastencdec_mojo_attn.so``
(``pixi run -m mojo/pixi.toml build``). Set ``FASTENCDEC_MOJO_ATTN`` to point at
another build. Everything here is optional: without the shared library (or on an
unsupported configuration) ``module()`` is ``None`` and ``attn_backend="auto"``
falls back to the vendored C++ kernel.
"""

import importlib.util
import os

import torch

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_DEFAULT_PATH = os.path.join(_REPO_ROOT, "mojo", "build", "fastencdec_mojo_attn.so")

# The kernel specializes on head_dim at compile time; anything else raises.
_HEAD_DIMS = {32, 64, 128}
_DTYPE_CODES = {torch.float32: 0, torch.bfloat16: 1}

_BUILT = None


def _load():
    path = os.environ.get("FASTENCDEC_MOJO_ATTN", _DEFAULT_PATH)
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("fastencdec_mojo_attn", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def module():
    """Return the compiled Mojo kernel module, or None if it is unavailable."""
    global _BUILT
    if _BUILT is None:
        try:
            _BUILT = _load()
        except Exception:  # noqa: BLE001
            _BUILT = False
    return _BUILT or None


def supported(dtype, block_size: int, head_dim: int) -> bool:
    """Return True if the Mojo kernel can run this dtype / page size / head dim."""
    return (
        dtype in _DTYPE_CODES
        and head_dim in _HEAD_DIMS
        and block_size % 16 == 0
        and module() is not None
    )


def require():
    """Return the Mojo kernel module, raising RuntimeError if it is unavailable."""
    built = module()
    if built is None:
        raise RuntimeError(
            "Mojo attention backend requested but fastencdec_mojo_attn.so is "
            "missing; build it with `pixi run --manifest-path mojo/pixi.toml build`."
        )
    return built


_ATTENTION = None


def _attention():
    """The extension's single entry point, resolved once (saves a lookup per call)."""
    global _ATTENTION
    if _ATTENTION is None:
        _ATTENTION = require().attention
    return _ATTENTION


_PHYSICAL = None
# Read once: an os.environ.get on the hot path costs ~0.35 us per call, i.e. >1% of
# a short-context attention call. Harnesses that sweep the worker count should
# patch ``_OVERRIDE`` (or ``_workers``) directly, not the environment.
_OVERRIDE = os.environ.get("FASTENCDEC_MOJO_WORKERS")


def _physical_cores() -> int:
    """Physical cores visible to this process, from the topology sysfs files."""
    global _PHYSICAL
    if _PHYSICAL is None:
        cores = set()
        try:
            for cpu in range(os.cpu_count() or 1):
                base = f"/sys/devices/system/cpu/cpu{cpu}/topology"
                with open(f"{base}/core_id") as f:
                    core = f.read().strip()
                with open(f"{base}/physical_package_id") as f:
                    cores.add((f.read().strip(), core))
        except OSError:
            cores = set()
        _PHYSICAL = len(cores) or (os.cpu_count() or 1)
    return _PHYSICAL


def _workers(num_tasks: int, pages: int = 1) -> int:
    """Workers for one call: the physical-core cap, inline only for tiny work.

    The kernel dispatches on torch's own OpenMP pool (``GOMP_parallel``), so a
    fork-join costs ~2.4 us warm and ~2.7 us with a 1 ms gap -- there is no
    sleeping second pool to wake, which is why the pool now wins almost
    everywhere: measured cold, one worker (inline, 0.36 us) beats 8 only up to
    ~32 KV positions per call (24.5 vs 31.0 us at batch 1 / ctx 64), while at
    batch 1 / ctx 128 the pool is already 1.3x faster and at batch 8 / ctx 256
    it is 8x. The cap at the *physical* core count matters on SMT boxes (8
    workers on a 4-core Xeon: 1.5x slower at ctx 1024); torch already sizes its
    pool that way, so it only bites when a parent sets a larger
    ``OMP_NUM_THREADS``. ``FASTENCDEC_MOJO_WORKERS`` overrides everything.
    """
    if _OVERRIDE:
        return max(1, min(int(_OVERRIDE), num_tasks))
    workers = min(torch.get_num_threads(), _physical_cores(), num_tasks)
    if workers > 1 and num_tasks * pages <= 32:
        return 1
    return workers


def _call(
    query,
    key,
    value,
    k_cache,
    v_cache,
    out,
    slots,
    block_table,
    lens,
    scale,
    do_store: bool,
):
    """Run one kernel call; ``key``/``value``/``slots`` are ignored without a store."""
    attention = _attention()
    zero = 0
    num_seqs, num_heads, head_dim = query.shape
    q_stride = query.stride()
    if q_stride[-1] != 1 or q_stride[-2] != head_dim:
        query = query.contiguous()
        q_stride = query.stride()
    # The fused q/k/v projection hands us chunks of one buffer: the head dim is
    # contiguous but the token stride is 3 * d_model, which the kernel takes
    # explicitly (a head-major layout check would silently copy).
    kv_stride = 0
    if do_store:
        k_stride = key.stride()
        if k_stride[-1] != 1 or k_stride[-2] != head_dim or value.stride() != k_stride:
            key, value = key.contiguous(), value.contiguous()
            k_stride = key.stride()
        kv_stride = k_stride[0]
    num_tasks = num_seqs * num_heads
    attention(
        query.data_ptr(),
        key.data_ptr() if do_store else zero,
        value.data_ptr() if do_store else zero,
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        out.data_ptr(),
        slots.data_ptr() if do_store else zero,
        block_table.data_ptr(),
        lens.data_ptr(),
        num_seqs,
        num_heads,
        k_cache.shape[1],
        head_dim,
        k_cache.shape[2],
        block_table.shape[1],
        q_stride[0],
        kv_stride,
        out.stride(0),
        _DTYPE_CODES[query.dtype],
        scale,
        _workers(num_tasks, block_table.shape[1]),
        do_store,
        True,
    )
    return out


def _paged(
    query, key, value, k_cache, v_cache, slots, page_table, lens, scale, do_store
):
    """Run one kernel call, allocating the output (geometry already checked)."""
    out = torch.empty(
        (query.shape[0], query.shape[1], query.shape[2]),
        dtype=query.dtype,
        device=query.device,
    )
    return _call(
        query,
        key,
        value,
        k_cache,
        v_cache,
        out,
        slots,
        page_table,
        lens,
        scale,
        do_store,
    )


def _lens_tensor(cache_seqlens, batch: int, device):
    """Return ``cache_seqlens`` as the int32 ``[batch]`` lengths the kernel reads."""
    if cache_seqlens is None:
        raise ValueError("Mojo attention: `cache_seqlens` is required")
    if isinstance(cache_seqlens, int):
        return torch.full((batch,), cache_seqlens, dtype=torch.int32, device=device)
    if cache_seqlens.dtype != torch.int32 or tuple(cache_seqlens.shape) != (batch,):
        raise ValueError(
            "Mojo attention: `cache_seqlens` must be an int or an int32 [batch] tensor"
        )
    return cache_seqlens


def _append_slots(page_table, lengths, block_size: int):
    """Slots writing one token at position ``lengths`` (flash's pre-append index)."""
    block = torch.div(lengths, block_size, rounding_mode="floor")
    page = page_table.gather(1, block.long().unsqueeze(1)).squeeze(1)
    return (page * block_size + lengths % block_size).long()


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens=None,
    cache_batch_idx=None,
    cache_leftpad=None,
    page_table=None,
    cu_seqlens_q=None,
    cu_seqlens_k_new=None,
    max_seqlen_q=None,
    rotary_seqlens=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,
    pack_gqa=None,
    sm_margin=0,
    return_softmax_lse=False,
    block_table=None,
    slot_mapping=None,
    check_inputs=True,
):
    """``flash_attn_with_kvcache`` over this backend's paged cache.

    Same arguments and defaults as flash-attn's function; the unsupported ones
    raise ``NotImplementedError`` rather than being silently ignored (``qv``,
    ``rotary_*``, ``cache_leftpad``, ``cu_seqlens_*``, ``max_seqlen_q``,
    ``*_descale``, ``window_size``, ``softcap``, ``attention_chunk``,
    ``return_softmax_lse``).  The tuning hints ``num_splits``, ``pack_gqa``,
    ``sm_margin``, ``scheduler_metadata`` and ``rotary_interleaved`` are
    accepted and ignored.  ``block_table`` is this repo's name for flash's
    ``page_table``.

    Deliberate differences:

    * ``q`` must be ``[batch, 1, nheads, headdim]``: the kernel decodes one
      query token per sequence, so ``seqlen`` may only be 1.
    * The cache is head-major ``[num_blocks, nheads_k, page_block_size,
      headdim]`` (``attention.self_cache_shape``), not flash's page-major
      layout, because K is stored transposed inside the page.  Must be
      contiguous.
    * ``causal`` is accepted and has no effect: the one query sits at the last
      cached position, where the bottom-right-aligned mask keeps every key.
    * With ``k``/``v``, they are written at ``cache_seqlens`` (the length
      *before* the write, flash's convention).  Passing ``slot_mapping``
      instead writes them at those slots and reads ``cache_seqlens`` as the
      length *after* the write -- the engine's Context convention, and it skips
      computing the slots per call.
    * ``check_inputs=False`` skips the argument validation for callers whose
      tensors are already known-good (the engine's per-layer path): the checks
      cost a few microseconds per call, which at 24 calls per decode step is
      measurable end to end.
    """
    if check_inputs:
        if (
            qv is not None
            or rotary_cos is not None
            or rotary_sin is not None
            or cache_leftpad is not None
            or cu_seqlens_q is not None
            or cu_seqlens_k_new is not None
            or max_seqlen_q is not None
            or rotary_seqlens is not None
            or q_descale is not None
            or k_descale is not None
            or v_descale is not None
        ):
            raise NotImplementedError(
                "Mojo attention: qv/rotary_*/cache_leftpad/cu_seqlens_*/"
                "max_seqlen_q/*_descale are not supported"
            )
        if window_size != (-1, -1):
            raise NotImplementedError(
                "Mojo attention: sliding windows are not supported"
            )
        if softcap:
            raise NotImplementedError("Mojo attention: softcapping is not supported")
        if attention_chunk:
            raise NotImplementedError(
                "Mojo attention: `attention_chunk` is not supported"
            )
        if return_softmax_lse:
            raise NotImplementedError(
                "Mojo attention: `return_softmax_lse` is not supported"
            )
        if q.dim() != 4:
            raise ValueError(
                "Mojo attention: q must be [batch, seqlen, nheads, headdim]"
            )
        batch, q_len, num_heads, head_dim = q.shape
        if q_len != 1:
            raise NotImplementedError(
                "Mojo attention decodes one query token per sequence; "
                f"got seqlen={q_len}"
            )
        if page_table is not None and block_table is not None:
            raise ValueError("Mojo attention: pass page_table or block_table, not both")
        page_table = page_table if page_table is not None else block_table
        if page_table is None:
            raise ValueError("Mojo attention: pass `page_table` (or `block_table`)")
        if cache_batch_idx is not None:
            page_table = page_table.index_select(0, cache_batch_idx.long())
        if page_table.dtype != torch.int32 or page_table.dim() != 2:
            raise ValueError(
                "Mojo attention: page_table must be int32 [batch, max_blocks]"
            )
        if page_table.shape[0] != batch:
            raise ValueError("Mojo attention: page_table has the wrong batch size")
        page_table = page_table.contiguous()
        if k_cache.dim() != 4 or v_cache.shape != k_cache.shape:
            raise ValueError(
                "Mojo attention: k_cache/v_cache must be "
                "[num_blocks, nheads_k, page_block_size, headdim] and match"
            )
        num_kv_heads, block_size, cache_dim = k_cache.shape[1:]
        if cache_dim != head_dim:
            raise ValueError(
                "Mojo attention: head_dim mismatch between q and the caches"
            )
        if num_heads % num_kv_heads:
            raise ValueError("Mojo attention: nheads must be divisible by nheads_k")
        if not (k_cache.is_contiguous() and v_cache.is_contiguous()):
            raise ValueError(
                "Mojo attention: the paged caches must be contiguous "
                "(head-major layout, see `attention.self_cache_shape`)"
            )
        if q.dtype != k_cache.dtype or v_cache.dtype != q.dtype:
            raise ValueError("Mojo attention: q and the caches must share a dtype")
        if not supported(q.dtype, block_size, head_dim):
            raise ValueError(
                f"Mojo attention cannot run dtype={q.dtype}, page={block_size}, "
                f"head_dim={head_dim}"
            )
        if k is not None and (
            k.dim() != 4
            or k.shape[:2] != (batch, 1)
            or k.shape[2] != num_kv_heads
            or v.shape != k.shape
            or k.dtype != q.dtype
            or v.dtype != q.dtype
        ):
            raise ValueError(
                "Mojo attention: k/v must be [batch, 1, nheads_k, headdim] and "
                "match the cache"
            )
        if slot_mapping is not None and (
            k is None or slot_mapping.dtype != torch.int64
        ):
            raise ValueError("Mojo attention: slot_mapping needs k/v and must be int64")
    else:
        batch, _, num_heads, head_dim = q.shape
        page_table = page_table if page_table is not None else block_table
        num_kv_heads, block_size = k_cache.shape[1], k_cache.shape[2]
    scale = head_dim**-0.5 if softmax_scale is None else softmax_scale
    query = q.squeeze(1)
    if k is not None:
        if slot_mapping is None:
            before = _lens_tensor(cache_seqlens, batch, q.device)
            slots = _append_slots(page_table, before, block_size)
            lens = before + 1
        else:
            slots = slot_mapping
            lens = _lens_tensor(cache_seqlens, batch, q.device)
        out = _paged(
            query,
            k.squeeze(1),
            v.squeeze(1),
            k_cache,
            v_cache,
            slots,
            page_table,
            lens,
            scale,
            do_store=True,
        )
    else:
        out = _paged(
            query,
            None,
            None,
            k_cache,
            v_cache,
            None,
            page_table,
            _lens_tensor(cache_seqlens, batch, q.device),
            scale,
            do_store=False,
        )
    return out.unsqueeze(1)


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """``flash_attn_func`` for the single query token this kernel supports.

    With ``q`` of length 1 the causal mask keeps every key, so ``k``/``v`` may be
    any length and ``causal`` does not change the result.  For ``seqlen > 1``
    (e.g. the BART encoder) this raises ``NotImplementedError``: dense
    multi-token attention is not ported, so the engine keeps its dense path
    (SDPA on CPU, FlashAttention-2 on CUDA).  ``dropout_p``, ``alibi_slopes``,
    ``window_size``, ``softcap`` and ``return_attn_probs`` are rejected;
    ``deterministic`` is accepted and ignored (the forward pass always is).
    """
    if dropout_p:
        raise NotImplementedError("Mojo attention: inference only, dropout_p must be 0")
    if alibi_slopes is not None:
        raise NotImplementedError("Mojo attention: `alibi_slopes` is not supported")
    if window_size != (-1, -1):
        raise NotImplementedError("Mojo attention: sliding windows are not supported")
    if softcap:
        raise NotImplementedError("Mojo attention: softcapping is not supported")
    if return_attn_probs:
        raise NotImplementedError(
            "Mojo attention: `return_attn_probs` is not supported"
        )
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            "Mojo attention: q/k/v must be [batch, seqlen, nheads, headdim]"
        )
    batch, q_len, num_heads, head_dim = q.shape
    if q_len != 1:
        raise NotImplementedError(
            f"Mojo attention decodes one query token per sequence; got seqlen={q_len}"
        )
    kv_len, num_kv_heads = k.shape[1], k.shape[2]
    if (
        kv_len < 1
        or k.shape[0] != batch
        or k.shape[3] != head_dim
        or v.shape != k.shape
        or num_heads % num_kv_heads
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise ValueError(
            "Mojo attention: k/v must be [batch, seqlen, nheads_k, headdim] and match q"
        )
    if not supported(q.dtype, 16, head_dim):
        raise ValueError(
            f"Mojo attention cannot run dtype={q.dtype}, head_dim={head_dim}"
        )
    scale = head_dim**-0.5 if softmax_scale is None else softmax_scale
    # The kernel is paged: stage K/V into a temporary one-page-per-16-tokens
    # cache, then run the usual single-token paged attention over it.
    block = 16
    blocks = (kv_len + block - 1) // block
    shape = (batch * blocks, num_kv_heads, block, head_dim)
    k_cache = torch.empty(shape, dtype=q.dtype, device=q.device)
    v_cache = torch.empty(shape, dtype=q.dtype, device=q.device)
    page_table = torch.arange(batch * blocks, dtype=torch.int32, device=q.device)
    page_table = page_table.reshape(batch, blocks)
    pos = torch.arange(kv_len, dtype=torch.int64, device=q.device)
    slots = (page_table[:, pos // block] * block + pos % block).reshape(-1)
    reshape_and_cache(
        k.reshape(batch * kv_len, num_kv_heads, head_dim),
        v.reshape(batch * kv_len, num_kv_heads, head_dim),
        k_cache,
        v_cache,
        slots,
    )
    lens = torch.full((batch,), kv_len, dtype=torch.int32, device=q.device)
    out = _paged(
        q.squeeze(1),
        None,
        None,
        k_cache,
        v_cache,
        None,
        page_table,
        lens,
        scale,
        do_store=False,
    )
    return out.unsqueeze(1)


def self_attention(
    query, key, value, k_cache, v_cache, slot_mapping, block_table, context_lens, scale
):
    """Engine alias: store K/V at ``slot_mapping`` (lengths *after* the store).

    Same call as ``flash_attn_with_kvcache(..., k=, v=, slot_mapping=)`` with the
    arguments the runner already validated, so it skips the wrapper's checks.
    """
    return _paged(
        query,
        key,
        value,
        k_cache,
        v_cache,
        slot_mapping,
        block_table,
        context_lens,
        scale,
        do_store=True,
    )


def cross_attention(query, k_cache, v_cache, block_table, cache_seqlens, scale):
    """Engine alias: attend over a staged, paged encoder K/V cache."""
    return _paged(
        query,
        None,
        None,
        k_cache,
        v_cache,
        None,
        block_table,
        cache_seqlens,
        scale,
        do_store=False,
    )


def reshape_and_cache(key, value, k_cache, v_cache, slot_mapping) -> None:
    """Scatter key/value [T, H, D] into the paged cache at slot_mapping."""
    # The `q`/`dst`/`lens` slots of the shared entry point are unused here: the
    # kernel only stores when ``do_attend`` is False.
    attention = _attention()
    num_tokens, num_heads, head_dim = key.shape
    if key.stride()[-1] != 1 or key.stride()[-2] != head_dim:
        key, value = key.contiguous(), value.contiguous()
    attention(
        key.data_ptr(),
        key.data_ptr(),
        value.data_ptr(),
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        0,
        slot_mapping.data_ptr(),
        0,
        0,
        num_tokens,
        num_heads,
        k_cache.shape[1],
        head_dim,
        k_cache.shape[2],
        0,
        0,
        key.stride(0),
        0,
        _DTYPE_CODES[key.dtype],
        1.0,
        _workers(num_tokens * num_heads),
        True,
        False,
    )
