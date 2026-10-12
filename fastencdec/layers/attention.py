"""Device-agnostic attention ops.

Three attention kernels are exposed with one API each, identical on every
device, and the implementation is chosen *inside* the op from the tensor device,
dtype and head dim:

* :func:`encoder_self_attention` -- dense, no cache; runs once per request.
* :func:`decoder_self_attention` -- paged, growing KV cache.
* :func:`decoder_cross_attention` -- against the layer's fixed per-request
  encoder K/V.

The candidates are FlashAttention-2 on CUDA (``kernels-community/flash-attn2``),
the vendored vLLM CPU kernel and the PyTorch SDPA fallback (also what XPU and
unsupported shapes use).  ``models/bart.py`` only calls these ops, and every op
reads the per-step :class:`~fastencdec.utils.context.Context`, so the model and
the runner are hardware-agnostic.

``set_backend`` overrides the automatic choice (``"auto"`` default, or
``"vllm"``/``"flash"``/``"sdpa"``); it exists for tests and debugging.
"""

import torch
import torch.nn.functional as F

from ..utils.context import get_context
from . import cpu_attn, flash_attn

_BACKENDS = ("auto", "vllm", "flash", "sdpa")
_PREFER = "auto"

# Kernel-specific metadata derived once per step (the runner builds a fresh
# Context each forward, so identity is the step key).
_DERIVED: tuple = (None, None, None)


def set_backend(name: str) -> None:
    """Force an attention backend, or ``"auto"`` (the default)."""
    if name not in _BACKENDS:
        raise ValueError(f"attn_backend must be one of {_BACKENDS}, got {name!r}")
    global _PREFER
    _PREFER = name


# --------------------------------------------------------------- cache layout
def _paged_backend(device, dtype, head_dim, block_size) -> str:
    """The paged backend for a ``(device, dtype, head_dim, block_size)`` tuple."""
    if _PREFER == "flash" and device.type != "cuda":
        raise RuntimeError("FlashAttention-2 requires a CUDA device")
    if _PREFER == "vllm" and device.type != "cpu":
        raise RuntimeError("the vendored vLLM attention kernel is CPU-only")
    if _PREFER == "sdpa":
        return "sdpa"
    if device.type == "cuda":
        # FlashAttention-2's paged kernel needs fp16/bf16, a supported head dim
        # and a page size divisible by 256.
        if not flash_attn.supported(dtype, head_dim) or block_size % 256 != 0:
            if _PREFER == "flash":
                raise RuntimeError(
                    f"FlashAttention-2 cannot run head_dim={head_dim} "
                    f"dtype={dtype} block_size={block_size}"
                )
            return "sdpa"
        if _PREFER == "flash" or flash_attn.module() is not None:
            if _PREFER == "flash":
                flash_attn.require()
            return "flash"
        return "sdpa"
    if device.type == "cpu" and _PREFER in ("auto", "vllm"):
        isa = cpu_attn.select_isa(dtype, block_size, head_dim)
        if isa is not None and (_PREFER == "vllm" or cpu_attn.module() is not None):
            if _PREFER == "vllm":
                cpu_attn.require()
            return "vllm"
        if _PREFER == "vllm":
            raise RuntimeError(
                f"vLLM CPU attention does not support head_dim={head_dim} "
                f"block_size={block_size} dtype={dtype}"
            )
    return "sdpa"


def self_cache_shape(
    num_blocks, block_size, num_heads, head_dim, device, dtype
) -> tuple:
    """Shape of the paged self-attention cache the active kernel expects.

    Page-major ``[blocks, block_size, heads, head_dim]`` for FlashAttention-2,
    head-major ``[blocks, heads, block_size, head_dim]`` otherwise.
    """
    device = torch.device(device)
    if _paged_backend(device, dtype, head_dim, block_size) == "flash":
        return (num_blocks, block_size, num_heads, head_dim)
    return (num_blocks, num_heads, block_size, head_dim)


def store_kvcache(key, value, k_cache, v_cache, slot_mapping, block_size, page_major):
    """Scatter new K/V ``[T, H, D]`` into the paged cache."""
    block = torch.div(slot_mapping, block_size, rounding_mode="floor")
    pos = slot_mapping % block_size
    if page_major:
        k_cache[block, pos, :, :] = key
        v_cache[block, pos, :, :] = value
    else:
        k_cache[block, :, pos, :] = key
        v_cache[block, :, pos, :] = value


def _store(key, value, k_cache, v_cache, slot_mapping, block_size, backend, isa=None):
    """Write new K/V with the cheapest scatter for ``backend``."""
    if backend == "vllm":
        # The vendored kernel does the scatter in C++ (much faster than the
        # Python advanced-index assignment).
        cpu_attn.module().reshape_and_cache(
            key, value, k_cache, v_cache, slot_mapping, isa
        )
    else:
        store_kvcache(
            key,
            value,
            k_cache,
            v_cache,
            slot_mapping,
            block_size,
            page_major=(backend == "flash"),
        )


# ------------------------------------------------------------------ encoder
def _dense_flash(query) -> bool:
    if _PREFER == "sdpa":
        return False
    if not query.is_cuda:
        if _PREFER == "flash":
            raise RuntimeError("FlashAttention-2 requires a CUDA device")
        return False
    if not flash_attn.supported(query.dtype, query.shape[-1]):
        if _PREFER == "flash":
            raise RuntimeError(
                f"FlashAttention-2 does not support {query.dtype} head_dim="
                f"{query.shape[-1]}"
            )
        return False
    if _PREFER == "flash":
        flash_attn.require()
    return flash_attn.module() is not None


def encoder_self_attention(query, key, value, scale=None):
    """Dense non-causal self-attention ``[B, H, S, D]`` (encoder, no cache)."""
    if _dense_flash(query):
        out = flash_attn.require().flash_attn_func(
            query.transpose(1, 2).contiguous(),
            key.transpose(1, 2).contiguous(),
            value.transpose(1, 2).contiguous(),
            causal=False,
            softmax_scale=scale,
        )
        return out.transpose(1, 2)
    return F.scaled_dot_product_attention(
        query, key, value, is_causal=False, scale=scale
    )


# ------------------------------------------------------------ decoder self
def _key_slot_ids(ctx, block_size):
    """(block_ids, pos_in_block) for every cached key, ordered by sequence."""
    lens = ctx.context_lens.long()
    max_len = int(lens.max())
    pos = torch.arange(max_len, device=lens.device)
    blocks = ctx.block_tables.long()[:, pos // block_size]
    pos_in_block = pos % block_size
    mask = pos.unsqueeze(0) < lens.unsqueeze(1)
    return blocks[mask], pos_in_block.expand(lens.numel(), -1)[mask]


def _dynamic_causal(ctx):
    """Per-sequence causal flag for the vLLM kernel (1 = prefill).

    A sequence was fresh this step iff its cached length is one (the decoder
    prompt is a single token); ``is_prefill`` short-circuits the homogeneous case.
    """
    if ctx.is_prefill:
        return torch.ones(
            ctx.context_lens.numel(), dtype=torch.int32, device=ctx.context_lens.device
        )
    return (ctx.context_lens == 1).to(torch.int32)


def _self_derived(ctx, backend, num_heads, head_dim, dtype, block_size):
    global _DERIVED
    key = (backend, num_heads, head_dim, dtype, block_size)
    if _DERIVED[0] is ctx and _DERIVED[1] == key:
        return _DERIVED[2]
    device = ctx.context_lens.device
    num = ctx.context_lens.numel()
    data = {
        "cache_seqlens": ctx.context_lens.to(torch.int32),
        "query_start_loc": torch.arange(num + 1, dtype=torch.int32, device=device),
    }
    if backend == "sdpa":
        data["key_slot_ids"] = _key_slot_ids(ctx, block_size)
    elif backend == "vllm":
        isa = cpu_attn.select_isa(dtype, block_size, head_dim)
        data["isa"] = isa
        data["dynamic_causal"] = _dynamic_causal(ctx)
        data["metadata"] = cpu_attn.module().build_metadata(
            num,
            num_heads,
            num_heads,
            head_dim,
            data["cache_seqlens"],
            dtype,
            data["query_start_loc"],
            False,
            -1,
            isa,
            True,
            data["dynamic_causal"],
        )
    _DERIVED = (ctx, key, data)
    return data


def decoder_self_attention(query, key, value, k_cache, v_cache, scale, block_size):
    """Paged causal self-attention with a growing KV cache.

    ``query/key/value`` are ``[num_seqs, H, D]`` (one new token per sequence);
    the op writes the new K/V at ``Context.slot_mapping`` and attends over
    ``Context.block_tables`` / ``Context.context_lens``.
    """
    ctx = get_context()
    num_heads, head_dim = query.shape[1], query.shape[2]
    backend = _paged_backend(query.device, query.dtype, head_dim, block_size)
    data = _self_derived(ctx, backend, num_heads, head_dim, query.dtype, block_size)
    _store(
        key,
        value,
        k_cache,
        v_cache,
        ctx.slot_mapping,
        block_size,
        backend,
        data.get("isa"),
    )
    if backend == "flash":
        out = flash_attn.require().flash_attn_with_kvcache(
            query.unsqueeze(1),
            k_cache,
            v_cache,
            cache_seqlens=data["cache_seqlens"],
            block_table=ctx.block_tables,
            causal=True,
            softmax_scale=scale,
        )
        return out.squeeze(1)
    if backend == "vllm":
        return cpu_attn.module().attention_forward(
            query,
            k_cache,
            v_cache,
            data["query_start_loc"],
            data["cache_seqlens"],
            scale,
            False,
            -1,
            ctx.block_tables,
            data["metadata"],
            data["dynamic_causal"],
        )
    return _sdpa_paged(query, k_cache, v_cache, data["key_slot_ids"], ctx, scale)


# ----------------------------------------------------------- decoder cross
def _cross_sdpa(query, layer, ctx, scale):
    outputs = []
    for i, request_id in enumerate(ctx.request_ids):
        key, value = layer.encoder_kv_cache[request_id]  # [H, enc_len, D]
        outputs.append(cross_sdpa(query[i : i + 1], key, value, scale))
    return torch.cat(outputs, dim=0)


def _cross_staged(layer, ctx, backend, block_size, num_heads, head_dim, dtype):
    """Stage the layer's dense per-request encoder K/V into a paged cache.

    Built once per distinct active request set and memoized on the layer, so
    beams share their request's K/V and it is never duplicated per step.
    """
    req_key = (backend, tuple(ctx.request_ids))
    staged = getattr(layer, "cross_staged", None)
    if staged is not None and staged[0] == req_key:
        return staged[1]
    device = ctx.context_lens.device
    seen, rows = {}, []
    for request_id in ctx.request_ids:
        if request_id not in seen:
            seen[request_id] = len(rows)
            rows.append(request_id)
    enc_lens = [layer.encoder_kv_cache[rid][0].shape[1] for rid in rows]
    pages = [(length + block_size - 1) // block_size for length in enc_lens]
    max_pages, total = max(pages), sum(pages)
    table = torch.zeros(len(rows), max_pages, dtype=torch.int32, device=device)
    pid = 0
    for i, page_num in enumerate(pages):
        table[i, :page_num] = torch.arange(
            pid, pid + page_num, dtype=torch.int32, device=device
        )
        pid += page_num
    if backend == "flash":
        shape = (total, block_size, num_heads, head_dim)
    else:
        shape = (total, num_heads, block_size, head_dim)
    k_cache = torch.zeros(shape, dtype=dtype, device=device)
    v_cache = torch.zeros(shape, dtype=dtype, device=device)
    isa = (
        cpu_attn.select_isa(dtype, block_size, head_dim) if backend == "vllm" else None
    )
    for i, request_id in enumerate(rows):
        key, value = layer.encoder_kv_cache[request_id]  # [H, L, D]
        length = key.shape[1]
        pos = torch.arange(length, device=device)
        blocks = table[i, pos // block_size].long()
        slots = blocks * block_size + (pos % block_size)
        _store(
            key.transpose(0, 1).contiguous(),
            value.transpose(0, 1).contiguous(),
            k_cache,
            v_cache,
            slots,
            block_size,
            backend,
            isa,
        )
    row_of_seq = torch.tensor(
        [seen[rid] for rid in ctx.request_ids], dtype=torch.long, device=device
    )
    block_table = table[row_of_seq]
    cache_seqlens = torch.tensor(
        [enc_lens[seen[rid]] for rid in ctx.request_ids],
        dtype=torch.int32,
        device=device,
    )
    metadata = None
    if backend == "vllm":
        num = len(ctx.request_ids)
        metadata = cpu_attn.module().build_metadata(
            num,
            num_heads,
            num_heads,
            head_dim,
            cache_seqlens,
            dtype,
            torch.arange(num + 1, dtype=torch.int32, device=device),
            False,
            -1,
            isa,
            True,
            None,
        )
    data = (k_cache, v_cache, block_table, cache_seqlens, metadata)
    layer.cross_staged = (req_key, data)
    return data


def decoder_cross_attention(query, layer, scale, block_size):
    """Cross-attention against the layer's fixed per-request encoder K/V.

    ``query`` is ``[num_seqs, H, D]``.  SDPA indexes the dense per-request cache
    directly; the vLLM and FlashAttention-2 kernels read a paged cache that is
    staged from it once per active request set.
    """
    ctx = get_context()
    num_heads, head_dim = query.shape[1], query.shape[2]
    backend = _paged_backend(query.device, query.dtype, head_dim, block_size)
    if backend == "sdpa":
        return _cross_sdpa(query, layer, ctx, scale)
    k_cache, v_cache, block_table, cache_seqlens, metadata = _cross_staged(
        layer, ctx, backend, block_size, num_heads, head_dim, query.dtype
    )
    if backend == "flash":
        out = flash_attn.require().flash_attn_with_kvcache(
            query.unsqueeze(1),
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=False,
            softmax_scale=scale,
        )
        return out.squeeze(1)
    return cpu_attn.module().attention_forward(
        query,
        k_cache,
        v_cache,
        torch.arange(query.shape[0] + 1, dtype=torch.int32, device=query.device),
        cache_seqlens,
        scale,
        False,
        -1,
        block_table,
        metadata,
        None,
    )


# ------------------------------------------------------------------ fallbacks
def _sdpa(query, key, value, causal, scale):
    # [L, H, D] -> [1, H, L, D].  ``contiguous`` matters on aarch64: SDPA with
    # BF16 non-contiguous inputs takes a ~20x slower path.
    out = F.scaled_dot_product_attention(
        query.transpose(0, 1).unsqueeze(0).contiguous(),
        key.transpose(0, 1).unsqueeze(0).contiguous(),
        value.transpose(0, 1).unsqueeze(0).contiguous(),
        is_causal=causal,
        scale=scale,
    )
    return out.squeeze(0).transpose(0, 1)


def cross_sdpa(query, key, value, scale):
    """Cross-attention over encoder K/V stored as contiguous ``[H, S, D]``."""
    out = F.scaled_dot_product_attention(
        query.transpose(0, 1).unsqueeze(0).contiguous(),
        key.unsqueeze(0),
        value.unsqueeze(0),
        is_causal=False,
        scale=scale,
    )
    return out.squeeze(0).transpose(0, 1)


def _gather_all(cache, key_slot_ids):
    block_ids, pos_in_block = key_slot_ids
    return cache[block_ids, :, pos_in_block, :]


def _sdpa_paged(query, k_cache, v_cache, key_slot_ids, ctx, scale):
    if query.is_cuda:
        return _sdpa_paged_batched(query, k_cache, v_cache, key_slot_ids, ctx, scale)
    k_all = _gather_all(k_cache, key_slot_ids)
    v_all = _gather_all(v_cache, key_slot_ids)
    outputs, offset = [], 0
    for i in range(ctx.context_lens.numel()):
        length = int(ctx.context_lens[i])
        outputs.append(
            _sdpa(
                query[i : i + 1],
                k_all[offset : offset + length],
                v_all[offset : offset + length],
                causal=length == 1,
                scale=scale,
            )
        )
        offset += length
    return torch.cat(outputs, dim=0)


def _sdpa_paged_batched(query, k_cache, v_cache, key_slot_ids, ctx, scale):
    """Batched block-diagonal SDPA over the gathered cache (CUDA).

    The per-sequence loop is launch-bound on a GPU; this attends all queries in
    one masked softmax.  There is one query per sequence, so query ``i`` attends
    to every cached key of sequence ``i``.
    """
    device = query.device
    k_all = _gather_all(k_cache, key_slot_ids)  # [Tk, H, D]
    v_all = _gather_all(v_cache, key_slot_ids)
    lens = ctx.context_lens.long()
    starts = torch.zeros_like(lens)
    if lens.numel() > 1:
        starts[1:] = torch.cumsum(lens, 0)[:-1]
    hi = starts + lens - 1  # last key index of each sequence
    keys = torch.arange(k_all.shape[0], device=device)
    mask = (keys.unsqueeze(0) >= starts.unsqueeze(1)) & (
        keys.unsqueeze(0) <= hi.unsqueeze(1)
    )
    scores = torch.einsum("thd,shd->hts", query.float(), k_all.float()) * scale
    scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hts,shd->thd", probs, v_all.float())
    return out.to(query.dtype)
