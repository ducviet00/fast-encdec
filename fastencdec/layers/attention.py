import torch
import torch.nn.functional as F

from ..utils.context import get_context
from . import cpu_attn, cpu_ops, flash_attn

_BACKENDS = ("auto", "vllm", "flash", "sdpa")
_PREFER = "auto"
# Per-step kernel metadata memoized by Context identity.
_DERIVED: tuple = (None, None, None)


def set_backend(name: str) -> None:
    """Override the attention backend ('auto', 'vllm', 'flash', or 'sdpa')."""
    if name not in _BACKENDS:
        raise ValueError(f"attn_backend must be one of {_BACKENDS}, got {name!r}")
    global _PREFER
    _PREFER = name


def preferred() -> str:
    """Return the backend preference set by ``set_backend`` ('auto' unless overridden)."""
    return _PREFER


def _paged_backend(device, dtype, head_dim, block_size) -> str:
    """Resolve the active paged attention backend for (device, dtype, head_dim, block_size)."""
    if _PREFER == "flash" and device.type != "cuda":
        raise RuntimeError("FlashAttention-2 requires a CUDA device")
    if _PREFER == "vllm" and device.type != "cpu":
        raise RuntimeError("the vendored vLLM attention kernel is CPU-only")
    if _PREFER == "sdpa":
        return "sdpa"
    if device.type == "cuda":
        if not flash_attn.supported(dtype, head_dim) or block_size % 256 != 0:
            if _PREFER == "flash":
                raise RuntimeError(
                    f"FlashAttention-2 cannot run head_dim={head_dim} "
                    f"dtype={dtype} block_size={block_size}"
                )
            return "sdpa"
        if _PREFER == "flash":
            flash_attn.require()
            return "flash"
        return "flash" if flash_attn.module() is not None else "sdpa"
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


def backend(device, dtype, head_dim, block_size) -> str:
    """Return the paged kernel name ('flash', 'vllm', or 'sdpa') for the given configuration."""
    return _paged_backend(torch.device(device), dtype, head_dim, block_size)


def self_cache_shape(
    num_blocks, block_size, num_heads, head_dim, device, dtype
) -> tuple:
    """Return page-major [blocks, bs, H, D] for FlashAttention-2 or head-major [blocks, H, bs, D] otherwise."""
    if backend(device, dtype, head_dim, block_size) == "flash":
        return (num_blocks, block_size, num_heads, head_dim)
    return (num_blocks, num_heads, block_size, head_dim)


def store_kvcache(key, value, k_cache, v_cache, slot_mapping, block_size, page_major):
    """Scatter new key/value [T, H, D] into the paged cache at slot_mapping."""
    if page_major:
        # Flat view indexing avoids div/mod kernels for page-major caches.
        k_cache.view(-1, *k_cache.shape[-2:])[slot_mapping] = key
        v_cache.view(-1, *v_cache.shape[-2:])[slot_mapping] = value
    else:
        block = torch.div(slot_mapping, block_size, rounding_mode="floor")
        pos = slot_mapping % block_size
        k_cache[block, :, pos, :] = key
        v_cache[block, :, pos, :] = value


def _store(key, value, k_cache, v_cache, slot_mapping, block_size, backend, isa=None):
    """Scatter new key/value into the paged cache using the fastest path for backend."""
    if backend == "vllm":
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


def _dense_flash(query) -> bool:
    """Return True if dense encoder self-attention should use FlashAttention-2."""
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
    """Run dense non-causal self-attention [B, H, S, D] for the encoder."""
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


def _key_slot_ids(ctx, block_size):
    """Build (block_ids, pos_in_block) gather indices for all cached keys in the batch."""
    lens = ctx.context_lens.long()
    pos = torch.arange(int(lens.max()), device=lens.device)
    blocks = ctx.block_tables.long()[:, pos // block_size]
    pos_in_block = pos % block_size
    mask = pos.unsqueeze(0) < lens.unsqueeze(1)
    return blocks[mask], pos_in_block.expand(lens.numel(), -1)[mask]


def _dynamic_causal(ctx):
    """Build per-sequence causal flags for the vLLM CPU kernel (1 = prefill when context_len == 1)."""
    if ctx.is_prefill:
        return torch.ones(
            ctx.context_lens.numel(), dtype=torch.int32, device=ctx.context_lens.device
        )
    return (ctx.context_lens == 1).to(torch.int32)


def _self_derived(ctx, backend, num_heads, head_dim, dtype, block_size):
    """Compute and memoize per-step self-attention kernel metadata for ctx."""
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
    if backend == "vllm":
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
    elif backend == "sdpa":
        data["key_slot_ids"] = _key_slot_ids(ctx, block_size)
    _DERIVED = (ctx, key, data)
    return data


def decoder_self_attention(query, key, value, k_cache, v_cache, scale, block_size):
    """Write new K/V into the paged cache and run causal self-attention over cached keys."""
    ctx = get_context()
    num_heads, head_dim = query.shape[1], query.shape[2]
    backend = _paged_backend(query.device, query.dtype, head_dim, block_size)
    if backend == "flash":
        # Read Context directly (no module-level memo) so torch.compile does not guard/recompile.
        store_kvcache(
            key,
            value,
            k_cache,
            v_cache,
            ctx.slot_mapping,
            block_size,
            page_major=True,
        )
        out = flash_attn.require().flash_attn_with_kvcache(
            query.unsqueeze(1),
            k_cache,
            v_cache,
            cache_seqlens=ctx.context_lens,
            block_table=ctx.block_tables,
            causal=True,
            softmax_scale=scale,
        )
        return out.squeeze(1)

    if backend == "vllm":
        # Opaque custom op: the kernel path reads Context and mutates the caches,
        # so torch.compile never traces it (no graph breaks, no cache re-materialization).
        return cpu_ops.self_attention(
            query, key, value, k_cache, v_cache, scale, block_size
        )
    # SDPA fallback: read the cached K/V through the block table.
    data = _self_derived(ctx, "sdpa", num_heads, head_dim, query.dtype, block_size)
    store_kvcache(
        key, value, k_cache, v_cache, ctx.slot_mapping, block_size, page_major=False
    )
    return _sdpa_paged(query, k_cache, v_cache, data["key_slot_ids"], ctx, scale)


def _cross_sdpa(query, layer, ctx, scale):
    """Run per-sequence SDPA cross-attention against each request's dense encoder K/V."""
    return torch.cat(
        [
            cross_sdpa(query[i : i + 1], *layer.encoder_kv_cache[rid], scale)
            for i, rid in enumerate(ctx.request_ids)
        ],
        dim=0,
    )


def _cross_staged(layer, ctx, backend, block_size, num_heads, head_dim, dtype):
    """Stage dense per-request encoder K/V into a paged cache, memoized per active request set."""
    req_key = (backend, tuple(ctx.request_ids))
    staged = getattr(layer, "cross_staged", None)
    if staged is not None and staged[0] == req_key:
        return staged[1]
    device = ctx.context_lens.device
    rows = list(dict.fromkeys(ctx.request_ids))
    seen = {rid: i for i, rid in enumerate(rows)}
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
    shape = (
        (total, block_size, num_heads, head_dim)
        if backend == "flash"
        else (total, num_heads, block_size, head_dim)
    )
    k_cache = torch.zeros(shape, dtype=dtype, device=device)
    v_cache = torch.zeros(shape, dtype=dtype, device=device)
    isa = (
        cpu_attn.select_isa(dtype, block_size, head_dim) if backend == "vllm" else None
    )
    for i, request_id in enumerate(rows):
        key, value = layer.encoder_kv_cache[request_id]
        pos = torch.arange(key.shape[1], device=device)
        slots = table[i, pos // block_size].long() * block_size + (pos % block_size)
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
    """Run decoder cross-attention against the layer's cached per-request encoder K/V."""
    ctx = get_context()
    num_heads, head_dim = query.shape[1], query.shape[2]
    backend = _paged_backend(query.device, query.dtype, head_dim, block_size)
    if backend == "vllm":
        # Opaque custom op: staging reads the layer's request-keyed state, so it runs
        # outside the compiled graph (see cpu_ops.cross_attention).
        handle = getattr(layer, "cpu_cross_handle", None)
        if handle is None:
            raise RuntimeError(
                "cross-attention layer is not registered for the vLLM CPU kernel; "
                "construct it as BartDecoderCrossAttention or call "
                "register_cross_layer() on it"
            )
        return cpu_ops.cross_attention(query, scale, handle, block_size)
    if backend == "sdpa":
        return _cross_sdpa(query, layer, ctx, scale)
    if ctx.cross_block_tables is not None:
        # CUDA-graph path: preallocated static cross cache and block tables.
        k_cache, v_cache = layer.cross_cache
        block_table, cache_seqlens = ctx.cross_block_tables, ctx.cross_cache_seqlens
    else:
        k_cache, v_cache, block_table, cache_seqlens, _ = _cross_staged(
            layer, ctx, "flash", block_size, num_heads, head_dim, query.dtype
        )
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


def _sdpa(query, key, value, causal, scale):
    """Run single-sequence SDPA over contiguous [L, H, D] tensors."""
    # .contiguous() avoids the ~20x slower non-contiguous BF16 SDPA path on aarch64.
    out = F.scaled_dot_product_attention(
        query.transpose(0, 1).unsqueeze(0).contiguous(),
        key.transpose(0, 1).unsqueeze(0).contiguous(),
        value.transpose(0, 1).unsqueeze(0).contiguous(),
        is_causal=causal,
        scale=scale,
    )
    return out.squeeze(0).transpose(0, 1)


def cross_sdpa(query, key, value, scale):
    """Run single-sequence cross-attention over contiguous [H, S, D] encoder K/V."""
    out = F.scaled_dot_product_attention(
        query.transpose(0, 1).unsqueeze(0).contiguous(),
        key.unsqueeze(0),
        value.unsqueeze(0),
        is_causal=False,
        scale=scale,
    )
    return out.squeeze(0).transpose(0, 1)


def _gather_all(cache, key_slot_ids):
    """Gather cached keys or values at (block_ids, pos_in_block)."""
    block_ids, pos_in_block = key_slot_ids
    return cache[block_ids, :, pos_in_block, :]


def _sdpa_paged(query, k_cache, v_cache, key_slot_ids, ctx, scale):
    """Run paged self-attention via PyTorch SDPA (batched on CUDA, per-sequence on CPU)."""
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
    """Run batched block-diagonal SDPA over the gathered cache on CUDA."""
    device = query.device
    k_all = _gather_all(k_cache, key_slot_ids)
    v_all = _gather_all(v_cache, key_slot_ids)
    lens = ctx.context_lens.long()
    starts = torch.zeros_like(lens)
    if lens.numel() > 1:
        starts[1:] = torch.cumsum(lens, 0)[:-1]
    hi = starts + lens - 1
    keys = torch.arange(k_all.shape[0], device=device)
    mask = (keys.unsqueeze(0) >= starts.unsqueeze(1)) & (
        keys.unsqueeze(0) <= hi.unsqueeze(1)
    )
    scores = torch.einsum("thd,shd->hts", query.float(), k_all.float()) * scale
    scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hts,shd->thd", probs, v_all.float())
    return out.to(query.dtype)
