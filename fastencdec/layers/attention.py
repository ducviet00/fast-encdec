import torch
import torch.nn.functional as F

from ..utils.context import get_context
from . import cpu_attn, cpu_ops, flash_attn, mojo_ops

try:  # the Mojo backend is an optional, separately installed library
    import cpu_attention
except ImportError:  # pragma: no cover - exercised by installing without it
    cpu_attention = None

_BACKENDS = ("auto", "vllm", "flash", "sdpa", "mojo")
# Backends whose ops are opaque custom ops: they read the per-step Context and
# mutate the caches, so torch.compile must never trace them.
_OPAQUE = ("vllm", "mojo")
_PREFER = "auto"
# Per-step gather index for the SDPA fallback, memoized by Context identity.
_DERIVED: tuple = (None, None, None)


def uses_custom_ops(backend: str) -> bool:
    """Return True if the backend runs as opaque custom ops (torch.compile-safe)."""
    return backend in _OPAQUE


def set_backend(name: str) -> None:
    """Override the attention backend ('auto', 'vllm', 'mojo', 'flash', or 'sdpa')."""
    if name not in _BACKENDS:
        raise ValueError(f"attn_backend must be one of {_BACKENDS}, got {name!r}")
    global _PREFER
    _PREFER = name


def _paged_backend(device, dtype, head_dim, block_size) -> str:
    """Resolve the active paged attention backend for (device, dtype, head_dim, block_size)."""
    if _PREFER == "flash" and device.type != "cuda":
        raise RuntimeError("FlashAttention-2 requires a CUDA device")
    if _PREFER in _OPAQUE and device.type != "cpu":
        raise RuntimeError("the CPU attention kernels are CPU-only")
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
    if device.type == "cpu" and _PREFER == "mojo":
        if cpu_attention is None:
            raise RuntimeError(
                "attn_backend='mojo' needs the cpu_attention library "
                "(pip install -e ../cpu-attention.mojo)"
            )
        if cpu_attention.supported(dtype, block_size, head_dim):
            return "mojo"
        raise RuntimeError(
            f"the Mojo attention kernel does not support head_dim={head_dim} "
            f"block_size={block_size} dtype={dtype}"
        )
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


def _store(key, value, k_cache, v_cache, slot_mapping, block_size, backend):
    """Scatter new key/value into the paged cache using the fastest path for backend."""
    if backend == "vllm":
        cpu_attn.reshape_and_cache(key, value, k_cache, v_cache, slot_mapping)
    elif backend == "mojo":
        cpu_attention.reshape_and_cache(key, value, k_cache, v_cache, slot_mapping)
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


def _sdpa_slots(ctx, block_size):
    """Memoize the per-step gather index for the SDPA fallback."""
    global _DERIVED
    if _DERIVED[0] is ctx and _DERIVED[1] == block_size:
        return _DERIVED[2]
    slots = _key_slot_ids(ctx, block_size)
    _DERIVED = (ctx, block_size, slots)
    return slots


def decoder_self_attention(query, key, value, k_cache, v_cache, scale, block_size):
    """Write new K/V into the paged cache and run causal self-attention over cached keys."""
    ctx = get_context()
    backend = _paged_backend(query.device, query.dtype, query.shape[2], block_size)
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

    if backend in _OPAQUE:
        # Opaque custom op: the kernel path reads Context and mutates the caches,
        # so torch.compile never traces it.
        ops = cpu_ops if backend == "vllm" else mojo_ops
        return ops.self_attention(query, key, value, k_cache, v_cache, scale)
    slots = _sdpa_slots(ctx, block_size)
    _store(key, value, k_cache, v_cache, ctx.slot_mapping, block_size, backend)
    return _sdpa_paged(query, k_cache, v_cache, slots, ctx, scale)


def cross_sdpa_batched(query, encoder_kv_cache, request_ids, scale):
    """Run per-sequence SDPA cross-attention against dense encoder K/V."""
    return torch.cat(
        [
            cross_sdpa(query[i : i + 1], *encoder_kv_cache[rid], scale)
            for i, rid in enumerate(request_ids)
        ],
        dim=0,
    )


def stage_cross_kv(encoder_kv_cache, request_ids, block_size, backend):
    """Stage dense per-request encoder K/V [H, L, D] into a paged cache.

    Returns ``(k_cache, v_cache, block_table, cache_seqlens)``.
    """
    rows = list(dict.fromkeys(request_ids))
    seen = {rid: i for i, rid in enumerate(rows)}
    enc_lens = [encoder_kv_cache[rid][0].shape[1] for rid in rows]
    pages = [(length + block_size - 1) // block_size for length in enc_lens]
    max_pages, total = max(pages), sum(pages)
    key0 = encoder_kv_cache[rows[0]][0]
    device, dtype = key0.device, key0.dtype
    num_heads, _, head_dim = key0.shape
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
    for i, request_id in enumerate(rows):
        key, value = encoder_kv_cache[request_id]
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
        )
    row_of_seq = torch.tensor(
        [seen[rid] for rid in request_ids], dtype=torch.long, device=device
    )
    block_table = table[row_of_seq]
    cache_seqlens = torch.tensor(
        [enc_lens[seen[rid]] for rid in request_ids],
        dtype=torch.int32,
        device=device,
    )
    return k_cache, v_cache, block_table, cache_seqlens


def _stage(layer, request_ids, backend, block_size):
    """Stage a layer's encoder K/V, reusing it while the active request set is unchanged."""
    key = (backend, tuple(request_ids))
    if layer.cross_staged is not None and layer.cross_staged_key == key:
        return layer.cross_staged
    layer.cross_staged = stage_cross_kv(
        layer.encoder_kv_cache, request_ids, block_size, backend
    )
    layer.cross_staged_key = key
    return layer.cross_staged


def stage_cross(layers, request_ids, backend="vllm"):
    """Stage every decoder layer's encoder K/V for a paged CPU kernel."""
    staged = []
    for layer in layers:
        attn = layer.encoder_attn
        staged.append(_stage(attn, request_ids, backend, attn.block_size))
    return staged


def decoder_cross_attention(query, layer, scale, block_size):
    """Run decoder cross-attention against the layer's cached per-request encoder K/V."""
    ctx = get_context()
    backend = _paged_backend(query.device, query.dtype, query.shape[2], block_size)
    if backend in _OPAQUE:
        # Opaque custom op: the staged tensors come from the per-step Context, so
        # neither they nor the kernel calls enter the compiled graph.
        ops = cpu_ops if backend == "vllm" else mojo_ops
        return ops.cross_attention(query, layer.layer_idx, scale)
    if backend == "sdpa":
        return cross_sdpa_batched(query, layer.encoder_kv_cache, ctx.request_ids, scale)
    if ctx.cross_block_tables is not None:
        # CUDA-graph path: preallocated static cross cache and block tables.
        k_cache, v_cache = layer.cross_cache
        block_table, cache_seqlens = ctx.cross_block_tables, ctx.cross_cache_seqlens
    else:
        k_cache, v_cache, block_table, cache_seqlens = _stage(
            layer, ctx.request_ids, "flash", block_size
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
