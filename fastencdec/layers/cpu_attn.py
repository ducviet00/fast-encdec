"""Wrapper over the ``cpu-attention`` kernels (flash-attn's API on CPU).

The paged caches are the library's own head-major layout
``[num_blocks, heads, block_size, head_dim]`` with K transposed inside the
page, so only the library writes them (``reshape_and_cache`` or the ``k``/``v``
arguments of ``flash_attn_with_kvcache``).
"""

import torch

HEAD_DIMS = (32, 64, 128)
DTYPES = (torch.float32, torch.bfloat16)

_MODULE = None
_RESOLVED = False


def supported(dtype, head_dim: int, block_size: int | None = None) -> bool:
    """Return True if the kernels take this (dtype, head_dim, page size)."""
    page_ok = block_size is None or block_size % 16 == 0
    return dtype in DTYPES and head_dim in HEAD_DIMS and page_ok


def _load():
    """Import ``cpu_attention`` and load its kernel, or return None."""
    try:
        import cpu_attention
        from cpu_attention import _kernel

        _kernel.runtime()  # loads the best ISA tier and checks torch's layout
    except Exception:  # noqa: BLE001
        return None
    return cpu_attention


def module():
    """Return the loaded ``cpu_attention`` package, or None if unavailable."""
    # A plain global instead of functools.lru_cache: Dynamo traces cache-wrapped
    # bodies once per call site.
    global _MODULE, _RESOLVED
    if not _RESOLVED:
        _MODULE = _load()
        _RESOLVED = True
    return _MODULE


def require():
    """Return the ``cpu_attention`` package, raising RuntimeError if unavailable."""
    mod = module()
    if mod is None:
        raise RuntimeError(
            "CPU attention backend requested but the cpu-attention package is "
            "not installed or its kernel is not built; install it (`uv sync "
            "--extra cpu`) or use attn_backend='auto'/'sdpa'."
        )
    return mod


def reshape_and_cache(key, value, k_cache, v_cache, slot_mapping) -> None:
    """Scatter new key/value [T, H, D] into the paged cache at slot_mapping."""
    require().reshape_and_cache(key, value, k_cache, v_cache, slot_mapping)


def self_attention(
    query, key, value, k_cache, v_cache, block_table, context_lens, scale
):
    """Append each sequence's new K/V [B, H, D] and attend over its whole cache.

    ``context_lens`` is the length after the write; the single query sits at the
    last position, so prefill (length 1) and decode need no causal mask.
    """
    out = require().flash_attn_with_kvcache(
        query.unsqueeze(1),
        k_cache,
        v_cache,
        k=key.unsqueeze(1),
        v=value.unsqueeze(1),
        cache_seqlens=context_lens - 1,
        page_table=block_table,
        softmax_scale=scale,
    )
    return out.squeeze(1)


def cross_attention(query, k_cache, v_cache, block_table, cache_seqlens, scale):
    """Run non-causal cross-attention over a staged, paged encoder K/V cache."""
    out = require().flash_attn_with_kvcache(
        query.unsqueeze(1),
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        page_table=block_table,
        softmax_scale=scale,
    )
    return out.squeeze(1)


def dense_attention(query, key, value, scale):
    """Run dense non-causal attention over [B, S, H, D] tensors."""
    return require().flash_attn_func(query, key, value, softmax_scale=scale)
