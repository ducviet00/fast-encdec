"""CPU paged attention built on ``torch.nn.functional.scaled_dot_product_attention``.

No padding is used: each sequence in the batch is attended independently by
slicing ``[query_start_loc[i]:query_start_loc[i + 1]]`` and gathering its own
K/V blocks from the paged cache.
"""

import torch
import torch.nn.functional as F

from ..utils.context import get_context


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping,
) -> None:
    """Scatter new K/V ``[T, H, D]`` into the flat paged cache."""
    k_cache.view(-1, *key.shape[1:])[slot_mapping] = key
    v_cache.view(-1, *value.shape[1:])[slot_mapping] = value


def _sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool,
    scale: float,
) -> torch.Tensor:
    # [L, H, D] -> [1, H, L, D]
    out = F.scaled_dot_product_attention(
        query.transpose(0, 1).unsqueeze(0),
        key.transpose(0, 1).unsqueeze(0),
        value.transpose(0, 1).unsqueeze(0),
        is_causal=causal,
        scale=scale,
    )
    return out.squeeze(0).transpose(0, 1)


def cross_sdpa(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float
) -> torch.Tensor:
    """Cross-attention over encoder K/V already stored as contiguous [H, S, D].

    Transposing once at encode time keeps the (long) key tensor contiguous for
    the flash kernel, which is markedly faster than a strided view.
    """
    out = F.scaled_dot_product_attention(
        query.transpose(0, 1).unsqueeze(0),
        key.unsqueeze(0),
        value.unsqueeze(0),
        is_causal=False,
        scale=scale,
    )
    return out.squeeze(0).transpose(0, 1)


def paged_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Write new K/V into the paged cache, then attend causally over it.

    ``query``/``key``/``value`` are ``[T, H, D]``; the new K/V are scattered at
    ``context.slot_mapping`` before the cached rows are gathered
    (``context.context_lens`` is the length *after* the write).  The gather is
    a single ``index_select`` per cache covering the whole batch, using the
    flat ``context.key_slot_ids`` built once per step.
    """
    ctx = get_context()
    store_kvcache(key, value, k_cache, v_cache, ctx.slot_mapping)
    k_all = k_cache.view(-1, k_cache.shape[2], k_cache.shape[3]).index_select(
        0, ctx.key_slot_ids
    )
    v_all = v_cache.view(-1, v_cache.shape[2], v_cache.shape[3]).index_select(
        0, ctx.key_slot_ids
    )
    outputs = []
    offset = 0
    for i in range(ctx.num_seqs):
        start, end = ctx.query_start_loc[i], ctx.query_start_loc[i + 1]
        length = ctx.context_lens[i]
        query_i = query[start:end]
        key_i = k_all[offset : offset + length]
        value_i = v_all[offset : offset + length]
        # Prefill feeds the whole sequence (queries == keys, causal);
        # decode feeds a single token that may attend to every cached key.
        causal = query_i.shape[0] == length
        outputs.append(_sdpa(query_i, key_i, value_i, causal, scale))
        offset += length
    return torch.cat(outputs, dim=0)


def cached_cross_attention(query: torch.Tensor, layer, scale: float) -> torch.Tensor:
    """Cross-attention against the per-request encoder K/V cached on ``layer``."""
    ctx = get_context()
    outputs = []
    for i in range(ctx.num_seqs):
        start, end = ctx.query_start_loc[i], ctx.query_start_loc[i + 1]
        key, value = layer.encoder_kv_cache[ctx.request_ids[i]]
        outputs.append(cross_sdpa(query[start:end], key, value, scale))
    return torch.cat(outputs, dim=0)
