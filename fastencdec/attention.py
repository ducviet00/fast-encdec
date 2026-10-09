"""CPU paged attention built on ``torch.nn.functional.scaled_dot_product_attention``.

No padding is used: each sequence in the batch is attended independently by
slicing ``[query_start_loc[i]:query_start_loc[i + 1]]`` and gathering its own
K/V blocks from the paged cache.
"""

import torch
import torch.nn.functional as F

from .context import get_context


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


def _gather(
    cache: torch.Tensor, block_table, length: int, block_size: int
) -> torch.Tensor:
    """Collect ``length`` contiguous K/V rows for one sequence."""
    num_blocks = (length + block_size - 1) // block_size
    blocks = torch.tensor(block_table[:num_blocks], dtype=torch.long)
    return cache.index_select(0, blocks).reshape(-1, *cache.shape[2:])[:length]


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
    ``context.slot_mapping`` before each sequence gathers its cached rows
    (``context.context_lens`` is the length *after* the write).
    """
    ctx = get_context()
    store_kvcache(key, value, k_cache, v_cache, ctx.slot_mapping)
    block_size = k_cache.shape[1]
    outputs = []
    for i in range(ctx.num_seqs):
        start, end = ctx.query_start_loc[i], ctx.query_start_loc[i + 1]
        query_i = query[start:end]
        length = ctx.context_lens[i]
        key = _gather(k_cache, ctx.block_tables[i], length, block_size)
        value = _gather(v_cache, ctx.block_tables[i], length, block_size)
        # Prefill feeds the whole sequence (queries == keys, causal);
        # decode feeds a single token that may attend to every cached key.
        causal = query_i.shape[0] == length
        outputs.append(_sdpa(query_i, key, value, causal, scale))
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
