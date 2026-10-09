"""Paged self-attention + cached cross-attention.

The decoder self-attention is paged and laid out as
``[num_blocks, num_heads, block_size, head_dim]`` (vLLM's layout).  When the
vendored vLLM CPU kernel is active (``Context.attn_isa``) it writes and reads
that cache directly; otherwise a pure-PyTorch SDPA fallback gathers each
sequence's rows and attends per sequence.
"""

import torch
import torch.nn.functional as F

from ..utils.context import get_context
from . import cpu_attn


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    """Scatter new K/V ``[T, H, D]`` into the ``[blocks, H, block_size, D]`` cache."""
    block = torch.div(slot_mapping, block_size, rounding_mode="floor")
    pos = slot_mapping % block_size
    k_cache[block, :, pos, :] = key
    v_cache[block, :, pos, :] = value


def _sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool,
    scale: float,
) -> torch.Tensor:
    # [L, H, D] -> [1, H, L, D].  ``contiguous`` matters on aarch64: SDPA with
    # BF16 non-contiguous inputs takes a ~20x slower path (see
    # BartEncoderSelfAttention.forward).
    out = F.scaled_dot_product_attention(
        query.transpose(0, 1).unsqueeze(0).contiguous(),
        key.transpose(0, 1).unsqueeze(0).contiguous(),
        value.transpose(0, 1).unsqueeze(0).contiguous(),
        is_causal=causal,
        scale=scale,
    )
    return out.squeeze(0).transpose(0, 1)


def cross_sdpa(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float
) -> torch.Tensor:
    """Cross-attention over encoder K/V already stored as contiguous [H, S, D]."""
    out = F.scaled_dot_product_attention(
        query.transpose(0, 1).unsqueeze(0).contiguous(),
        key.unsqueeze(0),
        value.unsqueeze(0),
        is_causal=False,
        scale=scale,
    )
    return out.squeeze(0).transpose(0, 1)


def _gather_all(cache: torch.Tensor, key_slot_ids) -> torch.Tensor:
    """Gather every cached key with one advanced index -> ``[T, H, D]``."""
    block_ids, pos_in_block = key_slot_ids
    return cache[block_ids, :, pos_in_block, :]


def _sdpa_paged(query, key, value, k_cache, v_cache, ctx, scale):
    store_kvcache(key, value, k_cache, v_cache, ctx.slot_mapping, k_cache.shape[2])
    k_all = _gather_all(k_cache, ctx.key_slot_ids)
    v_all = _gather_all(v_cache, ctx.key_slot_ids)
    outputs = []
    offset = 0
    for i in range(ctx.num_seqs):
        start, end = ctx.query_start_loc[i], ctx.query_start_loc[i + 1]
        length = ctx.context_lens[i]
        query_i = query[start:end]
        key_i = k_all[offset : offset + length]
        value_i = v_all[offset : offset + length]
        # Prefill feeds the whole sequence (queries == keys, causal); decode
        # feeds a single token that may attend to every cached key.
        causal = query_i.shape[0] == length
        outputs.append(_sdpa(query_i, key_i, value_i, causal, scale))
        offset += length
    return torch.cat(outputs, dim=0)


def _vllm_paged(query, key, value, k_cache, v_cache, ctx, scale):
    mod = cpu_attn.module()
    mod.reshape_and_cache(key, value, k_cache, v_cache, ctx.slot_mapping, ctx.attn_isa)
    return mod.attention_forward(
        query,
        k_cache,
        v_cache,
        ctx.cpu_query_start_loc,
        ctx.cpu_seq_lens,
        scale,
        False,
        -1,
        ctx.block_table,
        ctx.attn_metadata,
        ctx.dynamic_causal,
    )


def paged_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Write new K/V into the paged cache, then attend causally over it."""
    ctx = get_context()
    if ctx.attn_isa:
        return _vllm_paged(query, key, value, k_cache, v_cache, ctx, scale)
    return _sdpa_paged(query, key, value, k_cache, v_cache, ctx, scale)


def cached_cross_attention(query: torch.Tensor, layer, scale: float) -> torch.Tensor:
    """Cross-attention against the per-request encoder K/V cached on ``layer``."""
    ctx = get_context()
    paged = layer.encoder_paged_cache
    if ctx.attn_isa and paged is not None:
        k_cache, v_cache = paged
        return cpu_attn.module().attention_forward(
            query,
            k_cache,
            v_cache,
            ctx.cpu_query_start_loc,
            ctx.cross_seq_lens,
            scale,
            False,
            -1,
            ctx.cross_block_table,
            ctx.cross_metadata,
            None,
        )
    outputs = []
    for i in range(ctx.num_seqs):
        start, end = ctx.query_start_loc[i], ctx.query_start_loc[i + 1]
        key, value = layer.encoder_kv_cache[ctx.request_ids[i]]
        outputs.append(cross_sdpa(query[start:end], key, value, scale))
    return torch.cat(outputs, dim=0)
