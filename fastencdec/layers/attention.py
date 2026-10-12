"""Paged self-attention + cached cross-attention.

The decoder self-attention is paged.  Three backends consume it:

* the vendored vLLM CPU kernel (``Context.attn_backend == "vllm"``) and the
  pure-PyTorch SDPA fallback (``"sdpa"``) use the ``[blocks, heads, block_size,
  head_dim]`` layout; SDPA overlaps every sequence's rows in one masked softmax
  on CUDA and attends per sequence on CPU,
* FlashAttention-2 on CUDA (``"flash"``) uses the page-major ``[blocks,
  block_size, heads, head_dim]`` layout its paged decode kernel expects:
  ``flash_attn_with_kvcache`` reads the cache straight through the block table.

Cross-attention uses dense per-request encoder K/V.  With the vLLM kernel it is
staged into a head-major paged cache, and with FlashAttention-2 into a
page-major one; SDPA attends per sequence against the dense cache.
"""

import torch
import torch.nn.functional as F

from ..utils.context import get_context
from . import cpu_attn, flash_attn


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    block_major: bool = False,
) -> None:
    """Scatter new K/V ``[T, H, D]`` into the paged cache.

    ``block_major=False`` writes the vLLM/SDPA layout
    ``[blocks, H, block_size, D]``; ``block_major=True`` writes the
    FlashAttention-2 page layout ``[blocks, block_size, H, D]``.
    """
    block = torch.div(slot_mapping, block_size, rounding_mode="floor")
    pos = slot_mapping % block_size
    if block_major:
        k_cache[block, pos, :, :] = key
        v_cache[block, pos, :, :] = value
    else:
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


def dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool,
    scale: float | None = None,
) -> torch.Tensor:
    """Non-paged ``[B, H, S, D]`` attention (encoder self-attention).

    Uses FlashAttention-2 on CUDA when the backend is active and the head dim is
    supported; otherwise PyTorch SDPA (which itself dispatches to a flash kernel
    on CUDA).
    """
    if (
        flash_attn.is_active()
        and query.is_cuda
        and flash_attn.supported(query.dtype, query.shape[-1])
    ):
        mod = flash_attn.module()
        if mod is not None:
            out = mod.flash_attn_func(
                query.transpose(1, 2).contiguous(),
                key.transpose(1, 2).contiguous(),
                value.transpose(1, 2).contiguous(),
                causal=causal,
                softmax_scale=scale,
            )
            return out.transpose(1, 2)
    return F.scaled_dot_product_attention(
        query, key, value, is_causal=causal, scale=scale
    )


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
    if query.is_cuda:
        return _sdpa_paged_batched(query, k_cache, v_cache, ctx, scale)
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


def _sdpa_paged_batched(query, k_cache, v_cache, ctx, scale):
    """Batched block-diagonal SDPA over the gathered cache (CUDA).

    The per-sequence loop is launch-bound on a GPU; this instead gathers every
    cached key once and attends all queries in one masked softmax, with the
    causal/padding mask built from the sequence lengths.
    """
    device = query.device
    k_all = _gather_all(k_cache, ctx.key_slot_ids)  # [Tk, H, D]
    v_all = _gather_all(v_cache, ctx.key_slot_ids)
    qsl = ctx.query_start_loc
    lens = torch.tensor(ctx.context_lens, dtype=torch.long, device=device)
    qlens = torch.tensor(
        [qsl[i + 1] - qsl[i] for i in range(ctx.num_seqs)],
        dtype=torch.long,
        device=device,
    )
    starts = torch.zeros_like(lens)
    if lens.numel() > 1:
        starts[1:] = torch.cumsum(lens, 0)[:-1]

    seq_of_q = torch.repeat_interleave(torch.arange(ctx.num_seqs, device=device), qlens)
    pos_in_seq = torch.arange(query.shape[0], device=device) - torch.repeat_interleave(
        torch.tensor(qsl[:-1], device=device), qlens
    )
    # Highest absolute key index each query attends (inclusive): decode (q_len 1)
    # sees every cached key, prefill (q_len == len) sees its causal prefix.
    hi = starts[seq_of_q] + (lens - qlens)[seq_of_q] + pos_in_seq
    keys = torch.arange(k_all.shape[0], device=device)
    mask = (keys.unsqueeze(0) >= starts[seq_of_q].unsqueeze(1)) & (
        keys.unsqueeze(0) <= hi.unsqueeze(1)
    )
    scores = torch.einsum("thd,shd->hts", query.float(), k_all.float()) * scale
    scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hts,shd->thd", probs, v_all.float())
    return out.to(query.dtype)


def _flash_paged(query, key, value, k_cache, v_cache, ctx, scale):
    # ``k_cache``/``v_cache`` are ``[blocks, block_size, H, D]``: write the new
    # token ourselves, then let the paged decode kernel read the whole cache.
    store_kvcache(
        key,
        value,
        k_cache,
        v_cache,
        ctx.slot_mapping,
        k_cache.shape[1],
        block_major=True,
    )
    out = flash_attn.require().flash_attn_with_kvcache(
        query.unsqueeze(1),
        k_cache,
        v_cache,
        cache_seqlens=ctx.cache_seqlens,
        block_table=ctx.block_table,
        # Bottom-right aligned: one decode query attends to every cached key,
        # a prefill to exactly its prefix (same rule as ``_sdpa_paged``).
        causal=True,
        softmax_scale=scale,
    )
    return out.squeeze(1)


def _flash_cross(query, layer, ctx, scale):
    """Cross-attention against the per-request encoder paged cache."""
    k_cache, v_cache = layer.encoder_flash_cache
    out = flash_attn.require().flash_attn_with_kvcache(
        query.unsqueeze(1),
        k_cache,
        v_cache,
        cache_seqlens=ctx.cross_cache_seqlens,
        block_table=ctx.cross_block_table,
        causal=False,
        softmax_scale=scale,
    )
    return out.squeeze(1)


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
    if ctx.attn_backend == "flash":
        return _flash_paged(query, key, value, k_cache, v_cache, ctx, scale)
    if ctx.attn_backend == "vllm":
        return _vllm_paged(query, key, value, k_cache, v_cache, ctx, scale)
    return _sdpa_paged(query, key, value, k_cache, v_cache, ctx, scale)


def cached_cross_attention(query: torch.Tensor, layer, scale: float) -> torch.Tensor:
    """Cross-attention against the per-request encoder K/V cached on ``layer``."""
    ctx = get_context()
    if ctx.attn_backend == "flash":
        return _flash_cross(query, layer, ctx, scale)
    if ctx.attn_backend == "vllm":
        paged = layer.encoder_paged_cache
        if paged is not None:
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
