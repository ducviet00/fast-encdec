"""Opaque torch.compile ops for the ``cpu-attention`` kernels.

The op bodies run at call time, not at trace time, so they can read the
per-step Context and call the kernel while Dynamo sees a single op (see
``TORCH_COMPILE_FREEZING.md``).
"""

import torch

from ..utils.context import get_context
from . import cpu_attn


@torch.library.custom_op(
    "fastencdec::cpu_self_attention", mutates_args={"k_cache", "v_cache"}
)
def self_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Write K/V and run causal self-attention with the CPU kernel."""
    ctx = get_context()
    return cpu_attn.self_attention(
        query,
        key,
        value,
        k_cache,
        v_cache,
        ctx.block_tables,
        ctx.context_lens,
        scale,
    )


@self_attention.register_fake
def _self_attention_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return torch.empty_like(query)


@torch.library.custom_op("fastencdec::cpu_cross_attention", mutates_args={})
def cross_attention(
    query: torch.Tensor,
    layer_idx: int,
    scale: float,
) -> torch.Tensor:
    """Run cross-attention over the staged encoder K/V for decoder layer ``layer_idx``."""
    staged = get_context().cross_staged
    if staged is None:
        raise RuntimeError("cross-attention cache is not staged for this step")
    k_cache, v_cache, block_table, cache_seqlens = staged[layer_idx]
    return cpu_attn.cross_attention(
        query, k_cache, v_cache, block_table, cache_seqlens, scale
    )


@cross_attention.register_fake
def _cross_attention_fake(
    query: torch.Tensor, layer_idx: int, scale: float
) -> torch.Tensor:
    return torch.empty_like(query)


@torch.library.custom_op("fastencdec::cpu_dense_attention", mutates_args={})
def dense_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float
) -> torch.Tensor:
    """Run dense non-causal attention over [B, S, H, D] with the CPU kernel."""
    return cpu_attn.dense_attention(query, key, value, scale)


@dense_attention.register_fake
def _dense_attention_fake(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float
) -> torch.Tensor:
    return query.new_empty(query.shape)
