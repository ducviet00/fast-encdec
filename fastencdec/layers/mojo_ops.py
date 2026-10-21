"""Opaque torch.compile ops for the Mojo paged-attention kernel.

Same shape as :mod:`fastencdec.layers.cpu_ops`: the op bodies run at call time,
so they can read the per-step Context and call the kernel while Dynamo sees a
single op (see ``TORCH_COMPILE_FREEZING.md``).
"""

import cpu_attention
import torch

from ..utils.context import get_context


@torch.library.custom_op(
    "fastencdec::mojo_self_attention", mutates_args={"k_cache", "v_cache"}
)
def self_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Write K/V and run attention over the paged cache with the Mojo kernel."""
    if cpu_attention is None:
        raise RuntimeError(
            "attn_backend='mojo' needs the cpu_attention library "
            "(pip install -e ../cpu-attention.mojo)"
        )
    ctx = get_context()
    return cpu_attention.self_attention(
        query,
        key,
        value,
        k_cache,
        v_cache,
        ctx.slot_mapping,
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


@torch.library.custom_op("fastencdec::mojo_cross_attention", mutates_args={})
def cross_attention(
    query: torch.Tensor,
    layer_idx: int,
    scale: float,
) -> torch.Tensor:
    """Run cross-attention over the staged encoder K/V for decoder layer ``layer_idx``."""
    if cpu_attention is None:
        raise RuntimeError(
            "attn_backend='mojo' needs the cpu_attention library "
            "(pip install -e ../cpu-attention.mojo)"
        )
    staged = get_context().cross_staged
    if staged is None:
        raise RuntimeError("cross-attention cache is not staged for this step")
    k_cache, v_cache, block_table, cache_seqlens = staged[layer_idx]
    return cpu_attention.cross_attention(
        query, k_cache, v_cache, block_table, cache_seqlens, scale
    )


@cross_attention.register_fake
def _cross_attention_fake(
    query: torch.Tensor, layer_idx: int, scale: float
) -> torch.Tensor:
    return torch.empty_like(query)
