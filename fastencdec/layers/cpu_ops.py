"""Opaque torch.compile ops for the vendored vLLM CPU attention kernel.

The op bodies run at call time, not at trace time, so they can read the
per-step Context, build metadata, stage cross K/V, and mutate the paged caches
while Dynamo sees a single op (see ``TORCH_COMPILE_FREEZING.md``).
"""

import weakref

import torch

from ..utils.context import get_context
from . import cpu_attn

# Cross-attention layers by integer handle; weak values drop freed models.
_CROSS_LAYERS: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_NEXT_CROSS_HANDLE = 0


def register_cross_layer(layer) -> None:
    """Give a cross-attention layer a stable handle for the opaque op.

    Called from ``BartDecoderCrossAttention.__init__`` — **never** from traced
    code: the FX-graph cache can replay a compiled decoder without running the
    trace, so any trace-time registration would be skipped.
    """
    global _NEXT_CROSS_HANDLE
    if getattr(layer, "cpu_cross_handle", None) is not None:
        return
    layer.cpu_cross_handle = _NEXT_CROSS_HANDLE
    _CROSS_LAYERS[_NEXT_CROSS_HANDLE] = layer
    _NEXT_CROSS_HANDLE += 1


def _cross_layer(handle: int):
    """Resolve a registered cross-attention layer by handle."""
    try:
        return _CROSS_LAYERS[handle]
    except KeyError:
        raise RuntimeError(
            "cross-attention layer handle is no longer alive; keep the model "
            "referenced while its decoder runs"
        ) from None


@torch.library.custom_op(
    "fastencdec::cpu_vllm_self_attention", mutates_args={"k_cache", "v_cache"}
)
def self_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    block_size: int,
) -> torch.Tensor:
    """Write K/V into the paged cache and run causal self-attention."""
    from .attention import _self_derived

    ctx = get_context()
    num_heads, head_dim = query.shape[1], query.shape[2]
    data = _self_derived(ctx, "vllm", num_heads, head_dim, query.dtype, block_size)
    cpu_attn.module().reshape_and_cache(
        key, value, k_cache, v_cache, ctx.slot_mapping, data["isa"]
    )
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


@self_attention.register_fake
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    block_size: int,
) -> torch.Tensor:
    return torch.empty_like(query)


@torch.library.custom_op("fastencdec::cpu_vllm_cross_attention", mutates_args={})
def cross_attention(
    query: torch.Tensor, scale: float, handle: int, block_size: int
) -> torch.Tensor:
    """Run cross-attention over the layer's staged encoder K/V."""
    from .attention import _cross_staged

    layer = _cross_layer(handle)
    ctx = get_context()
    num_heads, head_dim = query.shape[1], query.shape[2]
    k_cache, v_cache, block_table, cache_seqlens, metadata = _cross_staged(
        layer, ctx, "vllm", block_size, num_heads, head_dim, query.dtype
    )
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


@cross_attention.register_fake
def _(query: torch.Tensor, scale: float, handle: int, block_size: int) -> torch.Tensor:
    return torch.empty_like(query)
