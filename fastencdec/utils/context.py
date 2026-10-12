"""Global per-step metadata read by the attention ops.

The model runner calls :func:`set_context` once per forward; the device-agnostic
attention ops in :mod:`fastencdec.layers.attention` read it with
:func:`get_context`.  Every field is a plain tensor (or list) on the compute
device, so the same context drives the vendored vLLM CPU kernel, FlashAttention-2
on CUDA and the PyTorch SDPA fallback; the ops derive their kernel-specific
metadata from it.

The decoder is always fed a single token per sequence (the prompt is one
``decoder_start_token_id``), so query lengths are implied by ``context_lens``.
"""

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class Context:
    """Metadata for one model forward pass.

    Attributes:
        is_prefill: True when every sequence is fresh (nothing cached yet).
        context_lens: int32 ``[num_seqs]`` cached length per sequence, after the
            write.
        block_tables: int32 ``[num_seqs, max_blocks]`` paged self-attention block
            ids (a token at position ``p`` is in ``block_tables[i, p // bs]`` at
            offset ``p % bs``).
        slot_mapping: int64 ``[num_seqs]`` flat cache slots for the new tokens.
        request_ids: ``[num_seqs]`` request id per sequence; cross-attention
            looks up the layer's dense per-request encoder K/V by it.
        cross_cache_seqlens / cross_block_tables: static paged cross-attention
            metadata (set only by the CUDA-graph path, where the encoder cache is
            a fixed buffer the graph can reference).
    """

    is_prefill: bool = False
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    slot_mapping: torch.Tensor | None = None
    request_ids: list[int] | None = None
    cross_cache_seqlens: torch.Tensor | None = None
    cross_block_tables: torch.Tensor | None = None


_CONTEXT: Context | None = None


def get_context() -> Context:
    return _CONTEXT


def set_context(**fields) -> None:
    global _CONTEXT
    _CONTEXT = Context(**fields)
