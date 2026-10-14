from dataclasses import dataclass

import torch


@dataclass(slots=True)
class Context:
    # Per-step metadata shared across attention layers; cross_* fields are for CUDA graphs.
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
