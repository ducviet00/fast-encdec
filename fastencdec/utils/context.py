from dataclasses import dataclass

import torch


@dataclass(slots=True)
class Context:
    """Per-step metadata shared across attention layers; cross_* fields are for cross-attention."""

    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    slot_mapping: torch.Tensor | None = None
    request_ids: list[int] | None = None
    cross_staged: list | None = None
    cross_cache_seqlens: torch.Tensor | None = None
    cross_block_tables: torch.Tensor | None = None


_CONTEXT: Context | None = None


def get_context() -> Context:
    """Return the active step's Context."""
    return _CONTEXT


def set_context(**fields) -> None:
    """Set the global Context for the current model forward pass."""
    global _CONTEXT
    _CONTEXT = Context(**fields)
