"""Global step context read by the model's attention layers.

The model runner calls :func:`set_context` once per forward; attention layers
read the metadata with :func:`get_context`.  This avoids threading a dozen
arguments through every module and keeps the model code close to the HF
reference.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class Context:
    """Metadata for one model forward pass.

    Attributes:
        slot_mapping: Physical cache slots for the new tokens, shape [T].
        query_start_loc: Cumulative new-token counts, length num_seqs + 1.
        context_lens: Total cached length per sequence, length num_seqs.
        key_slot_ids: Flat physical cache slots for every cached key, ordered by
            sequence (seq 0's positions, then seq 1's, ...).  Attention gathers
            all layers' K/V with one ``index_select`` per cache instead of one
            per sequence.
        request_ids: Request id per sequence (used for cross-attn cache).
        num_seqs: Number of sequences in the batch.
    """

    slot_mapping: list
    query_start_loc: list
    context_lens: list
    key_slot_ids: "torch.Tensor"
    request_ids: list
    num_seqs: int


_CONTEXT: Context | None = None


def get_context() -> Context:
    return _CONTEXT


def set_context(
    slot_mapping,
    query_start_loc,
    context_lens,
    key_slot_ids,
    request_ids,
    num_seqs,
) -> None:
    global _CONTEXT
    _CONTEXT = Context(
        slot_mapping,
        query_start_loc,
        context_lens,
        key_slot_ids,
        request_ids,
        num_seqs,
    )
