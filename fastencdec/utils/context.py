"""Global step context read by the model's attention layers.

The model runner calls :func:`set_context` once per forward; attention layers
read the metadata with :func:`get_context`.  This avoids threading a dozen
arguments through every module and keeps the model code close to the HF
reference.
"""

from dataclasses import dataclass


@dataclass
class Context:
    """Metadata for one model forward pass.

    Attributes:
        slot_mapping: Physical cache slots for the new tokens, shape [T].
        query_start_loc: Cumulative new-token counts, length num_seqs + 1.
        context_lens: Total cached length per sequence, length num_seqs.
        block_tables: Physical block ids per sequence (list of tensors).
        request_ids: Request id per sequence (used for cross-attn cache).
        num_seqs: Number of sequences in the batch.
    """

    slot_mapping: list
    query_start_loc: list
    context_lens: list
    block_tables: list
    request_ids: list
    num_seqs: int


_CONTEXT: Context | None = None


def get_context() -> Context:
    return _CONTEXT


def set_context(
    slot_mapping,
    query_start_loc,
    context_lens,
    block_tables,
    request_ids,
    num_seqs,
) -> None:
    global _CONTEXT
    _CONTEXT = Context(
        slot_mapping,
        query_start_loc,
        context_lens,
        block_tables,
        request_ids,
        num_seqs,
    )
