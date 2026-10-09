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
        slot_mapping: Flat physical cache slots for the new tokens, ``[T]``.
        query_start_loc: Cumulative new-token counts, length num_seqs + 1.
        context_lens: Total cached length per sequence, length num_seqs.
        request_ids: Request id per sequence (used for cross-attn cache).
        num_seqs: Number of sequences in the batch.
        attn_isa: Kernel ISA (``"vec"``/``"vec16"``) or ``None`` for SDPA.
        block_table: int32 ``[num_seqs, max_blocks]`` paged block ids.
        key_slot_ids: ``(block_ids, pos_in_block)`` for every cached key,
            ordered by sequence (SDPA fallback gather).
        cpu_query_start_loc / cpu_seq_lens / dynamic_causal / attn_metadata:
            int32 inputs and scheduler metadata for the vendored kernel.
        cross_block_table / cross_seq_lens / cross_metadata: the same for the
            encoder (cross-attention) paged cache.
    """

    slot_mapping: object
    query_start_loc: list
    context_lens: list
    request_ids: list
    num_seqs: int
    attn_isa: str | None = None
    block_table: object | None = None
    key_slot_ids: object | None = None
    cpu_query_start_loc: object | None = None
    cpu_seq_lens: object | None = None
    dynamic_causal: object | None = None
    attn_metadata: object | None = None
    cross_block_table: object | None = None
    cross_seq_lens: object | None = None
    cross_metadata: object | None = None


_CONTEXT: Context | None = None


def get_context() -> Context:
    return _CONTEXT


def set_context(**fields) -> None:
    global _CONTEXT
    _CONTEXT = Context(**fields)
