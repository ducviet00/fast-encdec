"""Logits processors from ``transformers`` for the paged engine.

The engine builds one :class:`~transformers.LogitsProcessorList` per request
(see :func:`build_logits_processor`) in the same order as
``GenerationMixin._get_logits_processor``.  Greedy/sampling apply it to the step
logits and beam search to the log-probs, matching where ``GenerationMixin``
invokes the processors (``_sample`` vs ``_beam_search``).
"""

import torch
from transformers import LogitsProcessorList
from transformers.generation.logits_process import (
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    MinLengthLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


def build_logits_processor(
    params,
    *,
    max_length: int,
    begin_index: int,
    is_beam: bool,
) -> LogitsProcessorList:
    """Assemble the processors for one request's resolved ``SamplingParams``.

    Args:
        max_length: total decoder length at which a forced EOS fires.
        begin_index: decoder position at which ``begin_suppress_tokens`` apply.
        is_beam: keep at least ``num_eos + 1`` tokens in the warpers so beam
            search always has a non-EOS continuation to explore.
    """
    eos_ids = sorted(params.eos_ids())
    eos = torch.tensor(eos_ids, dtype=torch.long) if eos_ids else None

    processors = LogitsProcessorList()
    if params.repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(params.repetition_penalty))
    if params.no_repeat_ngram_size > 0:
        processors.append(NoRepeatNGramLogitsProcessor(params.no_repeat_ngram_size))
    if params.min_length > 0 and eos is not None:
        processors.append(MinLengthLogitsProcessor(params.min_length, eos))
    if params.forced_bos_token_id is not None:
        processors.append(ForcedBOSTokenLogitsProcessor(params.forced_bos_token_id))
    if params.forced_eos_token_id is not None:
        processors.append(
            ForcedEOSTokenLogitsProcessor(max_length, params.forced_eos_token_id)
        )
    if params.suppress_tokens:
        processors.append(SuppressTokensLogitsProcessor(params.suppress_tokens))
    if params.begin_suppress_tokens:
        processors.append(
            SuppressTokensAtBeginLogitsProcessor(
                params.begin_suppress_tokens, begin_index
            )
        )

    # The warpers only run when sampling; greedy keeps the processors above.
    if params.temperature > 0:
        min_tokens = len(eos_ids) + 1 if is_beam else 1
        if params.temperature != 1.0:
            processors.append(TemperatureLogitsWarper(params.temperature))
        if params.top_k > 0:
            processors.append(
                TopKLogitsWarper(params.top_k, min_tokens_to_keep=min_tokens)
            )
        if params.top_p < 1.0:
            processors.append(
                TopPLogitsWarper(params.top_p, min_tokens_to_keep=min_tokens)
            )
    return processors
