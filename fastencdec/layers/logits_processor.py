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
    device: str = "cpu",
) -> LogitsProcessorList:
    """Build a transformers LogitsProcessorList from SamplingParams."""
    eos_ids = sorted(params.eos_ids())
    eos = torch.tensor(eos_ids, dtype=torch.long, device=device) if eos_ids else None

    processors = LogitsProcessorList()
    if params.repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(params.repetition_penalty))
    if params.no_repeat_ngram_size > 0:
        processors.append(NoRepeatNGramLogitsProcessor(params.no_repeat_ngram_size))
    if params.min_length > 0 and eos is not None:
        processors.append(
            MinLengthLogitsProcessor(params.min_length, eos, device=device)
        )
    if params.forced_bos_token_id is not None:
        processors.append(ForcedBOSTokenLogitsProcessor(params.forced_bos_token_id))
    if params.forced_eos_token_id is not None:
        processors.append(
            ForcedEOSTokenLogitsProcessor(
                max_length, params.forced_eos_token_id, device=device
            )
        )
    if params.suppress_tokens:
        processors.append(
            SuppressTokensLogitsProcessor(params.suppress_tokens, device=device)
        )
    if params.begin_suppress_tokens:
        processors.append(
            SuppressTokensAtBeginLogitsProcessor(
                params.begin_suppress_tokens, begin_index, device=device
            )
        )

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
