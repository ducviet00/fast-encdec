"""LogitsProcessor matches transformers' LogitsProcessorList exactly.

Covers every knob build_logits_processor used to wire up, on random logits and
histories including the edge lengths (len < ngram, len == 1 for forced BOS,
len == max-1 for forced EOS, begin_index).
"""

import itertools

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

from fastencdec.layers.logits_processor import LogitsProcessor
from fastencdec.sampling_params import SamplingParams

VOCAB = 37
EOS = [2, 7]


def reference(params, input_ids, scores):
    """The old stack, built the way build_logits_processor built it."""
    eos = torch.tensor(sorted(EOS), dtype=torch.long)
    procs = LogitsProcessorList()
    if params.repetition_penalty != 1.0:
        procs.append(RepetitionPenaltyLogitsProcessor(params.repetition_penalty))
    if params.no_repeat_ngram_size > 0:
        procs.append(NoRepeatNGramLogitsProcessor(params.no_repeat_ngram_size))
    if params.min_length > 0:
        procs.append(MinLengthLogitsProcessor(params.min_length, eos))
    if params.forced_bos_token_id is not None:
        procs.append(ForcedBOSTokenLogitsProcessor(params.forced_bos_token_id))
    if params.forced_eos_token_id is not None:
        procs.append(ForcedEOSTokenLogitsProcessor(64, params.forced_eos_token_id))
    if params.suppress_tokens:
        procs.append(SuppressTokensLogitsProcessor(params.suppress_tokens))
    if params.begin_suppress_tokens:
        procs.append(
            SuppressTokensAtBeginLogitsProcessor(params.begin_suppress_tokens, 3)
        )
    if params.temperature > 0:
        min_tokens = len(EOS) + 1 if params.num_beams > 1 else 1
        if params.temperature != 1.0:
            procs.append(TemperatureLogitsWarper(params.temperature))
        if params.top_k > 0:
            procs.append(TopKLogitsWarper(params.top_k, min_tokens_to_keep=min_tokens))
        if params.top_p < 1.0:
            procs.append(TopPLogitsWarper(params.top_p, min_tokens_to_keep=min_tokens))
    return procs(input_ids, scores)


def check(params, rows):
    torch.manual_seed(rows)
    input_ids = torch.randint(0, VOCAB, (4, rows))
    scores = torch.randn(4, VOCAB) * 3
    want = reference(params, input_ids, scores.clone())
    processor = LogitsProcessor(
        params,
        vocab_size=VOCAB,
        device="cpu",
        max_length=64,
        begin_index=3,
        is_beam=params.num_beams > 1,
    )
    got = processor(input_ids, scores.clone())
    assert torch.equal(got, want), f"mismatch rows={rows} params={params}"


def check_padded(params, true_lens):
    torch.manual_seed(sum(true_lens) * 131 + len(true_lens))
    rows = [torch.randint(0, VOCAB, (L,)) for L in true_lens]
    scores = torch.randn(len(rows), VOCAB) * 3
    want = torch.stack(
        [
            reference(params, r.unsqueeze(0), s.unsqueeze(0).clone()).squeeze(0)
            for r, s in zip(rows, scores)
        ]
    )
    width = max(true_lens)
    pad = torch.full((len(rows), width), -1, dtype=torch.long)
    for r, t in enumerate(rows):
        pad[r, width - len(t) :] = t
    processor = LogitsProcessor(
        params,
        vocab_size=VOCAB,
        device="cpu",
        max_length=64,
        begin_index=3,
        is_beam=params.num_beams > 1,
    )
    got = processor(pad, scores.clone(), torch.tensor(true_lens))
    assert torch.equal(got, want), f"padded mismatch lens={true_lens} params={params}"


def main():
    cases = [
        SamplingParams(eos_token_id=EOS, repetition_penalty=1.3),
        SamplingParams(eos_token_id=EOS, no_repeat_ngram_size=1),
        SamplingParams(eos_token_id=EOS, no_repeat_ngram_size=2),
        SamplingParams(eos_token_id=EOS, no_repeat_ngram_size=3),
        SamplingParams(eos_token_id=EOS, min_length=9),
        SamplingParams(eos_token_id=EOS, forced_bos_token_id=0),
        SamplingParams(eos_token_id=EOS, forced_eos_token_id=2),
        SamplingParams(eos_token_id=EOS, suppress_tokens=[5, 11]),
        SamplingParams(eos_token_id=EOS, begin_suppress_tokens=[5, 11]),
        SamplingParams(
            eos_token_id=EOS,
            min_length=9,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            forced_bos_token_id=0,
            suppress_tokens=[5],
        ),
        SamplingParams(eos_token_id=EOS, temperature=0.7, top_k=5),
        SamplingParams(eos_token_id=EOS, temperature=0.7, top_p=0.9),
        SamplingParams(eos_token_id=EOS, temperature=0.7, top_k=5, top_p=0.9),
        SamplingParams(
            eos_token_id=EOS, num_beams=4, temperature=0.7, top_k=5, top_p=0.9
        ),
    ]
    lengths = [1, 2, 3, 5, 9, 12, 63]
    for params, rows in itertools.product(cases, lengths):
        check(params, rows)
    print(f"processor parity ok ({len(cases)} param sets x {len(lengths)} lengths)")

    for lens in (
        [1, 2, 5],
        [2, 5, 9],
        [1, 1, 12],
        [3, 3, 3],
        [5, 9, 12],
        [1, 3, 9, 12],
        [63, 5],
    ):
        for params in cases:
            check_padded(params, lens)
    print(f"padded parity ok ({len(cases)} param sets x 7 shapes)")


if __name__ == "__main__":
    main()
