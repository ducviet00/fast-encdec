"""No-repeat n-gram masking on the device (Triton, CUDA).

``transformers``' ``NoRepeatNGramLogitsProcessor`` walks every row in Python and
calls ``input_ids[i].tolist()``, which synchronises the device and serialises the
decode loop.  This kernel applies the same mask in one launch, so the processor
can run inside the pipelined decode and the beam ranking with no host
round-trip.

The rule (matching ``GenerationMixin``): once a row has at least ``n`` tokens,
let ``prefix`` be its last ``n - 1`` tokens; every earlier occurrence of
``prefix`` bans the token that followed it.  The CPU path keeps the
``transformers`` processor.
"""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only install: fall back to the transformers processor
    triton = None
    tl = None


def triton_available() -> bool:
    return triton is not None


if triton is not None:

    @triton.jit
    def _ban_repeated_ngrams_kernel(
        input_ids_ptr,
        scores_ptr,
        stride_ids_b,
        stride_scores_b,
        cur_len,
        vocab_size,
        ngram_size: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        block_idx = tl.program_id(1)

        # Each lane handles one window start position in the sequence.
        win_start = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        num_windows = cur_len - ngram_size + 1
        valid_window = win_start < num_windows

        row_ids_ptr = input_ids_ptr + batch_idx * stride_ids_b
        prefix_start = cur_len - ngram_size + 1

        # Compare the (ngram_size - 1) prefix tokens.
        is_match = valid_window
        for k in range(ngram_size - 1):
            p_tok = tl.load(row_ids_ptr + prefix_start + k)
            w_tok = tl.load(row_ids_ptr + win_start + k, mask=is_match, other=-1)
            is_match = is_match & (w_tok == p_tok)

        # Load the last token of matching windows and ban it in-place in scores.
        banned_tok = tl.load(
            row_ids_ptr + win_start + ngram_size - 1, mask=is_match, other=vocab_size
        )
        can_ban = is_match & (banned_tok >= 0) & (banned_tok < vocab_size)

        row_scores_ptr = scores_ptr + batch_idx * stride_scores_b
        tl.store(row_scores_ptr + banned_tok, -float("inf"), mask=can_ban)


class TritonNoRepeatNGramLogitsProcessor:
    """Device-side ``NoRepeatNGramLogitsProcessor`` (same mask, one launch).

    ``input_ids`` must be contiguous along the sequence and ``scores`` along the
    vocab; both callers build them that way and mutate ``scores`` in place.
    """

    def __init__(self, ngram_size: int):
        if triton is None:
            raise RuntimeError("triton is required for the device n-gram processor")
        self.ngram_size = ngram_size

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        cur_len = input_ids.shape[-1]
        if cur_len < self.ngram_size:
            return scores
        batch_size, vocab_size = scores.shape
        num_windows = cur_len - self.ngram_size + 1
        block = 256
        grid = (batch_size, triton.cdiv(num_windows, block))
        _ban_repeated_ngrams_kernel[grid](
            input_ids,
            scores,
            input_ids.stride(0),
            scores.stride(0),
            cur_len,
            vocab_size,
            ngram_size=self.ngram_size,
            BLOCK_SIZE=block,
        )
        return scores
