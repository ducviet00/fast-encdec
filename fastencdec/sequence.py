"""Sequence and sampling-parameter dataclasses for the paged engine."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class SamplingParams:
    """Decoding options.  ``num_beams > 1`` switches to beam search."""

    max_tokens: int = 64
    num_beams: int = 1
    temperature: float = 0.0  # 0 => greedy
    top_p: float = 1.0
    eos_token_id: int | list[int] | None = None
    # Generation heuristics (HuggingFace-compatible defaults).
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    min_length: int = 0

    def eos_ids(self) -> set[int]:
        if self.eos_token_id is None:
            return set()
        if isinstance(self.eos_token_id, int):
            return {self.eos_token_id}
        return set(self.eos_token_id)


@dataclass(eq=False)
class Sequence:
    """One decoder hypothesis.

    ``token_ids`` holds the decoder prompt followed by generated tokens.
    ``num_cached_tokens`` counts how many of them already have K/V in the paged
    cache, so the tokens to feed next step are ``token_ids[num_cached_tokens:]``.
    Beam search keeps several sequences per request, each with its own
    ``block_table`` (blocks are shared via reference counting).
    """

    request_id: int
    decoder_prompt_ids: list[int]
    encoder_token_ids: list[int]
    sampling: SamplingParams
    token_ids: list[int] = field(default_factory=list)
    block_table: list[int] = field(default_factory=list)
    num_cached_tokens: int = 0
    cum_logprob: float = 0.0
    # Multimodal input (Florence-2): ``[3, H, W]`` for the request's image.
    pixel_values: "torch.Tensor | None" = None

    def __post_init__(self):
        if not self.token_ids:
            self.token_ids = list(self.decoder_prompt_ids)

    @property
    def num_generated(self) -> int:
        return len(self.token_ids) - len(self.decoder_prompt_ids)

    def generated_ids(self) -> list[int]:
        return self.token_ids[len(self.decoder_prompt_ids) :]
