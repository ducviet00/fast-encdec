"""Sequence and sampling-parameter dataclasses for the paged engine."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class SamplingParams:
    """Decoding options.  ``num_beams > 1`` switches to beam search.

    The defaults are neutral and let you pin every value explicitly.
    :meth:`from_generation_config` builds one from a checkpoint's
    ``GenerationConfig``; ``LLM.generate`` uses that when no params are passed,
    so checkpoint heuristics (min-length, n-grams, forced BOS/EOS, ...) apply by
    default while an explicit ``SamplingParams`` stays in full control.
    """

    max_tokens: int = 64
    num_beams: int = 1
    temperature: float = 0.0  # <= 0 => greedy
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    eos_token_id: int | list[int] | None = None
    # Generation heuristics (HuggingFace-compatible defaults).
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    min_length: int = 0
    forced_bos_token_id: int | None = None
    forced_eos_token_id: int | None = None
    suppress_tokens: list[int] | None = None
    begin_suppress_tokens: list[int] | None = None

    def eos_ids(self) -> set[int]:
        if self.eos_token_id is None:
            return set()
        if isinstance(self.eos_token_id, int):
            return {self.eos_token_id}
        return set(self.eos_token_id)

    @classmethod
    def from_generation_config(
        cls,
        config,
        eos_token_id: int | None = None,
        decoder_prompt_len: int = 1,
    ) -> "SamplingParams":
        """Map a ``transformers.GenerationConfig`` onto :class:`SamplingParams`.

        ``GenerationConfig`` leaves unset fields as ``None`` in recent
        transformers, so each field falls back to a neutral default here.
        """
        temperature = config.temperature
        if not config.do_sample:
            temperature = 0.0  # greedy
        elif temperature is None:
            temperature = 1.0

        if config.max_new_tokens is not None:
            max_tokens = config.max_new_tokens
        elif config.max_length is not None:
            max_tokens = config.max_length - decoder_prompt_len
        else:
            max_tokens = 64

        as_list = lambda tokens: None if tokens is None else list(tokens)
        return cls(
            max_tokens=max_tokens,
            num_beams=_or(config.num_beams, 1),
            temperature=temperature,
            top_p=_or(config.top_p, 1.0),
            top_k=_or(config.top_k, 0),
            repetition_penalty=_or(config.repetition_penalty, 1.0),
            length_penalty=_or(config.length_penalty, 1.0),
            no_repeat_ngram_size=_or(config.no_repeat_ngram_size, 0),
            min_length=_or(config.min_length, 0),
            eos_token_id=_or(config.eos_token_id, eos_token_id),
            forced_bos_token_id=config.forced_bos_token_id,
            forced_eos_token_id=config.forced_eos_token_id,
            suppress_tokens=as_list(config.suppress_tokens),
            begin_suppress_tokens=as_list(config.begin_suppress_tokens),
        )


def _or(value, fallback):
    return fallback if value is None else value


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
