"""One decoder hypothesis, plus its KV-cache bookkeeping."""

from typing import TYPE_CHECKING

from ..sampling_params import SamplingParams

if TYPE_CHECKING:
    import torch


class Sequence:
    """One decoder hypothesis.

    ``token_ids`` holds the decoder prompt followed by generated tokens.
    ``num_cached_tokens`` counts how many of them already have K/V in the paged
    cache, so the tokens to feed next step are ``token_ids[num_cached_tokens:]``.
    Beam search keeps several sequences per request, each with its own
    ``block_table`` (blocks are shared via reference counting).
    """

    def __init__(
        self,
        request_id: int,
        decoder_prompt_ids: list[int],
        encoder_token_ids: list[int],
        sampling: SamplingParams,
        pixel_values: "torch.Tensor | None" = None,
    ):
        self.request_id = request_id
        self.decoder_prompt_ids = list(decoder_prompt_ids)
        self.encoder_token_ids = list(encoder_token_ids)
        self.sampling = sampling
        # Multimodal input (Florence-2): ``[3, H, W]`` for the request's image.
        self.pixel_values = pixel_values

        self.token_ids = list(decoder_prompt_ids)
        self.block_table: list[int] = []
        self.num_cached_tokens = 0
        self.cum_logprob = 0.0

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)

    @property
    def num_generated(self) -> int:
        return len(self.token_ids) - len(self.decoder_prompt_ids)

    @property
    def num_completion_tokens(self) -> int:
        return self.num_generated

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[len(self.decoder_prompt_ids) :]
