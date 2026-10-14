import torch

from ..sampling_params import SamplingParams


class Sequence:
    def __init__(
        self,
        request_id: int,
        decoder_prompt_ids: list[int],
        encoder_token_ids: list[int],
        sampling: SamplingParams,
        pixel_values: torch.Tensor | None = None,
    ):
        self.request_id = request_id
        self.decoder_prompt_ids = list(decoder_prompt_ids)
        self.encoder_token_ids = list(encoder_token_ids)
        self.sampling = sampling
        self.pixel_values = pixel_values

        # token_ids[num_cached_tokens:] are the uncached tokens to feed next step.
        self.token_ids = list(decoder_prompt_ids)
        self.block_table: list[int] = []
        self.num_cached_tokens = 0
        self.cum_logprob = 0.0

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)

    @property
    def num_generated(self) -> int:
        return len(self.token_ids) - len(self.decoder_prompt_ids)

    num_completion_tokens = num_generated

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[len(self.decoder_prompt_ids) :]
