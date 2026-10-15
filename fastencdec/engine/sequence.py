import torch

from ..sampling_params import SamplingParams


class Sequence:
    """State and paged KV-cache bookkeeping for a single decoder hypothesis."""

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
        """Append a newly generated token ID to the sequence."""
        self.token_ids.append(token_id)

    @property
    def num_generated(self) -> int:
        """Return the number of tokens generated after the decoder prompt."""
        return len(self.token_ids) - len(self.decoder_prompt_ids)

    num_completion_tokens = num_generated

    @property
    def completion_token_ids(self) -> list[int]:
        """Return the generated token IDs excluding the decoder prompt."""
        return self.token_ids[len(self.decoder_prompt_ids) :]
