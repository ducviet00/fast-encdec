"""fast-encdec: a tiny CPU BART inference library with paged KV cache,
continuous batching and beam search."""

__version__ = "0.1.0"

import torch
from transformers import AutoConfig, AutoTokenizer

from .block_manager import BlockManager
from .engine import LLMEngine
from .loader import load_hf_weights
from .models.bart import BartForConditionalGeneration
from .sequence import SamplingParams


class LLM:
    """Minimal offline BART engine.

    Args:
        model_path: HF repo id or local path of a BART checkpoint.
        num_blocks: number of paged KV blocks (total KV capacity).
        block_size: tokens per KV block.
        max_num_seqs: maximum sequences (beams) in flight at once.
        dtype: ``torch.float32`` or ``torch.bfloat16``.
    """

    def __init__(self, model_path: str, *, num_blocks: int = 512,
                 block_size: int = 16, max_num_seqs: int = 32,
                 dtype: torch.dtype = torch.float32):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        self.model = BartForConditionalGeneration(config)
        load_hf_weights(self.model, model_path)
        self.model.to(dtype)

        self.engine = LLMEngine(
            self.model, BlockManager(num_blocks, block_size),
            max_num_seqs=max_num_seqs, dtype=dtype)
        self.decoder_start_token_id = config.decoder_start_token_id
        self.eos_token_id = config.eos_token_id

    def _decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @torch.no_grad()
    def generate(self, encoder_prompts, sampling_params: SamplingParams | None = None,
                 decoder_prompts: list[str] | None = None) -> list[str]:
        """Summarize/translate ``encoder_prompts`` (str or list of str)."""
        if isinstance(encoder_prompts, str):
            encoder_prompts = [encoder_prompts]
        if sampling_params is None:
            sampling_params = SamplingParams()
        if sampling_params.eos_token_id is None:
            sampling_params.eos_token_id = self.eos_token_id

        encoder_ids = self.tokenizer(
            encoder_prompts, add_special_tokens=True, padding=False)["input_ids"]

        request_ids = []
        for i, encoder_id in enumerate(encoder_ids):
            if decoder_prompts is not None:
                decoder_ids = self.tokenizer(
                    decoder_prompts[i], add_special_tokens=False)["input_ids"]
            else:
                decoder_ids = [self.decoder_start_token_id]
            request_ids.append(
                self.engine.add_request(encoder_id, decoder_ids, sampling_params))

        self.engine.run()
        return [self._decode(self.engine.results[rid]) for rid in request_ids]
