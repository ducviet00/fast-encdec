"""fast-encdec: a tiny CPU BART inference library with paged KV cache,
continuous batching and beam search.  Also supports Florence-2 (DaViT vision
encoder + BART language model) through :class:`Florence2LLM`."""

__version__ = "0.1.0"

import torch
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from .block_manager import BlockManager
from .engine import LLMEngine
from .loader import load_hf_weights
from .models.bart import BartForConditionalGeneration
from .models.florence2 import Florence2ForConditionalGeneration
from .sequence import SamplingParams


def _load_generation_config(model_path: str) -> GenerationConfig:
    """Load the checkpoint's ``GenerationConfig``, or a bare default."""
    try:
        return GenerationConfig.from_pretrained(model_path)
    except OSError:
        return GenerationConfig()


class LLM:
    """Minimal offline BART engine.

    Args:
        model_path: HF repo id or local path of a BART checkpoint.
        num_blocks: number of paged KV blocks (total KV capacity).
        block_size: tokens per KV block.
        max_num_seqs: maximum sequences (beams) in flight at once.
        dtype: ``torch.float32`` or ``torch.bfloat16``.
    """

    def __init__(
        self,
        model_path: str,
        *,
        num_blocks: int = 512,
        block_size: int = 16,
        max_num_seqs: int = 32,
        dtype: torch.dtype = torch.float32,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        if config.model_type == "florence2":
            raise ValueError(
                "use Florence2LLM for Florence-2 checkpoints (LLM is BART-only)"
            )
        self.model = BartForConditionalGeneration(config)
        load_hf_weights(self.model, model_path)
        self.model.to(dtype)

        self.generation_config = _load_generation_config(model_path)
        self.decoder_start_token_id = config.decoder_start_token_id
        self.eos_token_id = config.eos_token_id
        self.default_sampling_params = SamplingParams.from_generation_config(
            self.generation_config, self.eos_token_id
        )
        self.engine = LLMEngine(
            self.model,
            BlockManager(num_blocks, block_size),
            max_num_seqs=max_num_seqs,
            dtype=dtype,
            eos_token_id=self.eos_token_id,
        )

    def _decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @torch.no_grad()
    def generate(
        self,
        encoder_prompts,
        sampling_params: SamplingParams | None = None,
        decoder_prompts: list[str] | None = None,
    ) -> list[str]:
        """Summarize/translate ``encoder_prompts`` (str or list of str)."""
        if isinstance(encoder_prompts, str):
            encoder_prompts = [encoder_prompts]
        if sampling_params is None:
            sampling_params = self.default_sampling_params

        encoder_ids = self.tokenizer(
            encoder_prompts, add_special_tokens=True, padding=False
        )["input_ids"]

        request_ids = []
        for i, encoder_id in enumerate(encoder_ids):
            if decoder_prompts is not None:
                decoder_ids = self.tokenizer(
                    decoder_prompts[i], add_special_tokens=False
                )["input_ids"]
            else:
                decoder_ids = [self.decoder_start_token_id]
            request_ids.append(
                self.engine.add_request(encoder_id, decoder_ids, sampling_params)
            )

        self.engine.run()
        return [self._decode(self.engine.results[rid]) for rid in request_ids]


class Florence2LLM:
    """Minimal offline Florence-2 engine on CPU.

    Florence-2 pairs a DaViT vision encoder with a BART language model.  The
    image is encoded once per request and its visual tokens are spliced into
    the encoder input at the ``<image>`` placeholder positions; decoding then
    runs on the same paged-cache engine as BART.

    Args:
        model_path: HF repo id or local path of a Florence-2 checkpoint
            (e.g. ``florence-community/Florence-2-base``).
        num_blocks: number of paged KV blocks (total KV capacity).
        block_size: tokens per KV block.
        max_num_seqs: maximum sequences (beams) in flight at once.
        dtype: ``torch.float32`` or ``torch.bfloat16``.
        compile_mm_encoder: ``torch.compile`` the vision tower + projector
            (~1.3-1.5x on that part).  Off by default: it changes the BF16
            outputs slightly and compiles once per batch size on first use.
    """

    def __init__(
        self,
        model_path: str,
        *,
        num_blocks: int = 512,
        block_size: int = 16,
        max_num_seqs: int = 32,
        dtype: torch.dtype = torch.float32,
        compile_mm_encoder: bool = False,
    ):
        from transformers import AutoProcessor
        from transformers import Florence2ForConditionalGeneration as HFFlorence2

        self.processor = AutoProcessor.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        self.model = Florence2ForConditionalGeneration(config)
        load_hf_weights(self.model, model_path, hf_model_cls=HFFlorence2)
        self.model.to(dtype)
        if compile_mm_encoder:
            self.model.compile_mm_encoder()

        self.generation_config = _load_generation_config(model_path)
        self.decoder_start_token_id = config.text_config.decoder_start_token_id
        self.eos_token_id = config.text_config.eos_token_id
        self.default_sampling_params = SamplingParams.from_generation_config(
            self.generation_config, self.eos_token_id
        )
        self.engine = LLMEngine(
            self.model,
            BlockManager(num_blocks, block_size),
            max_num_seqs=max_num_seqs,
            dtype=dtype,
            eos_token_id=self.eos_token_id,
        )

    @torch.no_grad()
    def generate(
        self,
        prompts,
        images,
        sampling_params: SamplingParams | None = None,
    ) -> list[str]:
        """Run ``prompts`` (task strings like ``"<CAPTION>"``) on ``images``.

        ``prompts``/``images`` may be single items or equal-length lists; a
        single image is broadcast across multiple prompts.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        if not isinstance(images, list):
            images = [images]
        if len(images) == 1 and len(prompts) > 1:
            images = images * len(prompts)
        if len(images) != len(prompts):
            raise ValueError("prompts and images must have the same length")
        if sampling_params is None:
            sampling_params = self.default_sampling_params

        request_ids = []
        for prompt, image in zip(prompts, images):
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            request_ids.append(
                self.engine.add_request(
                    inputs["input_ids"][0].tolist(),
                    [self.decoder_start_token_id],
                    sampling_params,
                    pixel_values=inputs["pixel_values"][0],
                )
            )

        self.engine.run()
        tokenizer = self.processor.tokenizer
        return [
            tokenizer.decode(self.engine.results[rid], skip_special_tokens=True)
            for rid in request_ids
        ]
