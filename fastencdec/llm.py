"""The unified offline :class:`LLM` facade for BART and Florence-2.

``LLM.generate`` mirrors the vLLM offline entrypoint: it takes prompts plus
optional ``SamplingParams`` and returns ``RequestOutput`` objects (the text is
``output.outputs[0].text``).
"""

import torch
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from .config import Config
from .engine.llm_engine import LLMEngine
from .models.bart import BartForConditionalGeneration
from .models.florence2 import Florence2ForConditionalGeneration
from .outputs import CompletionOutput, RequestOutput
from .sampling_params import SamplingParams
from .utils.loader import load_hf_weights


def _load_generation_config(model_path: str) -> GenerationConfig:
    """Load the checkpoint's ``GenerationConfig``, or a bare default."""
    try:
        return GenerationConfig.from_pretrained(model_path)
    except OSError:
        return GenerationConfig()


class LLM:
    """Minimal offline engine for BART and Florence-2 checkpoints.

    The checkpoint's ``model_type`` picks the model: BART takes text prompts;
    Florence-2 takes a task prompt plus an image (its visual tokens are spliced
    into the encoder input).  Everything below this facade — paged KV cache,
    continuous batching, beam search — is shared.

    Args:
        model_path: HF repo id or local path of a BART or Florence-2 checkpoint.
        num_blocks: number of paged KV blocks (total KV capacity).
        block_size: tokens per KV block.
        max_num_seqs: maximum sequences (beams) in flight at once.
        dtype: ``torch.float32`` or ``torch.bfloat16``.
        attn_backend: ``"auto"`` (use the vendored vLLM CPU attention kernel
            when it builds, else the PyTorch SDPA path), ``"vllm"`` (require
            it) or ``"sdpa"`` (force the PyTorch path).
        compile_mm_encoder: ``torch.compile`` the Florence-2 vision tower +
            projector (~1.3-1.5x there).  Ignored for BART.  Off by default:
            it changes the BF16 outputs slightly and compiles once per batch
            size on first use.
    """

    def __init__(
        self,
        model_path: str,
        *,
        num_blocks: int = 512,
        block_size: int = 16,
        max_num_seqs: int = 32,
        dtype: torch.dtype = torch.float32,
        attn_backend: str = "auto",
        compile_mm_encoder: bool = False,
    ):
        hf_config = AutoConfig.from_pretrained(model_path)
        self.multimodal = hf_config.model_type == "florence2"

        if self.multimodal:
            from transformers import AutoProcessor
            from transformers import (
                Florence2ForConditionalGeneration as HFFlorence2,
            )

            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = Florence2ForConditionalGeneration(hf_config)
            load_hf_weights(self.model, model_path, hf_model_cls=HFFlorence2)
            self.model.to(dtype)
            if compile_mm_encoder:
                self.model.compile_mm_encoder()
            self.tokenizer = self.processor.tokenizer
            text_config = hf_config.text_config
            self.decoder_start_token_id = text_config.decoder_start_token_id
            self.eos_token_id = text_config.eos_token_id
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = BartForConditionalGeneration(hf_config)
            load_hf_weights(self.model, model_path)
            self.model.to(dtype)
            self.decoder_start_token_id = hf_config.decoder_start_token_id
            self.eos_token_id = hf_config.eos_token_id

        self.generation_config = _load_generation_config(model_path)
        self.default_sampling_params = SamplingParams.from_generation_config(
            self.generation_config, self.eos_token_id
        )
        config = Config(
            num_blocks=num_blocks,
            block_size=block_size,
            max_num_seqs=max_num_seqs,
            dtype=dtype,
            attn_backend=attn_backend,
        )
        self.engine = LLMEngine(self.model, config, eos_token_id=self.eos_token_id)

    def _decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def _parse_prompt(self, item):
        """Split one prompt into ``(text, prompt_token_ids, image)``.

        Accepts a plain string, or a vLLM-style dict with ``prompt`` /
        ``prompt_token_ids`` and an optional ``multi_modal_data`` (e.g.
        ``{"multi_modal_data": {"image": image}}``).
        """
        if isinstance(item, str):
            return item, None, None
        if not isinstance(item, dict):
            raise TypeError(f"prompt must be a str or dict, got {type(item).__name__}")
        text = item.get("prompt")
        token_ids = item.get("prompt_token_ids")
        if text is None and token_ids is None:
            raise ValueError("prompt dict needs 'prompt' or 'prompt_token_ids'")
        image = (item.get("multi_modal_data") or {}).get("image")
        return text, token_ids, image

    @torch.no_grad()
    def generate(
        self,
        prompts,
        sampling_params: SamplingParams | None = None,
        *,
        use_tqdm: bool = True,
        mm_processor_kwargs: dict | None = None,
    ) -> list[RequestOutput]:
        """Generate completions for ``prompts`` (vLLM-style entrypoint).

        ``prompts`` is a string, a dict, or a list of either.  A dict may carry
        ``prompt`` / ``prompt_token_ids`` and, for Florence-2, an image via
        ``{"multi_modal_data": {"image": image}}``.  Decoding starts from the
        model's ``decoder_start_token_id``; ``sampling_params`` defaults to the
        checkpoint's ``GenerationConfig``.

        Returns one :class:`RequestOutput` per prompt; the text is
        ``output.outputs[0].text``.

        ``use_tqdm`` is accepted for signature parity with vLLM but unused (the
        engine decodes the whole batch in one pass).
        """
        if isinstance(prompts, (str, dict)):
            prompts = [prompts]
        if sampling_params is None:
            sampling_params = self.default_sampling_params
        eos_ids = sampling_params.eos_ids()
        if not eos_ids and self.eos_token_id is not None:
            eos_ids = {self.eos_token_id}

        records = []
        engine_requests = []
        for item in prompts:
            encoder_prompt, encoder_ids, image = self._parse_prompt(item)
            decoder_ids = [self.decoder_start_token_id]

            if self.multimodal:
                if image is None:
                    raise ValueError(
                        "Florence-2 needs an image, e.g. "
                        '{"prompt": "<CAPTION>", "multi_modal_data": {"image": img}}'
                    )
                inputs = self.processor(
                    text=encoder_prompt,
                    images=image,
                    return_tensors="pt",
                    **(mm_processor_kwargs or {}),
                )
                encoder_ids = inputs["input_ids"][0].tolist()
                pixel_values = inputs["pixel_values"][0]
            else:
                if image is not None:
                    raise ValueError(
                        "images are only supported for Florence-2 checkpoints"
                    )
                if encoder_ids is None:
                    encoder_ids = self.tokenizer(
                        encoder_prompt, add_special_tokens=True, padding=False
                    )["input_ids"]
                pixel_values = None

            records.append((encoder_prompt, encoder_ids, decoder_ids))
            engine_requests.append((encoder_ids, decoder_ids, pixel_values))

        request_ids = self.engine.generate(engine_requests, sampling_params)

        results = []
        for request_id, (encoder_prompt, encoder_ids, decoder_ids) in zip(
            request_ids, records
        ):
            token_ids = self.engine.results[request_id]
            finished = bool(token_ids) and token_ids[-1] in eos_ids
            results.append(
                RequestOutput(
                    request_id=request_id,
                    prompt=self._decode(decoder_ids),
                    prompt_token_ids=decoder_ids,
                    outputs=[
                        CompletionOutput(
                            index=0,
                            text=self._decode(token_ids),
                            token_ids=token_ids,
                            finish_reason="stop" if finished else "length",
                            stop_reason=token_ids[-1] if finished else None,
                        )
                    ],
                    encoder_prompt=encoder_prompt,
                    encoder_prompt_token_ids=encoder_ids,
                )
            )
        return results
