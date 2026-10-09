"""fast-encdec: a tiny CPU BART inference library with paged KV cache,
continuous batching and beam search.  Florence-2 (DaViT vision encoder + BART
language model) checkpoints are served by the same :class:`LLM` facade.

``LLM.generate`` mirrors the vLLM offline entrypoint: it takes prompts plus
optional ``SamplingParams`` and returns ``RequestOutput`` objects (the text is
``output.outputs[0].text``).
"""

__version__ = "0.1.0"

from .llm import LLM
from .outputs import CompletionOutput, RequestOutput
from .sampling_params import SamplingParams

__all__ = ["LLM", "CompletionOutput", "RequestOutput", "SamplingParams"]
