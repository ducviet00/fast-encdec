# fast-encdec

A small, readable inference engine for **BART** and **Florence-2**, in the spirit of
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

## Key Features

* 🚀 **Fast offline inference** - up to 3.5x faster than `transformers.generate` on CPU and 3.4x on a GPU (beam search wins most; greedy on CPU is 0.9-2.0x)
* 📖 **Readable codebase** - paged KV cache, continuous batching and beam search in plain PyTorch
* 🧩 **Two models, one engine** - BART for summarisation, Florence-2 for vision-language tasks
* ⚡ **CPU and CUDA** - vendored vLLM CPU attention kernel; FlashAttention-2 and a CUDA graph on GPU

## Installation

```bash
git clone https://github.com/ducviet00/fast-encdec && cd fast-encdec
uv sync --extra cpu      # or --extra cu130 for CUDA 13
```

## Quick Start

See `examples/` for runnable scripts. The API mirrors vLLM's; `LLM` also takes
`device`, `dtype`, `num_blocks`, `block_size`, `attn_backend`, `use_cudagraph`
and `compile_mm_encoder`:

```python
from fastencdec import LLM, SamplingParams

llm = LLM("facebook/bart-large-cnn")
outputs = llm.generate(
    "A car is parked in front of a building ...",
    SamplingParams(max_tokens=64, num_beams=4),
)
print(outputs[0].outputs[0].text)
```

Florence-2 takes an image plus a task prompt, and the answer is parsed with the
processor:

```python
import torch
from PIL import Image

from fastencdec import LLM, SamplingParams

llm = LLM("florence-community/Florence-2-base", dtype=torch.bfloat16)
image = Image.open("car.jpg")
outputs = llm.generate(
    {"prompt": "<OD>", "multi_modal_data": {"image": image}},
    SamplingParams(max_tokens=256, num_beams=3),
)
parsed = llm.processor.post_process_generation(
    outputs[0].outputs[0].text, task="<OD>", image_size=image.size
)
```

## How It Works

* **Paged KV cache.** Keys and values live in fixed blocks; beams share blocks and
  only copy one when they write into a shared block.
* **Continuous batching.** Requests join and leave the running batch freely.
* **No padding.** One token per sequence per step, and attention reads exactly the
  cached length of each sequence, so mixed lengths waste nothing.
* **Beam search** runs on the same cache, ranked by length-penalised score.
* **Device-agnostic attention.** The model calls three attention ops; each picks
  its kernel from the device, dtype and head dim (vLLM CPU kernel, SDPA or
  FlashAttention-2).

## Benchmark

`benchmarks/benchmark.py` (BART) and `benchmarks/benchmark_florence2.py`
(Florence-2) compare this engine with `transformers.generate` on the same
machine, dtype, inputs and output tokens. `benchmarks/verify_parity.py` checks
the tokens and `benchmarks/throughput.py` measures decode-only throughput.

**Test configuration**

- Hardware: Xeon 8581C (Emerald Rapids), Xeon 6975P-C (Granite Rapids), Graviton 5, NVIDIA L4
- Model: `facebook/bart-large-cnn`, bf16 for both engines
- Three points, as encoder -> generated tokens: 128->512, 512->128, 256->256
- Configs: batch 1/8, beams 1/4

**Speedup vs `transformers`**

| point | config | Xeon 8581C | Xeon 6975P-C | Graviton 5 | L4 |
|---|---|---:|---:|---:|---:|
| 128->512 | batch=1, beams=1 | 1.12x | 1.10x | 1.05x | 3.19x |
| 128->512 | batch=8, beams=1 | 1.08x | 1.13x | 1.18x | 2.73x |
| 128->512 | batch=8, beams=4 | 2.23x | 2.10x | 1.78x | 1.09x |
| 512->128 | batch=1, beams=1 | 1.09x | 1.09x | 1.08x | 3.20x |
| 512->128 | batch=8, beams=1 | 0.95x | 0.93x | 1.12x | 2.37x |
| 512->128 | batch=8, beams=4 | 3.47x | 2.42x | 1.49x | 1.45x |
| 256->256 | batch=1, beams=1 | 1.10x | 1.10x | 1.10x | 3.27x |
| 256->256 | batch=8, beams=1 | 1.08x | 1.07x | 1.19x | 2.70x |
| 256->256 | batch=8, beams=4 | 1.99x | 1.86x | 1.64x | 1.15x |

**Throughput (tokens/s), 256->256**

| config | engine | Xeon 8581C | Xeon 6975P-C | Graviton 5 | L4 |
|---|---|---:|---:|---:|---:|
| batch=8, beams=1 | fast-encdec | 354 | 331 | 497 | 1841 |
| batch=8, beams=1 | transformers | 328 | 308 | 418 | 681 |
| batch=8, beams=4 | fast-encdec | 211 | 200 | 248 | 525 |
| batch=8, beams=4 | transformers | 106 | 107 | 151 | 457 |

Beam search is the biggest win: `transformers` reorders the whole beam cache every
step, while blocks are shared here. Greedy with a long encoder (512->128) is the
one case where the two engines are even, because the encoder is shared work.

**Florence-2** (real 640x480 photo, four tasks: caption, detection, referring
segmentation and phrase grounding)

| config | Xeon 8581C | Xeon 6975P-C | Graviton 5 | L4 |
|---|---:|---:|---:|---:|
| batch=1, beams=1 | 1.12x | 1.20x | 1.13x | 2.73x |
| batch=8, beams=1 | 1.66x | 1.95x | 1.24x | 3.41x |
| batch=8, beams=4 | 3.40x | 3.43x | 2.03x | 3.34x |

Numbers are the average of two runs. In fp32 this engine reproduces
`transformers` token for token; in bf16 the output length always matches.

## Tests

```bash
python tests/test_parity.py        # vs transformers, tiny random BART
python tests/test_florence2.py     # vs transformers, tiny random Florence-2
python tests/test_attn_backend.py  # vLLM CPU kernel vs SDPA
python tests/test_gpu_flash.py     # FlashAttention-2 vs SDPA (CUDA only)
```

## Limitations

* No prefix caching, chunked prefill, preemption or quantization.
* Every running sequence is processed each step, so a long prefill delays decoding.
* `num_blocks` is a hard KV budget; the block manager raises when it runs out.
* Beam search stops with `transformers`' `early_stopping=False` rule only.
