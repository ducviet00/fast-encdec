# fast-encdec

A small, readable inference engine for **BART** and **Florence-2**, in the spirit of
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

## Key Features

* 🚀 **Fast offline inference** - up to 5.1x faster than `transformers.generate` on CPU and 4.2x on a GPU (beam search drives the CPU wins; greedy is 1.2-3.0x on CPU and 2.4-3.6x on GPU)
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
| 128->512 | batch=1, beams=1 | 1.50x | 1.47x | 2.99x | 3.29x |
| 128->512 | batch=8, beams=1 | 1.53x | 1.42x | 1.75x | 2.75x |
| 128->512 | batch=8, beams=4 | 2.78x | 2.61x | 2.25x | 2.04x |
| 512->128 | batch=1, beams=1 | 1.49x | 1.39x | 2.73x | 3.23x |
| 512->128 | batch=8, beams=1 | 1.23x | 1.19x | 1.38x | 2.38x |
| 512->128 | batch=8, beams=4 | 5.08x | 4.22x | 1.88x | 2.80x |
| 256->256 | batch=1, beams=1 | 1.55x | 1.43x | 2.88x | 3.36x |
| 256->256 | batch=8, beams=1 | 1.40x | 1.31x | 1.60x | 2.76x |
| 256->256 | batch=8, beams=4 | 2.57x | 2.37x | 2.06x | 2.19x |

**Throughput (tokens/s), 256->256**

| config | engine | Xeon 8581C | Xeon 6975P-C | Graviton 5 | L4 |
|---|---|---:|---:|---:|---:|
| batch=8, beams=1 | fast-encdec | 420 | 408 | 711 | 1845 |
| batch=8, beams=1 | transformers | 299 | 312 | 444 | 668 |
| batch=8, beams=4 | fast-encdec | 234 | 254 | 338 | 975 |
| batch=8, beams=4 | transformers | 91 | 107 | 164 | 446 |

Beam search is still the biggest win: `transformers` reorders the whole beam cache
every step, while blocks are shared here.  Greedy used to be a near tie, especially
at `512->128` where the encoder is shared work; prepacking the CPU linear weights
(below) moved every greedy row.

Each cell is the average of two independent runs, and every run writes its raw
results with `--json`, so the numbers can be re-derived rather than trusted.  Every
CPU `nn.Linear` weight is prepacked into oneDNN's blocked layout at load
(`pack_weights` in the `AGENTS.md` gotchas), which removes the per-call weight
reorder `F.linear` pays.  On aarch64 that also routes the GEMM through oneDNN-ACL
instead of torch's fallback kernel, which is why Graviton 5's single-sequence rows
gain ~2.7-2.9x while the x86 rows gain ~1.2-1.5x.  The two x86 boxes additionally
run the engine's 32-token page size, which engages the full-width AVX-512 attention
kernel (see the page-size gotcha in `AGENTS.md`), while Graviton 5 and the L4 keep
theirs (16 and 256).  The L4 column is unchanged: the prepack is CPU-only.

**Florence-2** (real 640x480 photo, four tasks: caption, detection, referring
segmentation and phrase grounding)

| config | Xeon 8581C | Xeon 6975P-C | Graviton 5 | L4 |
|---|---:|---:|---:|---:|
| batch=1, beams=1 | 1.34x | 1.39x | 1.30x | 2.54x |
| batch=8, beams=1 | 1.83x | 2.08x | 1.29x | 3.60x |
| batch=8, beams=4 | 4.12x | 3.48x | 2.16x | 4.23x |

The Florence-2 numbers move less than BART's: the vision tower and encoder are
shared work, so the page size and the linear prepack only touch the decoder.  In
fp32 this engine reproduces `transformers` token for token; in bf16 the output
length always matches.

## Tests

```bash
python tests/test_parity.py        # vs transformers, tiny random BART
python tests/test_florence2.py     # vs transformers, tiny random Florence-2
python tests/test_attn_backend.py  # vLLM CPU kernel vs SDPA
python tests/test_packed_linear.py # packed CPU linears (fallback + token parity)
python tests/test_gpu_flash.py     # FlashAttention-2 vs SDPA (CUDA only)
```

## Limitations

* No prefix caching, chunked prefill, preemption or quantization.
* Every running sequence is processed each step, so a long prefill delays decoding.
* `num_blocks` is a hard KV budget; the block manager raises when it runs out.
* Beam search stops with `transformers`' `early_stopping=False` rule only.
