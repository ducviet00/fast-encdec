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
machine, dtype, inputs and output tokens; `benchmarks/benchmark_vllm.py` runs the
same BART points against vLLM.  `benchmarks/verify_parity.py` checks the tokens
and `benchmarks/throughput.py` measures decode-only throughput.

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

**vs vLLM (NVIDIA L4)**

`benchmarks/benchmark_vllm.py` runs the same points against vLLM, which serves
BART through two plugins: [`vllm-bart-plugin`](https://github.com/vllm-project/bart-plugin)
for encoder-decoder support and
[`vllm-beam-search-plugin`](https://github.com/neuralmagic/vllm-beam-search-plugin)
for the beam scheduler.  Both engines use their CUDA graph, run bf16 and decode
the same token count (vLLM's beam path appends one trailing EOS per sequence).
Greedy is measured with the beam plugin **uninstalled** -- it swaps in a custom
scheduler that disables vLLM's async scheduling -- so only the `beams=4` rows need
it loaded.

| point | config | ours ms | ours tok/s | vLLM ms | vLLM tok/s | speedup |
|---|---|---:|---:|---:|---:|---:|
| 128->512 | batch=1, beams=1 | 1457 | 351 | 1397 | 366 | 0.96x |
| 128->512 | batch=8, beams=1 | 1834 | 2233 | 1685 | 2431 | 0.92x |
| 128->512 | batch=8, beams=4 | 3197 | 1281 | 3458 | 1187 | **1.08x** |
| 128->512 | batch=8, beams=1, fp32 | 12011 | 341 | 3355 | 1221 | 0.28x |
| 512->128 | batch=1, beams=1 | 372 | 344 | 361 | 355 | 0.97x |
| 512->128 | batch=8, beams=1 | 560 | 1828 | 517 | 1980 | 0.92x |
| 512->128 | batch=8, beams=4 | 900 | 1138 | 1058 | 975 | **1.18x** |
| 512->128 | batch=8, beams=1, fp32 | 3190 | 321 | 985 | 1040 | 0.31x |
| 256->256 | batch=1, beams=1 | 727 | 352 | 699 | 366 | 0.96x |
| 256->256 | batch=8, beams=1 | 950 | 2155 | 870 | 2353 | 0.92x |
| 256->256 | batch=8, beams=4 | 1737 | 1179 | 1761 | 1167 | **1.01x** |
| 256->256 | batch=8, beams=1, fp32 | 6026 | 340 | 1723 | 1189 | 0.29x |

At a single new token per step in bf16, vLLM's decode loop is 3-9% leaner than
this one, which is what the greedy rows show.  Beam search pulls the other way:
this engine shares KV blocks between beams where the plugin copies a whole beam
group every step, so `beams=4` is a modest win (`1.01-1.18x`) despite each request
running four sequences.  fp32 is the clear miss: this engine has no
FlashAttention-2 path in fp32, so it loses both the fused attention *and* the
decode CUDA graph that depends on it and goes launch-bound, while vLLM keeps fp32
on its `TRITON_ATTN` backend.

**Greedy decode scaling (vs vLLM, no beam plugin)**

Greedy scales smoothly with batch on both engines and vLLM stays 3-15% ahead; the
widest gap is the encoder-heavy `512->128` point at `batch=32` (0.85x).  At
`batch=1` both are latency-bound (~350 tok/s) and nearly tie, and the gap opens as
each step starts re-reading more KV.

| point | batch | ours ms | ours tok/s | vLLM ms | vLLM tok/s | speedup |
|---|---|---:|---:|---:|---:|---:|
| 128->512 | 1 | 1457 | 351 | 1397 | 366 | 0.96x |
| 128->512 | 2 | 1567 | 654 | 1475 | 694 | 0.94x |
| 128->512 | 4 | 1676 | 1222 | 1510 | 1356 | 0.90x |
| 128->512 | 8 | 1834 | 2233 | 1685 | 2431 | 0.92x |
| 128->512 | 16 | 2273 | 3604 | 2094 | 3913 | 0.92x |
| 128->512 | 32 | 3170 | 5168 | 2977 | 5503 | 0.94x |
| 512->128 | 1 | 372 | 344 | 361 | 355 | 0.97x |
| 512->128 | 2 | 420 | 609 | 397 | 644 | 0.95x |
| 512->128 | 4 | 460 | 1112 | 441 | 1162 | 0.96x |
| 512->128 | 8 | 560 | 1828 | 517 | 1980 | 0.92x |
| 512->128 | 16 | 779 | 2630 | 693 | 2953 | 0.89x |
| 512->128 | 32 | 1230 | 3330 | 1051 | 3896 | 0.85x |
| 256->256 | 1 | 727 | 352 | 699 | 366 | 0.96x |
| 256->256 | 2 | 797 | 642 | 747 | 686 | 0.94x |
| 256->256 | 4 | 855 | 1197 | 775 | 1321 | 0.91x |
| 256->256 | 8 | 950 | 2155 | 870 | 2353 | 0.92x |
| 256->256 | 16 | 1202 | 3409 | 1100 | 3725 | 0.92x |
| 256->256 | 32 | 1729 | 4737 | 1563 | 5242 | 0.90x |

Reproduce with `--batches 1,2,4,8,16,32`.  Removing the beam plugin moved vLLM's
greedy ~1% at `batch=8` (`256->256` 870 ms / 2353 tok/s vs 876 ms / 2337 tok/s
with it) and ~0.4% in fp32; `batch=1` is inside the run-to-run noise.

**vs vLLM (Graviton 5, CPU)**

`benchmarks/benchmark_vllm.py --device cpu` runs the same comparison on CPU -
greedy only.  vLLM's beam plugin needs its V2 model runner *with async
scheduling*, and vLLM gates V2 on Triton, which has no CPU backend
(`BeamSearchScheduler requires MRV2 with async scheduling enabled`), so the CPU
rows need the pre-MRV2 `vllm-bart-plugin` v0.5.0 on `vllm==0.24.0+cpu`.  Both
engines decode the same token count on every row (the script asserts it), bf16
unless noted, one process per engine, warmup 1 + best-of-3.

| point | config | ours ms | ours tok/s | vLLM ms | vLLM tok/s | speedup |
|---|---|---:|---:|---:|---:|---:|
| 128->512 | batch=1, beams=1 | 3361 | 152.3 | 3473 | 147.4 | 1.03x |
| 128->512 | batch=8, beams=1 | 4761 | 860.4 | 5862 | 698.7 | 1.23x |
| 128->512 | batch=8, beams=1, fp32 | 8629 | 474.7 | 9734 | 420.8 | 1.13x |
| 512->128 | batch=1, beams=1 | 972 | 131.7 | 1829 | 70.0 | **1.88x** |
| 512->128 | batch=8, beams=1 | 2149 | 476.6 | 9016 | 113.6 | **4.20x** |
| 512->128 | batch=8, beams=1, fp32 | 5049 | 202.8 | 5603 | 182.8 | 1.11x |
| 256->256 | batch=1, beams=1 | 1730 | 148.0 | 1972 | 129.8 | 1.14x |
| 256->256 | batch=8, beams=1 | 2703 | 757.7 | 4915 | 416.7 | **1.82x** |
| 256->256 | batch=8, beams=1, fp32 | 5333 | 384.1 | 6024 | 340.0 | 1.13x |

| point | batch | ours ms | ours tok/s | vLLM ms | vLLM tok/s | speedup |
|---|---|---:|---:|---:|---:|---:|
| 128->512 | 1 | 3366 | 152.1 | 3612 | 141.7 | 1.07x |
| 128->512 | 2 | 3672 | 278.9 | 4501 | 227.5 | 1.23x |
| 128->512 | 4 | 4058 | 504.7 | 4942 | 414.4 | 1.22x |
| 128->512 | 8 | 4766 | 859.5 | 5975 | 685.5 | 1.25x |
| 128->512 | 16 | 7648 | 1071.2 | 9676 | 846.6 | 1.27x |
| 128->512 | 32 | 12757 | 1284.3 | 15775 | 1038.6 | 1.24x |
| 512->128 | 1 | 992 | 129.0 | 1877 | 68.2 | 1.89x |
| 512->128 | 2 | 1165 | 219.8 | 3062 | 83.6 | 2.63x |
| 512->128 | 4 | 1489 | 343.9 | 5029 | 101.8 | 3.38x |
| 512->128 | 8 | 2164 | 473.3 | 9070 | 112.9 | 4.19x |
| 512->128 | 16 | 3827 | 535.2 | 17522 | 116.9 | 4.58x |
| 512->128 | 32 | 7172 | 571.1 | 34178 | 119.8 | **4.77x** |
| 256->256 | 1 | 1713 | 149.5 | 2098 | 122.0 | 1.22x |
| 256->256 | 2 | 1910 | 268.0 | 2768 | 185.0 | 1.45x |
| 256->256 | 4 | 2188 | 467.9 | 3492 | 293.2 | 1.60x |
| 256->256 | 8 | 2695 | 760.0 | 4994 | 410.1 | 1.85x |
| 256->256 | 16 | 4369 | 937.5 | 8758 | 467.7 | 2.00x |
| 256->256 | 32 | 7627 | 1074.1 | 15709 | 521.5 | 2.06x |

This engine is ahead on every CPU row, because the two shapes fail differently for
vLLM.  Where the encoder dominates (`512->128`) its CPU path barely batches: it
goes 68 -> 120 tok/s from `batch=1` to `batch=32` while this engine goes
129 -> 571, so the margin grows with batch to 4.8x.  Where the decode dominates
(`128->512`) the two are close (1.03-1.27x) - a single-token CPU decode is the
same bandwidth-bound GEMM work in both engines.  fp32 is flat ~1.1x, since
neither engine has a fused attention path to lose there.

Beam search has no vLLM reference on CPU, so its rows are this engine's own: at
`batch=8, beams=4, bf16` it reaches 386 tok/s at `128->512`, 273 at `512->128`
and 362 at `256->256`.  Measure those against `transformers` with
`benchmarks/benchmark.py` instead.

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
