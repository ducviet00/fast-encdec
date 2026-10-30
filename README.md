# fast-encdec

A small, readable inference engine for **BART** and **Florence-2**, in the spirit of
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

## Key Features

* 🚀 **Fast offline inference** - up to 5.0x faster than `transformers.generate` on CPU and 4.2x on a GPU (beam search drives the CPU wins; greedy is 1.1-3.5x on CPU and 2.7-4.0x on GPU)
* 📖 **Readable codebase** - paged KV cache, continuous batching and beam search in plain PyTorch
* 🧩 **Two models, one engine** - BART for summarisation, Florence-2 for vision-language tasks
* ⚡ **CPU and CUDA** - [cpu-attention](https://github.com/ducviet00/cpu-attention) kernels on CPU; [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) and a CUDA graph on GPU

## Installation

```bash
git clone https://github.com/ducviet00/fast-encdec && cd fast-encdec
uv sync --extra cpu      # or --extra cu130 for CUDA 13
```

The `cpu` extra builds [cpu-attention](https://github.com/ducviet00/cpu-attention)
from git (Linux, a C toolchain; the Mojo compiler is fetched for the build only),
and `cu130` builds [flash-attn](https://github.com/Dao-AILab/flash-attention)
(needs `nvcc`, or a prebuilt wheel matching your torch).  Without them the engine
falls back to PyTorch SDPA, which is much slower.

## Quick Start

See `examples/` for runnable scripts. The API mirrors vLLM's; `LLM` also takes
`device`, `dtype`, `num_blocks`, `block_size`, `attn_backend`, `use_cudagraph`,
`compile` (frozen `torch.compile` on CPU, default on) and `compile_mm_encoder`:

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
  its kernel from the device, dtype and head dim (cpu-attention, SDPA or
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
| 128->512 | batch=1, beams=1 | 1.42x | 1.54x | 3.47x | 3.86x |
| 128->512 | batch=8, beams=1 | 1.44x | 1.48x | 1.91x | 3.16x |
| 128->512 | batch=8, beams=4 | 2.73x | 2.68x | 2.38x | 2.10x |
| 512->128 | batch=1, beams=1 | 1.43x | 1.47x | 2.97x | 3.99x |
| 512->128 | batch=8, beams=1 | 1.11x | 1.22x | 1.51x | 2.73x |
| 512->128 | batch=8, beams=4 | 4.98x | 4.48x | 2.05x | 2.88x |
| 256->256 | batch=1, beams=1 | 1.45x | 1.51x | 3.22x | 3.98x |
| 256->256 | batch=8, beams=1 | 1.33x | 1.35x | 1.77x | 3.12x |
| 256->256 | batch=8, beams=4 | 2.42x | 2.33x | 2.25x | 2.22x |

**Throughput (tokens/s), 256->256**

| config | engine | Xeon 8581C | Xeon 6975P-C | Graviton 5 | L4 |
|---|---|---:|---:|---:|---:|
| batch=8, beams=1 | fast-encdec | 361 | 411 | 731 | 2090 |
| batch=8, beams=1 | transformers | 272 | 304 | 413 | 670 |
| batch=8, beams=4 | fast-encdec | 204 | 248 | 353 | 985 |
| batch=8, beams=4 | transformers | 85 | 106 | 157 | 444 |

Beam search is still the biggest win: `transformers` reorders the whole beam cache
every step, while blocks are shared here.  Greedy stays ahead too, especially
at `512->128` where the encoder is shared work.

Two caveats when diffing this table against an older recording.  These are
ratios, and on the 4-core Xeon instances the denominator is the unstable side:
two runs of this same commit read 1.52x and 1.31x on `128->512, batch=1`
(ours 7856/8311 ms, HF 11926/10905 ms) and 2.24x vs 2.60x on `256->256, b=8,
beams=4` (ours 10646/9464 ms), i.e. a cell can move ~0.1-0.35 on its own.  And
the engine's own absolute times are flat across recordings where HF's are not
(Emerald `512->128, beams=4`: ours 5.90 s in a pre-fix session, 5.90/5.85 s
here; HF 30.8 s then, 30.0/28.6 s now), so a few tenths of drift in a Xeon
cell is the baseline, not the engine.

Each cell is the average of two independent runs, and every run writes its raw
results with `--json`, so the numbers can be re-derived rather than trusted.  On
CPU the engine runs the cpu-attention kernels behind opaque
`torch.library.custom_op`s, which makes the decoder + lm head and the encoder
trace as **frozen `torch.compile` graphs**: Inductor's freezing pass packs every
GEMM into oneDNN's blocked layouts — the same op a manual weight prepack used to
feed, produced at compile time instead of at load, with the per-layer pointwise
ops fused into the linear epilogues for free (`AGENTS.md` "CPU GEMM packing"
gotcha; CPU-only, fp32 and bf16, and fp32 there stays token-for-token identical
to `transformers`).  On aarch64 the packing routes
through oneDNN-ACL, which is why Graviton 5's single-sequence rows gain
~3.0-3.5x while the x86 rows gain ~1.2-1.6x.  The two x86 boxes run the
engine's 32-token page size, which engages the full-width AVX-512 attention
kernel (see the page-size gotcha in `AGENTS.md`), while Graviton 5 keeps 16.
The L4 column is unchanged by any of this: it runs FlashAttention-2 plus the
decode CUDA graph, and the CUDA graph is why its rows stay ahead even where the
CPU boxes tie (`beams=4`).  It is re-recorded with each table but does not move
with the CPU work: across recordings its own milliseconds stay flat to ~1%
(e.g. `512->128, batch=1`: 378 -> 381 ms) while the `transformers` baseline on
the 4-vCPU box moves 4-8%, which is what shifts the L4 ratios by up to ~0.2 —
including between two runs of the same commit (3.87x vs 4.11x).

**GCP matrix, torch 2.15 (cpu-attention, current engine)**

c4a-standard-8 (Axion / Neoverse-V2, 8 cores), c4-standard-8 (Xeon 8581C, 4
cores / 8 threads, AMX) and c4d-standard-8 (AMD EPYC 9B45 Turin, 4 cores / 8
threads, no AMX), torch 2.15.0+cpu, transformers 5.22.1, one spot VM each,
`benchmarks/benchmark.py` and `benchmarks/benchmark_florence2.py`, mean of two
runs, bf16 unless noted.  Cells are the speedup vs `transformers`, with ours /
HF tokens/s; † marks a cell whose two runs differ by more than 10%.

| point | config | c4a (Axion) | c4 (Xeon 8581C) | c4d (Turin) |
|---|---|---:|---:|---:|
| 128->512 | batch=1 | 1.26x (118 / 94) | 1.20x (79 / 66) | 1.23x (85 / 69) |
| 128->512 | batch=8 | 1.91x (580 / 303) | 1.66x (460 / 278) | 2.13x (508 / 239) |
| 128->512 | batch=8, beams=4 | 2.12x (282 / 133) | 3.04x (259 / 85) | 4.73x (321 / 68) |
| 128->512 | batch=8, fp32 | 1.50x (347 / 231) | 1.78x (236 / 133) | 2.00x (275 / 138) |
| 512->128 | batch=1 | 0.95x (84 / 88) | 0.84x (54 / 63) | 1.15x (67 / 58) † |
| 512->128 | batch=8 | 1.64x (398 / 243) | 1.41x (347 / 247) | 1.78x (303 / 170) † |
| 512->128 | batch=8, beams=4 | 2.02x (234 / 116) | 4.72x (207 / 44) | 5.82x (236 / 40) |
| 512->128 | batch=8, fp32 | 1.09x (167 / 154) | 1.21x (129 / 107) | 1.64x (141 / 86) |
| 256->256 | batch=1 | 1.15x (112 / 97) | 1.09x (71 / 65) † | 1.33x (86 / 65) |
| 256->256 | batch=8 | 1.87x (547 / 293) | 1.52x (438 / 288) | 1.84x (453 / 247) |
| 256->256 | batch=8, beams=4 | 2.15x (282 / 131) | 2.70x (254 / 94) | 4.22x (310 / 73) |
| 256->256 | batch=8, fp32 | 1.37x (304 / 222) | 1.61x (217 / 135) | 1.88x (245 / 131) |
| Florence-2 | batch=1 | 1.98x (37 / 19) | 3.14x (45 / 14) | 2.75x (33 / 12) |
| Florence-2 | batch=8 | 1.36x (149 / 110) | 2.23x (177 / 79) | 2.19x (155 / 71) |
| Florence-2 | batch=8, beams=4 | 2.17x (131 / 61) | 4.23x (144 / 34) | 4.26x (117 / 27) |
| Florence-2 | batch=8, fp32 | 1.22x (78 / 64) | 1.87x (67 / 36) | 1.47x (45 / 31) |

Batch=8 and beam search lead everywhere (1.4-2.1x greedy, 2.0-5.8x beams;
`transformers`' beam reorder is costliest on the x86 boxes).  On torch 2.15 the
weakest point is the encoder-heavy `512->128, batch=1`, which trails
`transformers` on c4a and c4: 2.15 sped up `transformers`' one-row GEMMs (see
below), so a single short decode has little left to win.  The table above this
one (Xeon 6975P-C, Graviton 5, ...) is a torch 2.14 recording and is not
directly comparable.

**L4: Dao-AILab flash-attn vs the old hub kernel (GCP g2-standard-32 spot, NVIDIA L4)**

torch 2.14.1+cu130, `flash-attn` 2.8.3.post2 built from source for sm80 (see the
CUDA gotcha in `AGENTS.md`), bf16 unless noted, mean of two interleaved runs per
tree.  "old" is the previous engine on `kernels-community/flash-attn2`; both use
the decode CUDA graph.

| point | config | old ms | new ms | new vs old | new vs HF |
|---|---|---:|---:|---:|---:|
| 128->512 | batch=1 | 1484 | 1482 | 1.00x | 4.01x |
| 128->512 | batch=8 | 1897 | 1896 | 1.00x | 3.19x |
| 128->512 | batch=8, beams=4 | 4365 | 4402 | 0.99x | 2.06x |
| 512->128 | batch=1 | 380 | 378 | 1.01x | 4.00x |
| 512->128 | batch=8 | 583 | 570 | 1.02x | 2.73x |
| 512->128 | batch=8, beams=4 | 967 | 956 | 1.01x | 2.81x |
| 256->256 | batch=1 | 752 | 776 | 0.97x | 3.89x |
| 256->256 | batch=8 | 988 | 978 | 1.01x | 3.14x |
| 256->256 | batch=8, beams=4 | 2086 | 2061 | 1.01x | 2.26x |

The two kernels are the same FlashAttention-2 code, so the swap is neutral
(`256->256, batch=1` has one 805 ms outlier run against 746).  fp32 stays at
0.51-0.55x vs `transformers` on both (no FlashAttention-2 fp32 path).
Florence-2 on the same box: 2.83x (batch=1), 3.79x (batch=8), 4.23x (beams=4),
0.83x (fp32).

**GCP c4a-standard-8 (Axion / Neoverse-V2, 8 cores) - cpu-attention, and torch 2.14 vs 2.15**

`benchmarks/benchmark.py`, bf16 unless noted, tokens/s, each cell the mean of two
interleaved runs on one spot VM (`transformers` from all four runs).  "old" is
the previous engine (vendored vLLM `csrc/cpu` kernel); "new" uses cpu-attention
for the decoder self/cross-attention and the encoder.

| point | config | torch 2.14: HF | old | new | new vs HF | torch 2.15: HF | new | new vs HF |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 128->512 | batch=1 | 31 | 89 | 97 | 3.17x | 97 | 129 | 1.34x |
| 128->512 | batch=8 | 304 | 551 | 589 | 1.93x | 303 | 584 | 1.91x |
| 128->512 | batch=8, beams=4 | 131 | 278 | 290 | 2.22x | 131 | 291 | 2.21x |
| 128->512 | batch=8, fp32 | 229 | 334 | 344 | 1.49x | 228 | 347 | 1.52x |
| 512->128 | batch=1 | 31 | 83 | 80 | 2.58x | 93 | 101 | 1.08x |
| 512->128 | batch=8 | 239 | 372 | 395 | 1.64x | 243 | 398 | 1.63x |
| 512->128 | batch=8, beams=4 | 113 | 222 | 233 | 2.08x | 116 | 236 | 2.03x |
| 512->128 | batch=8, fp32 | 151 | 177 | 184 | 1.21x | 153 | 175 | 1.14x |
| 256->256 | batch=1 | 31 | 89 | 92 | 2.96x | 100 | 122 | 1.21x |
| 256->256 | batch=8 | 292 | 519 | 529 | 1.82x | 290 | 538 | 1.87x |
| 256->256 | batch=8, beams=4 | 130 | 273 | 283 | 2.18x | 130 | 287 | 2.23x |
| 256->256 | batch=8, fp32 | 218 | 291 | 294 | 1.33x | 219 | 302 | 1.38x |

On torch 2.14 the c4a ratios reproduce the Graviton 5 column above (batch=8
within +-0.15, batch=1 0.2-0.4 lower), and cpu-attention is +2-7% over the
vendored kernel on nearly every row.  Torch 2.15 changes the batch=1 baseline,
not the engine: on aarch64 a one-row bf16 GEMM no longer goes through oneDNN,
which in 2.14 re-packed the whole weight on every call, so `transformers`
batch=1 triples (31 -> 93-100 tok/s) and the batch=1 ratio falls to 1.1-1.3x.
The engine's own batch=1 graph is compiled without freezing on aarch64 + torch
>= 2.15 (a packed weight would keep that GEMM on oneDNN), which is worth
1.29-1.39x there; batch=8 is unchanged.  `512->128, batch=1` is the noisy cell
(runs read 116/86 tok/s).

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
| 128->512 | batch=1, beams=1 | 1466 | 349 | 1405 | 364 | 0.96x |
| 128->512 | batch=8, beams=1 | 1859 | 2203 | 1706 | 2401 | 0.92x |
| 128->512 | batch=8, beams=4 | 3243 | 1263 | 3495 | 1174 | **1.08x** |
| 128->512 | batch=8, beams=1, fp32 | 12262 | 334 | 3377 | 1213 | 0.28x |
| 512->128 | batch=1, beams=1 | 375 | 341 | 366 | 349 | 0.98x |
| 512->128 | batch=8, beams=1 | 568 | 1801 | 517 | 1981 | 0.91x |
| 512->128 | batch=8, beams=4 | 914 | 1120 | 1052 | 981 | **1.15x** |
| 512->128 | batch=8, beams=1, fp32 | 3220 | 318 | 991 | 1033 | 0.31x |
| 256->256 | batch=1, beams=1 | 736 | 348 | 707 | 362 | 0.96x |
| 256->256 | batch=8, beams=1 | 961 | 2130 | 874 | 2343 | 0.91x |
| 256->256 | batch=8, beams=4 | 1758 | 1165 | 1740 | 1182 | 0.99x |
| 256->256 | batch=8, beams=1, fp32 | 6188 | 331 | 1732 | 1182 | 0.28x |

At a single new token per step in bf16, vLLM's decode loop is 1-9% leaner than
this one, which is what the greedy rows show.  Beam search pulls the other way:
this engine shares KV blocks between beams where the plugin copies a whole beam
group every step, so `beams=4` is a modest win (`0.99-1.15x`) despite each request
running four sequences.  fp32 is the clear miss: this engine has no
FlashAttention-2 path in fp32, so it loses both the fused attention *and* the
decode CUDA graph that depends on it and goes launch-bound, while vLLM keeps fp32
on its `TRITON_ATTN` backend.

**Greedy decode scaling (vs vLLM, no beam plugin)**

Greedy scales smoothly with batch on both engines and vLLM stays 1-14% ahead; the
widest gap is the encoder-heavy `512->128` point at `batch=32` (0.86x).  At
`batch=1` both are latency-bound (~350 tok/s) and nearly tie, and the gap opens as
each step starts re-reading more KV.

| point | batch | ours ms | ours tok/s | vLLM ms | vLLM tok/s | speedup |
|---|---|---:|---:|---:|---:|---:|
| 128->512 | 1 | 1463 | 350 | 1389 | 369 | 0.95x |
| 128->512 | 2 | 1572 | 651 | 1476 | 694 | 0.94x |
| 128->512 | 4 | 1679 | 1220 | 1534 | 1335 | 0.91x |
| 128->512 | 8 | 1859 | 2204 | 1708 | 2398 | 0.92x |
| 128->512 | 16 | 2305 | 3554 | 2114 | 3875 | 0.92x |
| 128->512 | 32 | 3209 | 5106 | 2997 | 5466 | 0.93x |
| 512->128 | 1 | 375 | 341 | 372 | 344 | 0.99x |
| 512->128 | 2 | 422 | 607 | 400 | 641 | 0.95x |
| 512->128 | 4 | 463 | 1105 | 441 | 1160 | 0.95x |
| 512->128 | 8 | 567 | 1807 | 520 | 1970 | 0.92x |
| 512->128 | 16 | 788 | 2600 | 701 | 2922 | 0.89x |
| 512->128 | 32 | 1243 | 3296 | 1067 | 3841 | 0.86x |
| 256->256 | 1 | 735 | 348 | 692 | 370 | 0.94x |
| 256->256 | 2 | 802 | 638 | 753 | 680 | 0.94x |
| 256->256 | 4 | 859 | 1192 | 779 | 1314 | 0.91x |
| 256->256 | 8 | 961 | 2131 | 876 | 2338 | 0.91x |
| 256->256 | 16 | 1215 | 3373 | 1096 | 3739 | 0.90x |
| 256->256 | 32 | 1737 | 4716 | 1561 | 5248 | 0.90x |

Reproduce with `--batches 1,2,4,8,16,32`.  These rows are measured with the beam
plugin uninstalled (the script picks `VLLM_PLUGINS=bart` for a `--batches` run),
since that plugin replaces vLLM's scheduler with one that disables async
scheduling; `batch=1` is inside the run-to-run noise either way.

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
| 128->512 | batch=1, beams=1 | 3358 | 152.5 | 3485 | 146.9 | 1.04x |
| 128->512 | batch=8, beams=1 | 4744 | 863.5 | 6017 | 680.7 | 1.27x |
| 128->512 | batch=8, beams=1, fp32 | 8784 | 466.3 | 9907 | 413.4 | 1.13x |
| 512->128 | batch=1, beams=1 | 1002 | 127.8 | 1839 | 69.6 | **1.84x** |
| 512->128 | batch=8, beams=1 | 2143 | 477.9 | 9049 | 113.2 | **4.22x** |
| 512->128 | batch=8, beams=1, fp32 | 5017 | 204.1 | 5656 | 181.1 | 1.13x |
| 256->256 | batch=1, beams=1 | 1765 | 145.1 | 1997 | 128.2 | 1.13x |
| 256->256 | batch=8, beams=1 | 2661 | 769.7 | 4985 | 410.8 | **1.87x** |
| 256->256 | batch=8, beams=1, fp32 | 5334 | 384.0 | 6109 | 335.3 | 1.15x |

| point | batch | ours ms | ours tok/s | vLLM ms | vLLM tok/s | speedup |
|---|---|---:|---:|---:|---:|---:|
| 128->512 | 1 | 3370 | 151.9 | 3542 | 144.6 | 1.05x |
| 128->512 | 2 | 3717 | 275.5 | 4496 | 227.8 | 1.21x |
| 128->512 | 4 | 4140 | 494.7 | 4950 | 413.7 | 1.20x |
| 128->512 | 8 | 4883 | 838.8 | 6022 | 680.2 | 1.23x |
| 128->512 | 16 | 7830 | 1046.2 | 9700 | 844.6 | 1.24x |
| 128->512 | 32 | 12916 | 1268.5 | 15827 | 1035.2 | 1.23x |
| 512->128 | 1 | 982 | 130.4 | 1861 | 68.8 | 1.90x |
| 512->128 | 2 | 1155 | 221.6 | 3042 | 84.1 | 2.63x |
| 512->128 | 4 | 1478 | 346.5 | 5015 | 102.1 | 3.39x |
| 512->128 | 8 | 2123 | 482.3 | 9039 | 113.3 | 4.26x |
| 512->128 | 16 | 3775 | 542.5 | 17496 | 117.1 | 4.63x |
| 512->128 | 32 | 6988 | 586.2 | 34082 | 120.2 | **4.88x** |
| 256->256 | 1 | 1736 | 147.5 | 2017 | 126.9 | 1.16x |
| 256->256 | 2 | 1944 | 263.4 | 2764 | 185.2 | 1.42x |
| 256->256 | 4 | 2228 | 459.7 | 3481 | 294.2 | 1.56x |
| 256->256 | 8 | 2779 | 737.1 | 4970 | 412.0 | 1.79x |
| 256->256 | 16 | 4426 | 925.4 | 8764 | 467.3 | 1.98x |
| 256->256 | 32 | 7672 | 1067.8 | 15712 | 521.4 | 2.05x |

This engine is ahead on every CPU row, because the two shapes fail differently for
vLLM.  Where the encoder dominates (`512->128`) its CPU path barely batches: it
goes 69 -> 120 tok/s from `batch=1` to `batch=32` while this engine goes
130 -> 586, so the margin grows with batch to 4.88x.  Where the decode dominates
(`128->512`) the two are close (1.05-1.27x) - a single-token CPU decode is the
same bandwidth-bound GEMM work in both engines.  fp32 is 1.13-1.15x, since
neither engine has a fused attention path to lose there.

Beam search has no vLLM reference on CPU, so its rows are this engine's own: at
`batch=8, beams=4, bf16` it reaches 386 tok/s at `128->512`, 279 at `512->128`
and 366 at `256->256`.  Measure those against `transformers` with
`benchmarks/benchmark.py` instead.

**Florence-2** (real 640x480 photo, four tasks: caption, detection, referring
segmentation and phrase grounding)

| config | Xeon 8581C | Xeon 6975P-C | Graviton 5 | L4 |
|---|---:|---:|---:|---:|
| batch=1, beams=1 | 1.88x | 1.80x | 1.47x | 2.75x |
| batch=8, beams=1 | 2.06x | 2.38x | 1.43x | 3.69x |
| batch=8, beams=4 | 3.68x | 3.86x | 2.35x | 4.21x |
| batch=8, beams=1, fp32 | 1.65x | 1.70x | 1.36x | 0.81x |

The Florence-2 numbers move less than BART's: the vision tower and text encoder
are shared work, so on CPU they are frozen compile graphs of their own (the
tower static per image shape, the text encoder dynamic over prompt length) and
decode rows ride the same decoder graph as BART.  That split is what the fp32
rows gained from — every task prompt used to get its own graph, which pushed the
engine past Dynamo's cache budget (1.36x before, 1.65x now on the 8581C).  In
fp32 this engine reproduces `transformers` token for token; in bf16 the output
length always matches.

## Tests

```bash
python tests/test_parity.py        # vs transformers, tiny random BART
python tests/test_florence2.py     # vs transformers, tiny random Florence-2
python tests/test_attn_backend.py  # cpu-attention vs SDPA (compiled path)
python tests/test_gpu_flash.py     # FlashAttention-2 vs SDPA (CUDA only)
```

## Limitations

* No prefix caching, chunked prefill, preemption or quantization.
* Every running sequence is processed each step, so a long prefill delays decoding.
* `num_blocks` is a hard KV budget; the block manager raises when it runs out.
* Beam search stops with `transformers`' `early_stopping=False` rule only.
