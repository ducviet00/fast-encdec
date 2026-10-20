# Mojo paged attention

A Mojo port of the hot path of `csrc/` — the vendored vLLM CPU attention
kernel — plus the Python bindings that let the engine call it as a torch
custom op (`attn_backend="mojo"`).

## What is (and is not) ported

Ported, from the x86 `VEC` implementation (`csrc/cpu/cpu_attn_vec.hpp`) with
its `TileGemm82` micro-kernel (`csrc/cpu/cpu_attn_impl.hpp`):

* the paged decoder **self-attention** (scatter K/V, then attend) and
  **cross-attention** over a staged encoder cache,
* the vLLM page layout and the fp32-accumulate numerics,
* `reshape_and_cache` in the same transposed-K layout, so both backends share
  cache tensors and `stage_cross_kv`.

Not ported (deliberately — the engine never calls them):

* multi-token queries / chunked prefill, sliding windows, ALiBi, softcap,
  attention sinks, FP8 KV cache, GQA (the kernel keeps the `q_heads_per_kv`
  mapping but BART/Florence-2 are MHA),
* the `VEC16` / `NEON` / `AMX` / `RVV` / `VSX` / `VXE` ISAs and the vLLM
  work-item scheduler. One task owns one `(sequence, head)` pair end to end, so
  there is no KV split, no scratch buffer and no cross-thread reduction.

The contract is the engine's (see `AGENTS.md` invariant 1): **one query token
per sequence, attending to every cached position**. The query is always the
last position, so the causal mask is exactly the cache tail.

## Build and run

```bash
cd mojo
pixi run build        # -> mojo/build/fastencdec_mojo_attn.so (links -lgomp)
pixi run test         # builds the standalone kernel test with -lgomp and runs it
```

The kernel dispatches on libgomp (`GOMP_parallel`), so the build needs
`-Xlinker -lgomp` and the backend is Linux-only; the pixi env is `mojo` alone
(no `max` package). `mojo run` cannot run the kernel — the JIT cannot resolve
the OpenMP symbols — which is why `pixi run test` builds a binary first.

The engine loads `mojo/build/fastencdec_mojo_attn.so` automatically; set
`FASTENCDEC_MOJO_ATTN=/path/to/other.so` to override. Without the library the
backend is simply unavailable and `attn_backend="auto"` keeps using the C++
kernel.

```bash
# from the repo root
PYTHONPATH=. .venv/bin/python tests/test_mojo_attn.py            # engine parity
PYTHONPATH=. .venv/bin/python benchmarks/attn_backend.py --mode kernel
PYTHONPATH=. .venv/bin/python benchmarks/benchmark.py --backend mojo --scenarios '256->256'
```

## Design

* **Parallelism: one task per (sequence, head), on torch's own pool.** The work
  is embarrassingly parallel across heads because every head owns a disjoint
  slice of the paged cache, so no reduction is needed. 128 tasks at batch 8 /
  16 heads, and a thread only ever streams its own head's pages, which keeps the
  access pattern prefetch-friendly. The dispatch is `GOMP_parallel` (libgomp,
  linked with `-Xlinker -lgomp`), i.e. **the OpenMP pool torch already keeps
  warm** rather than a second pool of our own: 2.4 µs per region warm and 2.7 µs
  with a 1 ms gap, against 6.3 µs / 103-140 µs for a private pool that has to be
  woken (see "Where the time goes"). The kernel is memory-bandwidth-bound, like
  the C++ one.
* **Page layout: vLLM's, unchanged.** A page is
  `[num_blocks, num_kv_heads, block_size * head_dim]` with K stored
  *transposed* inside the page (`[head_dim, block_size]`, row-major) and V
  row-major (`[block_size, head_dim]`). Q@K^T then reads one contiguous
  `block_size`-wide row of K per head dim and accumulates a TILE-wide vector,
  and P@V reads one contiguous `head_dim`-wide row of V per position. That is
  also why `reshape_and_cache` writes K column-wise, exactly like the C++
  kernel.
* **Tiles, groups and masks.** A page is one 32-token tile (vLLM's
  `BlockSizeAlignment`, hence `block_size % 32 == 0`; 16 on aarch64 pages). Four
  pages are scored into one **group** before the online-softmax step, which is
  how the C++ kernel amortizes its softmax over an L2-sized tile — per *page*
  softmax bookkeeping is 4x more max/exp/rescale work at short contexts. The
  last page of a sequence masks positions past `context_lens` with `-3.4e38`
  (not `-inf`: `exp()` underflows it to 0 and it cannot produce a NaN).
* **Online softmax per group.** `m_new = max(m, max(group))`,
  `l = l * exp(m - m_new) + sum(exp(group - m_new))`,
  `acc = acc * exp(m - m_new) + P @ V`. Probabilities stay in registers; the
  C++ kernel writes them to a buffer and reloads them as broadcast operands
  (a 4-byte reload out of a 64-byte store), which costs a store-forwarding
  stall per position. The C++ kernel also skips the rescale unless the max
  moves by more than 8 (an FA4 trick); rescaling unconditionally costs
  `head_dim / 16` multiplies per group.
* **Q@K^T as AVX512-BF16 dot products** (`llvm.x86.avx512bf16.dpbf16ps.512`,
  gated on `CompilationTarget` at compile time, fp32 broadcast-FMA otherwise).
  One instruction folds two head dims into each of 16 position lanes, so the
  query-key phase takes half the instructions of the widened fp32 form
  (measured 1.87x on this box: 34 vs 63 ns per 32-position tile). Two bf16
  values multiplied into fp32 are exact, so this is the same math as widening
  first — bf16 output stays bit-identical to the C++ kernel. It needs K stored
  as `[head_dim/2, block_size, 2]` so a `(k[2d], k[2d+1])` pair of one position
  is contiguous; the portable path keeps the plain `[head_dim, block_size]`
  layout. Q is not pre-scaled on this path (it would round through bf16);
  `scale` is applied to the logits instead.
* **P@V: fp32 on x86, bf16 dot on NEON.** A dot instruction needs the two
  contracted positions adjacent, so the ARM path stores V position-paired
  (`[block_size/2, head_dim, 2]`) and rounds the probabilities to bf16 — exactly
  what `csrc` does on ARM (`prob_buffer_t`). One `bfdot` then retires 8 MACs
  instead of 4, and the tile's contribution is folded into the fp32 accumulator
  once per tile. This is the only place the port gives up bit-exactness, and
  only on ARM (x86 keeps the fp32 P@V and stays bit-identical to `csrc/`).
* **Strides.** `q`/`k`/`v` arrive as chunks of the fused `qkv_proj` output, so
  a token sits `3 * d_model` floats apart, not `num_heads * head_dim`. The
  kernel takes the token strides explicitly, like the C++ entry point does;
  without that a batch of 2+ sequences reads the wrong rows.
* **Dispatch.** `head_dim` is a comptime parameter (32/64/128 specializations,
  anything else raises), `dtype` is fp32 or bf16 (fp16 is rejected — this
  project's CPUs have no `avx512_fp16`).

## Numerics

bf16 output is bit-identical to the vendored C++ kernel on the engine's
shapes, and fp32 agrees to ~4e-6 (different summation order, same fp32 math:
the cache is widened to fp32 and Q is pre-scaled by `1/sqrt(head_dim)`).

## Performance

Kernel call only (Ryzen 7 7840H, 8 threads, bf16, `head_dim` 64, page 32,
best-of-N over alternating rounds, `benchmarks/kernel_ab.py`). Each column
includes its own Python/pybind entry point, i.e. what the engine pays per layer.
`ratio` is Mojo over vLLM, so < 1 is faster (the range spans the two runs and
the drift-controlled median; one number = all of them agree):

| shape | vLLM `csrc/` | Mojo | ratio |
|---|---:|---:|---:|
| self, batch=1 ctx=256 | 15.6 µs | 7.9 µs | 0.41-0.51 |
| self, batch=8 ctx=256 | 37.4 µs | 20.0 µs | 0.52-0.53 |
| self, batch=8 ctx=1024 | 445 µs | 383 µs | 0.82-0.86 |
| self, batch=32 ctx=1024 | 2923 µs | 2410 µs | 0.82 |
| self, batch=1 ctx=4096 | 87.0 µs | 43.2 µs | 0.48-0.50 |
| cross, batch=1 ctx=256 | 12.8 µs | 7.4 µs | 0.57-0.58 |
| cross, batch=8 ctx=256 | 28.5 µs | 19.0 µs | **0.66** |
| cross, batch=8 ctx=1024 | 449 µs | 364 µs | 0.78-0.81 |
| cross, batch=32 ctx=1024 | 2769 µs | 2342 µs | 0.85 |
| cross, batch=1 ctx=4096 | 75.6 µs | 45.2 µs | 0.58-0.60 |

Run-to-run spread on this laptop is +-10%, so treat the ratios as bands rather
than exact values. Since the kernel moved onto torch's OpenMP pool (§ "Where
the time goes") the short-context rows gained the most — the dispatch used to
cost 6.3 µs warm / 103-140 µs cold and now costs 2.4 / 2.7 — so the target row
`cross b8/ctx256` went from 1.07-1.30 to **0.66**. The engine runs one self and
one cross call per decoder layer, so the per-layer aggregate is 0.7-0.85 of
vLLM's: ahead at every shape now, widest where the call is short.

Three-box run — `benchmarks/attn_backend.py --mode kernel` plus
`benchmarks/kernel_ab.py` (alternating vLLM/Mojo rounds), two runs each, torch
threads = the box's physical cores, Mojo workers = `_workers()` (the same
number). Ratio `mojo/vllm`, so < 1 is faster. **These two boxes were measured
with the old private pool**, so their rows are conservative: the libgomp switch
removes ~4 µs of dispatch per call, which is 3-15% of the short-context calls
there (the laptop row above was re-measured and shows the size of the gain):

| box (threads / workers) | self b8/256 | **cross b8/256** | self b8/1024 | cross b8/1024 | self b32/1024 |
|---|---:|---:|---:|---:|---:|
| Ryzen 7840H (8 / 8), libgomp | 0.52-0.53 | **0.66** | 0.82-0.86 | 0.78-0.81 | 0.82 |
| Xeon 6975P-C (4 / 4) | 0.67-0.70 | **0.71-0.84** | 0.89-0.90 | 1.00-1.02 | 0.85-0.90 |
| Graviton 5 (8 / 8) | 0.43-0.48 | **0.43-0.51** | 0.60-0.79 | 0.65-0.77 | 0.86-0.95 |

At `b1` the two server boxes are 0.70-0.83 (`self`) and 0.99-1.14 (`cross`); at
`b1/ctx4096` 0.86-1.09 and 0.77-0.99. On the laptop the same points are 0.41-0.51
(`self b1/256`), 0.57-0.60 (`cross b1/256`) and 0.48-0.60 at ctx 4096. The
1024-token rows are memory-roofline bound on both servers (231-298 GB/s measured
on Graviton, 133 GB/s from L3 on the Xeon) and insensitive to the kernel, which
is also where the run-to-run spread is largest (+-10-25% on Graviton).

## End to end (engine)

`benchmarks/benchmark.py --scenarios '256->256'` and
`benchmarks/benchmark_florence2.py`, dev Ryzen 7840H, 8 threads, one engine per
process, Mojo and the vendored C++ backend measured in the same session:

BART-large-CNN, batch x 256 generated tokens:

| config | Mojo ms | Mojo tok/s | C++ ms | C++ tok/s | Mojo / C++ | Mojo vs HF |
|---|---:|---:|---:|---:|---:|---:|
| batch=1, beams=1, bf16 | 3324 | 77.0 | 3611 | 70.9 | **0.92** | 1.66x |
| batch=8, beams=1, bf16 | 5060 | 404.7 | 5157 | 397.2 | 0.98 | 1.75x |
| batch=8, beams=4, bf16 | 7661 | 267.3 | 8054 | 254.3 | 0.95 | 4.16x |
| batch=8, beams=1, fp32 | 8527 | 240.2 | 8977 | 228.1 | 0.95 | 2.95x |

Florence-2-base on a real 640x480 photo, four tasks (caption / detection /
referring-expression segmentation / caption-to-phrase grounding), EOS-stopped so
the output lengths are the tasks' own (13/22/256/22):

| config | Mojo ms | Mojo tok/s | C++ ms | C++ tok/s | Mojo / C++ | Mojo vs HF |
|---|---:|---:|---:|---:|---:|---:|
| batch=1, beams=1, bf16 | 398 | 32.7 | 427 | 30.5 | 0.93 | 1.51x |
| batch=8, beams=1, bf16 | 4344 | 144.1 | 4454 | 140.6 | 0.98 | 2.02x |
| batch=8, beams=4, bf16 | 4873 | 128.5 | 4953 | 126.4 | 0.98 | 4.99x |
| batch=8, beams=1, fp32 | 10070 | 62.2 | 10170 | 61.6 | 0.99 | 2.28x |

So end to end the backend is now **ahead of the C++ one on every row**: 2-8% on
BART (widest at batch 1), 1-7% on Florence-2. Before the move onto torch's
OpenMP pool it was 3-9% behind on BART and 2-6% on Florence-2; the difference is
the per-call dispatch (2.4 µs vs 6.3 warm, and no cold wake), which is most of a
short attention call. Both engines beat HF by 1.5-5x on these points.

## Where the time goes (and what not to redo)

Measured on the Ryzen box with the engine's call pattern, not the microbenchmark.

**The dispatch is the whole story, and it is solved.** An attention call is a
fine-grained parallel region with ~0.8-1.5 ms of torch GEMMs between calls, so a
pool that sleeps pays a wake every time. Measured per dispatch:

| dispatch | µs |
|---|---:|
| inline, 1 worker (no region) | 0.36 |
| **`GOMP_parallel` on torch's pool, warm** | **2.4** |
| **`GOMP_parallel`, 1 ms idle gap, no torch work** | **2.7** |
| `max.algorithm.parallelize`, private pool, warm | 6.3 |
| same, 1 ms gap (the engine's cadence) | 103-140 |
| a hand-rolled spin-then-condvar pool, 1 ms gap | 103 |

The last two rows are why the backend no longer uses MAX's runtime: a private
pool either sleeps (103-140 µs wake, 24 times per step) or spins (which steals
GEMM cycles — MAX's default 200 µs spin cost 4.5% end to end and was worse than
useless). Running the kernel on the pool that *torch* keeps busy between calls
removes the wake entirely, and it also removes the need for the `max` package:
`mojo/pixi.toml` depends on `mojo` alone now.

What that is worth, measured in one harness (8 matmuls ~ 0.8 ms between calls,
i.e. the engine's cadence; per-call numbers include the call itself):

| b8/ctx256 | attention call | GEMM-phase interference |
|---|---:|---:|
| C++ backend | 90.7 µs | +25.9 µs |
| this backend | **29.8-49.6 µs** | +39.2 µs |

And end to end on bart-large-cnn, 8x128 tokens, same session:

| | tok/s |
|---|---:|
| this backend, before the dispatch change | 344 |
| this backend now | **392-396** |
| C++ backend | 364-377 |

So the backend is now **faster than the C++ one end to end on this laptop**
(+7.5% at 8x128, +4.8% at batch 1 x 128), and its attention call is ~3x cheaper
than vLLM's at `cross b8/256`.

Remaining host-side cost, all measured: `torch.empty_like` 0.53 µs, 9
`data_ptr()` 0.53 µs, pybind with 23 args 0.68 µs, the Python wrapper ~3.1 µs
(callable and worker override resolved once, one `stride()` per tensor). Tiny
calls run **inline** (`_workers()` returns 1 when `num_tasks * pages <= 32`:
24.5 µs vs 31.0 µs at batch 1 / ctx 64); from batch 1 / ctx 128 up the pool wins
(1.3x), and 8x at batch 8 / ctx 256.

**On a 4-physical-core Granite Rapids box the picture inverts.** The kernel is
bandwidth-bound (133 GB/s from L3 at this shape, 62-70 GB/s from DRAM, both
measured by pointing every sequence at one shared page set), the C++ kernel is
compute-bound at 81 GB/s, and the port beats it on every row at 4 threads —
self 0.67 / 0.94 / 0.86, cross 0.72 (b8/enc256) / 0.90 (b8/enc1024). There,
prefetching K/V (3 strides), `GROUP` 2/8 and 6/8 workers were all
neutral-to-worse; `_workers()` is capped at the physical core count (sysfs,
cached) — torch already sizes its pool that way, so the cap only matters when
a parent sets a larger `OMP_NUM_THREADS`. `block_size=128` buys ~9% at ctx=1024
(config-level: `attn_backend.py --block-size 128`).

## Layout

```
mojo/
  pixi.toml                     Mojo nightly env (no MAX dependency)
  src/attention.mojo            the kernel (Job, store, attention, dispatch)
  src/fastencdec_mojo_attn.mojo Python extension module (PyInit_...)
  tests/test_attention.mojo     standalone parity test against a scalar reference
```
