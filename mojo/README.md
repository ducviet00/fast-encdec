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
pixi run build        # -> mojo/build/fastencdec_mojo_attn.so
pixi run test         # standalone kernel test (no torch)
```

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

* **Parallelism: one task per (sequence, head).** The work is embarrassingly
  parallel across heads because every head owns a disjoint slice of the paged
  cache, so no reduction is needed. `max.algorithm.parallelize` runs the tasks
  on MAX's runtime thread pool; the engine passes `torch.get_num_threads()`
  workers. With `num_seqs * num_heads` tasks (128 for batch 8 / 16 heads) the
  pool stays busy even at batch 1, and a thread only ever streams its own
  head's pages, which keeps the access pattern prefetch-friendly. The kernel
  is memory-bandwidth-bound, like the C++ one.
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

Kernel call only (Ryzen 7 7840H, 8 threads, bf16, `head_dim` 64, page 32, best
of 60 alternating rounds). Each column includes its own Python/pybind entry
point, i.e. what the engine pays per layer. `ratio` is Mojo over vLLM, so < 1
is faster:

| shape | vLLM `csrc/` | Mojo | ratio |
|---|---:|---:|---:|
| self, batch=1 ctx=256 | 15.1 µs | 12.8 µs | 0.84 |
| self, batch=8 ctx=256 | 45.4 µs | 33.1 µs | 0.73 |
| self, batch=8 ctx=1024 | 451 µs | 419 µs | 0.93 |
| self, batch=32 ctx=1024 | 2933 µs | 2679 µs | 0.91 |
| self, batch=1 ctx=4096 | 77.8 µs | 59.1 µs | 0.76 |
| cross, batch=1 ctx=256 | 12.8 µs | 13.7 µs | 1.07 |
| cross, batch=8 ctx=256 | 28.5 µs | 31.5 µs | 1.11 |
| cross, batch=8 ctx=1024 | 422 µs | 413 µs | 0.92-0.98 |
| cross, batch=32 ctx=1024 | 2789 µs | 2531 µs | 0.91 |
| cross, batch=1 ctx=4096 | 84 µs | 77 µs | 0.91 |

Run-to-run spread on this laptop is +-10%, so treat the ratios as bands rather
than exact values; the shortest-batch cross rows in particular move between 1.07
and 1.30.  The engine runs one self and one cross call per decoder layer, so the
per-layer aggregate is 0.87-0.97 of vLLM's — ahead at every shape except the
batch-1/short-context cross call.

Final three-box run of the merged tree — `benchmarks/attn_backend.py --mode kernel`
plus `benchmarks/kernel_ab.py` (alternating vLLM/Mojo rounds), two runs each,
torch threads = the box's physical cores, Mojo workers = `_workers()` (the same
number). Ratio `mojo/vllm`, so < 1 is faster:

| box (threads / workers) | self b8/256 | **cross b8/256** | self b8/1024 | cross b8/1024 | self b32/1024 |
|---|---:|---:|---:|---:|---:|
| Ryzen 7840H (8 / 8) | 0.76-0.88 | 1.02-1.18 | 0.84-0.92 | 0.92-0.96 | 0.86 |
| Xeon 6975P-C (4 / 4) | 0.67-0.70 | **0.71-0.84** | 0.89-0.90 | 1.00-1.02 | 0.85-0.90 |
| Graviton 5 (8 / 8) | 0.43-0.48 | **0.43-0.51** | 0.60-0.79 | 0.65-0.77 | 0.86-0.95 |

At `b1` the two server boxes are 0.70-0.83 (`self`) and 0.99-1.14 (`cross`); at
`b1/ctx4096` 0.86-1.09 and 0.77-0.99. The 1024-token rows are memory-roofline
bound on both servers (231-298 GB/s measured on Graviton, 133 GB/s from L3 on
the Xeon) and insensitive to the kernel, which is also where the run-to-run
spread is largest (+-10-25% on Graviton).

## End to end (engine)

`benchmarks/benchmark.py --scenarios '256->256'` and
`benchmarks/benchmark_florence2.py`, dev Ryzen 7840H, 8 threads, one engine per
process, Mojo and the vendored C++ backend measured in the same session:

BART-large-CNN, batch x 256 generated tokens:

| config | Mojo ms | Mojo tok/s | C++ ms | C++ tok/s | Mojo / C++ | Mojo vs HF |
|---|---:|---:|---:|---:|---:|---:|
| batch=1, beams=1, bf16 | 3922 | 65.3 | 3600 | 71.1 | 1.09 | 1.28x |
| batch=8, beams=1, bf16 | 5550 | 369.0 | 5259 | 389.4 | 1.06 | 1.60x |
| batch=8, beams=4, bf16 | 8452 | 242.3 | 7990 | 256.3 | 1.06 | 3.70x |
| batch=8, beams=1, fp32 | 9827 | 208.4 | 8702 | 235.3 | 1.13 | 2.54x |

Florence-2-base on a real 640x480 photo, four tasks (caption / detection /
referring-expression segmentation / caption-to-phrase grounding), EOS-stopped so
the output lengths are the tasks' own (13/22/256/22):

| config | Mojo ms | Mojo tok/s | C++ ms | C++ tok/s | Mojo / C++ | Mojo vs HF |
|---|---:|---:|---:|---:|---:|---:|
| batch=1, beams=1, bf16 | 432 | 30.1 | 410 | 31.7 | 1.05 | 1.39x |
| batch=8, beams=1, bf16 | 4689 | 133.5 | 4531 | 138.2 | 1.03 | 1.87x |
| batch=8, beams=4, bf16 | 5214 | 120.1 | 5018 | 124.8 | 1.04 | 4.65x |
| batch=8, beams=1, fp32 | 10654 | 58.8 | 10197 | 61.4 | 1.04 | 2.09x |

Both end-to-end tables are 3-13% behind the C++ backend on this laptop: a decode
step here is dominated by the fused GEMMs plus MAX's fork-join waking 8 threads
per attention call (the same ~140 us per-call tax per step that OpenMP charges
the C++ path), not by the attention kernel — which is why the same backend wins
on the 4-core Xeon and on Graviton (see the kernel table above).

## Where the time goes (and what not to redo)

Measured on the Ryzen box with the engine's call pattern, not the microbenchmark:

- **In-engine an attention call costs ~196 µs (ours) vs ~166 µs (vLLM), while the
  same call warm costs 30-40 µs.** With ~1.5 ms between calls a decode step the
  worker pool has gone to sleep, and the wake dominates: dispatching 128 no-op
  items costs 0.36 µs inline, 6.3 µs with a warm 8-thread pool, and a *median*
  103-140 µs when the pool is asleep (measured for MAX's runtime and for a
  hand-rolled pthread pool). That is a property of waking 8 threads on this box,
  not of MAX: a lean spin-then-condvar pool prototype measured 9.4 µs warm
  (worse than MAX's 6.3) and 103 µs cold (no better than MAX's 140).
- Consequences: keeping the pool hot across the step
  (`MODULAR_THREAD_BUSY_WAIT_US` >= the inter-call gap) makes the totals worse,
  because the spinning threads then compete with torch's GEMM threads; a
  4-worker cap gains ~2.6% end to end at batch 8 / short context but costs ~37%
  on the warm `cross b8/256` number (work is 1/workers bound there), so the
  worker count stays at `torch.get_num_threads()`.
- So at `cross b8/256` the warm kernel work is already at parity and the ~3 µs
  residual is MAX's handoff (6.3 µs) against OpenMP's (~2.5 µs) — the remaining
  lever is the kernel body, not the host: bf16 P@V with quantized probabilities
  (what MAX and vLLM's AMX path do) would halve the P@V multiply count at the
  cost of bit-exactness with `csrc/`.
- **On a 4-physical-core Granite Rapids box the picture inverts**: the kernel is
  bandwidth-bound (133 GB/s from L3 at this shape, 62-70 GB/s from DRAM, both
  measured by pointing every sequence at one shared page set), the C++ kernel is
  compute-bound at 81 GB/s, and the port beats it on every row at 4 threads —
  self 0.67 / 0.94 / 0.86, cross 0.72 (b8/enc256) / 0.90 (b8/enc1024). There,
  prefetching K/V (3 strides), `GROUP` 2/8 and 6/8 workers were all
  neutral-to-worse; `_workers()` is capped at the physical core count (sysfs,
  cached) because a larger `OMP_NUM_THREADS` from a parent otherwise oversubscribes
  MAX's fork-join and costs up to 4.3x. `block_size=128` buys ~9% at ctx=1024
  (config-level: `attn_backend.py --block-size 128`).

## Layout

```
mojo/
  pixi.toml                     Mojo + MAX (for `parallelize`) nightly env
  src/attention.mojo            the kernel (Job, store, attention, dispatch)
  src/fastencdec_mojo_attn.mojo Python extension module (PyInit_...)
  tests/test_attention.mojo     standalone parity test against a scalar reference
```
