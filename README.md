# fast-encdec

A tiny, readable BART inference library for **offline inference on CPU and GPU**,
in the spirit of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

It implements the pieces that actually matter for seq2seq decoding:

- **Paged KV cache** with block reference counting and copy-on-write.
- **Continuous batching** — requests join and leave the running batch freely.
- **No padding** — attention is per-sequence, so variable-length prompts and
  batches waste nothing.
- **Beam search** — implemented on top of the same paged cache; beams share
  blocks and only copy a block when they write into a shared one.
- **One fused attention kernel per step** — on CPU the decoder self- and
  cross-attention run on vLLM's hand-written kernels (NEON BFMMLA on aarch64,
  AVX-512 VEC/VEC16 on x86), vendored in `csrc/` and compiled on first use; on
  CUDA they run on **FlashAttention-2** (`flash_attn_with_kvcache`, reading the
  paged block table directly, no per-step gather). A pure PyTorch
  `scaled_dot_product_attention` path is the fallback (`attn_backend="sdpa"`);
  BF16 GEMMs go through oneDNN on CPU.

The model code is a near-verbatim copy of the HuggingFace BART reference; the
attention kernels are swapped and the attention q/k/v projections are fused into
one GEMM (`utils/loader` concatenates HF's parts, so the weights still load 1:1).
**Florence-2** is supported too: the same
paged BART decoder is paired with the DaViT vision encoder and the multimodal
projector imported verbatim from `transformers`.

## Layout

Mirrors nano-vllm's package layout (`engine/`, `layers/`, `models/`, `utils/`).

```
fastencdec/
  __init__.py        # package exports (LLM, SamplingParams, RequestOutput, ...)
  config.py          # engine hyperparameters (Config)
  llm.py             # unified LLM facade (BART + Florence-2, vLLM-style API)
  outputs.py         # RequestOutput / CompletionOutput (vLLM-shaped)
  sampling_params.py # SamplingParams
  engine/
    sequence.py      # Sequence
    block_manager.py # paged blocks: allocate / fork / copy-on-write / free
    scheduler.py     # continuous batching
    model_runner.py  # batching, KV cache, encoder caching, forward
    llm_engine.py    # LLMEngine: step loop, sampling, beam search
    cudagraph.py     # decode CUDA graph (on by default on CUDA + flash)
  layers/
    attention.py     # paged self-attention + cached cross-attention dispatch
    cpu_attn.py      # JIT build + wrapper for the vendored vLLM CPU kernels
    flash_attn.py    # lazy loader for kernels-community/flash-attn2 (CUDA)
    logits_processor.py # transformers LogitsProcessorList from SamplingParams
    sampler.py       # greedy / multinomial sampling (Sampler)
  models/
    bart.py          # BART model (encoder / decoder / cross-attn)
    florence2.py     # Florence-2 (DaViT + projector + BART language model)
  utils/
    context.py       # per-step metadata read by attention layers
    loader.py        # HuggingFace -> model weight loading
csrc/                # vendored vLLM CPU attention kernels (+ bindings.cpp)
```

## Usage

`generate` mirrors the vLLM offline API: prompts plus optional `SamplingParams`,
returning `RequestOutput` objects (text via `output.outputs[0].text`).

```python
from fastencdec import LLM, SamplingParams

llm = LLM("facebook/bart-large-cnn")

# greedy
output = llm.generate("The quick brown fox ...", SamplingParams(max_tokens=32))[0]
print(output.outputs[0].text)

# beam search
output = llm.generate(
    "The quick brown fox ...", SamplingParams(max_tokens=32, num_beams=4)
)[0]
print(output.outputs[0].text)

# batched (continuous batching): different-length inputs run together
outputs = llm.generate(
    [article_a, article_b, article_c], SamplingParams(max_tokens=32, num_beams=4)
)
for output in outputs:
    print(output.outputs[0].text)
```

`dtype=torch.bfloat16` halves memory and speeds up the GEMMs on CPUs with
`avx512_bf16` (e.g. AMD Zen 4, Intel Sapphire Rapids). Do **not** use
`float16` on Zen 4 (no AVX512-FP16).

Attention runs on the vendored vLLM CPU kernels by default; the first call
compiles `csrc/` (1–3 min) and caches the `.so`. Pass `attn_backend="sdpa"` to
force the pure-PyTorch path, or `attn_backend="vllm"` to require the kernel
(raising instead of silently falling back).

### GPU (CUDA)

Pass `device="cuda"` to run the whole engine on a GPU. The model, the paged KV
cache and every per-step tensor move to the device, the decoder self- and
cross-attention run on **FlashAttention-2** (the prebuilt
[`kernels-community/flash-attn2`](https://huggingface.co/kernels-community/flash-attn2)
kernel, fetched once via the HuggingFace `kernels` library and cached), and the
DaViT vision tower / BART encoder run through it too:

```python
llm = LLM(
    "facebook/bart-large-cnn",
    num_blocks=1024,
    dtype=torch.bfloat16,
    device="cuda",
)
```

`attn_backend` behaves the same but picks the CUDA kernel: `"auto"` uses
FlashAttention-2 when it can (CUDA, fp16/bf16, a supported head dim, a
compatible block size), else the PyTorch SDPA path; `"flash"` requires it
(raising otherwise) and `"sdpa"` forces SDPA. FlashAttention-2 only supports
fp16/bf16 with head dims `{32, 64, 96, 128, 160, 192, 224, 256}`, and its
*paged* decode kernel needs a page size divisible by 256, so `block_size`
defaults to 256 on CUDA — fp32, an odd head dim (like the tiny-random-BART test
model's 4) or an explicit `block_size=16` falls back to SDPA. The kernel is
loaded lazily, so a CPU-only install never imports it.

On GPU the decoder cache uses FlashAttention-2's page-major
`[blocks, block_size, heads, head_dim]` layout and `flash_attn_with_kvcache`
reads it straight through the block table (no gather, one call per layer), with
`cache_seqlens` carrying the per-sequence context length. Cross-attention stages
each request's encoder K/V into the same kind of paged cache once per active
batch, so beams share it and never duplicate K/V. All the paging machinery
(reference-counted blocks, copy-on-write, continuous batching, beam search) is
identical on both devices.

The decode step is additionally captured into a **CUDA graph** — one graph per
batch-size bucket, replayed each step — which removes the per-step host launch
overhead that dominates small-batch latency. It is **on by default** and only
takes effect on CUDA with the FlashAttention-2 backend (the runner ANDs the flag
with the device/backend check, so CPU and SDPA runs silently use the eager path);
it keeps a fixed preallocated encoder cache. Pass `use_cudagraph=False` to opt
out, e.g. for eager-vs-graph comparisons. See `fastencdec/engine/cudagraph.py` and
the AGENTS gotchas.

`compile_model=True` (default off) ports nano-vllm's `torch.compile` trick: it
fuses the two tensor-only pointwise sequences a BART layer has — the residual
add + layer norm and the gelu + fc1 — instead of wrapping the decoder (which
fragments on the flash attention calls). It fuses cleanly (~7% fewer kernels,
0 graph breaks) but is worth **~0 on GPU time** here, because BART's
`native_layer_norm`/gelu are already single fused kernels and decode is bound by
FlashAttention + GEMMs. It is off by default; nano-vllm gains 1.22× from the
same trick because Qwen3 is full of RMSNorm / gated MLP / rotary ops to fuse, and
a T5 model here (RMS norm + gated FFN) would have the same kind of ops.

### Generation heuristics

Decoding runs through HuggingFace's logits processors
(`LogitsProcessorList`), so `min_length`, `no_repeat_ngram_size`,
`repetition_penalty`, `forced_bos_token_id` / `forced_eos_token_id`,
`suppress_tokens` and the sampling warpers (temperature / top-k / top-p)
behave exactly as in `generate`. When no `SamplingParams` is passed they are
taken from the checkpoint's `generation_config.json`:

```python
llm.generate(article)  # checkpoint defaults
llm.generate(article, sampling_params=SamplingParams(max_tokens=48))  # explicit
```

For `bart-large-cnn` the default is `num_beams=4`, `length_penalty=2.0`,
`no_repeat_ngram_size=3`, `min_length=56`, `forced_bos_token_id=0`,
`forced_eos_token_id=2`. Passing a `SamplingParams` overrides the defaults
entirely, so reproduce a "clean" reference (e.g. for parity tests) with:

```python
SamplingParams(
    max_tokens=48, num_beams=1, length_penalty=2.0, no_repeat_ngram_size=3, min_length=8
)
```

`length_penalty` ranks finished hypotheses, `min_length` masks EOS (counting
the decoder prompt, as `generate` does), and `no_repeat_ngram_size` bans tokens
that would repeat an n-gram.

## Florence-2

Florence-2 combines a DaViT vision encoder with a BART language model.  The
vision tower and the multimodal projector are imported from `transformers`, so
the checkpoint loads 1:1; only the language model uses the paged decoder here.
Images are encoded once per request and their visual tokens are spliced into the
encoder input at the `<image>` placeholders.

```python
import torch
from PIL import Image

from fastencdec import LLM, SamplingParams

llm = LLM("florence-community/Florence-2-base", dtype=torch.bfloat16)

image = Image.open("photo.jpg")
output = llm.generate(
    {"prompt": "<CAPTION>", "multi_modal_data": {"image": image}},
    SamplingParams(max_tokens=32),
)[0]
print(output.outputs[0].text)

# several tasks on one image, or several images:
tasks = ["<CAPTION>", "<OD>"]
outputs = llm.generate(
    [{"prompt": task, "multi_modal_data": {"image": image}} for task in tasks],
    SamplingParams(max_tokens=32),
)
for output in outputs:
    print(output.outputs[0].text)
```

Requires `Pillow` for image loading/processing.  The tiny
`hf-tiny-v2/tiny-random-Florence2ForConditionalGeneration` checkpoint is only
usable with hand-built inputs (see `tests/test_florence2.py`) because its
processor and vision config disagree.

Pass `compile_mm_encoder=True` to `torch.compile` the DaViT tower + projector
(≈1.3–1.5× on the encoder, ≈1.15× end-to-end). It is off by default because
inductor's BF16 fusion shifts the outputs slightly — greedy tokens can diverge
from HuggingFace — and the first call compiles once per batch size:

```python
llm = LLM("florence-community/Florence-2-base", compile_mm_encoder=True)
```

## How a step works

```
scheduler.schedule()            -> all running sequences (+ admit waiting)
model_runner.run(seqs)
    _encode(fresh seqs)         batched encoder (grouped by length/image shape)
                                -> dense cross-attn K/V per request
    _prepare(seqs)              input_ids / positions / slot_mapping / blocks
    set_context(...)            Context: is_prefill, context_lens,
                                block_tables, slot_mapping, request_ids
    decoder forward             ops pick vLLM-CPU / flash-attn2-CUDA / SDPA
llm_engine.step()               compute logits -> sample / beam-expand -> free finished
```

`LLMEngine.run()` loops `step()` until `is_finished()`.

Each sequence only feeds the token not yet in the cache, which is one token per
sequence per step (the decoder prompt is a single token). Attention reads exactly
`context_len` keys per sequence from its own block table, so there is no padding.

The model calls three device-agnostic ops (`layers/attention.py`):
`encoder_self_attention` (dense, once per request), `decoder_self_attention`
(paged, growing cache) and `decoder_cross_attention`. Each op picks its kernel
from the tensor device, dtype and head dim — FlashAttention-2 on CUDA, the
vendored vLLM CPU kernel, or PyTorch SDPA. Each self-attention projects q, k and
v with **one `[dim -> 3*dim]` GEMM** (`qkv_proj`) instead of three small ones —
at decode shapes those are launch-bound, so fusing them is ~3× faster there —
and cross-attention shares a `kv_proj` for its key/value. `utils/loader`
concatenates the HuggingFace `q_proj`/`k_proj`/`v_proj` weights into the fused
projections, so the checkpoint still loads 1:1. Decoder self-attention is paged;
**cross-attention is dense**: each decoder layer keeps
`encoder_kv_cache[request_id] -> (k, v)` of shape `[heads, enc_len, dim]`,
computed once per request and shared by all of its beams. The paged kernels
stage that dense K/V into their own page-major (flash) or head-major (vLLM)
cache, memoized on the layer, so it is rebuilt only when the active request set
changes; SDPA reads the dense cache directly.

For Florence-2 the encoder input is not pure token embeddings: the DaViT tower
and the multimodal projector produce one visual token per image patch (plus one
summary token), which are scattered into the `<image>` placeholder positions
before the BART encoder runs. The vision graph therefore runs in `_encode`,
once per request.

## Beam search

Each beam is an ordinary `Sequence` with its own block table. When hypotheses
fork, the child shares the parent's blocks (refcount++) and copy-on-write
happens lazily on the next append, so forking is O(1) in cache size. This is
why beam search needs almost no extra machinery: the scheduler already moves
sequences in and out of the running batch.

Selection is standard: every step expands each live beam to its top `2*num_beams`
tokens, keeps the best `num_beams` continuations, and moves finished (EOS)
hypotheses aside. The best finished hypothesis (by length-penalized log-prob)
is returned.

## Correctness

- `tests/test_parity.py`: greedy and beam output matches HuggingFace
  token-for-token on a tiny random BART.
- `tests/test_florence2.py`: image features, encoder hidden states, and
  greedy/beam output match HuggingFace on a tiny random Florence-2, plus
  batched-vs-single equivalence with mixed image sizes.
- `tests/test_gpu_flash.py` (CUDA only): the FlashAttention-2 paged/cross paths
  match the SDPA path, on the raw attention kernel and end-to-end through a tiny
  BART.
- Greedy decoding matches HuggingFace on `facebook/bart-large-cnn` (the
  `bart-large-cnn` generation config — `forced_bos_token_id`,
  `forced_eos_token_id`, `min_length`, `no_repeat_ngram_size`,
  `length_penalty` — must be neutralized when comparing).
- Beam search is standard but stops once `num_beams` hypotheses finish, which
  can differ from HF `early_stopping=False` in rare cases.

## Benchmark

```bash
PYTHONPATH=. python benchmarks/benchmark.py                # BART
PYTHONPATH=. python benchmarks/benchmark.py --device cuda  # BART on a GPU
PYTHONPATH=. python benchmarks/benchmark_florence2.py      # Florence-2
PYTHONPATH=. python benchmarks/throughput.py               # decode throughput
PYTHONPATH=. python benchmarks/verify_parity.py            # token-for-token parity
```

Both engines run on the same machine at the **same dtype, inputs and output
length**, so they do identical work; every row reports each engine's throughput
and the speedup (`ours ÷ HF`).  Three points cover where the time goes — a long
output, a long encoder and a square one (BART's position table caps both
lengths at 1022 tokens):

| scenario | encoder tokens | generated tokens |
|---|---:|---:|
| `128->512` | 128 | 512 |
| `512->128` | 512 | 128 |
| `256->256` | 256 | 256 |

**Boxes.** Xeon 8581C (Emerald Rapids, `amx_bf16`, 4 cores) · Xeon 6975P-C
(Granite Rapids, `amx_bf16`, 4 cores) · Graviton 5 (Neoverse-V3, 8 cores) ·
NVIDIA L4 (sm89, 24 GB).

### BART (`facebook/bart-large-cnn`)

Three bf16 generation configs per scenario, plus one fp32 row for verification.
BART decodes a fixed length (EOS suppressed via `min_length` / `min_new_tokens`),
so both engines always decode the same number of tokens.  Cells are
`ours tok/s · HF tok/s (speedup)`.

**128->512**

| box | b=1, beams=1 | b=8, beams=1 | b=8, beams=4 | b=8, beams=1, fp32 |
|---|---:|---:|---:|---:|
| **Xeon 8581C (Emerald)** | 63 · 57 (1.12×) | 348 · 322 (1.08×) | 210 · 94 (2.23×) | 165 · 143 (1.15×) |
| **Xeon 6975P-C (Granite)** | 60 · 54 (1.10×) | 340 · 302 (1.13×) | 206 · 98 (2.10×) | 150 · 136 (1.10×) |
| **Graviton 5** | 49 · 46 (1.05×) | 517 · 436 (1.18×) | 264 · 149 (1.78×) | 250 · 208 (1.20×) |
| **NVIDIA L4** | 290 · 91 (3.19×) | 1897 · 695 (2.73×) | 500 · 458 (1.09×) | 379 · 692 (0.55×) |

**512->128**

| box | b=1, beams=1 | b=8, beams=1 | b=8, beams=4 | b=8, beams=1, fp32 |
|---|---:|---:|---:|---:|
| **Xeon 8581C (Emerald)** | 60 · 55 (1.09×) | 265 · 279 (0.95×) | 165 · 51 (3.47×) | 114 · 111 (1.02×) |
| **Xeon 6975P-C (Granite)** | 57 · 52 (1.09×) | 255 · 273 (0.93×) | 155 · 64 (2.42×) | 106 · 109 (0.98×) |
| **Graviton 5** | 47 · 43 (1.08×) | 337 · 302 (1.12×) | 196 · 132 (1.49×) | 145 · 145 (1.00×) |
| **NVIDIA L4** | 284 · 89 (3.20×) | 1596 · 673 (2.37×) | 565 · 389 (1.45×) | 353 · 619 (0.57×) |

**256->256**

| box | b=1, beams=1 | b=8, beams=1 | b=8, beams=4 | b=8, beams=1, fp32 |
|---|---:|---:|---:|---:|
| **Xeon 8581C (Emerald)** | 63 · 57 (1.10×) | 354 · 328 (1.08×) | 211 · 106 (1.99×) | 150 · 144 (1.05×) |
| **Xeon 6975P-C (Granite)** | 58 · 53 (1.10×) | 331 · 308 (1.07×) | 200 · 107 (1.86×) | 140 · 136 (1.03×) |
| **Graviton 5** | 49 · 44 (1.10×) | 497 · 418 (1.19×) | 248 · 151 (1.64×) | 224 · 199 (1.12×) |
| **NVIDIA L4** | 289 · 88 (3.27×) | 1841 · 681 (2.70×) | 525 · 457 (1.15×) | 374 · 681 (0.55×) |

Each cell is the mean of two independent runs; they agreed token-for-token
everywhere, with throughput repeatability of ≤1 % on the L4 and ≤8 % on the CPU
boxes.

On CPU the default attention is the vendored **vLLM CPU kernel** (Apache-2.0, see
`csrc/`): one fused call per step for the whole batch, over a paged KV cache.  HF
re-concatenates its cache every step and reorders a full beam cache when
`beams > 1`, which is where the beam-search win comes from.  On CUDA the decode
step is captured into a CUDA graph (on by default); fp32 has no FlashAttention-2
path, so those rows trail — use bf16 on a GPU.

### Florence-2 (`florence-community/Florence-2-base`)

A real 640×480 photo (`car.jpg` from the HuggingFace documentation images) and
four task prompts whose outputs have very different lengths — `<CAPTION>` ~13,
`<OD>` ~22, `<CAPTION_TO_PHRASE_GROUNDING>` ~22 and a referring-expression
segmentation capped at 256 tokens — so the batch has a wide length spread,
which is what the paged, unpadded decoder is for.  Both engines stop at EOS.
The encoder length is fixed by the image (768 px → 577 visual tokens), so only
the generation configs vary:

| box | b=1, beams=1 | b=8, beams=1 | b=8, beams=4 | b=8, beams=1, fp32 |
|---|---:|---:|---:|---:|
| **Xeon 8581C (Emerald)** | 31 · 28 (1.12×) | 175 · 106 (1.66×) | 140 · 41 (3.40×) | 72 · 46 (1.56×) |
| **Xeon 6975P-C (Granite)** | 30 · 25 (1.20×) | 165 · 85 (1.95×) | 134 · 39 (3.43×) | 73 · 45 (1.63×) |
| **Graviton 5** | 28 · 25 (1.13×) | 155 · 125 (1.24×) | 133 · 66 (2.03×) | 63 · 54 (1.17×) |
| **NVIDIA L4** | 301 · 110 (2.73×) | 1044 · 307 (3.41×) | 636 · 190 (3.34×) | 212 · 246 (0.86×) |

Each cell is the mean of two runs (repeatability ≤3 %, ≤7 % on Graviton).  The
four tasks make the batch a 13/22/256/22-token spread (the segmentation is capped
by `max_tokens=256`), which is the case the paged decoder is for; the two engines
agree on the token totals except under beam search, where they can pick slightly
different hypotheses.

### Throughput

`benchmarks/throughput.py` measures a decode-only workload — N requests, a fixed
encoder length and output length, EOS suppressed — i.e. an offline
summarisation service.  On the L4 (`bart-large-cnn`, bf16, enc=512):

| requests | max_tokens | eager tok/s | graph tok/s |
|---|---:|---:|---:|
| 32 | 256 | 2601 | 3395 |
| 128 | 256 | 4546 | 4654 |

At batch ≥ 32 the decode is GPU-bound re-reading the KV cache every step
(`flash_fwd_splitkv_kernel` is ~2/3 of the CUDA time), so eager and the graph
converge and the graph's win is small-batch latency.

### Fairness

`benchmarks/verify_parity.py` compares the tokens directly:

```bash
PYTHONPATH=. python benchmarks/verify_parity.py --dtype float32
```

In **fp32** the engine reproduces HF's tokens exactly — BART and Florence-2,
greedy and beam, token for token.  In **bf16** the output length always matches
and the tokens agree except for occasional single-step differences: expected,
since the two engines run different attention kernels.

## Limitations

- The scheduler returns the whole running set each step; there is no token
  budget / chunked prefill. Keep `max_num_seqs >= num_beams` so a request's
  beams are always batched together.
- The attention ops assume exactly one decoder prompt token per sequence (the
  Context carries per-sequence cached lengths, not query lengths); `generate`
  rejects a longer decoder prompt.
- No prefix caching, no preemption/recompute, no quantization.  The decode CUDA
  graph is on by default on CUDA + FlashAttention-2 (see the GPU section); the CPU
  path never uses it.
- The cross-attention cache is dense and scales with `batch × enc_len`; it is
  not counted against `num_blocks`. For Florence-2 the projected visual tokens
  are part of that encoder sequence (577 tokens for a 768px image at base), so
  long image contexts are the dominant cost.
- `num_blocks` is the total KV budget; the block manager raises if exhausted.
- Attention runs on the vendored vLLM CPU kernel by default (one fused call per
  step); `attn_backend="sdpa"` selects the pure-PyTorch per-sequence loop, which
  is the fallback when the kernel cannot be compiled or the head size is
  unsupported. On CUDA the decoder attention runs on FlashAttention-2's paged
  decode kernel (`flash_attn_with_kvcache`), which reads the block table
  directly, so there is no per-step gather.
