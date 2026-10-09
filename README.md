# fast-encdec

A tiny, readable BART inference library for **CPU offline inference**, in the
spirit of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

It implements the pieces that actually matter for seq2seq decoding:

- **Paged KV cache** with block reference counting and copy-on-write.
- **Continuous batching** — requests join and leave the running batch freely.
- **No padding** — attention is per-sequence, so variable-length prompts and
  batches waste nothing.
- **Beam search** — implemented on top of the same paged cache; beams share
  blocks and only copy a block when they write into a shared one.
- **One fused attention kernel per step** — the decoder self- and
  cross-attention run on vLLM's hand-written CPU kernels (NEON BFMMLA on
  aarch64, AVX-512 VEC/VEC16 on x86), vendored in `csrc/` and compiled on first
  use. A pure PyTorch `scaled_dot_product_attention` path is the fallback
  (`attn_backend="sdpa"`); BF16 GEMMs go through oneDNN.

The model code is a near-verbatim copy of the HuggingFace BART reference; only
the attention kernels are swapped.  **Florence-2** is supported too: the same
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
  layers/
    attention.py     # paged self-attention + cached cross-attention dispatch
    cpu_attn.py      # JIT build + wrapper for the vendored vLLM CPU kernels
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

llm = LLM("facebook/bart-large-cnn", num_blocks=256, block_size=16)

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
from PIL import Image
from fastencdec import LLM, SamplingParams

llm = LLM("florence-community/Florence-2-base", num_blocks=1024, block_size=16)

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
                                -> cross-attn K/V per request
    build input_ids/positions/slot_mapping
    decoder forward             self-attn reads/writes paged cache,
                                cross-attn reads cached encoder K/V
    set_context(...)            global metadata read by attention
llm_engine.step()               compute logits -> sample / beam-expand -> free finished
```

`LLMEngine.run()` loops `step()` until `is_finished()`.

Each sequence only feeds the tokens not yet in the cache
(`token_ids[num_cached_tokens:]`): the whole prompt on prefill, one token per
decode step. Attention gathers exactly `context_len` keys per sequence from its
own block table, so there is no padding.

Decoder self-attention is paged. **Cross-attention is dense**: each decoder
layer keeps `encoder_kv_cache[request_id] -> (k, v)` of shape
`[heads, enc_len, dim]`, computed once per request and shared by all of its
beams. With the vLLM backend that dense K/V is staged into a paged cache
(`EncoderPagedCache`) so cross-attention uses the same kernel; the staging is
rebuilt only when the set of active requests changes.

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
- Greedy decoding matches HuggingFace on `facebook/bart-large-cnn` (the
  `bart-large-cnn` generation config — `forced_bos_token_id`,
  `forced_eos_token_id`, `min_length`, `no_repeat_ngram_size`,
  `length_penalty` — must be neutralized when comparing).
- Beam search is standard but stops once `num_beams` hypotheses finish, which
  can differ from HF `early_stopping=False` in rare cases.

## Benchmark

```bash
PYTHONPATH=. python benchmarks/benchmark.py --quick                 # BART, bf16
PYTHONPATH=. python benchmarks/benchmark.py --dtype float32         # BART, fp32
PYTHONPATH=. python benchmarks/benchmark_florence2.py --quick       # Florence-2, bf16
PYTHONPATH=. python benchmarks/benchmark_florence2.py --quick --dtype float32
```

It compares fast-encdec against HuggingFace `generate` on the same machine, at
the **same dtype, same inputs and same output tokens**, so both engines do
identical work (latency, req/s, tok/s, speedup).  BART decodes a fixed length
(EOS suppressed via `min_length` / `min_new_tokens`); Florence-2 stops at EOS and
the reported token totals confirm both engines emitted the same number.

Attention runs on the vendored vLLM CPU kernel by default, so the whole batch's
self- and cross-attention is one fused call per step instead of a per-sequence
SDPA loop; the paged cache and block sharing keep the beam-search win (each step
avoids HF's per-step cache reordering).

### Fairness

`benchmarks/verify_parity.py` compares the tokens directly:

```bash
PYTHONPATH=. python benchmarks/verify_parity.py --dtype float32
PYTHONPATH=. python benchmarks/verify_parity.py --dtype bfloat16
```

In **fp32** the engine reproduces HF's tokens exactly — BART and Florence-2,
greedy and beam, token for token.  In **bf16** the output length always matches
and the tokens agree except for occasional single-step differences: expected,
since HF runs its own batched SDPA while fast-encdec runs the vLLM CPU kernel.

### Hardware

| box | CPU | threads | kernel |
|---|---|---|---|
| x86 dev | AMD Ryzen 7 7840H (Zen 4, AVX-512 + `avx512_bf16`, no AMX) | 8 | `vec16` |
| Graviton 5 | AWS Graviton 5, 8× Arm Neoverse-V3 (BF16 / I8MM / SVE2) | 8 | `neon` |

Both boxes run torch 2.14 CPU with `block_size=16` and the vendored kernel
(`attn_backend="auto"`).

### BART (`facebook/bart-large-cnn`)

Speedup vs HuggingFace `generate` at the same dtype, 48 output tokens, encoder
length 256 unless noted (`benchmark.py --quick`):

| sweep | config | x86 bf16 | x86 fp32 | Graviton bf16 | Graviton fp32 |
|---|---|---:|---:|---:|---:|
| batch, beams=1 | batch=1 | 1.13× | 1.21× | 1.31× | 1.07× |
| batch, beams=1 | batch=8 | 1.28× | 1.06× | 2.55× | 1.00× |
| batch, beams=4 | batch=8 | 2.35× | 2.14× | 3.67× | 1.34× |
| beams, batch=8 | beams=4 | 2.36× | 2.12× | 3.68× | 1.35× |
| enc length, beams=1 | enc=128 | 1.19× | 1.07× | 2.10× | 1.01× |
| enc length, beams=1 | enc=256 | 1.39× | 1.05× | 2.60× | 1.01× |

Greedy is a modest win (HF's batched SDPA is already good and the encoder is
GEMM-bound on both engines); beam search wins big, because the paged cache shares
blocks instead of reordering a full beam cache each step.  FP32 leaves less
headroom — HF is compute-bound there rather than kernel-bound — so the beam win
is smaller than in BF16.

**BF16 vs FP32.** On the x86 box BF16 is ~2.4× faster than FP32; on Graviton it
was only ~1.2× until a fix. The cause was PyTorch's CPU SDPA, which on aarch64
runs ~20× slower for BF16 *non-contiguous* inputs — and the encoder self-attention
fed it transposed views. Calling `.contiguous()` there (and in the SDPA fallback)
took the Graviton BF16 batch=8/beams=1 run from 2237 ms to 1207 ms, making BF16
**~2.3× faster than FP32** on Graviton.  (The x86 box is a thermally-limited
laptop, so its absolute fp32 numbers vary a few percent run-to-run.)

### Florence-2 (`florence-community/Florence-2-base`)

Mixed `<CAPTION>` / `<OD>` / `<REFERRING_EXPRESSION_SEGMENTATION>the car` on a
real 640×480 photo, so the batch has a wide output-length spread
(`benchmark_florence2.py --quick`).  `ours tok == hf tok` in every row, i.e. both
engines emitted the same number of tokens:

| sweep | config | x86 bf16 | x86 fp32 | Graviton bf16 | Graviton fp32 |
|---|---|---:|---:|---:|---:|
| greedy | batch=3 | 1.21× | 1.05× | 1.26× | 1.03× |
| greedy | batch=6 | 1.19× | 1.25× | 1.37× | 1.02× |
| beam | beams=1 | 1.19× | 1.05× | 1.25× | 1.00× |
| beam | beams=3 | 1.51× | 1.37× | 1.44× | 1.10× |
| continuous | window=6 | 1.21× | 1.25× | 1.37× | 1.02× |

The win comes from **not padding**: the mixed batch has a 13/22/96-token spread,
and fast-encdec never decodes the padding positions HF carries; beam search
benefits most from paged block sharing.  The DaViT vision encoder (~0.5 s/image)
is a large cost shared by both engines, which caps the end-to-end gain.

`LLM(..., compile_mm_encoder=True)` compiles the shared vision tower
(~1.3–1.5×, ≈1.15× end-to-end); it may change the decoded tokens, so it is off
by default.

## Limitations

- The scheduler returns the whole running set each step; there is no token
  budget / chunked prefill. Keep `max_num_seqs >= num_beams` so a request's
  beams are always batched together.
- No prefix caching, no preemption/recompute, no CUDA graphs, no quantization.
- The cross-attention cache is dense and scales with `batch × enc_len`; it is
  not counted against `num_blocks`. For Florence-2 the projected visual tokens
  are part of that encoder sequence (577 tokens for a 768px image at base), so
  long image contexts are the dominant cost.
- `num_blocks` is the total KV budget; the block manager raises if exhausted.
- Attention runs on the vendored vLLM CPU kernel by default (one fused call per
  step); `attn_backend="sdpa"` selects the pure-PyTorch per-sequence loop, which
  is the fallback when the kernel cannot be compiled or the head size is
  unsupported.
