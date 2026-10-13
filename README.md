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
    cudagraph.py     # optional decode-only CUDA graph (prototype)
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
PYTHONPATH=. python benchmarks/benchmark.py --quick                 # BART, bf16
PYTHONPATH=. python benchmarks/benchmark.py --dtype float32         # BART, fp32
PYTHONPATH=. python benchmarks/benchmark.py --quick --device cuda   # BART on GPU
PYTHONPATH=. python benchmarks/benchmark_florence2.py --quick       # Florence-2, bf16
PYTHONPATH=. python benchmarks/benchmark_florence2.py --quick --dtype float32
PYTHONPATH=. python benchmarks/benchmark_florence2.py --quick --device cuda
```

The scripts compare fast-encdec against HuggingFace `generate` on the same
machine, at the **same dtype, same inputs and same output tokens**, so both
engines do identical work.  Every row reports the end-to-end decode throughput of
both engines and the speedup.  BART decodes a fixed length (EOS suppressed via
`min_length` / `min_new_tokens`); Florence-2 stops at EOS, and the reported token
totals confirm both engines emitted the same number.

The tables below are sets of **sweeps**.  A *sweep* varies one knob — batch size,
beam width, encoder length or dtype — while the rest are held fixed, and each row
is one *config*: a point in that sweep.  So "config" fully describes the point
(batch, beams, encoder length and dtype), and the last column is
`speedup = ours tok/s ÷ HF tok/s`.

On CPU, attention runs on the vendored vLLM kernel by default, so the whole
batch's self- and cross-attention is one fused call per step instead of a
per-sequence SDPA loop; the paged cache and block sharing keep the beam-search win
(each step avoids HF's per-step cache reordering).

Decode throughput (a nano-vllm-style workload — N requests, fixed encoder length
and output length, EOS suppressed) is measured separately with
`benchmarks/throughput.py` (the decode CUDA graph is on by default on CUDA;
`--no-cudagraph` disables it).

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
The same holds with `--device cuda` for BART (fp32 and bf16 token-for-token
against HF on the L4); Florence-2 in bf16 can differ in a few tokens because the
compiled vision tower shifts the encoder features.

### CPU

`benchmarks/benchmark.py --quick --threads 8` on `facebook/bart-large-cnn`, bf16
for both engines (fp32 where noted), 48 output tokens, encoder length 256 unless
noted.  fast-encdec runs the vendored vLLM CPU kernel (`vec16` on x86, `neon` on
aarch64), HF its own batched SDPA; "ours" and "HF" are end-to-end decode
throughput, and `speedup = ours ÷ HF`:

| box | config | HF tok/s | ours tok/s | speedup |
|---|---|---:|---:|---:|
| **AMD Ryzen 7 7840H** (Zen 4, `avx512_bf16`, 16 CPUs) | batch=1, beams=1 | 41.1 | 46.7 | 1.14× |
| | batch=8, beams=1 | 200.6 | 252.8 | 1.26× |
| | batch=8, beams=4 | 74.0 | 173.2 | 2.34× |
| | batch=8, beams=1, enc=128 | 251.5 | 328.1 | 1.30× |
| | batch=8, beams=4, fp32 | 28.4 | 61.3 | 2.16× |
| **Intel Xeon 8581C** (Emerald Rapids, `amx_bf16`, c4-standard-8) | batch=1, beams=1 | 50.0 | 56.8 | 1.14× |
| | batch=8, beams=1 | 283.2 | 306.2 | 1.08× |
| | batch=8, beams=4 | 112.0 | 185.4 | 1.66× |
| | batch=8, beams=1, enc=128 | 342.8 | 362.8 | 1.06× |
| | batch=8, beams=4, fp32 | 25.6 | 63.2 | 2.47× |
| **Intel Xeon 6975P-C** (Granite Rapids, `amx_bf16`, m8i.2xlarge) | batch=1, beams=1 | 46.1 | 51.4 | 1.12× |
| | batch=8, beams=1 | 264.4 | 285.4 | 1.08× |
| | batch=8, beams=4 | 108.7 | 178.2 | 1.64× |
| | batch=8, beams=1, enc=128 | 317.0 | 346.6 | 1.09× |
| | batch=8, beams=4, fp32 | 29.4 | 63.6 | 2.17× |
| **AWS Graviton 5** (Neoverse-V3, BF16/I8MM/SVE2, m9g.2xlarge) | batch=1, beams=1 | 42.6 | 46.2 | 1.08× |
| | batch=8, beams=1 | 316.3 | 346.3 | 1.09× |
| | batch=8, beams=4 | 160.7 | 214.5 | 1.33× |
| | batch=8, beams=1, enc=128 | 389.9 | 450.8 | 1.16× |
| | batch=8, beams=4, fp32 | 71.2 | 94.8 | 1.33× |

Every box runs 8 threads (`--threads 8`) with `block_size=16` and the vendored
kernel (`attn_backend="auto"`).  Beam search is the big win (1.3–2.3×) because the
paged cache shares blocks instead of reordering a full beam cache each step;
greedy is a modest 1.06–1.30× (HF's batched SDPA is already good and the encoder
is GEMM-bound on both engines).  Graviton 5 has the highest absolute throughput and
the smallest *relative* win — HF is fast there too — while fp32 leaves the least
headroom (HF is compute-bound rather than kernel-bound), so its greedy rows are
near parity and only beam search still pays.

**BF16 vs FP32.** On the x86 boxes BF16 is ~3× faster than FP32 (the fp32 rows
above are batch=8/beams=4: 2.16–2.47× vs HF, but ~60 tok/s against ~175 tok/s in
bf16). On Graviton BF16 was only ~1.2× faster than FP32 until a fix: PyTorch's CPU
SDPA runs ~20× slower for BF16 *non-contiguous* inputs on aarch64, and the encoder
self-attention fed it transposed views. Calling `.contiguous()` there (and in the
SDPA fallback) took the Graviton BF16 batch=8/beams=1 run from 2237 ms to 1207 ms,
making BF16 **~2.3× faster than FP32** there — visible above as Graviton's fp32 row
holding up best (94.8 tok/s).  (The Ryzen box is a thermally-limited laptop, so its
absolute numbers vary a few percent run-to-run.)

### Attention implementation

`benchmarks/attn_backend.py` isolates the decoder attention and runs it
end-to-end.  Three implementations: the vendored **vLLM CPU kernel** (the
engine's default), PyTorch **SDPA** (`attn_backend="sdpa"`) and **eager** — SDPA
pinned to PyTorch's math kernel, i.e. the naive matmul/softmax/matmul, and HF's
`attn_implementation="eager"`.

Microseconds per decoder attention call (bf16, `block_size=16`, 8 threads).  On
CPU our SDPA/eager paths are the per-sequence fallback loop (CUDA's is batched),
so those columns include that Python overhead:

| box | op | vllm | sdpa | eager | vllm vs eager |
|---|---|---:|---:|---:|---:|
| Ryzen 7 7840H | self, batch=8, ctx=256 | 32 | 2073 | 2549 | 80× |
| | self, batch=8, ctx=1024 | 467 | 7471 | 9594 | 21× |
| | self, batch=32, ctx=1024 | 2984 | 28210 | 29371 | 10× |
| | cross, batch=8, enc=256 | 37 | 437 | 468 | 13× |
| Xeon 8581C (Emerald Rapids) | self, batch=8, ctx=256 | 96 | 2223 | 3082 | 32× |
| | self, batch=8, ctx=1024 | 387 | 5598 | 9990 | 26× |
| | self, batch=32, ctx=1024 | 1814 | 17923 | 29025 | 16× |
| | cross, batch=8, enc=256 | 102 | 220 | 737 | 7× |
| Xeon 6975P-C (Granite Rapids) | self, batch=8, ctx=256 | 129 | 2004 | 3111 | 24× |
| | self, batch=8, ctx=1024 | 494 | 7657 | 12402 | 25× |
| | self, batch=32, ctx=1024 | 2199 | 27580 | 41405 | 19× |
| | cross, batch=8, enc=256 | 134 | 217 | 763 | 6× |
| Graviton 5 | self, batch=8, ctx=256 | 71 | 645 | 1491 | 21× |
| | self, batch=8, ctx=1024 | 363 | 1667 | 3930 | 11× |
| | self, batch=32, ctx=1024 | 1324 | 7005 | 15546 | 12× |
| | cross, batch=8, enc=256 | 62 | 114 | 926 | 15× |

End-to-end, batch=8, greedy, `ours` = this engine and `HF` = `generate` (tok/s):

| box | workload | ours/vllm | ours/sdpa | ours/eager | HF/sdpa | HF/eager |
|---|---|---:|---:|---:|---:|---:|
| Ryzen 7 7840H | enc=256, out=48 | **256** | 184 | 155 | 199 | 202 |
| | enc=64, out=256 | **400** | 225 | 192 | 262 | 274 |
| Xeon 8581C | enc=256, out=48 | **306** | 247 | 173 | 282 | 253 |
| | enc=64, out=256 | **437** | 275 | 200 | 368 | 332 |
| Xeon 6975P-C | enc=256, out=48 | **283** | 230 | 163 | 273 | 244 |
| | enc=64, out=256 | **409** | 258 | 181 | 345 | 315 |
| Graviton 5 | enc=256, out=48 | **345** | 300 | 171 | 318 | 238 |
| | enc=64, out=256 | **579** | 402 | 209 | 503 | 366 |

The vendored kernel is **5.7–80× faster per op** than the SDPA/eager fallbacks
(the ratio shrinks as the op grows, since the fallback's per-sequence loop
amortizes while the GEMM work does not), which is worth **+39 % to +93 %
end-to-end** over our own `attn_backend="sdpa"` and 1.04–1.53× over HF.  Our SDPA
fallback is itself *slower* than HF (0.75–0.94×) because on CPU it attends one
sequence at a time — hence the kernel is the default.  HF's own `eager` and
`sdpa` are close on x86 (either can win by a few %), while on Graviton SDPA is
clearly better than eager (503 vs 366 tok/s decode-heavy).  Graviton 5 and
Emerald Rapids are the fastest boxes for the kernel; the Ryzen laptop wins the
smallest case (ctx=256, 32 µs) on clock speed but falls behind at ctx=1024.

### Florence-2 (`florence-community/Florence-2-base`)

Mixed `<CAPTION>` / `<OD>` / `<REFERRING_EXPRESSION_SEGMENTATION>the car` on a
real 640×480 photo, so the batch has a wide output-length spread
(`benchmark_florence2.py --quick`, Ryzen + Graviton 5).  `ours tok == hf tok` in
every row, i.e. both engines emitted the same number of tokens:

| config | Ryzen bf16 | Ryzen fp32 | Graviton bf16 | Graviton fp32 |
|---|---:|---:|---:|---:|
| greedy, batch=3 | 1.27× | 1.08× | 1.26× | 1.03× |
| greedy, batch=6 | 1.27× | 1.37× | 1.37× | 1.02× |
| beam, batch=3, beams=1 | 1.27× | 1.11× | 1.25× | 1.00× |
| beam, batch=3, beams=3 | 1.21× | 1.41× | 1.44× | 1.10× |
| continuous, N=6, window=6 | 1.32× | 1.37× | 1.37× | 1.02× |

The win comes from **not padding**: the mixed batch has a 13/22/96-token spread,
and fast-encdec never decodes the padding positions HF carries; beam search
benefits most from paged block sharing.  The DaViT vision encoder (~0.5 s/image)
is a large cost shared by both engines, which caps the end-to-end gain.

`LLM(..., compile_mm_encoder=True)` compiles the shared vision tower
(~1.3–1.5×, ≈1.15× end-to-end); it may change the decoded tokens, so it is off
by default.

### GPU (CUDA, NVIDIA L4)

`benchmarks/benchmark.py --quick --device cuda` — same workload, bf16 unless
noted, FlashAttention-2 backend, `block_size=256`, `num_blocks=64`.  The decode
step is captured into a **CUDA graph** (on by default on CUDA + flash — see the
CUDA usage section above):

| box | config | HF tok/s | ours tok/s | speedup |
|---|---|---:|---:|---:|
| **NVIDIA L4** (sm89, 24 GB, 4 vCPU) | batch=1, beams=1 | 89.4 | 272.1 | 3.04× |
| | batch=8, beams=1 | 677.8 | 1641.0 | 2.42× |
| | batch=8, beams=4 | 474.7 | 598.9 | 1.26× |
| | batch=8, beams=1, enc=128 | 686.2 | 1747.8 | 2.55× |
| | batch=8, beams=4, fp32 | 332.3 | 187.2 | 0.56× |

Greedy is now well ahead of HF (2.4–3.0×): the CUDA graph removes the per-step
host launch overhead that dominated small batches — eager, the same rows were
~1.05× (see the AGENTS baseline).  Beam search is 1.26× because the paged cache
writes new K/V separately where HF's cache append is fused, and at 32 sequences
(8 requests × 4 beams) the step is real GPU work rather than launch overhead.
FP32 trails (0.56×) because FlashAttention-2 is fp16/bf16-only and the SDPA
fallback is slower — use `dtype=torch.bfloat16` on a GPU.

Decode throughput (`benchmarks/throughput.py`, `bart-large-cnn`, bf16, enc=512,
EOS suppressed, so both engines decode a fixed token count; the graph is on by
default, `--no-cudagraph` disables it):

| requests | max_tokens | eager tok/s | cudagraph tok/s |
|---|---:|---:|---:|
| 32 | 256 | 2601 | 3395 |
| 128 | 256 | 4546 | 4654 |

At batch ≥ 32 the decode is GPU-bound re-reading the KV cache each step
(`flash_fwd_splitkv_kernel` is ~2/3 of the CUDA time), so eager and the graph
converge; the graph's win is small-batch latency.

Florence-2 on the same box (`benchmark_florence2.py --quick --device cuda`, mixed
caption/detection/segmentation batch, vision tower compiled):

| box | config | HF tok/s | ours tok/s | speedup |
|---|---|---:|---:|---:|
| **NVIDIA L4** | greedy, batch=3 | 170.2 | 589.8 | 3.46× |
| | greedy, batch=6 | 308.8 | 911.5 | 2.95× |
| | beam, batch=3, beams=1 | 181.8 | 574.8 | 3.16× |
| | beam, batch=3, beams=3 | 146.0 | 376.9 | 2.58× |
| | continuous, N=6, window=6 | 308.7 | 916.9 | 2.97× |

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
