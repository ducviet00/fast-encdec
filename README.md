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
- **Pure PyTorch on CPU** — no flash-attn; attention is
  `torch.nn.functional.scaled_dot_product_attention`, so BF16 GEMMs go through
  oneDNN's `avx512_core_bf16` path on modern x86.

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
    sequence.py      # Sequence / SequenceStatus
    block_manager.py # paged blocks: allocate / fork / copy-on-write / free
    scheduler.py     # continuous batching
    model_runner.py  # batching, KV cache, encoder caching, forward
    llm_engine.py    # LLMEngine: step loop, sampling, beam search
  layers/
    attention.py     # CPU paged self-attention + cached cross-attention
    logits_processor.py # transformers LogitsProcessorList from SamplingParams
    sampler.py       # greedy / multinomial sampling (Sampler)
  models/
    bart.py          # BART model (encoder / decoder / cross-attn)
    florence2.py     # Florence-2 (DaViT + projector + BART language model)
  utils/
    context.py       # per-step metadata read by attention layers
    loader.py        # HuggingFace -> model weight loading
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
beams.

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
PYTHONPATH=. python benchmarks/benchmark.py            # full sweep
PYTHONPATH=. python benchmarks/benchmark.py --quick    # fast smoke
```

It sweeps batch size, beam width, dtype and encoder length, comparing fast-encdec
against HuggingFace `generate` on the same machine, at the **same dtype, same
inputs and same decoding**, with a fixed output length (EOS suppressed) so both
engines do identical work (latency, req/s, tok/s, speedup).

The same-precision comparison is the honest one. The paged cache and block
sharing win on **beam search** (each step avoids HF's per-step cache reordering),
while greedy decoding is near parity — HF's batched SDPA edges out our
per-sequence attention when there is no padding or beam work to exploit.

Full sweep — `facebook/bart-large-cnn`, 8 threads, BF16 for both engines, 48
tokens, encoder length 256:

| config | ours ms | hf ms | speedup |
|---|---:|---:|---:|
| batch=1, beams=1 | 1172 | 1157 | 0.99× |
| batch=8, beams=1 | 2050 | 1912 | 0.93× |
| batch=16, beams=1 | 3153 | 2848 | 0.90× |
| batch=1, beams=4 | 1338 | 1531 | 1.14× |
| batch=8, beams=4 | 3778 | 5175 | 1.37× |
| batch=16, beams=4 | 6672 | 10832 | 1.62× |

Other axes: at batch=8/beams=4 the same-precision speedup is ~1.4× in BF16 and
~1.9× in FP32; HF gains more from BF16 than we do (it speeds up 2.6× going
FP32→BF16, ours 1.8×). For greedy decoding (batch=8, beams=1) our per-sequence
attention is ~9% slower than HF's batched SDPA (0.91× at enc=128), but the gap
narrows as the encoder grows (0.95× at enc=1000) since the encoder is
GEMM-bound and both engines share the same oneDNN matmuls.

### Florence-2

`benchmarks/benchmark_florence2.py` mixes captioning (short), detection
(medium) and segmentation (long) requests on a real photo, so the batch has a
large output-length spread:

```bash
PYTHONPATH=. python benchmarks/benchmark_florence2.py --quick
PYTHONPATH=. python benchmarks/benchmark_florence2.py --batches 3,6,12 --beams 1,3,5
```

`florence-community/Florence-2-base`, BF16 for both engines, `<CAPTION>`/`<OD>`/`<REFERRING_EXPRESSION_SEGMENTATION>`,
max 256 tokens, ~13/22/256 output tokens:

| sweep | ours ms | hf ms | speedup |
|---|---:|---:|---:|
| greedy batch=3 | 3751 | 4357 | 1.16× |
| greedy batch=6 | 5868 | 6634 | 1.13× |
| greedy batch=12 | 10237 | 11734 | 1.15× |
| beams=1 | 2558 | 2812 | 1.10× |
| beams=3 | 2910 | 4554 | 1.56× |
| beams=5 | 3240 | 5906 | 1.82× |
| N=24, running window=12 | 18250 | 22314 | 1.22× |

The win comes from **not padding**: the mixed batch has a 13/22/256-token
spread, and fast-encdec never decodes the padding positions HF carries.
Beam search benefits most, because the paged cache shares blocks instead of
reordering a full beam cache each step (up to 1.82×); greedy is a more modest
1.1–1.2×, since the DaViT vision encoder (~0.5 s/image here) is a large cost
shared by both engines and our per-sequence CPU attention can be slower than
HF's batched SDPA.

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
- Attention is a Python per-sequence loop; a batched SDPA path would recover
  most of the remaining Florence-2 gap (see the benchmark above).
