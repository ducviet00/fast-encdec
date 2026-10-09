# AGENTS.md — fast-encdec handover

Guidance for AI agents (and humans) working on this repository.

## 1. What this is

`fast-encdec` is a **small, readable CPU inference library for BART** (seq2seq),
in the spirit of nano-vllm. It implements only what matters for offline
encoder-decoder decoding:

- **Paged KV cache** with block reference counting and copy-on-write.
- **Continuous batching** (requests join/leave the running batch freely).
- **No padding** — attention is per-sequence, so variable-length prompts and
  batches waste no compute.
- **Beam search** on top of the same paged cache.
- **Pure PyTorch** on CPU (SDPA attention; BF16 GEMMs go through oneDNN's
  `avx512_core_bf16` on modern x86).

The model (`fastencdec/models/bart.py`) is a near-verbatim copy of the
HuggingFace BART reference; only the attention kernels differ.

**Florence-2** (`fastencdec/models/florence2.py`) reuses that paged BART as its
language model and pairs it with the DaViT vision encoder
(`Florence2VisionBackbone`) and `Florence2MultiModalProjector`, both imported
verbatim from `transformers`. Visual tokens are spliced into the encoder input
at the `<image>` placeholders; the decoder is unchanged.

### Non-goals / do not add without discussion

- No CUDA / GPU support, no flash-attn.
- No quantization, prefix caching, chunked prefill, or CUDA graphs.
- **No vendored vLLM kernels.** An earlier revision copied vLLM's `csrc/cpu`
  CPU_ATTN kernel in as an optional backend; it was removed by maintainer
  decision. Do not re-add it (or any other external kernel dependency) without
  explicit approval.

## 2. Environment

- Python env used for development: `/home/ducviet00/.venvs/torch-cpu/bin/python`
  (torch 2.14.0+cpu, transformers 5.17, safetensors).
- Dev CPU: AMD Ryzen 7 7840H (Zen 4) — has `avx512_bf16`, **no** AMX, **no**
  `avx512_fp16`. Prefer `bfloat16` or `float32`; never `float16`.
- HF models are cached locally (`facebook/bart-large-cnn`,
  `hf-internal-testing/tiny-random-BartForConditionalGeneration`).
- Run from the repo root with `PYTHONPATH=.`, or `pip install -e .`.

## 3. Layout

```
fastencdec/
  __init__.py        LLM facade (load weights, tokenize, drive engine)
  context.py         global per-step Context read by attention layers
  sequence.py        Sequence / SamplingParams
  block_manager.py   paged blocks: allocate / fork / copy-on-write / free
  attention.py       CPU paged self-attention + cached cross-attention
  models/bart.py     BART model (encoder / decoder / cross-attn)
  models/florence2.py Florence-2 (DaViT + projector + BART language model)
  loader.py          HF checkpoint -> model weight loading
  model_runner.py    batching, KV cache, encoder caching, forward
  engine.py          Scheduler (continuous batching) + beam search
  logits_processor.py transformers LogitsProcessorList from SamplingParams
  sampler.py         greedy / multinomial sampling on processed logits
examples/summarize.py
examples/florence2.py
benchmarks/benchmark.py
tests/test_parity.py
tests/test_florence2.py
```

## 4. How a step works

```
Scheduler.schedule()              # admit waiting up to max_num_seqs; return ALL running
ModelRunner.run(seqs)
    fresh = [seq for seq in seqs if seq.num_cached_tokens == 0]
    _encode(fresh)                # batched encoder (grouped by length/image shape), cache cross-attn K/V
    _prepare(seqs)                # input_ids/positions/slot_mapping; ensure blocks/CoW
    set_context(Context(...))     # global read by attention
    decoder forward               # self-attn reads/writes paged cache, cross-attn reads cache
    compute_logits(last token)    # [num_seqs, vocab]
LLMEngine.run()                   # sample (greedy) or beam-expand; free finished
```

## 5. Core invariants (read before changing anything)

1. **Token bookkeeping.** `Sequence.token_ids` = decoder prompt + generated
   tokens. `num_cached_tokens` = how many of them have K/V in the paged cache.
   The tokens fed next step are `token_ids[num_cached_tokens:]`. After
   `_prepare`, `num_cached_tokens == len(token_ids)`; a sampled token is then
   appended, so between steps there is exactly one uncached token. Prefill
   (fresh sequence) feeds the whole prompt at once.
2. **Positions** are absolute indices into `token_ids` (0-based).
   `BartLearnedPositionalEmbedding` adds the `+2` BART offset internally.
3. **Paged self-attn layout** is `[num_blocks, block_size, num_heads, head_dim]`.
   A token at sequence position `p` lives at physical slot
   `block_table[p // block_size] * block_size + (p % block_size)`.
   `Context.context_lens[i]` is the total cached length *after* writing.
4. **Causal rule** (`attention.paged_attention`): `is_causal = query_len ==
   context_len`. Prefill has `query_len == context_len` (fresh sequence);
   decode has `query_len == 1 < context_len` (attend to every cached key).
   This **assumes a sequence is prefilled exactly once, when its cache is
   empty**. Do not add chunked prefill or preemption/recompute without updating
   the mask logic.
5. **Cross-attention is dense, not paged.** Each `BartDecoderCrossAttention`
   holds `encoder_kv_cache: dict[request_id -> (k, v)]`, each
   `[num_heads, enc_len, head_dim]` (head-first so the flash kernel reads a
   contiguous tensor). It is filled once in `_encode`, shared by all beams of a
   request (same `request_id`), and freed in `ModelRunner.clear_request`. It
   does not count against `num_blocks`.
6. **Beam search = sequences + block sharing.** Every beam is an ordinary
   `Sequence` with the request's `request_id`. `BlockManager.fork` shares the
   parent's blocks (refcount++); `BlockManager.ensure_capacity` copy-on-writes
   any block that will be written while shared. Children are created **before**
   parents are freed.
7. **Scheduling.** `Scheduler.schedule()` returns *all* running sequences each
   step, so a request's beams are never split. Keep `max_num_seqs >=
   num_beams`.
8. **Global context.** `context.set_context` is called once per forward;
   attention layers read it via `context.get_context()`. It is not thread-safe
   and assumes a single in-flight forward.
9. **EOS.** `LLMEngine.add_request` fills `SamplingParams.eos_token_id` from the
   model config when unset. Logits processing is delegated to `transformers`'
   `LogitsProcessorList` (built in `logits_processor.build_logits_processor`),
   so `min_length`, `no_repeat_ngram_size`, `repetition_penalty`, forced BOS/EOS,
   suppressed tokens and the sampling warpers match `generate`. `LLM.generate`
   uses the checkpoint's `GenerationConfig` (via
   `SamplingParams.from_generation_config`) when called without params; an
   explicit `SamplingParams` fully overrides those defaults. `length_penalty`
   ranks finished beams.
10. **Multimodal inputs (Florence-2).** `Sequence.pixel_values` carries the
    request image `[3, H, W]`; `_encode` groups fresh sequences by
    `(encoder length, image shape)` and the model scatters projected visual
    tokens into the `image_token_id` slots. `ModelRunner` reaches the model
    through `model.decoder`, `model.decoder_layers` and `model.encode(...)`,
    which both BART and Florence-2 expose.

## 6. Running things

```bash
P=/home/ducviet00/.venvs/torch-cpu/bin/python

# unit / parity (tiny random BART; greedy + beam vs HF)
PYTHONPATH=. $P tests/test_parity.py

# Florence-2 parity (tiny random Florence-2; image features + greedy/beam vs HF)
PYTHONPATH=. $P tests/test_florence2.py

# example
PYTHONPATH=. $P examples/summarize.py
PYTHONPATH=. $P examples/florence2.py

# benchmark (sweeps batch / beams / dtype / encoder length vs HF)
PYTHONPATH=. $P benchmarks/benchmark.py                 # full
PYTHONPATH=. $P benchmarks/benchmark.py --quick         # fast smoke
PYTHONPATH=. $P benchmarks/benchmark.py --json out.json

# Florence-2 benchmark (caption/detection/segmentation mix vs HF)
PYTHONPATH=. $P benchmarks/benchmark_florence2.py --quick
PYTHONPATH=. $P benchmarks/benchmark_florence2.py --batches 3,6,12 --beams 1,3,5
```

## 7. Gotchas

- **`bart-large-cnn` needs decoding heuristics.** Without `length_penalty=2.0`,
  `no_repeat_ngram_size=3`, `min_length=8` it collapses to short/repetitive
  output. Those come from `generation_config.json` automatically when
  `generate` is called without `SamplingParams`; the checkpoint also sets
  `forced_bos_token_id=0` / `forced_eos_token_id=2`, which are applied through
  `transformers`' logits processors (so an explicit `SamplingParams` leaves them
  at `None` to disable them).
- **Comparing to HF.** To get a clean reference, disable the checkpoint's
  generation config (`forced_bos_token_id=None`, `forced_eos_token_id=None`,
  `min_length=0`, `no_repeat_ngram_size=0`, `length_penalty=1.0`,
  `early_stopping=False`) and pass `num_beams` explicitly (otherwise HF falls
  back to the config's `num_beams=4`). `GenerationConfig` assignment of
  `forced_bos_token_id=None` is unreliable; passing it as a `generate()` kwarg
  works.
- **Beam parity.** Our beam search stops once `num_beams` hypotheses finish
  (standard early stopping). This can differ from HF `early_stopping=False` by
  a token or two. Greedy is exact.
- **Cross-attn memory** scales with `batch × enc_len` and lives outside
  `num_blocks` (BART-large ≈ 8 MB/layer/request at enc_len=1024 in fp32).
- **`num_blocks` is a hard budget**; `BlockManager._allocate` raises when
  exhausted.
- **BF16 vs FP32**: BF16 ≈ 2× on GEMM-heavy work via oneDNN; attention is
  FP32-accumulate in both.
- **Florence-2 image input** needs `Pillow` (approved dependency) for
  `AutoProcessor`. Use `florence-community/Florence-2-base` / `-large`; the
  `hf-tiny-v2/...` tiny checkpoint is inconsistent (config `image_token_id=4`
  vs tokenizer `51289`, and its vision 2D position embeddings overflow at the
  processor's 768px), so `tests/test_florence2.py` builds inputs by hand.
- **`compile_mm_encoder=True`** (opt-in) `torch.compile`s the vision tower +
  projector (~1.3-1.5x there, ~1.15x end-to-end). It is off by default because
  inductor's BF16 fusion shifts outputs (greedy tokens can diverge from HF) and
  it compiles once per batch size. `dynamic=True` compiles fast but gives no
  speedup; fp32 compile is exact but slower than plain BF16 eager.

## 8. Known limitations / TODO ideas

- No token budget / chunked prefill: every running sequence is processed each
  step, so a long prefill blocks decode.
- No prefix caching, no preemption/recompute, no quantization.
- Beam search: `num_beams` is per request; no length-normalized stopping
  heuristic beyond `length_penalty`.
- Missing samplers: top-k, repetition penalty, forced BOS/EOS.
- Cross-attention could be paged to count against `num_blocks` (dense is
  intentional for simplicity).
- Attention is a Python per-sequence loop; a batched SDPA path could help
  large batches.

## 9. Style

- Keep it small and readable; match the existing file structure and naming.
- Minimal comments; docstrings brief and direct. No dead code.
- Pure PyTorch only. No new third-party dependencies without discussion.
  (`Pillow` was approved for Florence-2 image processing; the DaViT vision
  tower and projector are imported from `transformers` rather than copied.)
- After changes: run `python -m compileall fastencdec`, `tests/test_parity.py`,
  `tests/test_florence2.py`, and a `benchmarks/benchmark.py --quick` smoke.
