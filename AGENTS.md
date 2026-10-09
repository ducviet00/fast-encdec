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
(`Florence2VisionBackbone`) and `Florence2MultiModalProjector`, vendored in
`fastencdec/models/davit.py` (an optimized copy of the `transformers`
reference). Visual tokens are spliced into the encoder input at the `<image>`
placeholders; the decoder is unchanged.

### Non-goals / do not add without discussion

- No CUDA / GPU support, no flash-attn.
- No quantization, prefix caching, chunked prefill, or CUDA graphs.
- **Vendored vLLM CPU attention kernel.** `csrc/cpu` is a copy of vLLM's
  `csrc/cpu` attention implementation (Apache-2.0, see `csrc/LICENSE`), built
  on first use by `fastencdec/layers/cpu_attn.py`. It is the default decoder
  self/cross-attention backend when it compiles; the pure-PyTorch SDPA loop is
  the fallback. Do not add *other* external kernels without explicit approval.

## 2. Environment

- Python env used for development: `/home/ducviet00/.venvs/torch-cpu/bin/python`
  (torch 2.14.0+cpu, transformers 5.17, safetensors).
- Dev CPU: AMD Ryzen 7 7840H (Zen 4) — has `avx512_bf16`, **no** AMX, **no**
  `avx512_fp16`. Prefer `bfloat16` or `float32`; never `float16`.
- HF models are cached locally (`facebook/bart-large-cnn`,
  `hf-internal-testing/tiny-random-BartForConditionalGeneration`).
- Run from the repo root with `PYTHONPATH=.`, or `pip install -e .`.

## 3. Layout

Mirrors nano-vllm's package layout (`engine/`, `layers/`, `models/`, `utils/`).

```
fastencdec/
  __init__.py          package exports (LLM, SamplingParams, RequestOutput, ...)
  config.py            Config (engine hyperparameters)
  llm.py               unified LLM facade (BART + Florence-2, vLLM-style API)
  outputs.py           RequestOutput / CompletionOutput (vLLM-shaped)
  sampling_params.py   SamplingParams
  engine/
    sequence.py        Sequence
    block_manager.py   paged blocks: allocate / fork / copy-on-write / free
    scheduler.py       continuous batching
    model_runner.py    batching, KV cache, encoder caching, forward
    llm_engine.py      LLMEngine: step loop, greedy/sampling, beam search
  layers/
    attention.py       paged self-attention + cached cross-attention dispatch
    cpu_attn.py        JIT build + wrapper for the vendored vLLM CPU kernels
    logits_processor.py transformers LogitsProcessorList from SamplingParams
    sampler.py         greedy / multinomial sampling (Sampler nn.Module)
  models/
    bart.py            BART model (encoder / decoder / cross-attn)
    davit.py           vendored + optimized DaViT vision tower + projector
    florence2.py       Florence-2 (DaViT + projector + BART language model)
  utils/
    context.py         global per-step Context read by attention layers
    loader.py          HF checkpoint -> model weight loading
csrc/                  vendored vLLM CPU attention kernels (+ bindings.cpp)
examples/summarize.py
examples/florence2.py
benchmarks/benchmark.py
tests/test_parity.py
tests/test_attn_backend.py
tests/test_florence2.py
tests/test_beam.py
```

## 4. How a step works

```
Scheduler.schedule()              # admit waiting up to max_num_seqs; return ALL running
ModelRunner.run(seqs)
    fresh = [seq for seq in seqs if seq.num_cached_tokens == 0]
    _encode(fresh)                # batched encoder (grouped by length/image shape), cache cross-attn K/V
    _prepare(seqs)                # input_ids/positions/slot_mapping; ensure blocks/CoW
    set_context(...)              # global Context read by attention
    decoder forward               # self-attn reads/writes paged cache, cross-attn reads cache
    compute_logits(last token)    # [num_seqs, vocab]
LLMEngine.step()                  # sample (greedy) or beam-expand; append; free finished
```

`LLMEngine.run()` loops `step()` until `is_finished()`.
`LLMEngine.generate(requests, params)` is the offline entrypoint: it adds the
whole batch under a single `SamplingParams` and runs to completion.

## 5. Core invariants (read before changing anything)

1. **Token bookkeeping.** `Sequence.token_ids` = decoder prompt + generated
   tokens. `num_cached_tokens` = how many of them have K/V in the paged cache.
   The tokens fed next step are `token_ids[num_cached_tokens:]`. After
   `_prepare`, `num_cached_tokens == len(token_ids)`; a sampled token is then
   appended, so between steps there is exactly one uncached token. Prefill
   (fresh sequence) feeds the whole prompt at once.
2. **Positions** are absolute indices into `token_ids` (0-based).
   `BartLearnedPositionalEmbedding` adds the `+2` BART offset internally.
3. **Paged self-attn layout** is `[num_blocks, num_heads, block_size, head_dim]`
   (vLLM's, so the vendored kernel can read it directly). A token at sequence
   position `p` lives in block `block_table[p // block_size]` at offset
   `p % block_size`. `Context.context_lens[i]` is the total cached length
   *after* writing. `Context.key_slot_ids` is `(block_ids, pos_in_block)` for
   every cached key (ordered by sequence), so the SDPA fallback gathers each
   layer's cache with one advanced-index instead of one slice per sequence.
   The vLLM kernel instead writes K/V itself (`cpu_attn_reshape_and_cache`) and
   reads the cache via `Context.block_table`; `Context.attn_isa` selects it.
4. **Causal rule** (`paged_attention` in `layers/attention.py`): `is_causal = query_len ==
   context_len`. Prefill has `query_len == context_len` (fresh sequence);
   decode has `query_len == 1 < context_len` (attend to every cached key).
   This **assumes a sequence is prefilled exactly once, when its cache is
   empty**. Do not add chunked prefill or preemption/recompute without updating
   the mask logic. The kernel backend encodes the same rule as
   `Context.dynamic_causal` (1 = prefill, 0 = decode), because one batch can mix
   both.
5. **Cross-attention is dense, not paged.** Each `BartDecoderCrossAttention`
   holds `encoder_kv_cache: dict[request_id -> (k, v)]`, each
   `[num_heads, enc_len, head_dim]` (head-first so the flash kernel reads a
   contiguous tensor). It is filled once in `_encode`, shared by all beams of a
   request (same `request_id`), and freed in `ModelRunner.clear_request`. It
   does not count against `num_blocks`. With the vLLM backend each layer's
   `EncoderPagedCache` stages that K/V into a paged cache for the kernel,
   rebuilt only when the set of active requests changes (once per offline
   batch).
6. **Beam search = sequences + block sharing.** Every beam is an ordinary
   `Sequence` with the request's `request_id`. `BlockManager.fork` shares the
   parent's blocks (refcount++); `BlockManager.ensure_capacity` copy-on-writes
   any block that will be written while shared. Children are created **before**
   parents are freed.
7. **Scheduling.** `Scheduler.schedule()` returns *all* running sequences each
   step, so a request's beams are never split. Keep `max_num_seqs >=
   num_beams`.
8. **Global context.** `utils/context.set_context(...)` is called once per
   forward; it builds the step's `Context` and attention layers read it via
   `get_context()`. It is not thread-safe and assumes a single in-flight
   forward.
9. **EOS.** `LLMEngine.generate` fills `SamplingParams.eos_token_id` from the
   model config when unset. Logits processing is delegated to `transformers`'
   `LogitsProcessorList` (built in `layers/logits_processor.build_logits_processor`),
   so `min_length`, `no_repeat_ngram_size`, `repetition_penalty`, forced BOS/EOS,
   suppressed tokens and the sampling warpers match `generate`. `LLM.generate`
   uses the checkpoint's `GenerationConfig` (via
   `SamplingParams.from_generation_config`) when called without params; an
   explicit `SamplingParams` fully overrides those defaults. `length_penalty`
   ranks finished beams. The whole batch shares one (stateless)
   `LogitsProcessorList`; greedy decoding applies it once per step to every
   request at the same decoder length.
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

# vendored vLLM kernel vs the SDPA path (tiny BART with head_dim=32)
PYTHONPATH=. $P tests/test_attn_backend.py

# token-by-token parity vs HF on the real checkpoints (fp32 exact; bf16 same length)
PYTHONPATH=. $P benchmarks/verify_parity.py --dtype float32

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

### Benchmark baseline

Snapshot of `PYTHONPATH=. $P benchmarks/benchmark.py --quick`
(`bart-large-cnn`, 8 threads, bfloat16 for **both** engines, `max_tokens=48`,
`num_blocks=1024`, `block_size=16`, on the dev CPU; attention on the vendored
vLLM CPU kernel, i.e. `attn_backend="auto"`). The HF baseline is loaded at the
same dtype and both engines decode exactly `max_tokens`, so the speedup is
engine-vs-engine. **Every commit must refresh this from a real run of the
script — replace the numbers with the actual `ours ms` / `speedup` output;
never hand-edit or guess.**

| sweep (bf16, enc=256 unless noted) | config | ours ms | speedup vs HF |
|---|---|---:|---:|
| batch, beams=1 | batch=1 | 1038 | 1.11x |
| batch, beams=1 | batch=2 | 951 | 1.33x |
| batch, beams=1 | batch=4 | 1115 | 1.31x |
| batch, beams=1 | batch=8 | 1486 | 1.29x |
| batch, beams=4 | batch=1 | 1081 | 1.41x |
| batch, beams=4 | batch=2 | 1238 | 1.59x |
| batch, beams=4 | batch=4 | 1544 | 1.91x |
| batch, beams=4 | batch=8 | 2219 | 2.31x |
| beams, batch=8 | beams=1 | 1518 | 1.28x |
| beams, batch=8 | beams=4 | 2222 | 2.35x |
| dtype, batch=8, beams=4 | float32 | 5962 | 2.15x |
| dtype, batch=8, beams=4 | bfloat16 | 2187 | 2.37x |
| enc length, batch=8, beams=1 | enc=128 | 1194 | 1.28x |
| enc length, batch=8, beams=1 | enc=256 | 1507 | 1.28x |

Snapshot of `PYTHONPATH=. $P benchmarks/benchmark_florence2.py --quick`
(`florence-community/Florence-2-base`, bf16 for both engines, 8 threads,
640×480 image → 768px, `max_tokens=96` / `beam_tokens=64`; vision tower +
projector on `torch.compile`, the new default). Refresh from a real run.

| sweep (bf16) | config | ours ms | speedup vs HF |
|---|---|---:|---:|
| greedy batch | batch=3 | 1923 | 1.32x |
| greedy batch | batch=6 | 3151 | 1.40x |
| beam (batch=3) | beams=1 | 1690 | 1.27x |
| beam (batch=3) | beams=3 | 1883 | 1.56x |
| continuous (N=6) | w=3 | 3416 | 1.30x |
| continuous (N=6) | w=6 | 3106 | 1.43x |

## 7. Gotchas

- **`bart-large-cnn` needs decoding heuristics.** Without `length_penalty=2.0`,
  `no_repeat_ngram_size=3`, `min_length=8` it collapses to short/repetitive
  output. Those come from `generation_config.json` automatically when
  `generate` is called without `SamplingParams`; the checkpoint also sets
  `forced_bos_token_id=0` / `forced_eos_token_id=2`, which are applied through
  `transformers`' logits processors (so an explicit `SamplingParams` leaves them
  at `None` to disable them).
- **Fair benchmarks.** `benchmark.py` loads HF at the run's `--dtype` (never
  fp32-for-HF vs bf16-for-us) and both engines decode exactly `max_tokens` (EOS
  suppressed via `min_length=max_tokens+1` / HF `min_new_tokens`). Same
  precision, the engine now leads both greedy and beam search on the vendored
  kernel (see the baseline table). `benchmarks/verify_parity.py` confirms the
  fairness directly: fp32 is token-for-token identical to HF, bf16 matches the
  output length (occasional single-step token differences come from the
  different attention kernels).
- **Comparing to HF.** To get a clean reference, disable the checkpoint's
  generation config (`forced_bos_token_id=None`, `forced_eos_token_id=None`,
  `min_length=0`, `no_repeat_ngram_size=0`, `length_penalty=1.0`,
  `early_stopping=False`) and pass `num_beams` explicitly (otherwise HF falls
  back to the config's `num_beams=4`). `GenerationConfig` assignment of
  `forced_bos_token_id=None` is unreliable; passing it as a `generate()` kwarg
  works.
- **Beam parity.** Beam search ranks all continuations globally, finalizes an
  EOS / max-length candidate only if it lands in the top `num_beams`, prunes the
  finished set to the best `num_beams` by length-normalized score, and stops
  with HF's `early_stopping=False` improvement heuristic; greedy is exact and
  beam matches `generate` in fp32. Note `_beam_search` applies the logits
  processors to the log-probs (whereas `_sample` applies them to the logits),
  so n-gram masking does not renormalize the scores — the engine mirrors this
  split in its `run` loop.
- **Cross-attn memory** scales with `batch × enc_len` and lives outside
  `num_blocks` (BART-large ≈ 8 MB/layer/request at enc_len=1024 in fp32).
- **`num_blocks` is a hard budget**; `BlockManager._allocate_block` raises when
  exhausted.
- **BF16 vs FP32**: BF16 ≈ 2× on GEMM-heavy work via oneDNN; attention is
  FP32-accumulate in both.
- **Florence-2 image input** needs `Pillow` (approved dependency) for
  `AutoProcessor`. Use `florence-community/Florence-2-base` / `-large`; the
  `hf-tiny-v2/...` tiny checkpoint is inconsistent (config `image_token_id=4`
  vs tokenizer `51289`, and its vision 2D position embeddings overflow at the
  processor's 768px), so `tests/test_florence2.py` builds inputs by hand.
- **`compile_mm_encoder`** (on by default) `torch.compile`s the vision tower +
  projector (~1.2x there, ~1.13x end-to-end after the vendored eager tower).
  Inductor's BF16 fusion shifts outputs, so greedy tokens can diverge from HF;
  pass `compile_mm_encoder=False` for exact parity or to avoid the per-shape
  compile latency (it compiles once per batch size on first use). `dynamic=True`
  compiles slowly and gives no speedup; fp32 compile is near-exact but slower
  than plain BF16 eager.
- **Attention backend.** `attn_backend="auto"` uses the vendored vLLM kernel
  when it compiles (the default), else SDPA. The kernel only handles
  `head_dim ∈ {32,48,64,80,96,112,128,160,192,224,256,512}`; other shapes (e.g.
  the tiny-random-BART test model, `head_dim=4`) fall back to SDPA. The first
  use compiles `csrc/` (~1-3 min) and caches the `.so` under torch's extensions
  dir; a missing compiler downgrades `"auto"` to SDPA but makes `"vllm"` fail.
- **BF16 SDPA needs contiguous inputs on aarch64.** PyTorch's CPU SDPA with BF16
  *non-contiguous* (transposed) inputs takes a path ~20x slower than the
  contiguous one (Graviton: 86 ms vs 4 ms for a BART-large encoder block); x86
  dispatches to the same flash kernel either way and is unaffected (slightly
  faster non-contiguous). The encoder self-attention builds q/k/v with
  `view().transpose(1, 2)`, so it calls `.contiguous()` before SDPA (the SDPA
  fallback in `layers/attention.py` does too), and so does the vendored DaViT
  window attention (`models/davit.py`). Without it, BF16 on Graviton ran at fp32
  speed; with it, BF16 is ~2.3x faster than fp32 there. Measured penalty for
  the non-contiguous BF16 vision shapes (Graviton, torch 2.14): ~5.5x for the
  window attention, ~19x for the text encoder shape; the vision tower was ~2x
  slower end-to-end until the window attention was made contiguous.

## 8. Known limitations / TODO ideas

- No token budget / chunked prefill: every running sequence is processed each
  step, so a long prefill blocks decode.
- No prefix caching, no preemption/recompute, no quantization.
- Beam search: `num_beams` is per request; stopping uses HF's
  `early_stopping=False` improvement heuristic only (no `early_stopping=True`
  / `"never"` modes, no beam sampling).
- Sampling: `SamplingParams` exposes temperature, top-p, top-k, repetition
  penalty, forced BOS/EOS and suppressed tokens via `transformers`' processors;
  no typical-p / min-p / epsilon / eta / bad-words / prefix-constrained.
- Cross-attention could be paged to count against `num_blocks` (dense is
  intentional for simplicity).
- Attention is a Python per-sequence loop; a batched SDPA path could help
  large batches.

## 9. Style

- Keep it small and readable; match the existing file structure and naming.
- Minimal comments; docstrings brief and direct. No dead code.
- Pure PyTorch plus the vendored vLLM CPU attention kernel in `csrc/` (approved;
  Apache-2.0). No new third-party dependencies without discussion. (`Pillow` was
  approved for Florence-2 image processing; the DaViT vision tower and projector
  are vendored in `fastencdec/models/davit.py`, an optimized copy of the
  `transformers` reference.)
- After changes: run `python -m compileall fastencdec`,
  `tests/test_parity.py`, `tests/test_attn_backend.py`,
  `tests/test_florence2.py`, `tests/test_beam.py`, `uvx ruff check` /
  `uvx ruff format --check`, and `benchmarks/benchmark.py --quick`.
- Every commit must update the "Benchmark baseline" table above with the real
  numbers from that `--quick` run (no stale or estimated values).
