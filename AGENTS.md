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
- **Pure PyTorch** on CPU, with an optional **CUDA** path for the decoder
  attention. On CPU, BF16 GEMMs go through oneDNN's `avx512_core_bf16` on
  modern x86; on CUDA the decoder self-/cross-attention (and the BART encoder
  self-attention) run on **FlashAttention-2**.

The model (`fastencdec/models/bart.py`) is a near-verbatim copy of the
HuggingFace BART reference; only the attention kernels differ.

**Florence-2** (`fastencdec/models/florence2.py`) reuses that paged BART as its
language model and pairs it with the DaViT vision encoder
(`Florence2VisionBackbone`) and `Florence2MultiModalProjector`, vendored in the
same `fastencdec/models/florence2.py` (an optimized copy of the `transformers`
reference). Visual tokens are spliced into the encoder input at the `<image>`
placeholders; the decoder is unchanged.

### Non-goals / do not add without discussion

- No flash-attn **outside** the CUDA backend: CPU stays pure PyTorch + the
  vendored vLLM kernel.
- No quantization, prefix caching or chunked prefill. CUDA graphs exist only as
  an opt-in **decode-only** prototype (`Config.use_cudagraph`, default off) — see
  `engine/cudagraph.py`; nothing on the default path depends on them.
- **Vendored vLLM CPU attention kernel.** `csrc/cpu` is a copy of vLLM's
  `csrc/cpu` attention implementation (Apache-2.0, see `csrc/LICENSE`), built
  on first use by `fastencdec/layers/cpu_attn.py`. It is the default CPU
  decoder self/cross-attention backend when it compiles; the pure-PyTorch SDPA
  loop is the fallback. Do not add *other* external kernels without explicit
  approval. On CUDA the only external kernel is
  `kernels-community/flash-attn2` (BSD-3), fetched by
  `fastencdec/layers/flash_attn.py`.

## 2. Environment

- Python env used for development: `/home/ducviet00/.venvs/torch-cpu/bin/python`
  (torch 2.14.0+cpu, transformers 5.17, safetensors).
- Dev CPU: AMD Ryzen 7 7840H (Zen 4) — has `avx512_bf16`, **no** AMX, **no**
  `avx512_fp16`. Prefer `bfloat16` or `float32`; never `float16`.
- CUDA benchmarking box: `ssh ducviet00@34.142.225.47` (NVIDIA L4, sm89, 24 GB,
  driver 580 / CUDA 13.0, 4 vCPU). The repo lives at `~/fast-encdec` and the env
  at `~/fegpu` (python 3.12, torch 2.11.0+cu128, transformers 5.19, `kernels`);
  there is no GPU on the dev box, so run `--device cuda` benchmarks there. Auth
  is flaky (public-key connections are sometimes refused) — retry. `build_essential`
  was installed for `torch.compile` of the Florence-2 vision tower.
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
    cudagraph.py       optional decode-only CUDA-graph capture/replay (prototype)
  layers/
    attention.py       device-agnostic attention ops (encoder/self/cross) + dispatch
    cpu_attn.py        JIT build + wrapper for the vendored vLLM CPU kernels
    flash_attn.py      lazy loader for the kernels-community/flash-attn2 kernel (CUDA)
    logits_processor.py transformers LogitsProcessorList from SamplingParams
    sampler.py         greedy / multinomial sampling (Sampler nn.Module)
  models/
    bart.py            BART model (encoder / decoder / cross-attn)
    florence2.py       Florence-2 (vendored DaViT + projector + BART LM)
  utils/
    context.py         device-agnostic per-step Context read by the attention ops
    loader.py          HF checkpoint -> model weight loading
csrc/                  vendored vLLM CPU attention kernels (+ bindings.cpp)
examples/summarize.py
examples/florence2.py
benchmarks/benchmark.py
benchmarks/benchmark_florence2.py
benchmarks/verify_parity.py
benchmarks/throughput.py
benchmarks/florence2_tasks.py
benchmarks/profile_florence2.py
tests/test_parity.py
tests/test_attn_backend.py
tests/test_florence2.py
tests/test_beam.py
tests/test_gpu_flash.py
```

## 4. How a step works

```
Scheduler.schedule()              # admit waiting up to max_num_seqs; return ALL running
ModelRunner.run(seqs)
    fresh = [seq for seq in seqs if seq.num_cached_tokens == 0]
    _encode(fresh)                # batched encoder (grouped by length/image shape) -> dense cross-attn K/V
    _prepare(seqs)                # input_ids/positions/slot_mapping; ensure blocks/CoW
    set_context(...)              # Context: is_prefill, context_lens, block_tables, slot_mapping, request_ids
    decoder forward               # ops dispatch to vLLM-CPU / flash-attn2-CUDA / SDPA
                                  # (or, with use_cudagraph, replay a captured decode graph)
    compute_logits                # one row per sequence (one new token each)
LLMEngine.step()                  # sample (greedy) or beam-expand; append; free finished
```

`LLMEngine.run()` loops `step()` until `is_finished()`.
`LLMEngine.generate(requests, params)` is the offline entrypoint: it adds the
whole batch under a single `SamplingParams` and runs to completion.

`models/bart.py` calls three device-agnostic ops from
`layers/attention.py` -- `encoder_self_attention` (dense, no cache),
`decoder_self_attention` (paged growing cache) and `decoder_cross_attention`
(fixed per-request encoder K/V).  Each op reads `utils/context.py` and picks its
own kernel; the runner and the model never name a backend
(`attention.set_backend` only overrides the choice for tests).

## 5. Core invariants (read before changing anything)

1. **Token bookkeeping.** `Sequence.token_ids` = decoder prompt + generated
   tokens. `num_cached_tokens` = how many of them have K/V in the paged cache.
   The tokens fed next step are `token_ids[num_cached_tokens:]`. After
   `_prepare`, `num_cached_tokens == len(token_ids)`; a sampled token is then
   appended, so between steps there is exactly one uncached token. Prefill (a
   fresh sequence) feeds its single decoder prompt token: **every step feeds
   exactly one token per sequence**, which the attention ops assume (and
   `generate` rejects a longer decoder prompt).
2. **Positions** are absolute indices into `token_ids` (0-based).
   `BartLearnedPositionalEmbedding` adds the `+2` BART offset internally.
3. **Paged self-attn cache.** `layers/attention.py` picks the layout the active
   kernel wants -- head-major `[num_blocks, heads, block_size, head_dim]` (vLLM
   CPU / SDPA) or page-major `[num_blocks, block_size, heads, head_dim]`
   (FlashAttention-2) -- in the single place both the op and the runner go
   through, `attention.self_cache_shape`. A token at sequence position `p` lives
   in `block_tables[i, p // block_size]` at offset `p % block_size` (flat slot
   `block * block_size + p % block_size`). `Context.context_lens[i]` is the total
   cached length *after* writing. The SDPA fallback builds its gather index from
   `Context.block_tables`/`context_lens` inside the op (memoized per step), so
   the Context stays plain tensors.
4. **Causal rule.** Prefill (fresh) has `query_len == context_len` (both 1);
   decode has `query_len == 1 < context_len` (attend to every cached key). This
   **assumes a sequence is prefilled exactly once, when its cache is empty**. Do
   not add chunked prefill or preemption/recompute without updating the mask
   logic. The vLLM kernel takes it as a per-sequence `dynamic_causal` (1 =
   prefill), derived in the op from `context_lens == 1` (`is_prefill` is only the
   homogeneous fast path, so mixed prefill/decode batches still work). The flash
   path passes `causal=True` to its paged decode kernel: the bottom-right-aligned
   mask gives the same result.
5. **Cross-attention is dense, not paged.** Each `BartDecoderCrossAttention`
   holds `encoder_kv_cache: dict[request_id -> (k, v)]`, each
   `[num_heads, enc_len, head_dim]`, filled once in `_encode`, shared by all
   beams of a request (`Context.request_ids`) and freed in
   `ModelRunner.clear_request`. It does not count against `num_blocks`. SDPA
   reads it directly; the vLLM and flash kernels lazily stage it into a paged
   cache -- page-major for flash, head-major for vLLM -- memoized on the layer as
   `layer.cross_staged`, keyed by `(backend, tuple(request_ids))`, so it is
   rebuilt only when the active request set changes and beams never duplicate
   K/V.
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

# CUDA flash backend parity (skips without a GPU + the flash-attn2 kernel)
PYTHONPATH=. $P tests/test_gpu_flash.py

# example
PYTHONPATH=. $P examples/summarize.py
PYTHONPATH=. $P examples/florence2.py

# benchmark (sweeps batch / beams / dtype / encoder length vs HF)
PYTHONPATH=. $P benchmarks/benchmark.py                 # full
PYTHONPATH=. $P benchmarks/benchmark.py --quick         # fast smoke
PYTHONPATH=. $P benchmarks/benchmark.py --json out.json
PYTHONPATH=. $P benchmarks/benchmark.py --quick --device cuda   # on a GPU box

# Florence-2 benchmark (caption/detection/segmentation mix vs HF)
PYTHONPATH=. $P benchmarks/benchmark_florence2.py --quick
PYTHONPATH=. $P benchmarks/benchmark_florence2.py --batches 3,6,12 --beams 1,3,5
PYTHONPATH=. $P benchmarks/benchmark_florence2.py --quick --device cuda

# decode throughput (nano-vllm-style: N requests, fixed enc length + output)
PYTHONPATH=. $P benchmarks/throughput.py --requests 128 --enc-len 512 --max-tokens 512
PYTHONPATH=. $P benchmarks/throughput.py --requests 32 --enc-len 512 --max-tokens 256 --use-cudagraph
PYTHONPATH=. $P benchmarks/throughput.py --requests 64 --enc-len 512 --max-tokens 256 --compile-model

# Florence-2 token/padding accounting (8 images x 4 tasks) and profiling
PYTHONPATH=. $P benchmarks/florence2_tasks.py --device cuda [--use-cudagraph]
PYTHONPATH=. $P benchmarks/profile_florence2.py --engine ours --device cuda [--use-cudagraph]
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
| batch, beams=1 | batch=1 | 1074 | 1.08x |
| batch, beams=1 | batch=2 | 1020 | 1.23x |
| batch, beams=1 | batch=4 | 1202 | 1.21x |
| batch, beams=1 | batch=8 | 1530 | 1.26x |
| batch, beams=4 | batch=1 | 1070 | 1.44x |
| batch, beams=4 | batch=2 | 1279 | 1.54x |
| batch, beams=4 | batch=4 | 1578 | 1.90x |
| batch, beams=4 | batch=8 | 2245 | 2.31x |
| beams, batch=8 | beams=1 | 1522 | 1.27x |
| beams, batch=8 | beams=4 | 2242 | 2.32x |
| dtype, batch=8, beams=4 | float32 | 6010 | 2.29x |
| dtype, batch=8, beams=4 | bfloat16 | 2257 | 2.28x |
| enc length, batch=8, beams=1 | enc=128 | 1381 | 1.11x |
| enc length, batch=8, beams=1 | enc=256 | 1552 | 1.25x |

Snapshot of `PYTHONPATH=. $P benchmarks/benchmark_florence2.py --quick`
(`florence-community/Florence-2-base`, bf16 for both engines, 8 threads,
640×480 image → 768px, `max_tokens=96` / `beam_tokens=64`; vision tower +
projector on `torch.compile`, the new default). Refresh from a real run.

| sweep (bf16) | config | ours ms | speedup vs HF |
|---|---|---:|---:|
| greedy batch | batch=3 | 1920 | 1.32x |
| greedy batch | batch=6 | 3099 | 1.41x |
| beam (batch=3) | beams=1 | 1719 | 1.26x |
| beam (batch=3) | beams=3 | 1888 | 1.56x |
| continuous (N=6) | w=3 | 3442 | 1.30x |
| continuous (N=6) | w=6 | 3105 | 1.44x |

Snapshot of `PYTHONPATH=. $P benchmarks/benchmark.py --quick --device cuda` on
the L4 box (`bart-large-cnn`, bf16 for both engines unless noted, flash backend,
`num_blocks=64`, `block_size=256`, `max_tokens=48`). Refresh from a real run.

| sweep (bf16, enc=256 unless noted) | config | ours ms | speedup vs HF |
|---|---|---:|---:|
| batch, beams=1 | batch=1 | 510 | 1.04x |
| batch, beams=1 | batch=2 | 539 | 1.03x |
| batch, beams=1 | batch=4 | 530 | 1.05x |
| batch, beams=1 | batch=8 | 554 | 1.00x |
| batch, beams=4 | batch=1 | 546 | 1.17x |
| batch, beams=4 | batch=2 | 598 | 1.11x |
| batch, beams=4 | batch=4 | 722 | 0.98x |
| batch, beams=4 | batch=8 | 959 | 0.84x |
| beams, batch=8 | beams=1 | 569 | 0.99x |
| beams, batch=8 | beams=4 | 953 | 0.83x |
| dtype, batch=8, beams=4 | float32 | 2034 | 0.56x |
| dtype, batch=8, beams=4 | bfloat16 | 952 | 0.83x |
| enc length, batch=8, beams=1 | enc=128 | 551 | 1.00x |
| enc length, batch=8, beams=1 | enc=256 | 558 | 1.00x |

Snapshot of `PYTHONPATH=. $P benchmarks/benchmark_florence2.py --quick
--device cuda` (`Florence-2-base`, bf16 for both engines, image 640×480 → 768px,
vision tower compiled, `num_blocks=256`, `block_size=256`).

| sweep (bf16) | config | ours ms | speedup vs HF |
|---|---|---:|---:|
| greedy batch | batch=3 | 620 | 1.23x |
| greedy batch | batch=6 | 692 | 1.24x |
| beam (batch=3) | beams=1 | 438 | 1.23x |
| beam (batch=3) | beams=3 | 581 | 1.14x |
| continuous (N=6) | w=3 | 876 | 0.99x |
| continuous (N=6) | w=6 | 682 | 1.28x |

On GPU the engine matches HF on greedy BART and leads on Florence-2; it trails on
beam search (the paged cache writes new K/V separately where HF's cache append is
fused) and fp32 (FlashAttention-2 needs fp16/bf16 and the SDPA fallback is
slower).

Decode throughput (`benchmarks/throughput.py`, `bart-large-cnn`, bf16, enc=512,
EOS suppressed, so both engines decode a fixed token count; `--use-cudagraph`
captures the decode step):

| requests | max_tokens | eager tok/s | cudagraph tok/s |
|---|---:|---:|---:|
| 32 | 256 | 2543 | 3224 |
| 64 | 256 | 3945 | 3926 |
| 128 | 256 | 4518 | 4482 |
| 128 | 512 | 4346 | 4379 |

At batch ≥ 32 the decode is GPU-bound and re-reads the KV cache every step
(`flash_fwd_splitkv_kernel` is ~2/3 of the CUDA time), so eager and the CUDA
graph converge and the graph's win is small-batch latency. For calibration,
nano-vllm / Qwen3-0.6B (a 28-layer decoder, ~2.3x BART-large's) runs ~1780 tok/s
on this box with the same kind of workload.

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
- **CUDA / FlashAttention-2.** `device="cuda"` needs the `kernels` package; the
  `kernels-community/flash-attn2` wheel is downloaded once (`get_kernel`, tries
  `version=3` then `2` then unversioned) and cached under the HF cache dir, so
  the first CUDA run needs network. Flash runs fp16/bf16 only, with
  `head_dim ∈ {32,64,96,128,160,192,224,256}`, and its **paged** kernel needs a
  block size divisible by 256 — hence the CUDA default `block_size` of 256. So
  fp32, an unsupported head dim (e.g. the tiny-random-BART `head_dim=4`), or an
  explicit `block_size=16` on CUDA silently demote `"auto"` to SDPA (`"flash"`
  raises instead). The backend uses `flash_attn_with_kvcache` (page-major cache
  read through the block table) for self- and cross-attention: the dedicated
  decode kernel is ~3x cheaper per call than `flash_attn_varlen_func`, which
  matters because GPU decode here is launch-bound. The BART encoder
  self-attention uses `flash_attn_func`. There is no per-backend flag: the ops
  in `layers/attention.py` choose the kernel from the tensor device/dtype/shape,
  and `attention.set_backend` only overrides that for tests.
- **GPU decode is KV-bandwidth-bound.** Every step re-reads each sequence's
  key/value cache, so `flash_fwd_splitkv_kernel` is ~2/3 of all CUDA time (true
  for nano-vllm too). At batch ≥ 32 the engine is GPU-bound (eager ≈ CUDA-graph);
  small batches are host-bound. `benchmarks/throughput.py` reports decode tok/s
  and `benchmarks/profile_florence2.py [--use-cudagraph]` breaks it down.
- **CUDA graph (`use_cudagraph=True`, default off).** `engine/cudagraph.py`
  captures `model.decoder` once per batch-size bucket (`[1,2,4,8,16,…,
  max_num_seqs]`) into a shared pool and replays it — decode only, CUDA+flash
  path only. The caches must be static: the self cache gets one extra scratch
  row and the encoder cache is a fixed preallocated buffer, so an undersized
  batch is padded with sentinel metadata (scratch block/slot, `cache_seqlens=1`)
  whose rows write nothing real. The graph reads its inputs from the static
  buffers only — the paged store must **not** use the per-step memo, because a
  memoized temporary is computed outside the capture and freed (that caused
  illegal indices), so `_self_derived` no longer precomputes `(block, pos)`. It
  falls back to eager when `len(seqs) > max_bs` and re-captures if the
  cross-cache capacity grows (new requests admitted mid-run).
- **`torch.compile` (`compile_model=True`, default off) fuses the pointwise
  sequences but does not pay off for BART.** How nano-vllm uses it: not a
  whole-model compile, but `@torch.compile` on four tensor-only functions —
  `RMSNorm.rms_forward` / `add_rms_forward`, `SiluAndMul.forward`,
  `RotaryEmbedding.forward` and `Sampler.forward` — each of which Dynamo turns
  into one fused kernel (0 graph breaks). Measured on the L4 by patching the
  decorators out, that is worth **1.22x** for Qwen3-0.6B, on top of either mode
  (eager 892 → 1092 tok/s, with the CUDA graph 1760 → 2144). Our flag ports the
  same idea at BART's granularity: `models/bart.py` compiles `_add_layer_norm`
  (residual add + layer norm) and `_activation_linear` (gelu + fc1), the only
  pointwise sequences a layer has. Wrapping the whole `decoder` instead
  *fragments* — Dynamo breaks on every flash `fwd_kvcache` call (~24 tiny regions
  per step) and recompiled every step while guarding the per-step metadata memo.
  The helpers themselves are clean (2 recompiles, 0 graph breaks, 3 triton
  kernels, ~7% fewer launches) yet worth **~0 GPU time**: `native_layer_norm` +
  add and gelu are already single fused aten kernels, and decode is bound by
  `flash_fwd_splitkv_kernel` (~1/3) + GEMMs. Eager 3911 → 3195 tok/s at n=64
  (Dynamo's per-call wrapper across ~48 calls/step), with the graph 3925 → 3902.
  Off by default — BART has no RMSNorm / gated MLP / rotary to fuse. Two changes
  made while getting there are kept: `flash_attn.module()` is a resolved-once
  global rather than `functools.lru_cache` (Dynamo inlines an `lru_cache` body and
  would trace the Hub download), and the flash self-attention path reads the
  Context directly instead of the per-step memo.
- **BF16 SDPA needs contiguous inputs on aarch64.** PyTorch's CPU SDPA with BF16
  *non-contiguous* (transposed) inputs takes a path ~20x slower than the
  contiguous one (Graviton: 86 ms vs 4 ms for a BART-large encoder block); x86
  dispatches to the same flash kernel either way and is unaffected (slightly
  faster non-contiguous). The encoder self-attention builds q/k/v with
  `view().transpose(1, 2)`, so it calls `.contiguous()` before SDPA (the SDPA
  fallback in `layers/attention.py` does too), and so does the vendored DaViT
  window attention (`models/florence2.py`). Without it, BF16 on Graviton ran at fp32
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
  are vendored in `fastencdec/models/florence2.py`, an optimized copy of the
  `transformers` reference.)
- After changes: run `python -m compileall fastencdec`,
  `tests/test_parity.py`, `tests/test_attn_backend.py`,
  `tests/test_florence2.py`, `tests/test_beam.py`, `uvx ruff check` /
  `uvx ruff format --check`, and `benchmarks/benchmark.py --quick`.
- Every commit must update the "Benchmark baseline" table above with the real
  numbers from that `--quick` run (no stale or estimated values).
