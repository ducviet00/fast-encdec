# AGENTS.md — fast-encdec handover

Guidance for AI agents (and humans) working on this repository.

## 1. What this is

`fast-encdec` is a **small, readable inference engine for BART and Florence-2**
(CPU and CUDA), in the spirit of nano-vllm. It implements only what matters
for offline encoder-decoder decoding:

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
HuggingFace BART reference; the attention kernels differ, and the attention
q/k/v projections are fused into one GEMM (`utils/loader` concatenates the HF
parts — see the gotchas).

**Florence-2** (`fastencdec/models/florence2.py`) reuses that paged BART as its
language model and pairs it with the DaViT vision encoder
(`Florence2VisionBackbone`) and `Florence2MultiModalProjector`, vendored in the
same `fastencdec/models/florence2.py` (an optimized copy of the `transformers`
reference). Visual tokens are spliced into the encoder input at the `<image>`
placeholders; the decoder is unchanged.

### Non-goals / do not add without discussion

- No flash-attn **outside** the CUDA backend: CPU stays pure PyTorch + the
  vendored vLLM kernel.
- No quantization, prefix caching or chunked prefill.  The decode CUDA graph is
  **on by default** on CUDA with the FlashAttention-2 backend (`Config.use_cudagraph`;
  ignored elsewhere) — see `engine/cudagraph.py`.  The CPU path never uses it.
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
- CUDA benchmarking box: `ssh ducviet00@35.247.129.222` (NVIDIA L4, sm89, 24 GB,
  driver 580 / CUDA 13.0, 4 vCPU). The repo lives at `~/fast-encdec` and the env
  at `~/fast-encdec/.venv` (CUDA torch, transformers, `kernels`);
  there is no GPU on the dev box, so run `--device cuda` benchmarks there. Auth
  is flaky (public-key connections are sometimes refused) — retry. `build_essential`
  was installed for `torch.compile` of the Florence-2 vision tower.
- CPU benchmarking boxes (the README matrix): `ssh ducviet00@34.142.225.47`
  (Xeon 8581C, Emerald Rapids, `amx_bf16`, 8 vCPU = 4 cores, `~/env-ours`), `ssh
  ubuntu@54.252.169.176` (Xeon 6975P-C, Granite Rapids, `amx_bf16`, 8 vCPU = 4
  cores, `~/venv`) and `ssh ubuntu@3.26.21.19` (Graviton 5 / Neoverse-V3, 8 cores, no
  SMT, `~/venv`).  Each has the repo at `~/fast-encdec` and `car.jpg` at
  `/tmp/fastencdec/florence2_bench.jpg`.  Copy with
  `tar czf - ... | ssh box 'tar xzf - -C ~/fast-encdec'` — and all three needed
  `gcc g++ make ninja-build python3-dev` before `csrc/` would build (torch's
  cpp_extension shells out to ninja, and the Python headers must be present).
  Sync the whole tree, not just `benchmarks/`: a stale `fastencdec/` package
  silently keeps producing numbers.
- HF models are cached locally (`facebook/bart-large-cnn`,
  `hf-internal-testing/tiny-random-BartForConditionalGeneration`,
  `florence-community/Florence-2-base`, `hf-internal-testing/tiny-random-Florence2ForConditionalGeneration`).
- Run from the repo root with `PYTHONPATH=.`, or install it: `uv sync --extra
  cpu` (the project `.venv`, torch from the PyTorch CPU index) or `--extra
  cu130` for CUDA.  `uv lock --check` and `uv sync --extra cpu --dry-run`
  both pass.

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
    cudagraph.py       decode-only CUDA-graph capture/replay (on by default on CUDA)
  layers/
    attention.py       device-agnostic attention ops (encoder/self/cross) + dispatch
    cpu_attn.py        JIT build + wrapper for the vendored vLLM CPU kernels
    flash_attn.py      lazy loader for the kernels-community/flash-attn2 kernel (CUDA)
    logits_processor.py  pure-torch LogitsProcessor (built once per generate)
    packed_linear.py   oneDNN blocked-weight CPU linears (packed once at load)
    sampler.py         greedy / multinomial sampling (Sampler nn.Module)
  models/
    bart.py            BART model (encoder / decoder / cross-attn)
    florence2.py       Florence-2 (vendored DaViT + projector + BART LM)
  utils/
    context.py         device-agnostic per-step Context read by the attention ops
    loader.py          HF checkpoint -> model weights (concatenates fused q/k/v)
csrc/                  vendored vLLM CPU attention kernels (+ bindings.cpp)
examples/summarize.py
examples/florence2.py
benchmarks/benchmark.py
benchmarks/benchmark_florence2.py
benchmarks/verify_parity.py
benchmarks/throughput.py
benchmarks/attn_backend.py
benchmarks/gemm.py
benchmarks/florence2_tasks.py
benchmarks/profile_florence2.py
tests/test_parity.py
tests/test_attn_backend.py
tests/test_florence2.py
tests/test_beam.py
tests/test_packed_linear.py
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

`LLMEngine._postprocess` groups sequences by request, then batches every beam
request that shares a (beam width, live beam count, decoder length) into one
`_beam_group`: a single `log_softmax` + logits-processor call + `topk` over
`[num_requests, beam_count * vocab]`, with two `.tolist()` syncs for the whole
group.  Beam postprocess is the host bottleneck on CUDA, so this batching is
what keeps decode from being launch/sync-bound (see the L4 beam rows).

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
   model config when unset. Logits masking is pure torch
   (`layers.logits_processor.LogitsProcessor`, mirroring `transformers`'
   processor semantics), so `min_length`, `no_repeat_ngram_size`,
   `repetition_penalty`, forced BOS/EOS, suppressed tokens and the sampling warpers match `generate`. `LLM.generate`
   uses the checkpoint's `GenerationConfig` (via
   `SamplingParams.from_generation_config`) when called without params; an
   explicit `SamplingParams` fully overrides those defaults. `length_penalty`
   ranks finished beams. The whole batch shares one (stateless) mask call; greedy decoding applies it once per step to every
   request at the same decoder length.
10. **Multimodal inputs (Florence-2).** `Sequence.pixel_values` carries the
    request image `[3, H, W]`; `_encode` groups fresh sequences by
    `(encoder length, image shape)` and the model scatters projected visual
    tokens into the `image_token_id` slots. `ModelRunner` reaches the model
    through `model.decoder`, `model.decoder_layers` and `model.encode(...)`,
    which both BART and Florence-2 expose.  `LLM` decodes text with
    `skip_special_tokens=False` for multimodal models — Florence-2's answer *is*
    special tokens (`<loc_...>` carries the boxes, and
    `processor.post_process_generation` needs them) — and with `True` for BART.

## 6. Running things

```bash
# either the dev venv or the project one (uv sync --extra cpu)
P=/home/ducviet00/.venvs/torch-cpu/bin/python   # or .venv/bin/python
export HF_HUB_OFFLINE=1                        # the models are cached; the
                                               # Hub rate-limits hard (429)

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

# benchmark (scenarios x generation configs vs HF, see the baseline below)
PYTHONPATH=. $P benchmarks/benchmark.py                 # BART, cpu
PYTHONPATH=. $P benchmarks/benchmark.py --device cuda   # BART on a GPU box
PYTHONPATH=. $P benchmarks/benchmark.py --scenarios '256->256'
PYTHONPATH=. $P benchmarks/benchmark.py --json out.json

# Florence-2 benchmark (real image; caption / detection / segmentation /
# phrase-grounding tasks, EOS-stopped)
PYTHONPATH=. $P benchmarks/benchmark_florence2.py
PYTHONPATH=. $P benchmarks/benchmark_florence2.py --device cuda

# decode throughput (nano-vllm-style: N requests, fixed enc length + output)
PYTHONPATH=. $P benchmarks/throughput.py --requests 128 --enc-len 512 --max-tokens 512
PYTHONPATH=. $P benchmarks/throughput.py --requests 32 --enc-len 512 --max-tokens 256 --no-cudagraph
PYTHONPATH=. $P benchmarks/throughput.py --requests 64 --enc-len 512 --max-tokens 256 --compile-model

# attention implementation: vendored vLLM CPU kernel vs SDPA vs eager
PYTHONPATH=. $P benchmarks/attn_backend.py                    # kernel + end-to-end
PYTHONPATH=. $P benchmarks/attn_backend.py --mode kernel      # ops only, no weights
PYTHONPATH=. $P benchmarks/attn_backend.py --mode e2e --device cuda

# bf16 GEMM throughput at the engine's BART-large shapes (decode is GEMM-bound)
PYTHONPATH=. $P benchmarks/gemm.py --threads 4,8

# Florence-2 token/padding accounting (8 images x 4 tasks) and profiling
PYTHONPATH=. $P benchmarks/florence2_tasks.py --device cuda [--no-cudagraph]
PYTHONPATH=. $P benchmarks/profile_florence2.py --engine ours --device cuda [--no-cudagraph]
```

### Benchmark baseline

Per-commit baseline: `PYTHONPATH=. $P benchmarks/benchmark.py --scenarios '256->256'`
on the dev CPU (`bart-large-cnn`, bf16 for **both** engines unless noted).  The
engine and HF decode the same token count, so the speedup is engine-vs-engine.
**Every commit must refresh this table from a real run of the script — never
hand-edit or guess.**

| config | ours ms | ours tok/s | hf ms | hf tok/s | speedup |
|---|---:|---:|---:|---:|---:|
| batch=1, beams=1, bf16 | 3756 | 68.2 | 6253 | 40.9 | 1.66x |
| batch=8, beams=1, bf16 | 5270 | 388.6 | 9251 | 221.4 | 1.76x |
| batch=8, beams=4, bf16 | 8116 | 252.3 | 31434 | 65.2 | 3.87x |
| batch=8, beams=1, fp32 | 8896 | 230.2 | 24371 | 84.0 | 2.74x |

One run, with the `pack_weights` prepack on (see the gotcha).  Re-running the same
script with packing disabled in the same session read 5428 / 6153 / 9340 / 23814
ms (1.12x / 1.46x / 3.37x / 1.03x), so the prepack is worth 1.46x / 1.17x /
1.16x / **2.64x** here — it is the whole of the fp32 row, where eager
BART-large only ties HF.  The latency-bound batch=1 row varies ~5% with this
laptop's clock, so its 1.63x is the least reliable cell.  This recording is at
the x86 default
`block_size=32`; at the previous 16 it read 1.16x / 1.49x / 3.44x / 1.03x — that
page size is the greedy-vs-beam trade-off in the gotcha below.  Absolute times
are only comparable within one recording: the box ran ~20% slower here than when
the previous table was taken, which moves the single-sequence row most, so on
this box only the with-in-run speedup is meaningful.  Built with the vendored
kernel — if `csrc/` fails to build (no `ninja`, see the gotcha) the same table
reads 1.02x / 0.89x / 1.33x / 0.91x: greedy loses its edge (0.89-1.02x) and the
beams=4 win shrinks 3.34x → 1.33x, which is why `cpu_attn.module()` is worth
checking before trusting any CPU number.  With the kernel this laptop is the
*fastest* box at these points (8 cores, high clocks), which is why its numbers
beat the 4-core Xeons' in the README matrix.

`benchmark_florence2.py` runs the same configs on a real 640x480 photo
(`car.jpg`, scp'd to `/tmp/fastencdec/` so the boxes never hit the Hub) and a
four-task mix -- caption / detection / referring-expression segmentation /
caption-to-phrase grounding -- so batch=8 and beams=4 divide it evenly.  EOS is
allowed, so the output lengths are the tasks' own (13/22/256/22, the
segmentation capped at `--max-tokens`); under beam search the two engines can
pick slightly different hypotheses, which the script reports instead of
failing.

The README carries the full three-point matrix (`128->512`, `512->128`,
`256->256`) for the four server boxes.  Each cell is the mean of two independent
runs, and every run writes its raw results with `--json`, so the numbers can be
re-derived rather than trusted (an audit that recomputes all 16 rows from those
JSONs and diffs them against the README passes).

What to expect from those tables: on CPU greedy is 0.9-2.0x (the encoder is
shared work and dominates the 512->128 point, where the Xeons sit at 0.94-0.95x)
and beam search is 1.7-3.9x, because the paged cache shares blocks where HF
reorders a full beam cache every step.  On CUDA greedy is 2.4-3.6x - the decode
CUDA graph removes the per-step launch overhead that dominated small batches
(eager it was ~1.05x) - while beam search is 2.0-4.2x.  fp32 trails on CUDA
(there is no FlashAttention-2 path).

Decode throughput (`benchmarks/throughput.py`, `bart-large-cnn`, bf16, enc=512,
EOS suppressed, so both engines decode a fixed token count; the graph is on by
default, `--no-cudagraph` disables it):

| requests | max_tokens | eager tok/s | cudagraph tok/s |
|---|---:|---:|---:|
| 32 | 256 | 2674 | 3423 |
| 128 | 256 | 4575 | 4657 |

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
- **Fair benchmarks.** `benchmark.py` runs both engines at the same dtype
  (never fp32-for-HF vs bf16-for-us) and both decode exactly the scenario's
  token count (EOS suppressed via `min_length=max_tokens+1` / HF
  `min_new_tokens`); it raises if the two engines decode different token
  totals. Same precision, the engine leads both greedy and beam search (see
  the baseline tables). `benchmarks/verify_parity.py` confirms the
  fairness directly: fp32 is token-for-token identical to HF, and bf16 keeps the
  output length with occasional single-step token differences from the different
  attention kernels (the same 12/12 pattern before and after the q/k/v fusion).
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
- **CPU packed linears (`pack_weights`, on by default).** At load, every CPU
  `nn.Linear` weight is rewritten once with
  `torch.ops.mkldnn._reorder_linear_weight` and run through
  `torch.ops.mkldnn._linear_pointwise` — oneDNN on x86, oneDNN-ACL on aarch64,
  the same pair Inductor's weight-prepack pass emits.  That removes the per-call
  weight reorder `F.linear` pays and is worth ~1.3x (Xeon Emerald) / ~1.55x
  (Graviton Neoverse-V3) end-to-end at `256->256, batch=8`; decode is the
  bandwidth-bound part, so the encoder and the fp32 rows gain less.  It is always
  on for CPU and logs the count; a host without the ops keeps `F.linear`.  The
  original `weight` is kept, so weight memory doubles.
- **Box choice / threads** (`benchmarks/gemm.py`, bf16 addmm at the engine's
  shapes).  The 4-core Xeons' AMX only pays on the large prefill GEMMs, and SMT
  *hurts* it there: Emerald does 5455 GFLOP/s on `[2048x1024]x[1024x4096]` at 4
  threads but 3051 at 8 (Granite 6263 -> 2737), while the 8-core boxes nearly
  double.  At decode shapes (one token per sequence) the GEMMs are
  memory/latency-bound, so AMX buys nothing and physical cores win: 8-thread
  `[8x1024]x[1024x4096]` gives Ryzen 940, Graviton 754, Granite 520, Emerald 471
  GFLOP/s — which is why the engine's decode benchmarks rank Graviton above the
  Xeons.  The benchmark scripts therefore default to `--threads 0` = torch's
  default = the number of *physical* cores (8 on the dev box, 4 on the Xeons).
  For the *engine* itself 8 threads is still defensible on the Xeons (it has
  non-GEMM parallel work too): Emerald's decode-heavy run is 436 tok/s at 8
  threads vs 411 at 4.
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
  `benchmarks/attn_backend.py` measures what the kernel is worth: at the default
  page size on the four CPU boxes (bf16) it is **~4-75x faster per decoder op**
  than the SDPA/"eager" fallbacks (75x at batch=8/ctx=256 on the 8-core laptop,
  ~10-21x at batch=32/ctx=1024, where the fallback's per-sequence loop
  amortizes) and **+15% to +65% end-to-end** over `attn_backend="sdpa"`; versus
  HF the engine's margin is the one in the benchmark matrix above (0.97-3.79x
  depending on the point and the beam width).
  The CPU SDPA fallback is itself slower than HF (0.75-0.94x) because it attends
  one sequence at a time (CUDA uses the batched path), so `"auto"` resolving to
  the kernel matters a lot.  **Check `cpu_attn.module()` before trusting CPU
  numbers**: if `csrc/` cannot build (no `ninja`, no Python headers) the engine
  silently degrades to SDPA and gets *slower than HF* (0.75-0.94x) with no error
  — `ninja` is now in the `cpu` extra for exactly this reason, and a bare
  `uv sync` without it is the usual cause.
- **CPU page size (`block_size`).** `block_size=None` picks the smallest page the
  device's best kernel accepts (`fastencdec.config.default_block_size`): 256 on
  CUDA, **32 on x86**, 16 on aarch64.  `csrc`'s full-width AVX-512 kernel
  (`select_isa` → `"vec"`) only engages when `block_size % 32 == 0`, so the
  old x86 default of 16 silently ran the half-width `"vec16"` one and left ~10%
  of the attention kernel unused (`benchmarks/attn_backend.py --mode kernel` on
  Emerald, batch=8: self/ctx=256 113.0 → 103.6 µs, cross/enc=1024 428.0 →
  390.9 µs, and every other shape 8-11% faster).  It is not free: two costs grow
  with the page size, both measured on the Xeons — the vendored kernel writes K
  *column-major with stride `block_size`*, so the per-step KV store went 56 →
  116 µs/call (Emerald), and a beam-search copy-on-write copies a whole page,
  165 → 283 µs/event (copying only the used slots does *not* help: the copy is
  dispatch-bound).  Net effect at 32 on x86 (`benchmarks/benchmark.py`, two runs
  per box): greedy gains 0.02-0.08 of speedup (Emerald `512->128, batch=8,
  beams=1` 0.95x → 1.00x, Granite `128->512, batch=8, beams=1` 1.10x → 1.18x)
  while beams=4 loses 0.03-0.09 (2.52x → 2.42x, 3.91x → 3.79x, 2.21x → 2.14x on
  Emerald).  aarch64 is unaffected (16 is already enough for `"neon"`, and a
  smaller page is a cheaper copy).  The same `num_blocks` budget now covers
  twice the KV tokens (and uses twice the memory).
- **CUDA / FlashAttention-2.** `device="cuda"` needs the `kernels` package; the
  `kernels-community/flash-attn2` wheel is downloaded once (`get_kernel`, tries
  `version=3` then `2` then unversioned) and cached under the HF cache dir, so
  the first CUDA run needs network. Flash runs fp16/bf16 only, with
  `head_dim ∈ {32,64,96,128,160,192,224,256}`, and its **paged** kernel needs a
  block size divisible by 256 — hence the CUDA default `block_size` of 256. So
  fp32, an unsupported head dim (e.g. the tiny-random-BART `head_dim=4`), or an
  explicit `block_size=16` on CUDA silently demote `"auto"` to SDPA (`"flash"`
  raises instead).  So does an unavailable *kernel*: with `HF_HUB_OFFLINE=1`
  and a cold ``kernels`` cache, `get_kernel` fails even though the Hub is
  reachable, `flash_attn.module()` is `None`, `"auto"` runs SDPA and the decode
  CUDA graph switches off with it — on the L4, `batch=8, beams=1, bf16` then
  measured 347 tok/s where the flash path gives 1817.  Check
  `flash_attn.module()` before trusting CUDA numbers, exactly as
  `cpu_attn.module()` gates the CPU ones.
  The backend uses `flash_attn_with_kvcache` (page-major cache
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
  and `benchmarks/profile_florence2.py [--no-cudagraph]` breaks it down.
- **CUDA graph (on by default on CUDA + flash).** `engine/cudagraph.py`
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
  cross-cache capacity grows (new requests admitted mid-run).  The buffers are
  rebuilt — and therefore the capture redone — **only when `bufs is None`**, i.e.
  after `_alloc_cross` grew the cross cache; a new request set with the same
  capacity just rewrites the cross rows in place, so the capture survives across
  `generate()` calls.  (Rebuilding the buffers for every new request set
  re-captured all buckets on every call: ~100 ms of the L4's 176-641 ms rows.)
- **`torch.compile` (`compile_model=True`, default off) fuses the pointwise
  sequences but does not pay off for BART.** How nano-vllm uses it: not a
  whole-model compile, but `@torch.compile` on four tensor-only functions —
  `RMSNorm.rms_forward` / `add_rms_forward`, `SiluAndMul.forward`,
  `RotaryEmbedding.forward` and `Sampler.forward` — each of which Dynamo turns
  into one fused kernel (0 graph breaks). Measured on the L4 by patching the
  decorators out (n=64, out=64, eager): 119876 → 26116 kernel launches (**−78%**),
  2334 → 1743 ms CUDA, and it is worth **1.22x** on top of either mode (eager
  892 → 1092 tok/s, with their CUDA graph 1760 → 2144). Our flag ports the
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
  Off by default — BART has no RMSNorm / gated MLP / rotary to fuse (ours saves
  7% of launches, 16161 → 14985, for ~0 GPU time). **Revisit it when a T5 model
  lands:** T5 is `T5LayerNorm` (RMS) + a relative-position bias + a gated FFN
  (`T5DenseGatedActDense`, T5 v1.1), i.e. the same hand-written multi-kernel
  elementwise chains (6-8 kernels each) that inductor collapses into one — the
  reason nano-vllm gets 1.22x. Wire `_add_layer_norm`/`_activation_linear`
  equivalents (or new helpers) for the T5 layers and re-measure; keep the
  plumbing until then. Two changes
  made while getting there are kept: `flash_attn.module()` is a resolved-once
  global rather than `functools.lru_cache` (Dynamo inlines an `lru_cache` body and
  would trace the Hub download), and the flash self-attention path reads the
  Context directly instead of the per-step memo.
- **`torch.compile` + graph freezing on CPU.** The Inductor weight-prepack
  recipe (`torch._inductor.config.cpp.weight_prepack = True`,
  `config.freezing = True`, then `torch.compile(model)`) does work, but only on
  models that are pure torch: on a plain `Linear+GELU+Linear` it emits
  `mkldnn::_linear_pointwise` and gives 1.48x (bf16) / 2.28x (fp32) on the Xeon
  and 1.67x / 1.97x on Graviton.  Two traps sit in front of it: freezing only
  happens when `not torch.is_grad_enabled()`, so the forward must be under
  `torch.no_grad()` (`torch/_inductor/compile_fx.py`), and `recompile_limit`
  (default 8) is below the 12 decoder layers, which silently drops back to
  eager.  It does not pay off on this engine: the attention is a pybind op that
  also reads the module-level `Context`, so Dynamo graph-breaks there — its own
  remedy is "wrap the custom kernel into an opaque custom op" — and the ~84
  subgraphs per step cost more than the packing buys (`256->256, batch=8`: eager
  6700 ms, compile without freezing 7490, compile + freezing 6950, manual
  prepack 5150).  `torch._dynamo.allow_in_graph` is not enough either: Dynamo
  still calls the wrapped function on fake tensors, which the kernel cannot run.
  Registering the attention as a `torch.library.custom_op` with
  `register_fake` *does* work — the whole decoder then compiles into one graph
  and lands within 2-14% of the manual prepack (5250 ms Xeon, 5030 ms Graviton,
  tokens identical) — but `decoder_cross_attention` takes the attention module
  as an argument, which cannot be a custom-op input, so it needs re-plumbing to
  pass the staged tensors.  CPU post-op fusion is not worth the plumbing either:
  `linear+gelu` (`_linear_pointwise(..., "gelu", [-1], "none")`, exact like
  `F.gelu`) saves 0.002-0.005 ms per `fc1`, under 1% of a step — which is why
  `compile_model` (Inductor fusing `_add_layer_norm`/`_activation_linear`)
  measures ~7% *slower* on CPU.  Florence-2 is the exception that proves the
  rule: `compile_mm_encoder` compiles the (pure-torch) DaViT tower fine, the
  packed forward is `torch.compiler.disable`d so Inductor never sees the opaque
  blocked weight (it fails on its `stride=[1, 0]` otherwise), which keeps the
  linear out of the graph — yet `compile + prepack` is still the fastest
  `get_image_features` (356 ms vs 369 compiled-unpacked vs 394 eager at 577
  tokens).
- **Fused q/k/v projections.** Each self-attention holds one `qkv_proj`
  (`[D -> 3D]`) where HF has `q_proj`/`k_proj`/`v_proj`, and cross-attention
  holds `kv_proj` + `q_proj` (K and V are both projected from the encoder
  output, so they share a GEMM). Three small `[D -> D]` GEMMs are launch-bound,
  so one `[D -> 3D]` GEMM is ~3x faster at decode shapes (B=64 on the L4:
  57 → 18 us) and at parity for the encoder's wide GEMMs (B=16384: 1.01x).
  `utils/loader` stacks the HF parts before its name match, driven by each
  module's `fused_projections` class attribute, so the loader stays
  model-agnostic. Measured effect: ~5% fewer kernel launches per step
  (16161 → 15357 at n=32/out=32) and +1-2% wall at batch 8-64, but ~0 at
  batch ≥ 128 where decode is GPU-bound. fp32 output stays token-for-token
  identical to HF (`verify_parity.py`).
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
- On **CPU** the SDPA fallback attends one sequence at a time, because the
  batched block-diagonal path (`_sdpa_paged_batched`) is only taken on CUDA;
  a batched CPU path would help there (this is why the vendored vLLM kernel
  matters so much on CPU).
- **T5** (encoder-decoder like BART, so the paged cache + continuous batching
  carry over): `T5LayerNorm` is RMS-without-mean, T5 v1.1 has a gated FFN
  (`T5DenseGatedActDense`) and T5 uses a relative-position bias instead of
  learned positional embeddings. Those per-layer pointwise ops are the
  multi-kernel kind `torch.compile` collapses, so re-measure `compile_model`
  there (see the gotcha) — BART has nothing to fuse, T5 does.

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
  `tests/test_florence2.py`, `tests/test_beam.py`,
  `tests/test_packed_linear.py`, `uvx ruff check` /
  `uvx ruff format --check`, and `benchmarks/benchmark.py --scenarios '256->256'`.
- Every commit must update the "Benchmark baseline" table above with the real
  numbers from that run (no stale or estimated values).  The full three-point
  matrix in the README is refreshed when the tables or the engine change
  materially.
