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
- **Frozen `torch.compile` on CPU** (with the optional **CUDA** path for the
  decoder attention). One dynamic graph covers decoder + lm head and one the
  encoder; Inductor weight-freezing packs every GEMM into oneDNN's blocked
  layouts (BF16 `avx512_core_bf16` on modern x86, ACL on aarch64); on CUDA the
  decoder self-/cross-attention (and the BART encoder self-attention) run on
  **FlashAttention-2**.

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
benchmarks/benchmark_vllm.py
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

# token-by-token parity vs HF on the real checkpoints (fp32 exact; bf16 same length).
# fp32 needs ~8 GB above loading both engines: each (model, beams) phase runs in
# its own subprocess, and fp32 KV budgets are snapped down to fit an 11 GB box.
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

# ours vs vLLM on CUDA (one process per engine: different venvs).  vLLM needs
# the bart-plugin; the beam rows additionally need vllm-beam-search-plugin.
# VLLM_USE_FLASHINFER_SAMPLER=0 because there is no nvcc on the L4 box.  The
# script picks VLLM_PLUGINS (bart for greedy, bart+beam_search for beam), and
# --batches runs a greedy-only batch sweep.  Raw JSONs merge into one table.
PYTHONPATH=. .venv/bin/python benchmarks/benchmark_vllm.py --engine ours --json ours.json
HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
    ~/vllmenv/bin/python benchmarks/benchmark_vllm.py --engine vllm --json vllm.json
python benchmarks/benchmark_vllm.py --compare ours.json vllm.json
# greedy batch sweep (beam plugin not loaded):
PYTHONPATH=. .venv/bin/python benchmarks/benchmark_vllm.py --engine ours \
    --batches 1,2,4,8,16,32 --json ours_greedy.json

# ours vs vLLM on CPU (`--device cpu`): greedy only — see the vLLM gotcha.
# The vLLM side needs the *pre-MRV2* BART plugin (v0.5.0) on a vLLM whose CPU
# build still has the V1 model runner (0.24.0+cpu from the v0.24.0 GitHub
# release wheel, deps from the PyTorch CPU index), in its own venv:
PYTHONPATH=. ~/venv/bin/python benchmarks/benchmark_vllm.py --engine ours \
    --device cpu --configs 'batch=1, beams=1, bf16;batch=8, beams=1, bf16;batch=8, beams=1, fp32' --json ours.json
HF_HUB_OFFLINE=1 ~/vllm024/bin/python benchmarks/benchmark_vllm.py --engine vllm \
    --device cpu --configs 'batch=1, beams=1, bf16;batch=8, beams=1, bf16;batch=8, beams=1, fp32' --json vllm.json
python benchmarks/benchmark_vllm.py --compare ours.json vllm.json

# Florence-2 benchmark (real image; caption / detection / segmentation /
# phrase-grounding tasks, EOS-stopped)
PYTHONPATH=. $P benchmarks/benchmark_florence2.py
PYTHONPATH=. $P benchmarks/benchmark_florence2.py --device cuda

# decode throughput (nano-vllm-style: N requests, fixed enc length + output)
PYTHONPATH=. $P benchmarks/throughput.py --requests 128 --enc-len 512 --max-tokens 512
PYTHONPATH=. $P benchmarks/throughput.py --requests 32 --enc-len 512 --max-tokens 256 --no-cudagraph
PYTHONPATH=. $P benchmarks/throughput.py --requests 64 --enc-len 512 --max-tokens 256

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
| batch=1, beams=1, bf16 | 3433 | 74.6 | 5800 | 44.1 | 1.69x |
| batch=8, beams=1, bf16 | 4999 | 409.6 | 8949 | 228.9 | 1.79x |
| batch=8, beams=4, bf16 | 7926 | 258.4 | 31790 | 64.4 | 4.01x |
| batch=8, beams=1, fp32 | 8673 | 236.1 | 25472 | 80.4 | 2.94x |

This recording is the frozen `torch.compile` default: one dynamic decode graph
(decoder + lm head) and one dynamic encoder graph, with Inductor freezing packing
every GEMM; the old manual `pack_weights` prepack is gone (see the CPU GEMM
packing gotcha).  The fp32 row compiles too and verifies token-exact vs HF
(`verify_parity.py --dtype float32`).  Absolute times shift between recordings
(~5% with this laptop's clock, worst in the latency-bound `batch=1` row), so
only the within-run speedup is meaningful; the `beams=4` win over HF (4x) is the
stable feature of the table.

Built with the vendored kernel — if `csrc/` fails to build (no `ninja`, see the
gotcha) `auto` falls back to SDPA and the compiled gate turns off with it, so
every row reads like the old unpacked-eager numbers (0.75-1.38x), which is why
`cpu_attn.module()` is worth checking before trusting any CPU number.  This
laptop still has the largest *speedup* at these points (8 cores, high clocks,
and a much slower HF baseline) even though the Xeons decode more tokens/s in
absolute terms.

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

`benchmarks/benchmark_vllm.py` is the same comparison against **vLLM** on the L4
(bf16, both engines on their CUDA graph; vLLM 0.28.0 plus `vllm-bart-plugin`, and
`vllm-beam-search-plugin` for the beam rows only; one process per engine because
they live in different venvs).  Each engine decodes its own token count - vLLM's
beam path appends one trailing EOS - and the speedup is `vLLM ms / ours ms`, so
>1 is ours ahead.  Greedy is measured with the beam plugin *uninstalled* (it swaps
in a custom scheduler that disables async scheduling - see the gotcha):

| point | config | ours ms | ours tok/s | vLLM ms | vLLM tok/s | speedup |
|---|---:|---:|---:|---:|---:|---:|
| 128->512 | batch=1, beams=1 | 1457 | 351 | 1397 | 366 | 0.96x |
| 128->512 | batch=8, beams=1 | 1834 | 2233 | 1685 | 2431 | 0.92x |
| 128->512 | batch=8, beams=4 | 3197 | 1281 | 3458 | 1187 | 1.08x |
| 128->512 | batch=8, beams=1, fp32 | 12011 | 341 | 3355 | 1221 | 0.28x |
| 512->128 | batch=1, beams=1 | 372 | 344 | 361 | 355 | 0.97x |
| 512->128 | batch=8, beams=1 | 560 | 1828 | 517 | 1980 | 0.92x |
| 512->128 | batch=8, beams=4 | 900 | 1138 | 1058 | 975 | 1.18x |
| 512->128 | batch=8, beams=1, fp32 | 3190 | 321 | 985 | 1040 | 0.31x |
| 256->256 | batch=1, beams=1 | 727 | 352 | 699 | 366 | 0.96x |
| 256->256 | batch=8, beams=1 | 950 | 2155 | 870 | 2353 | 0.92x |
| 256->256 | batch=8, beams=4 | 1737 | 1179 | 1761 | 1167 | 1.01x |
| 256->256 | batch=8, beams=1, fp32 | 6026 | 340 | 1723 | 1189 | 0.29x |

Greedy alone, sweeping the batch size (`--batches 1,2,4,8,16,32`, no beam plugin):

| point | batch | ours ms | ours tok/s | vLLM ms | vLLM tok/s | speedup |
|---|---:|---:|---:|---:|---:|---:|
| 128->512 | 1 | 1457 | 351 | 1397 | 366 | 0.96x |
| 128->512 | 2 | 1567 | 654 | 1475 | 694 | 0.94x |
| 128->512 | 4 | 1676 | 1222 | 1510 | 1356 | 0.90x |
| 128->512 | 8 | 1834 | 2233 | 1685 | 2431 | 0.92x |
| 128->512 | 16 | 2273 | 3604 | 2094 | 3913 | 0.92x |
| 128->512 | 32 | 3170 | 5168 | 2977 | 5503 | 0.94x |
| 512->128 | 1 | 372 | 344 | 361 | 355 | 0.97x |
| 512->128 | 2 | 420 | 609 | 397 | 644 | 0.95x |
| 512->128 | 4 | 460 | 1112 | 441 | 1162 | 0.96x |
| 512->128 | 8 | 560 | 1828 | 517 | 1980 | 0.92x |
| 512->128 | 16 | 779 | 2630 | 693 | 2953 | 0.89x |
| 512->128 | 32 | 1230 | 3330 | 1051 | 3896 | 0.85x |
| 256->256 | 1 | 727 | 352 | 699 | 366 | 0.96x |
| 256->256 | 2 | 797 | 642 | 747 | 686 | 0.94x |
| 256->256 | 4 | 855 | 1197 | 775 | 1321 | 0.91x |
| 256->256 | 8 | 950 | 2155 | 870 | 2353 | 0.92x |
| 256->256 | 16 | 1202 | 3409 | 1100 | 3725 | 0.92x |
| 256->256 | 32 | 1729 | 4737 | 1563 | 5242 | 0.90x |

The shape is the HF comparison shrunk down.  vLLM's single-token decode loop is
3-9% leaner in bf16, so greedy loses 0.92-0.97x; the gap widens with batch
(`512->128, batch=32` is 0.85x) and is ~0 at `batch=1`, where both are
latency-bound.  Block sharing between beams turns greedy's loss back into a small
win at `beams=4` (1.01-1.18x) where the plugin copy-on-writes a whole beam group
every step.  fp32 is the one-sided loss (0.28-0.31x): with no FlashAttention-2
path the engine also loses the decode CUDA graph that depends on it, so a batch=8
decode runs at the batch=1 rate (launch-bound), while vLLM keeps fp32 on
`TRITON_ATTN`.  Each run is one process per engine on an idle L4, warmup 1 + 3
repeats, best-of-3; the raw JSONs are written per row so the numbers can be
re-derived.

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
- **CPU GEMM packing: frozen `torch.compile`, always on (CPU, fp32/bf16, when
  the vendored kernel is active).** There is no manual prepack anymore.  When
  `compile_decoder` holds (`device.type == "cpu"` + dtype
  float32/bfloat16 + `backend == "vllm"`), the runner `torch.compile`s **two
  graphs**: `_DecodeStep` (decoder + lm head, one graph — lm-head GEMMs pack
  too) via opaque attention custom ops, and `model.encode` for the encoder
  (vision tower included for Florence-2).  Inductor's freezing pass packs all
  of them into `_linear_pointwise` (oneDNN blocked weights; ACL on aarch64),
  the same op the old manual prepack fed; the artifact also fuses pointwise
  chains (`linear+gelu`, residual-add into the linear epilogue) for free.  fp32
  compiles as well and — measured — stays **token-for-token identical to HF**
  (`mkl`/pointwise fp32 GEMM numerics are exact enough), so fp32 lost nothing.
  See `TORCH_COMPILE_FREEZING.md` for how the gate chain works
  (`compile_fx.py`: freezing needs `not torch.is_grad_enabled()`, so forwards
  run under `torch.no_grad()`).  Two rules: (1) the CPU SDPA fallback is NOT
  compiled — it reads the per-step `Context` globals, which tracing would bake
  in — so if `csrc/` cannot build, CPU is plain unpacked eager and
  *slower than HF* (0.75-0.94x SDPA-loop penalty; check `cpu_attn.module()`);
  (2) shapes must stay dynamic: a static graph **duplicates every frozen weight
  per compiled shape** (measured +0.74 GB bf16 / +1.6 GB fp32 per artifact for
  bart-large, per batch size or encoder length), so the decode graph is
  `dynamic=True` for one artifact across all batch sizes, and the BART encode
  graph is dynamic over batch+length; the Florence-2 encode graph must stay
  **static** because Inductor's dynamic-conv lowering mis-strides DaViT's
  windowed feature maps (`assert_tensor_metadata` failures) — acceptable there,
  since image shapes (and usually batch, within an LLM instance) are fixed.
  First use pays a one-time Inductor compile per graph (artifacts cached in
  `/tmp/torchinductor_*` across processes, and the FX cache can **replay a
  graph without running the trace** — no trace-time side effects allowed; see
  the FX-cache gotcha), and parallel workers are capped at 4 to avoid a GB-scale
  compile memory spike on small boxes.
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
- **`compile_mm_encoder`** (default True, **CUDA-only** since the CPU path
  compiles the whole `encode` call — vision tower included — as part of its
  encoder graph; see the CPU GEMM packing gotcha) `torch.compile`s the vision
  tower + projector (~1.2x there, ~1.13x end-to-end after the vendored eager
  tower).  Inductor's BF16 fusion shifts outputs, so greedy tokens can diverge
  from HF; pass `compile_mm_encoder=False` for exact parity or to avoid the
  per-shape compile latency (it compiles once per batch size on first use).
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
- **vLLM comparison (`benchmarks/benchmark_vllm.py`).** vLLM serves BART only
  through two plugins: `vllm-bart-plugin` (encoder-decoder model + the
  `encoder_prompt`/`decoder_prompt` dict prompt) and `vllm-beam-search-plugin`
  (`scheduler_cls=vllm_beam_search.scheduler.BeamSearchScheduler`, per-request
  `SamplingParams(extra_args={"beam_width": n})`).  Both pin vLLM to
  `>=0.24,<0.32`; the L4 box uses `vllm==0.28.0` in `~/vllmenv` (ours stays in the
  repo `.venv`), and `VLLM_USE_FLASHINFER_SAMPLER=0` is required (no nvcc, so
  FlashInfer's JIT sampler fails).  **Load the beam plugin only for beam runs**: it
  replaces vLLM's scheduler with a custom one that disables async scheduling, which
  cost plain greedy ~1% at batch 8 (and ~0.4% in fp32) on the L4.  The script picks
  `VLLM_PLUGINS` (`bart` vs `bart,beam_search`) and the scheduler from the requested
  configs, so a greedy (`--batches`) run never loads it.  Run one process per engine
  and merge the two JSONs with `--compare`.  What to expect: in bf16 vLLM's
  single-token decode is 3-9% leaner than ours (greedy 0.85-0.97x, widest with
  batch), our block sharing wins beam back (1.01-1.18x at `beams=4`), and fp32 is a
  ~3.4x loss because fp32 has no FlashAttention-2 path, so it also loses the decode
  CUDA graph and goes launch-bound (a batch=8 decode runs at the batch=1 rate).
  vLLM's beam path appends a trailing EOS, so it decodes N+1 tokens against our
  exactly-N; the reported speedup is `vLLM ms / ours ms`.
- **vLLM on CPU: greedy only.**  `--device cpu` compares the two engines on a CPU
  box, but **only for `beams=1`**: vLLM's CPU wheels disable Triton ("Triton is
  installed, but doesn't include CPU backend"), and vLLM gates its V2 model runner
  on it (`ValueError: Model Runner V2 requires Triton.`, `vllm/config/vllm.py`).
  `BeamSearchScheduler.__init__` raises `requires MRV2 with async scheduling
  enabled` without it, and `vllm-bart-plugin` ≥ v0.6 requires V2 outright, so
  current BART plugins cannot even *load* BART on CPU.  The working combination is
  **vLLM 0.24.0+cpu + `vllm-bart-plugin` v0.5.0** (the last release that only
  registers the model class and runs on the V1 model runner) + the beam plugin
  installed but never used; `--engine vllm --device cpu` therefore rejects
  `beams>1` instead of silently degrading to greedy.  The CPU backend sizes its KV
  pool from `VLLM_CPU_KVCACHE_SPACE` (the script defaults it to 8 GiB), not from
  `gpu_memory_utilization`.  Beam search on CPU has no vLLM reference - compare it
  against HF with `benchmarks/benchmark.py` instead.  What to expect on CPU (Graviton
  5, aarch64, 8 cores, bf16): ours leads every greedy row at 1.03-4.8x.  The wide
  end is `512->128`, where vLLM's CPU path barely batches (68 -> 120 tok/s from
  `batch=1` to 32, against 129 -> 571 for ours); the narrow end is `128->512`
  (1.03-1.27x), where a single-token decode dominates and both engines do the same
  bandwidth-bound GEMM work.  fp32 is ~1.1x for the same reason.  Ours' own
  `beams=4` rows on that box are 386 / 273 / 362 tok/s at `128->512` / `512->128` /
  `256->256`.
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
- **`torch.compile` does not pay off for BART on CUDA.** How nano-vllm uses it:
  not a whole-model compile, but `@torch.compile` on four tensor-only functions —
  `RMSNorm.rms_forward` / `add_rms_forward`, `SiluAndMul.forward`,
  `RotaryEmbedding.forward` and `Sampler.forward` — each of which Dynamo turns
  into one fused kernel (0 graph breaks). Measured on the L4 by patching the
  decorators out (n=64, out=64, eager): 119876 → 26116 kernel launches (**−78%**),
  2334 → 1743 ms CUDA, worth **1.22x** on top of either mode. Ported at BART's
  granularity it was worth nothing: the helpers compiled clean (2 recompiles, 0
  graph breaks, ~7% fewer launches) but `native_layer_norm`+add and gelu are
  already single fused aten kernels, and decode is bound by
  `flash_fwd_splitkv_kernel` (~1/3 of CUDA time) + GEMMs — eager 3911 → 3195
  tok/s from Dynamo's own per-call wrapper overhead; whole-decoder tracing
  fragments on every flash `fwd_kvcache` call (~24 regions/step). The per-helper
  wiring (`bart.enable_compile`, compiled `_add_layer_norm`/`_activation_linear`
  variants) is removed; both helpers are plain eager again. Revisit on CUDA only
  when a T5 model lands. Two changes from that work are kept:
  `flash_attn.module()` is a resolved-once global rather than
  `functools.lru_cache` (Dynamo inlines an `lru_cache` body and would trace the
  Hub download), and the flash self-attention path reads the Context directly
  instead of the per-step memo.
- **CPU: frozen `torch.compile` IS the packing path (always on).** The decode
  (decoder + lm head via `_DecodeStep`) and BART-encode graphs trace as one
  graph each because the vLLM CPU attention branches are opaque
  `torch.library.custom_op`s (`fastencdec::cpu_vllm_self_attention`,
  `::cpu_vllm_cross_attention`, in `layers/attention.py`, `register_fake`
  impls, `mutates_args` covering the self caches) — Dynamo never traces the
  pybind module, the Context reads, or the cross staging, so it is **one graph,
  zero breaks** and Inductor freezing (patched on around every compiled call via
  `torch._inductor.config.patch(freezing=True)`) packs all GEMMs into
  `_linear_pointwise` (decode artifact: 72 calls, 0 `aten.addmm`; even
  `linear+gelu` fuses into the epilogue). History that validates the design:
  compile+freezing on the *unplumbed* decoder broke on every pybind call — the
  remaining census after hoisting the ISA probes (`TORCH_COMPILE_FREEZING.md§4.2`)
  was ~53, all pybind, 0.91x eager (6467 vs 5860 ms) — which is what drove the
  custom-op re-plumb; `allow_in_graph` is not enough (Dynamo calls the function
  on fake tensors); and `layers/cpu_attn.py` resolves its capability flags once
  into module globals and caches the built module in a plain global — do not put
  the per-call dispatch back on `torch.cpu._is_avx512_supported` or `lru_cache`
  calls (Dynamo traces into each once per call site). The old manual
  `packed_linear` prepack (now deleted) sat ~5-9% behind the compiled rows.
  fp32 compiles the same way and verifies token-exact vs HF. The per-shape
  hazard is real (static artifacts duplicate frozen weights — see the CPU GEMM
  packing gotcha), so the graphs are dynamic; CPU post-op fusion is left to
  Inductor: `fallback` hand-fusions like `_linear_pointwise(..., "gelu", [-1])`
  save only 0.002-0.005 ms per `fc1`, which is why per-helper fusion measured
  ~7% *slower* on CPU. Florence-2's DaViT tower is pure torch, so the whole
  Florence-2 `encode` compiles on CPU too — static shapes only (dynamic convs
  mis-stride, see the packing gotcha); the old `packed forward`
  `torch.compiler.disable` workaround is gone with the manual prepack.
- **The Inductor FX graph cache replays without running the trace.**
  `/tmp/torchinductor_*` caches compiled artifacts across processes; a cached
  graph can be **replayed without ever executing the Python of a fresh Dynamo
  trace**.  Any trace-time side effect is therefore invisible on a cache hit —
  the concrete failure: the cross-attention layer registry used to register
  lazily inside the traced `decoder_cross_attention`, a cached replay called
  `fastencdec::cpu_vllm_cross_attention.default(..., 0, ...)` with a baked
  handle while the registry was still empty (`IndexError: list index out of
  range`).  Rule: no module/global/attribute mutation inside traced code;
  state must be established in `__init__` or at engine load
  (`register_cross_layer` is called from `BartDecoderCrossAttention.__init__`,
  and the lesson generalizes to anything Dynamo-visible).  When debugging a
  cached-replay failure, `rm -rf /tmp/torchinductor_<user>` first — a cache
  hit hides the trace path entirely.
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
  multi-kernel kind the frozen compile graphs collapse, so T5's per-step CPU
  gains would exceed BART's (BART had ~0 fusion value; T5 has real pointwise
  chains to fuse) — the always-on compile path would do the fusing
  automatically.

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
  `uvx ruff check` / `uvx ruff format --check`, and
  `benchmarks/benchmark.py --scenarios '256->256'`.
- Every commit must update the "Benchmark baseline" table above with the real
  numbers from that run (no stale or estimated values).  The full three-point
  matrix in the README is refreshed when the tables or the engine change
  materially.
