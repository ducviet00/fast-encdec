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
the attention kernels are swapped.

## Layout

```
fastencdec/
  context.py       # per-step metadata read by attention layers
  sequence.py      # Sequence / SamplingParams
  block_manager.py # paged blocks: allocate / fork / copy-on-write / free
  attention.py     # CPU paged self-attention + cached cross-attention
  models/bart.py   # BART model (encoder / decoder / cross-attn)
  loader.py        # HuggingFace -> model weight loading
  model_runner.py  # batching, KV cache, encoder caching, forward
  engine.py        # scheduler (continuous batching) + beam search loop
  sampler.py       # greedy / temperature / top-p / n-gram banning
  __init__.py      # LLM facade
```

## Usage

```python
from fastencdec import LLM, SamplingParams

llm = LLM("facebook/bart-large-cnn", num_blocks=256, block_size=16)

# greedy
print(llm.generate("The quick brown fox ...", SamplingParams(max_tokens=32)))

# beam search
print(llm.generate(
    "The quick brown fox ...",
    SamplingParams(max_tokens=32, num_beams=4),
))

# batched (continuous batching): different-length inputs run together
print(llm.generate([article_a, article_b, article_c],
                   SamplingParams(max_tokens=32, num_beams=4)))
```

`dtype=torch.bfloat16` halves memory and speeds up the GEMMs on CPUs with
`avx512_bf16` (e.g. AMD Zen 4, Intel Sapphire Rapids). Do **not** use
`float16` on Zen 4 (no AVX512-FP16).

### Generation heuristics

BART checkpoints ship decoding settings in `generation_config.json`. Pass them
explicitly when they matter — `bart-large-cnn` collapses to short/repetitive
output without them:

```python
SamplingParams(max_tokens=48, num_beams=4, length_penalty=2.0,
               no_repeat_ngram_size=3, min_length=8)
```

`length_penalty` is applied when ranking finished hypotheses, `min_length` masks
EOS until enough tokens are produced, and `no_repeat_ngram_size` bans tokens
that would repeat an n-gram. (`forced_bos_token_id` / `forced_eos_token_id` are
not implemented.)

## How a step works

```
scheduler.schedule()            -> all running sequences (+ admit waiting)
model_runner.run(seqs)
    _encode(fresh seqs)         batched encoder (grouped by length) -> cross-attn K/V per request
    build input_ids/positions/slot_mapping
    decoder forward             self-attn reads/writes paged cache,
                                cross-attn reads cached encoder K/V
    compute_logits(last token)  -> sample / beam-expand
```

Each sequence only feeds the tokens not yet in the cache
(`token_ids[num_cached_tokens:]`): the whole prompt on prefill, one token per
decode step. Attention gathers exactly `context_len` keys per sequence from its
own block table, so there is no padding.

Decoder self-attention is paged. **Cross-attention is dense**: each decoder
layer keeps `encoder_kv_cache[request_id] -> (k, v)` of shape
`[heads, enc_len, dim]`, computed once per request and shared by all of its
beams.

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
against HuggingFace `generate` on the same machine (latency, req/s, tok/s,
speedup).

Full sweep — `facebook/bart-large-cnn`, 8 threads, BF16, 48 tokens, encoder
length 256:

| config | ours ms | hf ms | speedup |
|---|---:|---:|---:|
| batch=1, beams=1 | 835 | 1695 | 2.0× |
| batch=8, beams=1 | 1916 | 4185 | 2.2× |
| batch=16, beams=1 | 3138 | 6038 | 1.9× |
| batch=1, beams=4 | 550 | 3319 | 6.0× |
| batch=8, beams=4 | 1992 | 12580 | 6.3× |
| batch=16, beams=4 | 3681 | 20853 | 5.7× |

Other axes: BF16 is ~1.7× faster than FP32 in our engine (1984 ms vs 3337 ms
at batch=8/beams=4); the speedup shrinks as the encoder grows (2.3× at
enc=128 → 1.7× at enc=1000) because the encoder is GEMM-bound and both engines
use the same oneDNN matmuls.

## Limitations

- The scheduler returns the whole running set each step; there is no token
  budget / chunked prefill. Keep `max_num_seqs >= num_beams` so a request's
  beams are always batched together.
- No prefix caching, no preemption/recompute, no CUDA graphs, no quantization.
- The cross-attention cache is dense and scales with `batch × enc_len`; it is
  not counted against `num_blocks`.
- `num_blocks` is the total KV budget; the block manager raises if exhausted.
