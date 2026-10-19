"""Attention-implementation benchmark: vendored vLLM CPU kernel vs SDPA vs eager.

Two sections:

* ``kernel`` -- the decoder attention ops in isolation, with BART-large shapes
  (16 heads x head_dim 64) driven through a synthetic Context: paged
  self-attention at decode shapes, and cross-attention over a cached encoder.
  Reports microseconds per op call for the three implementations.
* ``e2e`` -- the whole engine (``attn_backend="vllm"`` / ``"sdpa"``, and SDPA
  forced to PyTorch's math kernel for "eager") against HuggingFace
  (``attn_implementation="sdpa"`` / ``"eager"``), on a standard workload and a
  decode-heavy (attention-dominated) one.

The three implementations are:
  vllm  -- the vendored vLLM CPU kernel (`csrc/`), the engine's default;
  sdpa  -- PyTorch's `scaled_dot_product_attention` (its own kernel choice);
  eager -- PyTorch SDPA pinned to `SDPBackend.MATH`, i.e. the naive
           matmul/softmax/matmul attention, and HF's `eager` implementation.

Run:  PYTHONPATH=. python benchmarks/attn_backend.py                # both
      PYTHONPATH=. python benchmarks/attn_backend.py --mode kernel  # no weights
      PYTHONPATH=. python benchmarks/attn_backend.py --mode e2e --device cuda
"""

import argparse
import sys
import time

import torch

from fastencdec.config import default_block_size
from fastencdec.layers import attention, cpu_attn
from fastencdec.utils.context import set_context

HEADS, HEAD_DIM = 16, 64  # bart-large-cnn
MODEL = "facebook/bart-large-cnn"
DECODE = {"length_penalty": 2.0, "no_repeat_ngram_size": 3}
VARIANTS = ("vllm", "sdpa", "eager")


def _timed(fn, iters: int, warmup: int = 5) -> float:
    """Microseconds per call (best of ``iters`` runs, after ``warmup``)."""
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best * 1e6


def _call(variant, fn):
    """Run ``fn`` under the variant's backend (``eager`` pins SDPA to math)."""
    attention.set_backend("vllm" if variant == "vllm" else "sdpa")
    if variant == "eager":
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel([SDPBackend.MATH]):
            return fn()
    return fn()


# --------------------------------------------------------------- kernel section
def _self_case(num_seqs, ctx_len, block_size, dtype):
    """Tensors for one paged self-attention call (one new token per sequence)."""
    pages = (ctx_len + block_size - 1) // block_size
    blocks = torch.arange(num_seqs * pages, dtype=torch.int32).reshape(num_seqs, pages)
    slots = (
        blocks[:, (ctx_len - 1) // block_size].long() * block_size
        + (ctx_len - 1) % block_size
    )
    lens = torch.full((num_seqs,), ctx_len, dtype=torch.int32)
    shape = attention.self_cache_shape(
        num_seqs * pages, block_size, HEADS, HEAD_DIM, torch.device("cpu"), dtype
    )
    set_context(
        context_lens=lens,
        block_tables=blocks,
        slot_mapping=slots,
        request_ids=None,
        cross_cache_seqlens=None,
        cross_block_tables=None,
    )
    return (
        torch.randn(num_seqs, HEADS, HEAD_DIM, dtype=dtype),
        torch.randn(num_seqs, HEADS, HEAD_DIM, dtype=dtype),
        torch.randn(num_seqs, HEADS, HEAD_DIM, dtype=dtype),
        torch.zeros(shape, dtype=dtype),
        torch.zeros(shape, dtype=dtype),
        slots,
        blocks,
        lens,
    )


def _cross_case(num_seqs, enc_len, block_size, dtype):
    """Dense query + per-request encoder K/V, plus the vLLM staged paged cache."""
    kv_cache = {
        i: (
            torch.randn(HEADS, enc_len, HEAD_DIM, dtype=dtype),
            torch.randn(HEADS, enc_len, HEAD_DIM, dtype=dtype),
        )
        for i in range(num_seqs)
    }
    request_ids = list(range(num_seqs))
    staged = attention.stage_cross_kv(kv_cache, request_ids, block_size, "vllm")
    return (
        torch.randn(num_seqs, HEADS, HEAD_DIM, dtype=dtype),
        kv_cache,
        request_ids,
        staged,
    )


def _time_self(variant, num_seqs, ctx_len, block_size, dtype, scale, iters):
    q, k, v, k_cache, v_cache, slots, blocks, lens = _self_case(
        num_seqs, ctx_len, block_size, dtype
    )
    if variant == "vllm":
        run = lambda: cpu_attn.self_attention(
            q, k, v, k_cache, v_cache, slots, blocks, lens, scale
        )
    else:
        run = lambda: attention.decoder_self_attention(
            q, k, v, k_cache, v_cache, scale, block_size
        )
    return _timed(lambda: _call(variant, run), iters)


def _time_cross(variant, num_seqs, enc_len, block_size, dtype, scale, iters):
    q, kv_cache, request_ids, staged = _cross_case(num_seqs, enc_len, block_size, dtype)
    if variant == "vllm":
        k_cache, v_cache, block_table, cache_seqlens = staged
        run = lambda: cpu_attn.cross_attention(
            q, k_cache, v_cache, block_table, cache_seqlens, scale
        )
    else:
        run = lambda: attention.cross_sdpa_batched(q, kv_cache, request_ids, scale)
    return _timed(lambda: _call(variant, run), iters)


def kernel_section(dtype, block_size, iters):
    scale = HEAD_DIM**-0.5
    print(
        f"\n== kernel: decoder attention ops in isolation ({dtype}, "
        f"heads={HEADS} head_dim={HEAD_DIM}, block_size={block_size}) =="
    )
    header = (
        f"{'op':<26} "
        + " ".join(f"{v:>10}" for v in VARIANTS)
        + f" {'vllm vs eager':>14}"
    )
    print(header)
    print("-" * len(header))
    print(
        "note: on CPU the sdpa/eager columns are our per-sequence fallback loop\n"
        "      (CUDA uses the batched one), so they include that Python overhead;\n"
        "      the vllm column calls the kernel entry point directly, so it does\n"
        "      not pay the op's backend resolution and Context read;\n"
        "      the e2e section compares against HF's batched sdpa/eager instead."
    )
    cases = [
        ("self, batch=8 ctx=256", _time_self, (8, 256)),
        ("self, batch=8 ctx=1024", _time_self, (8, 1024)),
        ("self, batch=32 ctx=1024", _time_self, (32, 1024)),
        ("cross, batch=8 enc=256", _time_cross, (8, 256)),
        ("cross, batch=8 enc=1024", _time_cross, (8, 1024)),
    ]
    for label, fn, (num_seqs, length) in cases:
        times = {
            variant: fn(variant, num_seqs, length, block_size, dtype, scale, iters)
            for variant in VARIANTS
        }
        print(
            f"{label:<26} "
            + " ".join(f"{times[v]:>9.1f}u" for v in VARIANTS)
            + f" {times['eager'] / times['vllm']:>13.1f}x"
        )
    attention.set_backend("auto")


# ---------------------------------------------------------------- e2e section
def _prompt(tok, tokens):
    text = (
        "The quick brown fox jumps over the lazy dog. It was a sunny afternoon and "
        "the fox had been running through the meadow for hours, chasing butterflies "
        "and enjoying the warm weather before returning to its den. "
    )
    while len(tok(text, add_special_tokens=False)["input_ids"]) < tokens:
        text += " " + text
    ids = tok(text, add_special_tokens=False, truncation=True, max_length=tokens)[
        "input_ids"
    ]
    return tok.decode(ids)


def _time_ours(llm, prompts, variant, out, repeats):
    from fastencdec import SamplingParams

    params = SamplingParams(
        max_tokens=out, num_beams=1, min_length=out + 1, temperature=0.0, **DECODE
    )
    return _timed(
        lambda: _call(variant, lambda: llm.generate(prompts, params)), repeats, 1
    )


def _time_hf(model, enc, out, repeats):
    def run():
        with torch.no_grad():
            return model.generate(
                **enc,
                max_new_tokens=out,
                min_new_tokens=out,
                num_beams=1,
                do_sample=False,
                forced_bos_token_id=None,
                forced_eos_token_id=None,
                early_stopping=False,
                **DECODE,
            )

    return _timed(run, repeats, 1)


def e2e_section(args, dtype):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from fastencdec import LLM

    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(
        MODEL,
        num_blocks=args.num_blocks,
        dtype=dtype,
        attn_backend="auto",
        device=args.device,
        use_cudagraph=args.cudagraph,
    )
    hf = {
        impl: AutoModelForSeq2SeqLM.from_pretrained(
            MODEL, dtype=dtype, attn_implementation=impl
        ).eval()
        for impl in ("sdpa", "eager")
    }
    if args.device.startswith("cuda"):
        for model in hf.values():
            model.to(args.device)

    for enc_len, out in ((args.enc_len, args.max_tokens), (64, 256)):
        prompts = [_prompt(tok, enc_len)] * args.batch
        enc = tok(prompts, return_tensors="pt", padding=True)
        if args.device.startswith("cuda"):
            enc = {k: v.to(args.device) for k, v in enc.items()}
        kind = "standard" if enc_len >= 128 else "decode-heavy"
        print(
            f"\n== e2e: batch={args.batch} enc~{enc['input_ids'].shape[1]} out={out} "
            f"({dtype}, {kind}) =="
        )
        rows = [
            (f"ours/{v}", _time_ours(llm, prompts, v, out, args.repeats) / 1e3)
            for v in VARIANTS
        ]
        rows += [
            (f"HF/{impl}", _time_hf(model, enc, out, args.repeats) / 1e3)
            for impl, model in hf.items()
        ]
        tokens = args.batch * out
        ref = dict(rows)["HF/sdpa"]
        print(f"{'implementation':<16} {'ms':>9} {'tok/s':>9} {'vs HF/sdpa':>11}")
        for label, ms in rows:
            print(
                f"{label:<16} {ms:>9.0f} {tokens / ms * 1e3:>9.1f} {ref / ms:>10.2f}x"
            )
    attention.set_backend("auto")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["kernel", "e2e", "both"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument(
        "--threads",
        type=int,
        default=0,
        help="0 = torch default, i.e. the number of physical cores",
    )
    ap.add_argument("--block-size", type=int, default=0, help="0 = device default")
    ap.add_argument("--num-blocks", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--enc-len", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20, help="kernel-section iters")
    ap.add_argument("--cudagraph", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    threads = torch.get_num_threads()  # physical cores unless forced
    dtype = getattr(torch, args.dtype)
    block_size = args.block_size or default_block_size(args.device)
    print(
        f"attn_backend benchmark: device={args.device} dtype={args.dtype} "
        f"threads={threads} block_size={block_size}"
    )
    if args.mode in ("kernel", "both"):
        kernel_section(dtype, block_size, args.iters)
    if args.mode in ("e2e", "both"):
        e2e_section(args, dtype)


if __name__ == "__main__":
    sys.exit(main())
