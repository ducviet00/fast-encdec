"""Benchmark fast-encdec against HuggingFace ``generate`` on CPU.

Sweeps batch size, beam width, dtype and encoder length, reporting latency and
throughput for both engines.  Example:

    PYTHONPATH=. python benchmarks/benchmark.py --model facebook/bart-large-cnn
    PYTHONPATH=. python benchmarks/benchmark.py --quick
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch

from fastencdec import LLM, SamplingParams

PARAGRAPHS = [
    (
        "The quick brown fox jumps over the lazy dog. It was a sunny afternoon and "
        "the fox had been running through the meadow for hours, chasing butterflies "
        "and enjoying the warm weather before returning to its den."
    ),
    (
        "Scientists at the university announced on Monday that they had discovered "
        "a new species of butterfly in the Amazon rainforest. The insect, which has "
        "bright blue wings and a wingspan of nearly ten centimetres, was found "
        "during a three-week expedition led by researchers from three countries."
    ),
    (
        "The stock market rose sharply on Tuesday after the central bank announced "
        "it would keep interest rates unchanged for the foreseeable future. "
        "Investors had feared a rate hike that could slow the economy, and the "
        "decision was welcomed by businesses across the region."
    ),
    (
        "The city council approved a new plan to expand public transport, including "
        "two new tram lines and a network of protected bicycle lanes. Officials "
        "said the project would reduce traffic congestion and cut emissions over "
        "the next decade."
    ),
]

# Decoding heuristics that make BART checkpoints produce sensible summaries.
DECODE = {"length_penalty": 2.0, "no_repeat_ngram_size": 3, "min_length": 8}


def make_prompt(tokenizer, target_tokens: int) -> str:
    """Build a prompt whose encoder length is about ``target_tokens``."""
    text = " ".join(PARAGRAPHS)
    while len(tokenizer(text, add_special_tokens=False)["input_ids"]) < target_tokens:
        text += " " + text
    ids = tokenizer(
        text, add_special_tokens=False, truncation=True, max_length=target_tokens
    )["input_ids"]
    return tokenizer.decode(ids)


def timed(fn, warmup: int, repeats: int):
    """Return (best_seconds, last_result)."""
    for _ in range(warmup):
        fn()
    best = float("inf")
    last = None
    for _ in range(repeats):
        start = time.perf_counter()
        last = fn()
        best = min(best, time.perf_counter() - start)
    return best, last


def bench_ours(llm, prompts, params, warmup, repeats):
    dt, _ = timed(lambda: llm.generate(prompts, params), warmup, repeats)
    ids = [llm.engine.results[r] for r in sorted(llm.engine.results)[-len(prompts) :]]
    return dt, sum(len(x) for x in ids)


def bench_hf(hf, tokenizer, prompts, max_new, beams, warmup, repeats):
    enc = tokenizer(prompts, return_tensors="pt", padding=True)
    kwargs = dict(
        max_new_tokens=max_new,
        num_beams=beams,
        do_sample=False,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        early_stopping=False,
        **DECODE,
    )

    def run():
        with torch.no_grad():
            return hf.generate(**enc, **kwargs)

    dt, out = timed(run, warmup, repeats)
    generated = int((out != tokenizer.pad_token_id).sum())
    return dt, generated


def row(label, ours_dt, ours_tok, hf_dt, hf_tok, n):
    return (
        f"| {label:<22} | {ours_dt * 1e3:>9.0f} | {n / ours_dt:>8.2f} "
        f"| {ours_tok / ours_dt:>8.1f} | {hf_dt * 1e3:>9.0f} | {n / hf_dt:>8.2f} "
        f"| {hf_dt / ours_dt:>7.2f}x |"
    )


def table(title, results):
    lines = [
        f"\n### {title}",
        "",
        "| config | ours ms | ours req/s | ours tok/s | hf ms | hf req/s | speedup |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines += results
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/bart-large-cnn")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--num-blocks", type=int, default=1024)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--encoder-tokens", type=int, default=256)
    ap.add_argument("--quick", action="store_true", help="small sweep")
    ap.add_argument("--json", default=None, help="write raw results to a JSON file")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    dtype = getattr(torch, args.dtype)

    import transformers
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    transformers.logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf = AutoModelForSeq2SeqLM.from_pretrained(args.model, dtype=torch.float32).eval()

    def build_llm(dt):
        return LLM(
            args.model,
            num_blocks=args.num_blocks,
            block_size=args.block_size,
            max_num_seqs=64,
            dtype=dt,
        )

    llm = build_llm(dtype)
    prompt_cache = {}
    raw = []

    def prompts_for(n, enc_tokens):
        if enc_tokens not in prompt_cache:
            prompt_cache[enc_tokens] = make_prompt(tokenizer, enc_tokens)
        return [prompt_cache[enc_tokens]] * n

    def run(label, n, beams, llm_, dtype_name, enc_tokens):
        prompts = prompts_for(n, enc_tokens)
        params = SamplingParams(max_tokens=args.max_tokens, num_beams=beams, **DECODE)
        ours_dt, ours_tok = bench_ours(llm_, prompts, params, args.warmup, args.repeats)
        hf_dt, hf_tok = bench_hf(
            hf, tokenizer, prompts, args.max_tokens, beams, args.warmup, args.repeats
        )
        raw.append(
            {
                "label": label,
                "batch": n,
                "beams": beams,
                "dtype": dtype_name,
                "encoder_tokens": enc_tokens,
                "ours_ms": ours_dt * 1e3,
                "hf_ms": hf_dt * 1e3,
            }
        )
        return row(label, ours_dt, ours_tok, hf_dt, hf_tok, n)

    print("# fast-encdec benchmark\n", flush=True)
    print(
        f"model={args.model}  threads={args.threads}  dtype={args.dtype}  "
        f"max_tokens={args.max_tokens}  num_blocks={args.num_blocks}  "
        f"block_size={args.block_size}",
        flush=True,
    )

    batch_sizes = [1, 2, 4, 8] if args.quick else [1, 2, 4, 8, 16]
    beam_widths = [1, 4] if args.quick else [1, 2, 4]

    table(
        f"batch scaling (beams=1, {args.dtype}, enc={args.encoder_tokens})",
        [
            run(f"batch={n}", n, 1, llm, args.dtype, args.encoder_tokens)
            for n in batch_sizes
        ],
    )

    table(
        f"batch scaling (beams=4, {args.dtype}, enc={args.encoder_tokens})",
        [
            run(f"batch={n}", n, 4, llm, args.dtype, args.encoder_tokens)
            for n in batch_sizes
        ],
    )

    table(
        f"beam scaling (batch=8, {args.dtype}, enc={args.encoder_tokens})",
        [
            run(f"beams={b}", 8, b, llm, args.dtype, args.encoder_tokens)
            for b in beam_widths
        ],
    )

    table(
        f"dtype (batch=8, beams=4, enc={args.encoder_tokens})",
        [
            run(
                dt_name,
                8,
                4,
                build_llm(getattr(torch, dt_name)),
                dt_name,
                args.encoder_tokens,
            )
            for dt_name in ("float32", "bfloat16")
        ],
    )

    enc_lengths = [128, 256] if args.quick else [128, 256, 512, 1000]
    table(
        f"encoder length (batch=8, beams=1, {args.dtype})",
        [run(f"enc={e}", 8, 1, llm, args.dtype, e) for e in enc_lengths],
    )

    if args.json:
        with open(args.json, "w") as f:
            json.dump(raw, f, indent=2)
        print(f"\nwrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
