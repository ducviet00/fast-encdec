"""BART benchmark: this engine vs HuggingFace `generate`, scenarios x configs.

Scenarios (encoder tokens -> generated tokens).  ``bart-large-cnn`` has a
1024-entry position table and BART offsets it by +2, so both lengths cap at 1022:

=========  ============  ============
scenario   encoder       generated
=========  ============  ============
128->512   128           512
512->128   512           128
256->256   256           256
=========  ============  ============

Configs: three bf16 points plus an fp32 row for verification.  fp32 is
``batch=8, beams=1`` rather than beams=4 because both engines' KV caches double,
and 32 sequences x 1000 generated tokens would not fit the 11 GB dev box.

Both engines run the same dtype, inputs and output length (EOS suppressed), so
the speedup is engine-vs-engine; the tables report the throughput of each engine
and the speedup.

  PYTHONPATH=. python benchmarks/benchmark.py                 # cpu
  PYTHONPATH=. python benchmarks/benchmark.py --device cuda
  PYTHONPATH=. python benchmarks/benchmark.py --scenarios 256->256
"""

import argparse
import json
import time
from functools import partial

import torch

from fastencdec import LLM, SamplingParams

MODEL = "facebook/bart-large-cnn"
DECODE = {"length_penalty": 2.0, "no_repeat_ngram_size": 3}

SCENARIOS = (
    ("128->512", 128, 512),
    ("512->128", 512, 128),
    ("256->256", 256, 256),
)
CONFIGS = (
    ("batch=1, beams=1, bf16", 1, 1, "bfloat16"),
    ("batch=8, beams=1, bf16", 8, 1, "bfloat16"),
    ("batch=8, beams=4, bf16", 8, 4, "bfloat16"),
    ("batch=8, beams=1, fp32", 8, 1, "float32"),
)

PARAGRAPHS = [
    (
        "The quick brown fox jumps over the lazy dog. It was a sunny afternoon and "
        "the fox had been running through the meadow for hours, chasing butterflies "
        "and enjoying the warm weather before returning to its den."
    ),
    (
        "Scientists at the university announced on Monday that they had discovered "
        "a new species of butterfly in the Amazon rainforest. The insect, which has "
        "a wingspan of only two centimetres, was found near a river bank and has "
        "been named after the researcher who first spotted it."
    ),
    (
        "The city council approved the new transit plan after a long debate. The "
        "plan adds two tram lines, extends the bus network to the outer districts "
        "and introduces a single ticket for all modes of transport from next year."
    ),
]


def make_prompt(tokenizer, target_tokens: int) -> str:
    """A prompt whose encoder length is about ``target_tokens``."""
    text = " ".join(PARAGRAPHS)
    while len(tokenizer(text, add_special_tokens=False)["input_ids"]) < target_tokens:
        text += " " + text
    ids = tokenizer(
        text, add_special_tokens=False, truncation=True, max_length=target_tokens
    )["input_ids"]
    return tokenizer.decode(ids)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, warmup: int, repeats: int):
    """Return (best_seconds, every result) so the caller can check consistency."""
    for _ in range(warmup):
        fn()
    best = float("inf")
    results = []
    for _ in range(repeats):
        _sync()
        start = time.perf_counter()
        results.append(fn())
        _sync()
        best = min(best, time.perf_counter() - start)
    return best, results


def params_for(max_tokens: int, beams: int) -> SamplingParams:
    """Fixed-length decoding: ``min_length`` masks EOS for the whole run.

    The decoder prompt is one token, so ``max_tokens + 1`` is the length at which
    EOS becomes legal — i.e. it is never emitted.  Mirrors HF's
    ``min_new_tokens=max_tokens`` so both engines decode exactly ``max_tokens``.
    """
    return SamplingParams(
        max_tokens=max_tokens, num_beams=beams, min_length=max_tokens + 1, **DECODE
    )


def count_generated(rows, pad_token_id) -> int:
    total = 0
    for row in rows:
        toks = row.tolist()[1:]
        while toks and toks[-1] == pad_token_id:
            toks.pop()
        total += len(toks)
    return total


def ours_run(llm, prompts, params):
    """One generate; returns the number of tokens the engine decoded."""
    llm.generate(prompts, params)
    ids = sorted(llm.engine.results)[-len(prompts) :]
    return sum(len(llm.engine.results[r]) for r in ids)


def hf_run(hf, tokenizer, prompts, tokens, beams, warmup, repeats):
    enc = tokenizer(prompts, return_tensors="pt", padding=True)
    if next(hf.parameters()).is_cuda:
        enc = {k: v.to("cuda") for k, v in enc.items()}
    kwargs = dict(
        max_new_tokens=tokens,
        min_new_tokens=tokens,  # fixed length: EOS suppressed until max_new
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

    dt, outs = timed(run, warmup, repeats)
    counts = {count_generated(out, tokenizer.pad_token_id) for out in outs}
    if len(counts) != 1:
        raise RuntimeError(f"HF decoded different token counts across runs: {counts}")
    return dt, counts.pop()


def blocks_for(configs, enc_tokens: int, out_tokens: int, block_size: int) -> int:
    """``num_blocks`` covering the widest config of one scenario.

    Every running sequence holds ``enc + out`` tokens, so 32 sequences x 1000
    generated tokens is a lot of KV; sizing it per scenario keeps the footprint
    down (and fits the fp32 row on a small box).
    """
    worst = max(batch * beams for _, batch, beams, _ in configs)
    tokens = worst * (enc_tokens + out_tokens + 8)
    return (tokens + block_size - 1) // block_size + 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--device", default="cpu", help="cpu or cuda (cuda:N works)")
    ap.add_argument(
        "--threads",
        type=int,
        default=0,
        help="0 = torch default, i.e. the number of physical cores",
    )
    ap.add_argument(
        "--block-size",
        type=int,
        default=0,
        help="0 = device default (256 cuda, 16 cpu)",
    )
    ap.add_argument("--scenarios", default="", help="comma list, default all")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--json", default=None, help="write raw results to a JSON file")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    block_size = args.block_size or (256 if args.device.startswith("cuda") else 16)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    wanted = {s.strip() for s in args.scenarios.split(",") if s.strip()}
    scenarios = [s for s in SCENARIOS if not wanted or s[0] in wanted]

    print("# fast-encdec benchmark\n", flush=True)
    print(
        f"model={args.model}  device={args.device}  threads={torch.get_num_threads()}",
        flush=True,
    )

    results = {}
    for dtype_name in ("bfloat16", "float32"):
        configs = [c for c in CONFIGS if c[3] == dtype_name]
        if not configs:
            continue
        dtype = getattr(torch, dtype_name)
        for name, enc_tokens, out_tokens in scenarios:
            prompt = make_prompt(tokenizer, enc_tokens)
            enc_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            num_blocks = blocks_for(configs, enc_len, out_tokens, block_size)
            # One engine in memory at a time: 32 sequences x 1000 generated tokens
            # is ~4 GB of KV in fp32 across both engines, which a small box cannot
            # hold (and HF's cache is freed between scenarios either way).
            llm = LLM(
                args.model,
                num_blocks=num_blocks,
                block_size=block_size,
                dtype=dtype,
                device=args.device,
            )
            for label, batch, beams, _ in configs:
                dt, toks = timed(
                    partial(
                        ours_run, llm, [prompt] * batch, params_for(out_tokens, beams)
                    ),
                    args.warmup,
                    args.repeats,
                )
                if len(set(toks)) != 1:
                    raise RuntimeError(
                        f"{name}/{label}: decoded {set(toks)} tokens across runs"
                    )
                results[f"{name}|{label}"] = {
                    "scenario": name,
                    "config": label,
                    "dtype": dtype_name,
                    "enc_tokens": enc_len,
                    "out_tokens": out_tokens,
                    "ours_ms": dt * 1e3,
                    "ours_tok": toks[0],
                }
                print(".", end="", flush=True)
            del llm

            hf = AutoModelForSeq2SeqLM.from_pretrained(args.model, dtype=dtype).eval()
            if args.device.startswith("cuda"):
                hf = hf.to(args.device)
            for label, batch, beams, _ in configs:
                dt, tok = hf_run(
                    hf,
                    tokenizer,
                    [prompt] * batch,
                    out_tokens,
                    beams,
                    args.warmup,
                    args.repeats,
                )
                r = results[f"{name}|{label}"]
                # Two checks: repeated runs of one engine agree with each other,
                # and the two engines decode the same number of tokens.
                if tok != r["ours_tok"]:
                    raise RuntimeError(
                        f"{name}/{label}: ours decoded {r['ours_tok']} tokens, "
                        f"HF {tok} - both engines must decode the same number"
                    )
                r["hf_ms"] = dt * 1e3
                r["hf_tok_s"] = tok / dt
                r["ours_tok_s"] = r["ours_tok"] / (r["ours_ms"] / 1e3)
                r["speedup"] = r["hf_ms"] / r["ours_ms"]
                print(".", end="", flush=True)
                if args.json:  # durable: a later OOM must not lose the run
                    with open(args.json, "w") as f:
                        json.dump(list(results.values()), f, indent=2)
            del hf

    for name, enc_tokens, out_tokens in scenarios:
        rows = [
            r for key, r in results.items() if r["scenario"] == name and "hf_ms" in r
        ]
        print(
            f"\n\n### {name}: encoder {rows[0]['enc_tokens']} -> {out_tokens} generated"
        )
        print(
            "\n| config | ours ms | ours tok/s | hf ms | hf tok/s | speedup |"
            "\n|---|---:|---:|---:|---:|---:|"
        )
        for r in rows:
            print(
                f"| {r['config']:<22} | {r['ours_ms']:>8.0f} | {r['ours_tok_s']:>10.1f} "
                f"| {r['hf_ms']:>8.0f} | {r['hf_tok_s']:>9.1f} "
                f"| {r['speedup']:>7.2f}x |"
            )

    if args.json:
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
