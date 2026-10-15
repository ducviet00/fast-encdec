"""BART benchmark on CUDA: this engine vs vLLM, scenarios x configs.

vLLM runs BART through two plugins (no fork, no source patch):

- https://github.com/vllm-project/bart-plugin -- encoder-decoder model support
- https://github.com/neuralmagic/vllm-beam-search-plugin -- beam scheduler +
  sampler (``scheduler_cls=vllm_beam_search.scheduler.BeamSearchScheduler``;
  per-request ``SamplingParams(extra_args={"beam_width": n})``)

Methodology mirrors ``benchmarks/benchmark.py``: both engines run the same
dtype, prompts and output length, EOS suppressed, so the speedup is
engine-vs-engine.  Two deviations, both deliberate:

- No decoding heuristics (no ``length_penalty``/``no_repeat_ngram_size``).
  vLLM's greedy path has no n-gram blocking parameter, so heuristics stay off
  on both sides to keep the comparison like-for-like.
- vLLM beam search appends a trailing EOS on top of ``min_tokens`` (N+1 tokens
  per sequence vs our exactly-N); the extra token is <0.5% of a 256-token run,
  and each engine's tok/s is computed from its own decoded count.

The two engines live in different venvs, so run one process per engine::

  PYTHONPATH=. .venv/bin/python benchmarks/benchmark_vllm.py --engine ours --json ours.json
  # FlashInfer JIT needs nvcc (absent), so disable its sampler.
  HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_PLUGINS=bart,beam_search \\
      ~/vllmenv/bin/python benchmarks/benchmark_vllm.py --engine vllm --json vllm.json

Then merge the two raw JSONs into the comparison table::

  python benchmarks/benchmark_vllm.py --compare ours.json vllm.json

``--batches`` overrides ``--configs`` with a greedy (``beams=1``) batch sweep, which
is how the decode-scaling tables are produced::

  ... --engine ours --batches 1,2,4,8,16,32 --json ours_greedy.json
  python benchmarks/benchmark_vllm.py --compare ours_greedy.json vllm_greedy.json
"""

import argparse
import gc
import json
import os
import re
import time
from functools import partial

import torch

MODEL = "facebook/bart-large-cnn"

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
    """Return (best_seconds, per-repeat token counts)."""
    for _ in range(warmup):
        fn()
    best = float("inf")
    toks = []
    for _ in range(repeats):
        _sync()
        start = time.perf_counter()
        toks.append(fn())
        _sync()
        best = min(best, time.perf_counter() - start)
    return best, toks


def dump(args, rows) -> None:
    """Write raw rows now: a later OOM must not lose the whole run."""
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)


def blocks_for(configs, enc_tokens: int, out_tokens: int, block_size: int) -> int:
    """``num_blocks`` covering the widest config of one scenario."""
    worst = max(batch * beams for _, batch, beams, _ in configs)
    tokens = worst * (enc_tokens + out_tokens + 8)
    return (tokens + block_size - 1) // block_size + 8


def ours_run(llm, prompts, params) -> int:
    """One generate; returns the number of tokens the engine decoded."""
    outs = llm.generate(prompts, params, use_tqdm=False)
    return sum(len(o.outputs[0].token_ids) for o in outs)


def vllm_run(llm, prompts, params) -> int:
    """One vLLM generate; returns the number of tokens decoded."""
    outs = llm.generate(prompts, params)
    return sum(len(o.outputs[0].token_ids) for o in outs)


def run_ours(args, tokenizer, scenarios):
    from fastencdec import LLM, SamplingParams
    from fastencdec.config import default_block_size

    block_size = default_block_size("cuda")
    max_seqs = max(b * n for _, b, n, _ in args.configs)
    rows = []
    for dtype_name in ("bfloat16", "float32"):
        configs = [c for c in args.configs if c[3] == dtype_name]
        if not configs:
            continue
        for name, enc_tokens, out_tokens in scenarios:
            prompt = make_prompt(tokenizer, enc_tokens)
            enc_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            llm = LLM(
                args.model,
                num_blocks=blocks_for(configs, enc_len, out_tokens, block_size),
                block_size=block_size,
                max_num_seqs=max_seqs,
                dtype=getattr(torch, dtype_name),
                device="cuda",
            )
            for label, batch, beams, _ in configs:
                # min_length = out + 1 makes EOS illegal for the whole run, so
                # the engine decodes exactly ``out_tokens`` (decoder prompt = 1 token).
                params = SamplingParams(
                    max_tokens=out_tokens, num_beams=beams, min_length=out_tokens + 1
                )
                dt, toks = timed(
                    partial(ours_run, llm, [prompt] * batch, params),
                    args.warmup,
                    args.repeats,
                )
                assert len(set(toks)) == 1, f"{name}/{label}: ours decoded {set(toks)}"
                expect = batch * out_tokens
                assert toks[0] == expect, (
                    f"{name}/{label}: ours {toks[0]}, want {expect}"
                )
                row = {
                    "scenario": name,
                    "config": label,
                    "dtype": dtype_name,
                    "batch": batch,
                    "beams": beams,
                    "enc_tokens": enc_len,
                    "out_tokens": out_tokens,
                    "ms": dt * 1e3,
                    "tok": toks[0],
                    "tok_s": toks[0] / dt,
                }
                rows.append(row)
                print(
                    f"ours {name} {label}: {dt * 1e3:.0f} ms, {row['tok_s']:.1f} tok/s",
                    flush=True,
                )
                dump(args, rows)
            del llm
            torch.cuda.empty_cache()
    return rows


def run_vllm(args, tokenizer, scenarios):
    use_beam = any(beams > 1 for _, _, beams, _ in args.configs)
    # Read before importing vllm: no nvcc on this box, so FlashInfer's JIT sampler
    # must be off.  The beam plugin replaces the scheduler with a custom one that
    # disables async scheduling (and can slow plain greedy down), so load it only
    # when a beam config is actually requested.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_PLUGINS", "bart,beam_search" if use_beam else "bart")
    import vllm_bart_plugin  # noqa: F401  registers BART with vLLM
    from vllm import LLM, SamplingParams

    llm_kwargs = {
        "model": args.model,
        "max_model_len": 1024,
        "max_num_seqs": max(b * n for _, b, n, _ in args.configs),
        "gpu_memory_utilization": 0.7,
    }
    if use_beam:
        llm_kwargs["scheduler_cls"] = "vllm_beam_search.scheduler.BeamSearchScheduler"

    rows = []
    for dtype_name in ("bfloat16", "float32"):
        configs = [c for c in args.configs if c[3] == dtype_name]
        if not configs:
            continue
        llm = LLM(dtype=dtype_name, **llm_kwargs)
        for name, enc_tokens, out_tokens in scenarios:
            prompt = make_prompt(tokenizer, enc_tokens)
            enc_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            for label, batch, beams, _ in configs:
                prompts = [
                    {
                        "encoder_prompt": {
                            "prompt": "",
                            "multi_modal_data": {"text": prompt},
                        },
                        "decoder_prompt": "<s>",
                    }
                ] * batch
                params_kwargs = {
                    "temperature": 0.0,
                    "max_tokens": out_tokens,
                    "min_tokens": out_tokens,
                }
                if beams > 1:
                    params_kwargs["extra_args"] = {"beam_width": beams}
                params = SamplingParams(**params_kwargs)

                dt, toks = timed(
                    partial(vllm_run, llm, prompts, params), args.warmup, args.repeats
                )
                assert len(set(toks)) == 1, f"{name}/{label}: vllm decoded {set(toks)}"
                # Beam search appends a trailing EOS on top of min_tokens;
                # greedy is exact.
                lo, hi = batch * out_tokens, batch * (out_tokens + 1)
                assert lo <= toks[0] <= hi, (
                    f"{name}/{label}: vllm {toks[0]} not in [{lo},{hi}]"
                )
                row = {
                    "scenario": name,
                    "config": label,
                    "dtype": dtype_name,
                    "batch": batch,
                    "beams": beams,
                    "enc_tokens": enc_len,
                    "out_tokens": out_tokens,
                    "ms": dt * 1e3,
                    "tok": toks[0],
                    "tok_s": toks[0] / dt,
                }
                rows.append(row)
                print(
                    f"vllm {name} {label}: {dt * 1e3:.0f} ms, {row['tok_s']:.1f} tok/s",
                    flush=True,
                )
                dump(args, rows)
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def compare(ours_path: str, vllm_path: str) -> None:
    with open(ours_path) as f:
        ours = {(r["scenario"], r["config"]): r for r in json.load(f)}
    with open(vllm_path) as f:
        vllm = {(r["scenario"], r["config"]): r for r in json.load(f)}

    def order(row):
        batch = row.get("batch") or int(
            re.search(r"batch=(\d+)", row["config"]).group(1)
        )
        beams = row.get("beams", 1)
        return (batch, beams, 1 if row.get("dtype") == "float32" else 0)

    for name, enc_tokens, out_tokens in SCENARIOS:
        keys = sorted(
            (k for k in ours if k[0] == name and k in vllm),
            key=lambda k: order(ours[k]),
        )
        if not keys:
            continue
        print(f"\n### {name}: encoder {enc_tokens} -> {out_tokens} generated\n")
        print("| config | ours ms | ours tok/s | vllm ms | vllm tok/s | speedup |")
        print("|---|---:|---:|---:|---:|---:|")
        for k in keys:
            o, v = ours[k], vllm[k]
            print(
                f"| {k[1]:<22} | {o['ms']:>8.0f} | {o['tok_s']:>10.1f} "
                f"| {v['ms']:>7.0f} | {v['tok_s']:>10.1f} "
                f"| {v['ms'] / o['ms']:>7.2f}x |"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--engine", choices=("ours", "vllm"))
    ap.add_argument("--scenarios", default="", help="comma list, default all")
    ap.add_argument(
        "--configs",
        default="",
        help="semicolon list (labels contain commas), default all",
    )
    ap.add_argument(
        "--batches",
        default="",
        help="greedy (beams=1) batch sweep, comma list; overrides --configs",
    )
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--json", default=None, help="write raw results to a JSON file")
    ap.add_argument("--compare", nargs=2, metavar=("OURS", "VLLM"))
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return
    if not args.engine:
        ap.error("--engine is required (or use --compare)")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def select(spec, all_names, label, sep=","):
        wanted = {s.strip() for s in spec.split(sep) if s.strip()}
        bad = wanted - set(all_names)
        if bad:
            raise SystemExit(
                f"unknown {label}(s) {sorted(bad)}; choose from {all_names}"
            )
        return wanted

    scen_wanted = select(args.scenarios, [s[0] for s in SCENARIOS], "scenario")
    scenarios = [s for s in SCENARIOS if not scen_wanted or s[0] in scen_wanted]

    if args.batches:
        batches = [int(b) for b in args.batches.split(",") if b.strip()]
        args.configs = [
            (f"batch={b}, beams=1, bf16", b, 1, "bfloat16") for b in batches
        ]
    else:
        cfg_wanted = select(args.configs, [c[0] for c in CONFIGS], "config", sep=";")
        args.configs = [c for c in CONFIGS if not cfg_wanted or c[0] in cfg_wanted]

    rows = (
        run_ours(args, tokenizer, scenarios)
        if args.engine == "ours"
        else (run_vllm(args, tokenizer, scenarios))
    )
    if args.json:
        dump(args, rows)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
