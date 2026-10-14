"""BART: this engine vs the vLLM BART plugin (``vllm-project/bart-plugin``).

The two engines pin different torch builds -- this repo is developed against
torch 2.11+cu128, the plugin's vLLM wheel against 2.13+cu132 -- so they cannot
share an interpreter and this script drives both as worker processes:

  # in the engine's env, pointing at the plugin's env (see --vllm-python)
  PYTHONPATH=. python benchmarks/benchmark_vllm.py \\
      --vllm-python ~/vllmenv/bin/python --device cuda --scenarios '256->256'

``--device cpu`` needs a vLLM built for CPU; the beam rows are CUDA-only
because ``BeamSearchScheduler`` needs MRV2 async scheduling, which
``vllm/platforms/cpu.py`` forces off.

The plugin env is vLLM 0.28.0 plus both plugins -- the beam plugin's supported
BART path -- installed separately from this repo's env::

    uv venv ~/vllmenv --python 3.12
    uv pip install --python ~/vllmenv/bin/python 'vllm==0.28.0'
    uv pip install --python ~/vllmenv/bin/python -e ~/bart-plugin \\
        -e ~/vllm-beam-search-plugin

``--engine ours|vllm`` runs a single worker (model load + timing) and prints its
result as JSON; the driver calls it once per (engine, dtype, scenario) and joins
the two.  Both workers build the prompt with the same tokenizer call, so the
encoder length matches, and both mask EOS so each decodes exactly the
scenario's token count -- the speedup is engine-vs-engine, as in
``benchmark.py``.

Differences the plugin forces, reported rather than hidden:

* beam search is a *second* plugin (``vllm-beam-search-plugin``), selected by
  ``scheduler_cls``; vLLM's ``SamplingParams`` carries neither ``num_beams``
  nor ``length_penalty``/``no_repeat_ngram_size``, so those ride in
  ``extra_args`` (the ``vllm_xargs`` of the plugin's HTTP API);
* the same is true of the checkpoint's n-gram heuristic on the *greedy* rows:
  vLLM greedy cannot express ``no_repeat_ngram_size``, so our greedy rows
  apply the checkpoint's heuristics and the plugin's do not.  The token counts
  still match; only the sampled tokens can differ.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

MODEL = "facebook/bart-large-cnn"
# bart-large-cnn's decoding heuristics (see AGENTS.md gotchas).
DECODE = {"length_penalty": 2.0, "no_repeat_ngram_size": 3}

SCENARIOS = (
    ("128->512", 128, 512),
    ("512->128", 512, 128),
    ("256->256", 256, 256),
)
CONFIGS = (
    {"name": "batch=1, beams=1, bf16", "batch": 1, "beams": 1, "dtype": "bfloat16"},
    {"name": "batch=8, beams=1, bf16", "batch": 8, "beams": 1, "dtype": "bfloat16"},
    {"name": "batch=8, beams=4, bf16", "batch": 8, "beams": 4, "dtype": "bfloat16"},
    {"name": "batch=8, beams=1, fp32", "batch": 8, "beams": 1, "dtype": "float32"},
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

RESULT_MARKER = "##RESULT##"


def make_prompt(tokenizer, target_tokens: int) -> str:
    """A prompt whose encoder length is about ``target_tokens``."""
    text = " ".join(PARAGRAPHS)
    while len(tokenizer(text, add_special_tokens=False)["input_ids"]) < target_tokens:
        text += " " + text
    ids = tokenizer(
        text, add_special_tokens=False, truncation=True, max_length=target_tokens
    )["input_ids"]
    return tokenizer.decode(ids)


def blocks_for(configs, enc_tokens: int, out_tokens: int, block_size: int) -> int:
    """``num_blocks`` covering the widest config of one scenario."""
    worst = max(c["batch"] * c["beams"] for c in configs)
    tokens = worst * (enc_tokens + out_tokens + 8)
    return (tokens + block_size - 1) // block_size + 8


def timed(fn, warmup: int, repeats: int):
    """Return ``(best_seconds, every result)`` so the caller can check them."""
    for _ in range(warmup):
        fn()
    best = float("inf")
    results = []
    for _ in range(repeats):
        start = time.perf_counter()
        results.append(fn())
        best = min(best, time.perf_counter() - start)
    return best, results


def _finish(name: str, dt: float, counts: list[int]) -> dict:
    """One config's result; every repeat must decode the same token count."""
    if len(set(counts)) != 1:
        raise RuntimeError(f"{name}: decoded {sorted(set(counts))} tokens across runs")
    return {"ms": dt * 1e3, "tokens": counts[0], "tok_s": counts[0] / dt}


# --------------------------------------------------------------------------- #
# workers
# --------------------------------------------------------------------------- #
def worker_ours(args, prompt: str, enc_len: int, configs: list[dict]) -> dict:
    import torch

    from fastencdec import LLM, SamplingParams
    from fastencdec.config import default_block_size

    if args.threads:
        torch.set_num_threads(args.threads)
    dtype = getattr(torch, args.dtype)
    block_size = default_block_size(args.device)
    llm = LLM(
        args.model,
        num_blocks=blocks_for(configs, enc_len, args.out_tokens, block_size),
        block_size=block_size,
        max_num_seqs=max(c["batch"] * c["beams"] for c in configs),
        dtype=dtype,
        device=args.device,
    )
    results = {}
    for c in configs:
        params = SamplingParams(
            max_tokens=args.out_tokens,
            num_beams=c["beams"],
            min_length=args.out_tokens + 1,  # EOS is never legal: fixed length
            **DECODE,
        )
        prompts = [prompt] * c["batch"]

        def run(prompts=prompts, params=params):
            return sum(
                len(o.outputs[0].token_ids) for o in llm.generate(prompts, params)
            )

        dt, counts = timed(run, args.warmup, args.repeats)
        results[c["name"]] = _finish(c["name"], dt, counts)
    return results


def _vllm_prompt(text: str) -> dict:
    """The plugin's explicit encoder/decoder prompt shape."""
    return {
        "encoder_prompt": {"prompt": "", "multi_modal_data": {"text": text}},
        "decoder_prompt": "<s>",
    }


def worker_vllm(args, prompt: str, enc_len: int, configs: list[dict]) -> dict:
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    # the plugin's BART config requires the V2 model runner (and the GPU path
    # selects it); on CPU it is not the default, so ask for it explicitly.
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    if args.threads:
        os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))

    import torch
    from vllm import SamplingParams

    # One engine per worker process: greedy and beam search need different
    # schedulers (the scheduler is fixed at construction), and an in-process
    # engine does not release its KV pool on ``del``, so the driver runs this
    # worker once per group instead of building both here.
    is_beam = args.beam
    results = {}
    if is_beam and not torch.cuda.is_available():
        msg = (
            "n/a on CPU: BeamSearchScheduler needs MRV2 with async "
            "scheduling, and vllm/platforms/cpu.py forces async_scheduling "
            "off - this row needs the CUDA build"
        )
        for c in configs:
            results[c["name"]] = {"error": msg}
        return results
    try:
        llm = _build_vllm(args, enc_len, configs, is_beam)
    except Exception as exc:  # noqa: BLE001 - reported per config below
        for c in configs:
            results[c["name"]] = {"error": f"{type(exc).__name__}: {exc}"}
        return results

    for c in configs:
        extra = dict(DECODE) if is_beam else {}
        if is_beam:
            extra["beam_width"] = c["beams"]
        params = SamplingParams(
            temperature=0.0,
            max_tokens=args.out_tokens,
            min_tokens=args.out_tokens,  # EOS is never legal: fixed length
            ignore_eos=True,
            extra_args=extra,
        )
        prompts = [_vllm_prompt(prompt)] * c["batch"]

        def run(llm=llm, prompts=prompts, params=params):
            return sum(
                len(o.outputs[0].token_ids) for o in llm.generate(prompts, params)
            )

        try:
            dt, counts = timed(run, args.warmup, args.repeats)
            results[c["name"]] = _finish(c["name"], dt, counts)
        except Exception as exc:  # noqa: BLE001
            results[c["name"]] = {"error": f"{type(exc).__name__}: {exc}"}
    return results


def _build_vllm(args, enc_len: int, configs: list[dict], is_beam: bool):
    from vllm import LLM

    kwargs = {
        "model": args.model,
        "dtype": args.dtype,
        # bart-large-cnn has a 1024-entry position table; the plugin tokenizes
        # the encoder text to its real length, so this just needs to clear the
        # widest decoder here (it always does).
        "max_model_len": max(1024, enc_len + args.out_tokens + 16),
        "max_num_seqs": max(c["batch"] * c["beams"] for c in configs),
        # the plugin feeds the encoder text in as a "multimodal item", so the
        # scheduler's token budget must clear that (see the plugin's examples).
        "max_num_batched_tokens": 11024,
        "enforce_eager": args.vllm_eager,
    }
    if args.device.startswith("cuda"):
        kwargs["gpu_memory_utilization"] = 0.6
    if is_beam:
        from vllm_beam_search.scheduler import BeamSearchScheduler

        kwargs["scheduler_cls"] = BeamSearchScheduler
        kwargs["async_scheduling"] = True
    return LLM(**kwargs)


def run_worker(args) -> None:
    from transformers import AutoTokenizer

    configs = json.loads(args.configs)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt = make_prompt(tokenizer, args.enc_tokens)
    enc_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if args.engine == "ours":
        results = worker_ours(args, prompt, enc_len, configs)
    else:
        results = worker_vllm(args, prompt, enc_len, configs)
    print(
        RESULT_MARKER
        + json.dumps(
            {"engine": args.engine, "enc_tokens": enc_len, "results": results}
        ),
        flush=True,
    )


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def call_worker(
    args, python: str, engine: str, dtype: str, enc: int, out: int, configs, beam=False
):
    cmd = [
        python,
        __file__,
        "--engine",
        engine,
        "--model",
        args.model,
        "--dtype",
        dtype,
        "--enc-tokens",
        str(enc),
        "--out-tokens",
        str(out),
        "--configs",
        json.dumps(configs),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--threads",
        str(args.threads),
        "--device",
        args.device,
    ]
    if args.vllm_eager:
        cmd.append("--vllm-eager")
    if beam:
        cmd.append("--beam")
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=args.timeout, check=False
    )
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_MARKER):
            return json.loads(line[len(RESULT_MARKER) :])
    tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
    raise RuntimeError(f"{engine} worker failed (rc={proc.returncode}):\n{tail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["driver", "ours", "vllm"], default="driver")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--ours-python", default=sys.executable)
    ap.add_argument("--vllm-python", default=None, help="the plugin's interpreter")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cpu", help="cpu or cuda (cuda:N works)")
    ap.add_argument("--enc-tokens", type=int, default=256)
    ap.add_argument("--out-tokens", type=int, default=256)
    ap.add_argument("--configs", default="[]", help="JSON, worker mode only")
    ap.add_argument("--scenarios", default="", help="comma list, default all")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument(
        "--threads",
        type=int,
        default=4,
        help="threads for both engines (0 = each engine's own default)",
    )
    ap.add_argument("--timeout", type=int, default=3600, help="per-worker seconds")
    ap.add_argument(
        "--vllm-eager",
        action="store_true",
        help="disable vLLM CUDA graphs (vLLM defaults to them on CUDA)",
    )
    ap.add_argument(
        "--beam",
        action="store_true",
        help="worker mode: this group uses the BeamSearchScheduler",
    )
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.engine != "driver":
        run_worker(args)
        return
    if not args.vllm_python:
        raise SystemExit("--vllm-python is required (the plugin's interpreter)")

    wanted = {s.strip() for s in args.scenarios.split(",") if s.strip()}
    scenarios = [s for s in SCENARIOS if not wanted or s[0] in wanted]

    print("# fast-encdec vs vllm-bart-plugin\n", flush=True)
    print(
        f"model={args.model}  threads={args.threads}  "
        f"warmup={args.warmup} repeats={args.repeats}",
        flush=True,
    )

    results = {}
    for dtype in ("bfloat16", "float32"):
        configs = [c for c in CONFIGS if c["dtype"] == dtype]
        if not configs:
            continue
        for name, enc, out in scenarios:
            ours = call_worker(args, args.ours_python, "ours", dtype, enc, out, configs)
            # vLLM: one engine per group (fresh process), so the beam scheduler
            # never shares a device with the greedy engine's unreleased pool.
            greedy = [c for c in configs if c["beams"] == 1]
            beam = [c for c in configs if c["beams"] > 1]
            parts = []
            if greedy:
                parts.append(
                    call_worker(args, args.vllm_python, "vllm", dtype, enc, out, greedy)
                )
            if beam:
                parts.append(
                    call_worker(
                        args, args.vllm_python, "vllm", dtype, enc, out, beam, beam=True
                    )
                )
            for p in parts:
                if p["enc_tokens"] != ours["enc_tokens"]:
                    raise RuntimeError("workers tokenized the prompt differently")
            vllm = {name_: r for p in parts for name_, r in p["results"].items()}
            for c in configs:
                row = {
                    "scenario": name,
                    "config": c["name"],
                    "dtype": dtype,
                    "enc_tokens": ours["enc_tokens"],
                    "out_tokens": out,
                }
                row["ours"] = ours["results"][c["name"]]
                row["vllm"] = vllm[c["name"]]
                results[f"{name}|{c['name']}"] = row
            print(".", end="", flush=True)
            if args.json:
                with open(args.json, "w") as f:
                    json.dump(list(results.values()), f, indent=2)
    print()

    for name, enc, out in scenarios:
        rows = [r for r in results.values() if r["scenario"] == name]
        print(f"\n### {name}: encoder {rows[0]['enc_tokens']} -> {out} generated")
        print(
            "\n| config | ours ms | ours tok/s | vllm ms | vllm tok/s | speedup |"
            "\n|---|---:|---:|---:|---:|---:|"
        )
        for r in rows:
            o, v = r["ours"], r["vllm"]
            if "error" in v:
                print(
                    f"| {r['config']:<22} | {o['ms']:>8.0f} | {o['tok_s']:>10.1f} "
                    f"| n/a | n/a | n/a |"
                )
                continue
            print(
                f"| {r['config']:<22} | {o['ms']:>8.0f} | {o['tok_s']:>10.1f} "
                f"| {v['ms']:>8.0f} | {v['tok_s']:>10.1f} "
                f"| {v['ms'] / o['ms']:>7.2f}x |"
            )
        for r in rows:
            if "error" in r["vllm"]:
                print(f"\n*{r['config']}: {r['vllm']['error']}*")

    if args.json:
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
