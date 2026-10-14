"""Phase profile of a BART generate: fast-encdec vs the vLLM BART plugin.

The point is to split one generate's wall time into *encoder* / *per-step
decoder* / *logits* / *sampling*, so the two engines can be compared phase by
phase instead of only end to end.

fast-encdec is instrumented directly: CUDA events around the encoder, the
decoder forward (including the CUDA-graph replay) and the LM head, and host
timers around the per-step bookkeeping and the sampling, so ``decode`` is the
residual ``step - encode - logits``.  vLLM has no offline hooks -- its
``RequestOutput.metrics`` is ``None`` for an offline ``LLM`` -- so it is measured
without internals two ways:

* a decode-length sweep: ``total(n) = encoder + n * step`` (least squares), so
  the slope is the per-step cost and the intercept is the encoder plus fixed
  overhead;
* a ``torch.profiler`` kernel table (``--profiler``), which names the kernels
  and their device time.

Both engines are driven through the same prompt and the same fixed decode
length, as in ``benchmark_vllm.py``.

  PYTHONPATH=. python benchmarks/profile_bart.py --engine ours --device cuda
  ~/vllmenv/bin/python benchmarks/profile_bart.py --engine vllm --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import benchmark_vllm as bv
import torch


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class PhaseTimer:
    """GPU phases via CUDA events, host phases via ``perf_counter``.

    Event pairs are queued and read once at the end, so the instrumentation
    does not add a synchronise to every phase (which would inflate the
    host-bound ones).
    """

    def __init__(self):
        self.gpu_ms: dict[str, float] = defaultdict(float)
        self.host_ms: dict[str, float] = defaultdict(float)
        self.n: dict[str, int] = defaultdict(int)
        self._pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    def gpu(self, name: str):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.start = torch.cuda.Event(enable_timing=True)
                self.start.record()

            def __exit__(self, *exc):
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                timer._pending.append((name, self.start, end))
                timer.n[name] += 1

        return _Ctx()

    def host(self, name: str):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.t = time.perf_counter()

            def __exit__(self, *exc):
                timer.host_ms[name] += (time.perf_counter() - self.t) * 1e3
                timer.n[name] += 1

        return _Ctx()

    def flush(self) -> None:
        _sync()
        for name, start, end in self._pending:
            self.gpu_ms[name] += start.elapsed_time(end)
        self._pending.clear()


def _wrap_class(cls, attr: str, timer: PhaseTimer, name: str, kind: str):
    original = getattr(cls, attr)

    def wrapped(self, *args, **kwargs):
        scope = timer.gpu(name) if kind == "gpu" else timer.host(name)
        with scope:
            return original(self, *args, **kwargs)

    setattr(cls, attr, wrapped)
    return cls, attr, original


def instrument_ours(llm, timer: PhaseTimer) -> list:
    """Patch the engine's phases; returns the patches for :func:`restore`."""
    from fastencdec.engine import model_runner as mr
    from fastencdec.engine.llm_engine import LLMEngine

    patches = [
        _wrap_class(mr.ModelRunner, "_encode", timer, "encode", "gpu"),
        _wrap_class(mr.ModelRunner, "_prepare", timer, "prepare", "host"),
        _wrap_class(mr.ModelRunner, "run", timer, "step", "gpu"),
        _wrap_class(LLMEngine, "_postprocess", timer, "sample", "host"),
    ]

    model = llm.engine.runner.model
    original = model.compute_logits

    def compute_logits(hidden, _original=original):
        with timer.gpu("logits"):
            return _original(hidden)

    model.compute_logits = compute_logits
    patches.append((model, "compute_logits", original))
    return patches


def restore(patches) -> None:
    for holder, attr, original in patches:
        setattr(holder, attr, original)


def build_ours(args, prompt: str, enc_len: int):
    from fastencdec import LLM, SamplingParams
    from fastencdec.config import default_block_size

    block_size = default_block_size(args.device)
    num_blocks = bv.blocks_for(
        [{"batch": args.batch, "beams": args.beams}],
        enc_len,
        args.out_tokens,
        block_size,
    )
    llm = LLM(
        args.model,
        num_blocks=num_blocks,
        block_size=block_size,
        max_num_seqs=args.batch * args.beams,
        dtype=getattr(torch, args.dtype),
        device=args.device,
    )
    params = SamplingParams(
        max_tokens=args.out_tokens,
        num_beams=args.beams,
        min_length=args.out_tokens + 1,
        **bv.DECODE,
    )
    prompts = [prompt] * args.batch
    return llm, lambda: llm.generate(prompts, params)


def build_vllm(args, prompt: str, enc_len: int):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    # in-process engine: one engine only, and it lets torch.profiler see the
    # kernels (a separate engine-core process would hide them)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import vllm_bart_plugin  # noqa: F401  (registers the model)
    from vllm import LLM, SamplingParams

    kwargs = {
        "model": args.model,
        "dtype": args.dtype,
        "max_model_len": max(1024, enc_len + args.out_tokens + 16),
        "max_num_seqs": args.batch * args.beams,
        "max_num_batched_tokens": 11024,
        "enforce_eager": args.vllm_eager,
    }
    if args.device.startswith("cuda"):
        kwargs["gpu_memory_utilization"] = 0.6
    if args.beams > 1:
        from vllm_beam_search.scheduler import BeamSearchScheduler

        kwargs["scheduler_cls"] = BeamSearchScheduler
        kwargs["async_scheduling"] = True
    llm = LLM(**kwargs)

    extra = dict(bv.DECODE) if args.beams > 1 else {}
    if args.beams > 1:
        extra["beam_width"] = args.beams
    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.out_tokens,
        min_tokens=args.out_tokens,
        ignore_eos=True,
        extra_args=extra,
    )
    prompts = [bv._vllm_prompt(prompt)] * args.batch
    return llm, lambda: llm.generate(prompts, params)


def timed(run, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        run()
    best = float("inf")
    for _ in range(repeats):
        _sync()
        start = time.perf_counter()
        run()
        _sync()
        best = min(best, time.perf_counter() - start)
    return best * 1e3


def sweep(args, build, prompt, enc_len, lengths) -> dict[int, float]:
    """Time one generate at each decode length (fresh engine each time)."""
    out = {}
    for n in lengths:
        args.out_tokens = n
        _, run = build(args, prompt, enc_len)
        out[n] = timed(run, args.warmup, args.repeats)
    return out


def fit(lengths, times):
    """Least-squares ``total(n) = intercept + slope * n``."""
    mean_n = sum(lengths) / len(lengths)
    mean_t = sum(times) / len(times)
    num = sum((n - mean_n) * (t - mean_t) for n, t in zip(lengths, times))
    den = sum((n - mean_n) ** 2 for n in lengths)
    slope = num / den
    return mean_t - slope * mean_n, slope


def ours_phases(args, prompt, enc_len) -> None:
    llm, run = build_ours(args, prompt, enc_len)
    run()  # warmup: capture the CUDA graphs / build the encoder cache
    _sync()

    # Instrument only after warmup: the capture path must not record events
    # inside a graph, and the measured pass then replays it.
    timer = PhaseTimer()
    patches = instrument_ours(llm, timer)
    _sync()
    start = time.perf_counter()
    run()
    _sync()
    wall = (time.perf_counter() - start) * 1e3
    timer.flush()
    restore(patches)

    steps = timer.n["step"]
    encode = timer.gpu_ms["encode"]
    logits = timer.gpu_ms["logits"]
    step = timer.gpu_ms["step"]
    decode = step - encode - logits
    rows = [
        ("encoder", encode, timer.n["encode"]),
        ("prepare (host)", timer.host_ms["prepare"], timer.n["prepare"]),
        ("decoder", decode, steps),
        ("logits (LM head)", logits, steps),
        ("sample (host+sync)", timer.host_ms["sample"], timer.n["sample"]),
    ]
    print(
        f"engine=ours device={args.device} dtype={args.dtype} batch={args.batch} "
        f"beams={args.beams} enc={enc_len} out={args.out_tokens}"
    )
    print(f"steps={steps}  wall={wall:.1f} ms")
    print(f"{'phase':<20}{'total ms':>10}{'count':>8}{'per call ms':>13}")
    for name, total, count in rows:
        per = total / count if count else 0.0
        print(f"{name:<20}{total:>10.1f}{count:>8}{per:>13.3f}")
    acc = sum(r[1] for r in rows)
    print(
        f"{'attributed':<20}{acc:>10.1f}{'':>8}{wall - acc:>13.1f} (wall - attributed)"
    )


def vllm_single(args, prompt, enc_len) -> None:
    _, run = build_vllm(args, prompt, enc_len)
    ms = timed(run, args.warmup, args.repeats)
    print(
        f"engine=vllm device={args.device} dtype={args.dtype} batch={args.batch} "
        f"beams={args.beams} enc={enc_len} out={args.out_tokens}"
    )
    print(f"total={ms:.1f} ms   {args.batch * args.out_tokens / (ms / 1e3):.1f} tok/s")


def profiler_table(args, build, prompt, enc_len) -> None:
    from torch.profiler import ProfilerActivity, profile

    _, run = build(args, prompt, enc_len)
    run()
    _sync()
    acts = [ProfilerActivity.CPU]
    if args.device.startswith("cuda"):
        acts.append(ProfilerActivity.CUDA)
    with profile(activities=acts) as prof:
        run()
    _sync()
    events = prof.key_averages()
    launches = sum(
        e.count for e in events if e.key in ("cudaLaunchKernel", "cuLaunchKernel")
    )
    syncs = sum(e.count for e in events if e.key == "cudaStreamSynchronize")
    print(f"\n-- torch.profiler (launches={launches} syncs={syncs}) --")
    if args.device.startswith("cuda"):
        print(
            prof.key_averages().table(sort_by="device_time_total", row_limit=args.rows)
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["ours", "vllm"], default="ours")
    ap.add_argument("--model", default=bv.MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--enc-tokens", type=int, default=256)
    ap.add_argument("--out-tokens", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument(
        "--vllm-eager",
        action="store_true",
        help="disable vLLM CUDA graphs",
    )
    ap.add_argument("--sweep", action="store_true", help="decode-length sweep")
    ap.add_argument("--profiler", action="store_true", help="torch.profiler table")
    ap.add_argument("--rows", type=int, default=25)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt = bv.make_prompt(tokenizer, args.enc_tokens)
    enc_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

    if args.engine == "ours":
        ours_phases(args, prompt, enc_len)
        if args.sweep:
            times = sweep(args, build_ours, prompt, enc_len, sorted({1, 64, 128, 256}))
            intercept, slope = fit(list(times), list(times.values()))
            print("\n-- ours length sweep --")
            for n, t in times.items():
                print(f"{n:>12}{t:>12.1f}")
            print(
                f"fit: encoder+overhead ~= {intercept:.1f} ms   "
                f"decode step ~= {slope:.3f} ms/token"
            )
    else:
        if not args.profiler:
            vllm_single(args, prompt, enc_len)
        if args.sweep:
            # A separate process per length: a second mp engine core in the same
            # process fails to start, so the sweep is driven from the shell.
            print("(vllm sweep: run once per --out-tokens from the shell)")

    if args.profiler:
        build = build_ours if args.engine == "ours" else build_vllm
        profiler_table(args, build, prompt, enc_len)


if __name__ == "__main__":
    main()
