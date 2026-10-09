"""Benchmark Florence-2 on CPU: fast-encdec vs HuggingFace ``generate``.

The workload mixes three task types with very different output lengths:

    <CAPTION>                              short   (~15 tokens)
    <OD>                                   medium  (~25 tokens)
    <REFERRING_EXPRESSION_SEGMENTATION>... long    (hundreds of tokens)

HuggingFace decodes the whole padded batch for ``max_new_tokens`` steps, so the
short requests keep paying for the long ones.  fast-encdec uses a paged KV cache
with no padding: each request decodes only up to its own length, and once it
emits EOS it leaves the running batch (continuous batching).

Three sweeps are reported:
  1. batch scaling  - mixed tasks, greedy
  2. beam scaling   - mixed tasks, beam search (paged blocks are shared)
  3. continuous batching - many requests through a small running window vs HF
                           running the whole batch at once

    PYTHONPATH=. python benchmarks/benchmark_florence2.py --quick
    PYTHONPATH=. python benchmarks/benchmark_florence2.py --batches 3,6,12 --dtype bfloat16
"""

import argparse
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
from PIL import Image

from fastencdec import Florence2LLM, SamplingParams

DEFAULT_MODEL = "florence-community/Florence-2-base"
IMAGE_URL = ("https://huggingface.co/datasets/huggingface/documentation-images/"
             "resolve/main/transformers/tasks/car.jpg")

# (label, prompt).  Ordered short -> long; requests cycle through this list.
TASKS = [
    ("caption", "<CAPTION>"),
    ("detection", "<OD>"),
    ("segmentation", "<REFERRING_EXPRESSION_SEGMENTATION>the car"),
]


def load_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        urllib.request.urlretrieve(IMAGE_URL, path)
    return Image.open(path).convert("RGB")


def workload_of(image, n):
    """``n`` (label, prompt, image) tuples with a balanced task mix."""
    return [(TASKS[i % len(TASKS)][0], TASKS[i % len(TASKS)][1], image)
            for i in range(n)]


def timed(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    best, result = float("inf"), None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return best, result


def prepare_ours(llm, workload):
    prepared = []
    for _, prompt, image in workload:
        inputs = llm.processor(text=prompt, images=image, return_tensors="pt")
        prepared.append((inputs["input_ids"][0].tolist(),
                         inputs["pixel_values"][0]))
    return prepared


def run_ours(llm, prepared, max_tokens, beams, warmup, repeats):
    params = SamplingParams(max_tokens=max_tokens, num_beams=beams,
                            temperature=0.0, eos_token_id=llm.eos_token_id)

    def once():
        requests = [llm.engine.add_request(ids, [llm.decoder_start_token_id],
                                           params, pixel_values=pixels)
                    for ids, pixels in prepared]
        llm.engine.run()
        return [llm.engine.results[r] for r in requests]

    return timed(once, warmup, repeats)


def run_hf(model, processor, workload, max_tokens, beams, warmup, repeats):
    inputs = processor(text=[p for _, p, _ in workload],
                       images=[im for _, _, im in workload],
                       padding=True, return_tensors="pt")
    pad = processor.tokenizer.pad_token_id

    def once():
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens,
                                 num_beams=beams, do_sample=False,
                                 forced_bos_token_id=None,
                                 forced_eos_token_id=None,
                                 early_stopping=False)
        rows = []
        for row in out:  # strip decoder-start and trailing padding
            toks = row.tolist()[1:]
            while toks and toks[-1] == pad:
                toks.pop()
            rows.append(toks)
        return rows

    return timed(once, warmup, repeats)


def token_mix(workload, outputs):
    lengths = {}
    for (label, _, _), out in zip(workload, outputs):
        lengths.setdefault(label, []).append(len(out))
    return " ".join(f"{label}={sum(v) / len(v):.0f}" for label, v in lengths.items())


def row(batch, ours_s, hf_s, ours_tok, hf_tok, mix):
    return (f"{batch:>6} | {ours_s * 1e3:>9.0f} | {hf_s * 1e3:>9.0f} | "
            f"{hf_s / ours_s:>7.2f}x | {ours_tok / ours_s:>11.1f} | "
            f"{hf_tok / hf_s:>10.1f} | {ours_tok:>8d} | {hf_tok:>7d} | {mix}")


HEADER = (f"{'config':>6} | {'ours ms':>9} | {'hf ms':>9} | {'speedup':>8} | "
          f"{'ours tok/s':>11} | {'hf tok/s':>10} | {'ours tok':>8} | "
          f"{'hf tok':>7} | mix")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--image", default="/tmp/opencode/florence2_bench.jpg")
    parser.add_argument("--batches", default="3,6,12")
    parser.add_argument("--beams", default="1,3,5")
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--windows", default="6,12")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--beam-tokens", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float32", "bfloat16"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--compile-mm-encoder", action="store_true",
                        help="torch.compile the vision tower + projector (opt-in)")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    if args.quick:
        args.batches, args.beams = "3,6", "1,3"
        args.requests, args.windows = 6, "3,6"
        args.max_tokens, args.beam_tokens = 96, 64
        args.warmup, args.repeats = 0, 1

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    batches = [int(b) for b in args.batches.split(",") if b]
    beams_list = [int(b) for b in args.beams.split(",") if b]
    windows = [int(w) for w in args.windows.split(",") if w]
    max_seqs = max(max(batches), max(windows), args.requests)

    from transformers import AutoProcessor
    from transformers import Florence2ForConditionalGeneration as HFFlorence2

    image = load_image(args.image)
    print(f"model={args.model}  dtype={args.dtype}  image={image.size}  "
          f"max_tokens={args.max_tokens}  beam_tokens={args.beam_tokens}")

    llm = Florence2LLM(args.model, num_blocks=4096, block_size=16,
                       max_num_seqs=max_seqs, dtype=dtype,
                       compile_mm_encoder=args.compile_mm_encoder)
    processor = AutoProcessor.from_pretrained(args.model)
    hf = HFFlorence2.from_pretrained(args.model, dtype=dtype).eval()
    results = []

    # ---------------------------------------------------------- batch scaling
    print(f"\n== greedy batch scaling (mixed tasks) ==\n{HEADER}")
    for batch in batches:
        workload = workload_of(image, batch)
        prepared = prepare_ours(llm, workload)
        ours_s, ours_out = run_ours(llm, prepared, args.max_tokens, 1,
                                    args.warmup, args.repeats)
        hf_s, hf_out = run_hf(hf, processor, workload, args.max_tokens, 1,
                              args.warmup, args.repeats)
        ours_tok = sum(map(len, ours_out))
        hf_tok = sum(map(len, hf_out))
        print(row(batch, ours_s, hf_s, ours_tok, hf_tok,
                  token_mix(workload, ours_out)))
        results.append(dict(sweep="batch", config=batch, ours_ms=ours_s * 1e3,
                            hf_ms=hf_s * 1e3, speedup=hf_s / ours_s,
                            ours_tok=ours_tok, hf_tok=hf_tok))

    # ----------------------------------------------------------- beam scaling
    print(f"\n== beam scaling (batch=3, mixed tasks) ==\n{HEADER}")
    workload = workload_of(image, 3)
    prepared = prepare_ours(llm, workload)
    for beams in beams_list:
        ours_s, ours_out = run_ours(llm, prepared, args.beam_tokens, beams,
                                    args.warmup, args.repeats)
        hf_s, hf_out = run_hf(hf, processor, workload, args.beam_tokens, beams,
                              args.warmup, args.repeats)
        ours_tok = sum(map(len, ours_out))
        hf_tok = sum(map(len, hf_out))
        print(row(beams, ours_s, hf_s, ours_tok, hf_tok,
                  token_mix(workload, ours_out)))
        results.append(dict(sweep="beam", config=beams, ours_ms=ours_s * 1e3,
                            hf_ms=hf_s * 1e3, speedup=hf_s / ours_s,
                            ours_tok=ours_tok, hf_tok=hf_tok))

    # ----------------------------------------------------- continuous batching
    print(f"\n== continuous batching (N={args.requests} mixed, greedy) ==\n"
          f"{HEADER}")
    workload = workload_of(image, args.requests)
    prepared = prepare_ours(llm, workload)
    hf_s, hf_out = run_hf(hf, processor, workload, args.max_tokens, 1,
                          args.warmup, args.repeats)
    hf_tok = sum(map(len, hf_out))
    print(f"{'HF(all)':>6} | {'-':>9} | {hf_s * 1e3:>9.0f} | {'-':>8} | "
          f"{'-':>11} | {hf_tok / hf_s:>10.1f} | {'-':>8} | {hf_tok:>7d} | "
          f"{token_mix(workload, hf_out)}")
    for window in windows:
        llm.engine.scheduler.max_num_seqs = window
        ours_s, ours_out = run_ours(llm, prepared, args.max_tokens, 1,
                                    args.warmup, args.repeats)
        ours_tok = sum(map(len, ours_out))
        print(row(f"w={window}", ours_s, hf_s, ours_tok, hf_tok,
                  token_mix(workload, ours_out)))
        results.append(dict(sweep="continuous", config=window,
                            ours_ms=ours_s * 1e3, hf_ms=hf_s * 1e3,
                            speedup=hf_s / ours_s, ours_tok=ours_tok,
                            hf_tok=hf_tok))

    if args.json:
        import json
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
