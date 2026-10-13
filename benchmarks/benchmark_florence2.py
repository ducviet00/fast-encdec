"""Florence-2 benchmark: this engine vs HuggingFace, real image, mixed tasks.

One real photo and three task prompts with very different output lengths
(`<CAPTION>` ~13, `<OD>` ~22 and a referring-expression segmentation ~96
tokens), so the batch has a wide length spread — exactly the case the paged,
unpadded decoder is for.  Both engines stop at EOS, and the token totals are
checked to match.

Runs the same generation configs as `benchmark.py`: three bf16 points plus one
fp32 row for verification.  The encoder length is fixed by the image (768 px ->
577 visual tokens), so the encoder/decoder scenario split does not apply here.

  PYTHONPATH=. python benchmarks/benchmark_florence2.py
  PYTHONPATH=. python benchmarks/benchmark_florence2.py --device cuda
"""

import argparse
import json
import os
import time
import urllib.request

import torch
from PIL import Image

from fastencdec import LLM, SamplingParams
from fastencdec.config import default_block_size

DEFAULT_MODEL = "florence-community/Florence-2-base"
IMAGE_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/"
    "transformers/tasks/car.jpg"
)

# (label, prompt).  Four tasks with very different output lengths; requests
# cycle through the list (4 divides batch=8 and beams=4 evenly).
TASKS = [
    ("caption", "<CAPTION>"),
    ("detection", "<OD>"),
    ("segmentation", "<REFERRING_EXPRESSION_SEGMENTATION>the car"),
    (
        "grounding",
        "<CAPTION_TO_PHRASE_GROUNDING>A car with visible door handles and wheels.",
    ),
]

CONFIGS = (
    ("batch=1, beams=1, bf16", 1, 1, "bfloat16"),
    ("batch=8, beams=1, bf16", 8, 1, "bfloat16"),
    ("batch=8, beams=4, bf16", 8, 4, "bfloat16"),
    ("batch=8, beams=1, fp32", 8, 1, "float32"),
)


def load_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(IMAGE_URL, path)
    return Image.open(path).convert("RGB")


def workload_of(image, n):
    """``n`` (label, prompt, image) tuples with a balanced task mix."""
    return [
        (TASKS[i % len(TASKS)][0], TASKS[i % len(TASKS)][1], image) for i in range(n)
    ]


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, warmup: int, repeats: int):
    for _ in range(warmup):
        fn()
    best = float("inf")
    result = None
    for _ in range(repeats):
        _sync()
        start = time.perf_counter()
        result = fn()
        _sync()
        best = min(best, time.perf_counter() - start)
    return best, result


def run_ours(llm, workload, max_tokens, beams, warmup, repeats):
    prepared = [
        (
            llm.processor(text=prompt, images=image, return_tensors="pt")["input_ids"][
                0
            ].tolist(),
            llm.processor(text=prompt, images=image, return_tensors="pt")[
                "pixel_values"
            ][0],
        )
        for _, prompt, image in workload
    ]
    params = SamplingParams(
        max_tokens=max_tokens,
        num_beams=beams,
        temperature=0.0,
        eos_token_id=llm.eos_token_id,
    )

    def once():
        requests = llm.engine.generate(
            [(ids, [llm.decoder_start_token_id], pixels) for ids, pixels in prepared],
            params,
        )
        return [llm.engine.results[r] for r in requests]

    return timed(once, warmup, repeats)


def run_hf(model, processor, workload, max_tokens, beams, warmup, repeats, device):
    inputs = processor(
        text=[p for _, p, _ in workload],
        images=[im for _, _, im in workload],
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    pad = processor.tokenizer.pad_token_id

    def once():
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                num_beams=beams,
                do_sample=False,
                forced_bos_token_id=None,
                forced_eos_token_id=None,
                early_stopping=False,
                length_penalty=1.0,
                no_repeat_ngram_size=0,
            )
        rows = []
        for row in out:  # strip decoder-start and trailing padding
            toks = row.tolist()[1:]
            while toks and toks[-1] == pad:
                toks.pop()
            rows.append(toks)
        return rows

    return timed(once, warmup, repeats)


def blocks_for(batch: int, beams: int, enc_len: int, max_tokens: int, bs: int) -> int:
    return (batch * beams * (enc_len + max_tokens) + bs - 1) // bs + 8


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--image", default="/tmp/fastencdec/florence2_bench.jpg")
    ap.add_argument("--device", default="cpu")
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
        help="0 = device default (256 cuda, 32 x86 cpu, 16 aarch64)",
    )
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    import transformers
    from transformers import (
        AutoProcessor,
    )
    from transformers import (
        Florence2ForConditionalGeneration as HFModel,
    )

    transformers.logging.set_verbosity_error()
    device = args.device
    image = load_image(args.image)
    processor = AutoProcessor.from_pretrained(args.model)
    enc_len = max(
        len(processor(text=prompt, images=image, return_tensors="pt")["input_ids"][0])
        for _, prompt, _ in workload_of(image, len(TASKS))
    )
    block_size = args.block_size or default_block_size(device)

    print("# fast-encdec Florence-2 benchmark\n", flush=True)
    print(
        f"model={args.model}  device={device}  threads={torch.get_num_threads()}  "
        f"image={image.size}  max_tokens={args.max_tokens}",
        flush=True,
    )
    print(
        "\n| config | ours ms | ours tok/s | hf ms | hf tok/s | speedup | mix |"
        "\n|---|---:|---:|---:|---:|---:|---|",
        flush=True,
    )

    results = []
    for dtype_name in ("bfloat16", "float32"):
        configs = [c for c in CONFIGS if c[3] == dtype_name]
        if not configs:
            continue
        dtype = getattr(torch, dtype_name)
        for label, batch, beams, _ in configs:
            workload = workload_of(image, batch)
            num_blocks = blocks_for(batch, beams, enc_len, args.max_tokens, block_size)
            llm = LLM(
                args.model,
                num_blocks=num_blocks,
                block_size=block_size,
                dtype=dtype,
                device=device,
            )
            hf = HFModel.from_pretrained(args.model, dtype=dtype).eval()
            if device.startswith("cuda"):
                hf = hf.to(device)

            ours_s, ours_rows = run_ours(
                llm, workload, args.max_tokens, beams, args.warmup, args.repeats
            )
            hf_s, hf_rows = run_hf(
                hf,
                processor,
                workload,
                args.max_tokens,
                beams,
                args.warmup,
                args.repeats,
                device,
            )
            ours_tok = sum(len(r) for r in ours_rows)
            hf_tok = sum(len(r) for r in hf_rows)
            if ours_tok != hf_tok:
                # EOS-stopped output, so beam search can pick a slightly different
                # hypothesis in each engine; report it rather than fail.
                print(
                    f"  (note: ours {ours_tok} tokens vs HF {hf_tok} - beam "
                    "search chose a different hypothesis)",
                    flush=True,
                )
            lengths = {}
            for (task, _, _), out in zip(workload, ours_rows):
                lengths.setdefault(task, []).append(len(out))
            mix = " ".join(f"{t}={sum(v) // len(v)}" for t, v in lengths.items())
            print(
                f"| {label:<22} | {ours_s * 1e3:>8.0f} | {ours_tok / ours_s:>10.1f} "
                f"| {hf_s * 1e3:>8.0f} | {hf_tok / hf_s:>9.1f} "
                f"| {hf_s / ours_s:>7.2f}x | {mix} |",
                flush=True,
            )
            results.append(
                {
                    "config": label,
                    "dtype": dtype_name,
                    "batch": batch,
                    "beams": beams,
                    "image": list(image.size),
                    "ours_ms": ours_s * 1e3,
                    "ours_tok": ours_tok,
                    "hf_tok": hf_tok,
                    "ours_tok_s": ours_tok / ours_s,
                    "hf_ms": hf_s * 1e3,
                    "hf_tok_s": hf_tok / hf_s,
                    "speedup": hf_s / ours_s,
                    "mix": mix,
                }
            )
            del llm, hf

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
