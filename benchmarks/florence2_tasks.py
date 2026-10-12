"""Florence-2 batched accounting: 8 images x 4 tasks vs HuggingFace generate.

Runs 32 requests (caption / OCR / detection / segmentation on 8 images) with a
fixed ``max_new_tokens`` and greedy decoding on both engines, then reports per
task and in total:

* generated tokens (our engine never pads; HF pads the batch to the longest row)
* padding tokens
* generated tokens/s, and the padded-token rate HF actually pays for

    PYTHONPATH=. python benchmarks/florence2_tasks.py --device cpu
    PYTHONPATH=. python benchmarks/florence2_tasks.py --device cuda
"""

import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
from PIL import Image, ImageEnhance

from fastencdec import SamplingParams
from fastencdec.llm import LLM

MODEL = "florence-community/Florence-2-base"
IMAGE_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/"
    "main/transformers/tasks/car.jpg"
)
IMAGE_PATH = "/tmp/fastencdec/florence2_bench.jpg"

# (label, prompt). The order defines the task mix cycle.
TASKS = [
    ("caption", "<CAPTION>"),
    ("ocr", "<OCR>"),
    ("detection", "<OD>"),
    ("segmentation", "<REFERRING_EXPRESSION_SEGMENTATION>the car"),
]


def _synthetic():
    from PIL import ImageDraw

    img = Image.new("RGB", (640, 480), (235, 240, 250))
    d = ImageDraw.Draw(img)
    d.rectangle([40, 120, 300, 320], fill=(200, 60, 60), outline=(0, 0, 0), width=3)
    d.ellipse([340, 140, 560, 360], fill=(60, 120, 200), outline=(0, 0, 0), width=3)
    d.text((40, 40), "CAR 1234 STOP", fill=(0, 0, 0))
    return img


def load_images(n):
    """One real photo plus deterministic variants (synthetic fallback)."""
    base = None
    if os.path.exists(IMAGE_PATH):
        base = Image.open(IMAGE_PATH).convert("RGB")
    else:
        try:
            os.makedirs(os.path.dirname(IMAGE_PATH), exist_ok=True)
            urllib.request.urlretrieve(IMAGE_URL, IMAGE_PATH)
            base = Image.open(IMAGE_PATH).convert("RGB")
        except Exception:  # noqa: BLE001 - offline box
            base = _synthetic()
    w, h = base.size
    variants = [
        base,
        base.transpose(Image.FLIP_LEFT_RIGHT),
        ImageEnhance.Brightness(base).enhance(1.25),
        ImageEnhance.Contrast(base).enhance(1.3),
        base.rotate(4, expand=True),
        base.crop((w // 8, h // 8, 7 * w // 8, 7 * h // 8)),
        base.resize((max(64, w // 2), max(64, h // 2))),
        base.rotate(-6, expand=True),
    ]
    return variants[:n]


def timed(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    best, last = float("inf"), None
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        last = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - start)
    return best, last


def account_hf(out, pad):
    rows = []
    for row in out:
        toks = row.tolist()[1:]  # drop the decoder-start token
        n_pad = 0
        while toks and toks[-1] == pad:
            toks.pop()
            n_pad += 1
        rows.append((len(toks), n_pad))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--images", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--num-blocks", type=int, default=4096)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--compile-mm-encoder", action="store_true")
    ap.add_argument("--use-cudagraph", action="store_true", help="decode CUDA graph")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    images = load_images(args.images)
    workload = [
        (name, prompt, img) for img in images for name, prompt in TASKS
    ]  # 8 images x 4 tasks = 32 requests
    n = len(workload)

    from transformers import AutoProcessor
    from transformers import Florence2ForConditionalGeneration as HFFlorence2

    processor = AutoProcessor.from_pretrained(MODEL)
    block_size = 256 if str(args.device).startswith("cuda") else 16
    num_blocks = max(1, args.num_blocks * 16 // block_size)
    llm = LLM(
        MODEL,
        num_blocks=num_blocks,
        block_size=block_size,
        max_num_seqs=n,
        dtype=dtype,
        compile_mm_encoder=args.compile_mm_encoder,
        device=args.device,
        use_cudagraph=args.use_cudagraph,
    )
    hf = HFFlorence2.from_pretrained(MODEL, dtype=dtype).eval().to(args.device)

    prepared = []
    for _, prompt, img in workload:
        inputs = llm.processor(text=prompt, images=img, return_tensors="pt")
        prepared.append((inputs["input_ids"][0].tolist(), inputs["pixel_values"][0]))

    params = SamplingParams(
        max_tokens=args.max_tokens,
        num_beams=1,
        temperature=0.0,
        eos_token_id=llm.eos_token_id,
    )

    def run_ours():
        reqs = llm.engine.generate(
            [(ids, [llm.decoder_start_token_id], px) for ids, px in prepared], params
        )
        return [llm.engine.results[r] for r in reqs]

    ours_time, ours_out = timed(run_ours, args.warmup, args.repeats)

    inputs = processor(
        text=[p for _, p, _ in workload],
        images=[im for _, _, im in workload],
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(args.device) for k, v in inputs.items()}
    pad = processor.tokenizer.pad_token_id

    def run_hf():
        with torch.no_grad():
            return hf.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                num_beams=1,
                do_sample=False,
                forced_bos_token_id=None,
                forced_eos_token_id=None,
                length_penalty=1.0,
                no_repeat_ngram_size=0,
            )

    hf_time, hf_out = timed(run_hf, args.warmup, args.repeats)

    ours_per = [(len(toks), 0) for toks in ours_out]
    hf_per = account_hf(hf_out, pad)

    names = [name for name, _, _ in workload]
    report = {
        "device": args.device,
        "dtype": args.dtype,
        "requests": n,
        "max_new_tokens": args.max_tokens,
        "tasks": {},
        "timing": {},
    }

    print(
        f"\nmodel={MODEL}  device={args.device}  dtype={args.dtype}  "
        f"requests={n} ({args.images} images x {len(TASKS)} tasks)  "
        f"max_new_tokens={args.max_tokens}  greedy  "
        f"compile_mm_encoder={args.compile_mm_encoder}"
    )
    header = (
        f"{'task':<13}{'reqs':>5}{'ours gen':>10}{'ours/seq':>10}"
        f"{'hf gen':>9}{'hf/seq':>9}{'hf pad':>9}{'hf pad/seq':>11}"
    )
    print("\n" + header)
    print("-" * len(header))

    def agg(task):
        idx = [i for i, name in enumerate(names) if name == task]
        og = sum(ours_per[i][0] for i in idx)
        hg = sum(hf_per[i][0] for i in idx)
        hp = sum(hf_per[i][1] for i in idx)
        r = len(idx)
        return r, og, hg, hp

    tot = [0, 0, 0, 0]
    for task, _ in TASKS:
        r, og, hg, hp = agg(task)
        tot = [tot[0] + r, tot[1] + og, tot[2] + hg, tot[3] + hp]
        print(
            f"{task:<13}{r:>5}{og:>10}{og / r:>10.1f}{hg:>9}{hg / r:>9.1f}"
            f"{hp:>9}{hp / r:>11.1f}"
        )
    print("-" * len(header))
    print(
        f"{'TOTAL':<13}{tot[0]:>5}{tot[1]:>10}{tot[1] / tot[0]:>10.1f}"
        f"{tot[2]:>9}{tot[2] / tot[0]:>9.1f}{tot[3]:>9}{tot[3] / tot[0]:>11.1f}"
    )

    ours_gen, hf_gen, hf_pad = tot[1], tot[2], tot[3]
    hf_total = hf_gen + hf_pad
    print("\n== totals ==")
    print(
        f"  generated tokens : ours={ours_gen}  hf={hf_gen}  "
        f"(hf+padding={hf_total}, {hf_pad} padding = {hf_pad / hf_total:.1%} of positions)"
    )
    print(
        f"  tokens / seq     : ours={ours_gen / n:.1f}  hf={hf_gen / n:.1f}  "
        f"hf incl. padding={hf_total / n:.1f}  (padded width={int(hf_out.shape[1]) - 1})"
    )

    print("\n== timing ==")
    print(f"{'engine':<8}{'time ms':>10}{'gen tok/s':>12}{'padded tok/s':>14}")
    print(
        f"{'ours':<8}{ours_time * 1e3:>10.0f}{ours_gen / ours_time:>12.1f}"
        f"{ours_gen / ours_time:>14.1f}"
    )
    print(
        f"{'hf':<8}{hf_time * 1e3:>10.0f}{hf_gen / hf_time:>12.1f}"
        f"{hf_total / hf_time:>14.1f}"
    )
    print(f"  speedup (gen tokens/s): {hf_time / ours_time:.2f}x")

    report["tasks"] = {
        task: dict(zip(("reqs", "ours_gen", "hf_gen", "hf_pad"), agg(task)))
        for task, _ in TASKS
    }
    report["tasks"]["TOTAL"] = dict(zip(("reqs", "ours_gen", "hf_gen", "hf_pad"), tot))
    report["timing"] = {
        "ours_ms": ours_time * 1e3,
        "hf_ms": hf_time * 1e3,
        "speedup": hf_time / ours_time,
    }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
