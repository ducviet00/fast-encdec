"""torch.profiler breakdown of a Florence-2 greedy generate (ours and/or HF).

One warmup, then one profiled generate on the same 8-image x 4-task batch; prints
the top ops by self CPU time and by CUDA time, plus the totals, so fast-encdec and
HuggingFace can be compared op for op.

    PYTHONPATH=. python benchmarks/profile_florence2.py --engine both --device cpu
    PYTHONPATH=. python benchmarks/profile_florence2.py --engine ours --device cuda --out gpu.txt

Notes:
* The cpu-attention kernels run inside the ``fastencdec::cpu_*`` custom ops, so
  our CPU attention time shows up under those op names.
* ``wall`` is the profiled run (the profiler adds overhead); use
  ``florence2_tasks.py`` for clean timings.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import florence2_tasks as tasks
import torch
from torch.profiler import ProfilerActivity, profile


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_runner(engine, args, dtype, workload):
    """Return a zero-arg ``run`` closure for ``engine`` on ``workload``."""
    if engine == "ours":
        from fastencdec import SamplingParams
        from fastencdec.config import default_block_size
        from fastencdec.llm import LLM

        block_size = default_block_size(args.device)
        llm = LLM(
            tasks.MODEL,
            num_blocks=max(1, args.num_blocks * 16 // block_size),
            block_size=block_size,
            max_num_seqs=len(workload),
            dtype=dtype,
            compile_mm_encoder=args.compile_mm_encoder,
            device=args.device,
            use_cudagraph=args.use_cudagraph,
        )
        prepared = []
        for _, prompt, img in workload:
            inputs = llm.processor(text=prompt, images=img, return_tensors="pt")
            prepared.append(
                (inputs["input_ids"][0].tolist(), inputs["pixel_values"][0])
            )
        params = SamplingParams(
            max_tokens=args.max_tokens,
            num_beams=1,
            temperature=0.0,
            eos_token_id=llm.eos_token_id,
        )

        def run():
            reqs = llm.engine.generate(
                [(ids, [llm.decoder_start_token_id], px) for ids, px in prepared],
                params,
            )
            return [llm.engine.results[r] for r in reqs]

        return run

    from transformers import AutoProcessor
    from transformers import Florence2ForConditionalGeneration as HFFlorence2

    processor = AutoProcessor.from_pretrained(tasks.MODEL)
    hf = HFFlorence2.from_pretrained(tasks.MODEL, dtype=dtype).eval().to(args.device)
    inputs = processor(
        text=[p for _, p, _ in workload],
        images=[im for _, _, im in workload],
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    def run():
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

    return run


def profile_one(engine, args, dtype, workload):
    run = make_runner(engine, args, dtype, workload)
    run()  # warmup: torch.compile / cudnn autotune / kernel download
    _sync()

    acts = [ProfilerActivity.CPU]
    if str(args.device).startswith("cuda"):
        acts.append(ProfilerActivity.CUDA)
    _sync()
    start = time.perf_counter()
    with profile(activities=acts) as prof:
        run()
    _sync()
    wall = (time.perf_counter() - start) * 1e3

    events = prof.key_averages()
    counts = {e.key: e.count for e in events}
    use_cuda = "CUDA" in str(acts)
    total = {
        "wall_ms": wall,
        "self_cpu_ms": sum(e.self_cpu_time_total for e in events) / 1e3,
        "syncs": counts.get("cudaStreamSynchronize", 0),
        "launches": counts.get("cudaLaunchKernel", 0) + counts.get("cuLaunchKernel", 0),
    }

    cpu_table = prof.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=args.rows
    )
    lines = [
        (
            f"engine={engine} device={args.device} dtype={args.dtype} "
            f"requests={len(workload)} max_new_tokens={args.max_tokens}"
        ),
        "totals: " + "  ".join(f"{k}={v:.1f}" for k, v in total.items()),
        "(self CUDA total is the footer line of each table below)",
        "",
        "--- top by self CPU time ---",
        cpu_table,
    ]
    if use_cuda:
        lines += [
            "",
            "--- top by CUDA time ---",
            prof.key_averages().table(sort_by="device_time_total", row_limit=args.rows),
        ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="both", choices=["ours", "hf", "both"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--images", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--num-blocks", type=int, default=4096)
    ap.add_argument("--compile-mm-encoder", action="store_true")
    ap.add_argument(
        "--cudagraph",
        dest="use_cudagraph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="decode CUDA graph (default on; CUDA + flash only)",
    )
    ap.add_argument("--rows", type=int, default=20, help="rows per profiler table")
    ap.add_argument("--out", default=None, help="write the raw tables to this file")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    images = tasks.load_images(args.images)
    workload = [(name, prompt, img) for img in images for name, prompt in tasks.TASKS]

    engines = ["ours", "hf"] if args.engine == "both" else [args.engine]
    text = "\n\n".join(profile_one(e, args, dtype, workload) for e in engines)
    print("\n" + text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
