"""Prototype: pipelined greedy decode with GPU-resident tokens.

The engine's synchronous loop blocks on ``Sampler(...).tolist()`` every step and
then rebuilds ``input_ids`` / ``positions`` / ``slot_mapping`` on the host while
the GPU idles.  This prototype runs the same model + paged cache + CUDA graph,
but keeps the sampled token on the GPU and derives all per-step metadata on the
GPU, so the host only enqueues: one device sync for the whole decode instead of
one per step.

It reuses the engine's building blocks (``ModelRunner``, ``BlockManager``,
``DecodeGraphRunner``) and drives them directly, so the numbers are directly
comparable to ``LLM.generate``:

    PYTHONPATH=. python benchmarks/pipelined_proto.py --device cuda

Scope of this prototype (see the plan for the full design):

* the pipelined path is **CUDA + greedy + CUDA graph** only -- that is where the
  device round-trip is the bottleneck.  ``--device cpu`` and ``--beams > 1``
  fall back to the normal synchronous loop, which this script still measures so
  the two can be compared and the fallbacks stay correct;
* a fixed-length, lockstep batch -- every sequence decodes the same number of
  steps (EOS suppressed), which is the ``benchmark_vllm.py`` workload;
* no logits processors in the decode loop (a fixed length needs none).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import benchmark_vllm as bv
import torch

from fastencdec import LLM, SamplingParams
from fastencdec.config import default_block_size
from fastencdec.engine.sequence import Sequence


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def add_requests(engine, requests, params) -> list[int]:
    """Mirror ``LLMEngine.generate``'s admission, without running."""
    ids = []
    for encoder_ids, decoder_ids, pixel_values in requests:
        request_id = engine._next_request_id
        engine._next_request_id += 1
        engine.scheduler.add(
            Sequence(
                request_id=request_id,
                decoder_prompt_ids=decoder_ids,
                encoder_token_ids=encoder_ids,
                sampling=params,
                pixel_values=pixel_values,
            )
        )
        ids.append(request_id)
    return ids


def pipelined_decode(engine, seqs, steps: int, eos_id: int | None) -> torch.Tensor:
    """Decode ``steps`` tokens with the token kept on the GPU.

    ``seqs`` are lockstep (same length) and greedy.  Returns ``[B, steps]``
    device tokens; the caller syncs once.  ``eos_id`` is banned from the argmax
    (the fixed-length ``min_length`` the synchronous path applies through its
    logits processor), so the two paths are token-for-token comparable.
    """
    runner = engine.runner
    graph = runner.graph
    assert graph is not None and graph.ready(), "need CUDA + flash + cudagraph"
    model = runner.model
    bs = runner.block_size
    dev = runner.device
    batch = len(seqs)

    # The run length is known, so allocate the whole block table up front and
    # keep it static for the loop (no host allocation mid-decode).
    for seq in seqs:
        runner.block_manager.ensure_capacity(seq, steps, runner._copy_block)
    width = max(len(s.block_table) for s in seqs)
    block_tables = torch.tensor(
        [s.block_table + [0] * (width - len(s.block_table)) for s in seqs],
        dtype=torch.int32,
        device=dev,
    )
    graph.stage(seqs)
    graph.capture()  # no-op if already captured; rebuilds if the cross cache grew

    # The token sampled by the prefill step; the only host->device copy.
    decode_ids = torch.tensor(
        [s.token_ids[-1] for s in seqs], dtype=torch.long, device=dev
    )
    start = seqs[0].num_cached_tokens

    outputs = []
    for i in range(steps):
        pos = start + i
        fields = {
            "is_prefill": False,
            "context_lens": torch.full(
                (batch,), pos + 1, dtype=torch.int32, device=dev
            ),
            "slot_mapping": (block_tables[:, pos // bs] * bs + pos % bs).to(
                torch.int64
            ),
            "block_tables": block_tables,
            "request_ids": None,
        }
        positions = torch.full((batch,), pos, dtype=torch.long, device=dev)
        hidden = graph.replay(batch, decode_ids, positions, fields)
        logits = model.compute_logits(hidden)
        if eos_id is not None:
            logits[:, eos_id] = float("-inf")
        decode_ids = torch.argmax(logits, dim=-1)
        outputs.append(decode_ids)
    return torch.stack(outputs, dim=1)


def finalize(engine, seqs, tokens: torch.Tensor) -> None:
    """Write the decoded tokens back into the engine's bookkeeping."""
    rows = tokens.tolist()
    for row, seq in enumerate(seqs):
        seq.token_ids.extend(rows[row])
        seq.num_cached_tokens += len(rows[row])
        engine.results[seq.request_id] = seq.completion_token_ids
        engine.scheduler.remove(seq)
        engine.block_manager.deallocate(seq)
        engine._release(seq.request_id)


def build_requests(enc_ids: list[int], batch: int, decoder_start: int):
    return [(enc_ids, [decoder_start], None) for _ in range(batch)]


def timed(fn, warmup: int, repeats: int) -> tuple[float, int]:
    for _ in range(warmup):
        fn()
    best, count = float("inf"), 0
    for _ in range(repeats):
        _sync()
        start = time.perf_counter()
        count = fn()
        _sync()
        best = min(best, time.perf_counter() - start)
    return best * 1e3, count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=bv.MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--enc-tokens", type=int, default=256)
    ap.add_argument("--out-tokens", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt = bv.make_prompt(tokenizer, args.enc_tokens)
    enc_ids = tokenizer(prompt, add_special_tokens=True, padding=False)["input_ids"]
    enc_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

    block_size = default_block_size(args.device)
    llm = LLM(
        args.model,
        num_blocks=bv.blocks_for(
            [{"batch": args.batch, "beams": args.beams}],
            enc_len,
            args.out_tokens,
            block_size,
        ),
        block_size=block_size,
        max_num_seqs=args.batch * args.beams,
        dtype=getattr(torch, args.dtype),
        device=args.device,
    )
    engine = llm.engine
    decoder_start = llm.decoder_start_token_id
    params = SamplingParams(
        max_tokens=args.out_tokens,
        num_beams=args.beams,
        min_length=args.out_tokens + 1,
    )

    def normal_once() -> list[list[int]]:
        out = llm.generate([prompt] * args.batch, params, use_tqdm=False)
        return [o.outputs[0].token_ids for o in out]

    def pipelined_once() -> list[list[int]]:
        # Fixed-length decode needs no logits processor; the prefill step still
        # goes through ``_postprocess``, so hand it a passthrough.
        engine._logits_processor = lambda input_ids, scores: scores
        requests = build_requests(enc_ids, args.batch, decoder_start)
        add_requests(engine, requests, params)
        engine.step()  # prefill + graph capture + cross-cache staging
        seqs = engine.scheduler.schedule()
        if engine.is_finished() or len(seqs) != args.batch:
            raise RuntimeError("prototype needs a lockstep batch")
        tokens = pipelined_decode(engine, seqs, args.out_tokens - 1, llm.eos_token_id)
        finalize(engine, seqs, tokens)
        return [engine.results[s.request_id] for s in seqs]

    def run_normal() -> int:
        return sum(len(t) for t in normal_once())

    def run_pipelined() -> int:
        return sum(len(t) for t in pipelined_once())

    # The pipelined path needs an async device, greedy decoding and the graph.
    pipelined_ok = (
        args.device.startswith("cuda")
        and args.beams == 1
        and engine.runner.graph is not None
    )
    n = args.batch * args.out_tokens
    print(
        f"model={args.model} device={args.device} dtype={args.dtype} "
        f"batch={args.batch} beams={args.beams} enc={enc_len} out={args.out_tokens}"
    )
    norm_ms, norm_toks = timed(run_normal, args.warmup, args.repeats)
    if norm_toks != n:
        raise RuntimeError(f"normal decoded {norm_toks} tokens, expected {n}")
    print(f"{'engine':<12}{'ms':>10}{'tok/s':>10}{'tokens':>10}")
    print(f"{'normal':<12}{norm_ms:>10.1f}{n / (norm_ms / 1e3):>10.1f}{norm_toks:>10}")
    if not pipelined_ok:
        print(
            "\n(pipelined path is CUDA + greedy + CUDA graph only; this "
            "configuration uses the normal loop -- see the plan)"
        )
        return
    pipe_ms, pipe_toks = timed(run_pipelined, args.warmup, args.repeats)
    if pipe_toks != n:
        raise RuntimeError(f"pipelined decoded {pipe_toks} tokens, expected {n}")
    print(
        f"{'pipelined':<12}{pipe_ms:>10.1f}{n / (pipe_ms / 1e3):>10.1f}{pipe_toks:>10}"
    )
    print(f"\npipelined speedup over the current loop: {norm_ms / pipe_ms:.2f}x")

    normal_tokens = normal_once()
    pipelined_tokens = pipelined_once()
    exact = normal_tokens == pipelined_tokens
    print("token parity (normal vs pipelined):", "exact" if exact else "MISMATCH")
    if not exact:
        for i, (a, b) in enumerate(zip(normal_tokens, pipelined_tokens)):
            if a != b:
                diff = next(
                    (j for j, (x, y) in enumerate(zip(a, b)) if x != y),
                    min(len(a), len(b)),
                )
                print(f"  request {i}: len {len(a)} vs {len(b)}, first diff at {diff}")
                break


if __name__ == "__main__":
    main()
