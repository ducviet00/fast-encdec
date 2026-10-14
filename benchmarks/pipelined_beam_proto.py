"""Prototype: pipelined beam-search decode (CUDA, fixed-length, lockstep).

Companion to ``pipelined_proto.py`` (greedy).  Beam search has a second feedback
loop that greedy does not: the *fork* (which parent each beam continues) decides
the next step's KV layout, so the decoder cannot run until the fork is known.
This prototype moves everything that can be pipelined onto the device:

* per-step metadata (``positions`` / ``context_lens`` / ``slot_mapping``) is
  derived on the device from a position counter and the device block table;
* the ranking (``log_softmax`` + ``topk``) stays on the device, and the next
  step's tokens are fed straight back (no host round-trip for the token);
* the block-table fork is a device gather over a device block table;
* the copy-on-write allocation stays on the host, so one sync per step remains
  -- the allocator is host-side and the number of copies is data dependent
  (a fixed-size device CoW would need custom kernels, which this box cannot
  build: there is no nvcc).

The synchronous engine is measured alongside, so the two are comparable:

    PYTHONPATH=. python benchmarks/pipelined_beam_proto.py --device cuda --beams 4

Scope: CUDA + flash + CUDA graph, lockstep batch, fixed length (EOS suppressed),
so the live beam count stays ``num_beams`` for the whole run.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import benchmark_vllm as bv
import torch
from fastencdec import LLM, SamplingParams
from fastencdec.config import default_block_size
from fastencdec.engine.sequence import Sequence
from fastencdec.layers.logits_processor import build_logits_processor


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# Optional per-step record for debugging: [(parents, tokens), ...]
RECORD: list | None = None


def add_requests(engine, requests, params) -> list[int]:
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


def order_slots(seqs):
    """Group running sequences request-major / beam-minor (the reshape order)."""
    request_ids = []
    for seq in seqs:
        if seq.request_id not in request_ids:
            request_ids.append(seq.request_id)
    slots = [s for rid in request_ids for s in seqs if s.request_id == rid]
    return slots, request_ids


def pipelined_beam_decode(engine, seqs, steps, beams, eos_id):
    """Decode ``steps`` tokens; returns ``(hist, cum)`` device tensors."""
    runner = engine.runner
    graph = runner.graph
    assert graph is not None and graph.ready(), "need CUDA + flash + cudagraph"
    model = runner.model
    bm = engine.block_manager
    bs = runner.block_size
    dev = runner.device
    cache_layers = runner.cache_layers

    slots, request_ids = order_slots(seqs)
    n_requests = len(request_ids)
    n_slots = len(slots)
    assert n_slots == n_requests * beams, "expected a full lockstep beam batch"

    width = max(len(s.block_table) for s in slots)
    bt = torch.tensor(
        [s.block_table + [0] * (width - len(s.block_table)) for s in slots],
        dtype=torch.int32,
        device=dev,
    )
    ids = torch.tensor([s.token_ids[-1] for s in slots], dtype=torch.long, device=dev)
    cum = torch.tensor([s.cum_logprob for s in slots], dtype=torch.float32, device=dev)
    length = max(len(s.token_ids) for s in slots)
    hist = torch.zeros(n_slots, length + steps, dtype=torch.long, device=dev)
    for i, s in enumerate(slots):
        hist[i, : len(s.token_ids)] = torch.tensor(s.token_ids, device=dev)
    start = slots[0].num_cached_tokens

    live = Counter(int(b) for s in slots for b in s.block_table)
    for block, count in live.items():
        bm.blocks[block].ref_count = count
    host_bt = [list(s.block_table) for s in slots]

    # the cross-attention cache is staged per sequence row: the prefill staged 8
    # rows (one per request), the beam loop needs one row per beam slot.
    graph.stage(slots)
    graph.capture()

    def cow(write_block: int) -> None:
        """Make the block this step writes private (mirrors ``ensure_capacity``).

        Called *before* the replay: after a fork the siblings share their
        parent's partial block, so each needs its own copy before it writes.
        """
        groups: dict[int, list[int]] = {}
        for slot, table in enumerate(host_bt):
            groups.setdefault(table[write_block], []).append(slot)
        moved = []
        for block, sharing in groups.items():
            for slot in sharing[1:]:
                new_block = bm._allocate_block()
                host_bt[slot][write_block] = new_block
                moved.append((slot, new_block))
                live[new_block] = 1  # so an un-inherited block is freed below
                for layer in cache_layers:
                    layer.k_cache[new_block].copy_(layer.k_cache[block])
                    layer.v_cache[new_block].copy_(layer.v_cache[block])
        if moved:
            rows = torch.tensor([m[0] for m in moved], dtype=torch.long, device=dev)
            vals = torch.tensor([m[1] for m in moved], dtype=torch.int32, device=dev)
            bt[rows, write_block] = vals

    for i in range(steps):
        pos = start + i
        cow(pos // bs)  # before the write, not after
        fields = {
            "is_prefill": False,
            "context_lens": torch.full(
                (n_slots,), pos + 1, dtype=torch.int32, device=dev
            ),
            "slot_mapping": (bt[:, pos // bs] * bs + pos % bs).to(torch.int64),
            "block_tables": bt,
            "request_ids": None,
        }
        positions = torch.full((n_slots,), pos, dtype=torch.long, device=dev)
        hidden = graph.replay(n_slots, ids, positions, fields)
        logits = model.compute_logits(hidden)
        vocab = logits.shape[-1]
        lp = torch.log_softmax(logits.float(), dim=-1)
        if eos_id is not None:
            lp[:, eos_id] = float("-inf")
        scores = lp.reshape(n_requests, beams, vocab) + cum.reshape(
            n_requests, beams, 1
        )
        top_values, top_indices = torch.topk(scores.reshape(n_requests, -1), 2 * beams)

        # the one sync: read the fork so the host can allocate the CoW blocks
        parents = (top_indices[:, :beams] // vocab).tolist()
        if RECORD is not None:
            RECORD.append((parents, (top_indices[:, :beams] % vocab).tolist()))

        parent_slot_idx = [
            [r * beams + parents[r][b] for b in range(beams)] for r in range(n_requests)
        ]
        parent_slots = torch.tensor(
            parent_slot_idx, dtype=torch.long, device=dev
        ).reshape(-1)

        bt = bt[parent_slots]  # device fork (gather rows)
        host_bt = [list(host_bt[ps]) for row in parent_slot_idx for ps in row]

        refs = Counter(b for table in host_bt for b in table)
        for block in list(live):
            if block not in refs:
                bm._deallocate_block(block)
                live.pop(block, None)
        for block, count in refs.items():
            bm.blocks[block].ref_count = count
            live[block] = count
        ids = (top_indices[:, :beams] % vocab).reshape(-1)
        cum = top_values[:, :beams].reshape(-1).float()
        hist_next = hist[parent_slots]
        hist_next[:, pos + 1] = ids
        hist = hist_next

    return hist, cum, live


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=bv.MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--enc-tokens", type=int, default=256)
    ap.add_argument("--out-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt = bv.make_prompt(tokenizer, args.enc_tokens)
    enc_ids = tokenizer(prompt, add_special_tokens=True, padding=False)["input_ids"]
    enc_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

    block_size = default_block_size(args.device)
    llm = LLM(
        args.model,
        num_blocks=4
        * bv.blocks_for(
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
        eos_token_id=llm.eos_token_id,
    )

    def normal_once():
        out = llm.generate([prompt] * args.batch, params, use_tqdm=False)
        return [o.outputs[0].token_ids for o in out]

    def pipelined_once():
        engine._logits_processor = build_logits_processor(
            params,
            max_length=1 + args.out_tokens,
            begin_index=1,
            is_beam=True,
            device=engine.device,
        )
        add_requests(engine, [(enc_ids, [decoder_start], None)] * args.batch, params)
        engine.step()  # prefill + beam expansion (host) + graph capture
        seqs = engine.scheduler.schedule()
        hist, cum, live = pipelined_beam_decode(
            engine, seqs, args.out_tokens - 1, args.beams, llm.eos_token_id
        )
        slots, request_ids = order_slots(seqs)
        best = cum.reshape(len(request_ids), args.beams).argmax(-1)
        completion = []
        for r in range(len(request_ids)):
            row = hist[r * args.beams + int(best[r])]
            completion.append(row[1 : 1 + args.out_tokens].tolist())
        # free the blocks the device table actually holds (the sequences'
        # host block tables are stale), then drop the sequences
        for block in live:
            engine.block_manager._deallocate_block(block)
        for request_id in request_ids:
            engine._release(request_id)
        for seq in slots:
            engine.scheduler.remove(seq)
        return completion

    def timed(fn, warmup, repeats):
        for _ in range(warmup):
            fn()
        best = float("inf")
        for _ in range(repeats):
            _sync()
            s = time.perf_counter()
            fn()
            _sync()
            best = min(best, time.perf_counter() - s)
        return best * 1e3

    normal_ms = timed(normal_once, args.warmup, args.repeats)
    print(
        f"model={args.model} device={args.device} dtype={args.dtype} "
        f"batch={args.batch} beams={args.beams} enc={enc_len} out={args.out_tokens}"
    )
    n = args.batch * args.out_tokens
    print(f"{'engine':<12}{'ms':>10}{'tok/s':>10}")
    print(f"{'normal':<12}{normal_ms:>10.1f}{n / (normal_ms / 1e3):>10.1f}")

    pipelined_once()  # correctness / crash check before timing
    pipe_ms = timed(pipelined_once, args.warmup, args.repeats)
    print(f"{'pipelined':<12}{pipe_ms:>10.1f}{n / (pipe_ms / 1e3):>10.1f}")
    print(f"\npipelined beam speedup: {normal_ms / pipe_ms:.2f}x")

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
                print(f"    normal    [:8] {a[:8]}")
                print(f"    pipelined [:8] {b[:8]}")
                break


if __name__ == "__main__":
    main()
