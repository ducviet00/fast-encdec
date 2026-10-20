"""Parity for the Mojo paged-attention backend.

Same checks as ``tests/test_attn_backend.py`` (greedy + beam token parity
against the pure-PyTorch path, first-step logits, a changing request set, the
backend switch after ``torch.compile``, and the one-token decoder-prompt
guard), but with ``attn_backend="mojo"`` on the tiny BART model whose
``head_dim`` (32) the kernel supports. Skips when the shared library is not
built.

Run with:  PYTHONPATH=. python tests/test_mojo_attn.py
"""

import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from fastencdec.config import Config
from fastencdec.engine.llm_engine import LLMEngine
from fastencdec.engine.sequence import Sequence
from fastencdec.layers import attention, mojo_attn
from fastencdec.models.bart import BartForConditionalGeneration
from fastencdec.sampling_params import SamplingParams

VOCAB = 256
EOS = 2
BACKEND = "mojo"


def build_model():
    from transformers import BartConfig

    config = BartConfig(
        vocab_size=VOCAB,
        d_model=64,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        max_position_embeddings=128,
        decoder_start_token_id=1,
        bos_token_id=0,
        eos_token_id=EOS,
        pad_token_id=0,
    )
    torch.manual_seed(0)
    return BartForConditionalGeneration(config).eval()


def decode(model, backend, requests, params, block_size, dtype=torch.float32):
    config = Config(
        num_blocks=256,
        block_size=block_size,
        max_num_seqs=8,
        dtype=dtype,
        attn_backend=backend,
    )
    engine = LLMEngine(model, config, eos_token_id=EOS)
    ids = engine.generate(requests, params)
    return [engine.results[i] for i in ids]


def first_step_logits(model, backend, requests, block_size, dtype):
    """Prefill the batch once and return the first-step logits [batch, vocab]."""
    config = Config(
        num_blocks=256,
        block_size=block_size,
        max_num_seqs=8,
        dtype=dtype,
        attn_backend=backend,
    )
    engine = LLMEngine(model, config, eos_token_id=EOS)
    params = SamplingParams(max_tokens=1, temperature=0.0)
    for i, (encoder_ids, decoder_ids, pixel_values) in enumerate(requests):
        engine.scheduler.add(
            Sequence(i, decoder_ids, encoder_ids, params, pixel_values)
        )
    return engine.runner.run(engine.scheduler.schedule())


def mixed_batch_logits(model, backend, requests, block_size, dtype):
    """Step an engine whose request set changes as requests finish."""
    from fastencdec.layers.logits_processor import LogitsProcessor

    config = Config(
        num_blocks=256,
        block_size=block_size,
        max_num_seqs=2,
        dtype=dtype,
        attn_backend=backend,
    )
    engine = LLMEngine(model, config, eos_token_id=EOS)
    params = SamplingParams(max_tokens=5, temperature=0.0)
    engine._logits_processor = LogitsProcessor(
        params,
        vocab_size=model.lm_head.out_features,
        device=torch.device("cpu"),
        max_length=1 + params.max_tokens,
        begin_index=1,
        is_beam=False,
    )
    for i, (encoder_ids, decoder_ids, pixel_values) in enumerate(requests):
        engine.scheduler.add(
            Sequence(i, decoder_ids, encoder_ids, params, pixel_values)
        )
    logits = []
    while not engine.is_finished():
        seqs = engine.scheduler.schedule()
        step_logits = engine.runner.run(seqs)
        logits.append(step_logits)
        engine._postprocess(seqs, step_logits)
    return logits


def mixed_batch(model, requests, block_size, dtype):
    """The staged cross cache must follow a changing request set."""
    extra = requests + [
        (torch.randint(3, VOCAB, (21,)).tolist(), [1], None),
        (torch.randint(3, VOCAB, (11,)).tolist(), [1], None),
    ]
    ref = mixed_batch_logits(model, "sdpa", extra, block_size, dtype)
    got = mixed_batch_logits(model, BACKEND, extra, block_size, dtype)
    diff = max((a - b).abs().max().item() for a, b in zip(ref, got))
    ok = len(ref) == len(got) and diff <= 1e-3
    print(
        f"[{'OK' if ok else 'MISMATCH'}] mixed batch: {len(ref)} steps, "
        f"max|diff|={diff:.2e}"
    )
    return 0 if ok else 1


def backend_switch(model, requests, block_size, dtype):
    """A compiled engine must fall back to eager when the backend is overridden."""
    config = Config(
        num_blocks=256,
        block_size=block_size,
        max_num_seqs=8,
        dtype=dtype,
        attn_backend=BACKEND,
    )
    engine = LLMEngine(model, config, eos_token_id=EOS)
    if not engine.runner.compile_decoder:
        print("SKIP: compiled path unavailable")
        return 0
    params = SamplingParams(max_tokens=6, temperature=0.0)
    first = [engine.results[i] for i in engine.generate(requests, params)]
    # What benchmarks/attn_backend.py does to compare backends on one engine.
    attention.set_backend("sdpa")
    try:
        with mock.patch.object(
            attention, "stage_cross", wraps=attention.stage_cross
        ) as staged:
            got = [engine.results[i] for i in engine.generate(requests, params)]
    finally:
        attention.set_backend("auto")
    ok = first == got and not staged.called
    print(
        f"[{'OK' if ok else 'MISMATCH'}] backend switch after compile: {got} "
        f"(staged={staged.call_count})"
    )
    return 0 if ok else 1


def compile_off(model, requests, block_size, dtype):
    """compile=False must keep the CPU decoder eager and match the compiled path."""
    config = Config(
        num_blocks=256,
        block_size=block_size,
        max_num_seqs=8,
        dtype=dtype,
        attn_backend=BACKEND,
        compile=False,
    )
    engine = LLMEngine(model, config, eos_token_id=EOS)
    params = SamplingParams(max_tokens=6, temperature=0.0)
    eager = [engine.results[i] for i in engine.generate(requests, params)]
    compiled = decode(model, BACKEND, requests, params, block_size, dtype)
    ok = (
        not engine.runner.compile_decoder
        and engine.runner.decode_step is None
        and eager == compiled
    )
    print(f"[{'OK' if ok else 'MISMATCH'}] compile=False stays eager: {eager}")
    return 0 if ok else 1


def main():
    if mojo_attn.module() is None:
        print("SKIP: mojo/build/fastencdec_mojo_attn.so is not built")
        return
    if not mojo_attn.supported(torch.float32, 16, 32):
        print("SKIP: the Mojo kernel does not support head_dim=32")
        return
    model = build_model()
    torch.manual_seed(1)
    requests = [
        (torch.randint(3, VOCAB, (17,)).tolist(), [1], None),
        (torch.randint(3, VOCAB, (31,)).tolist(), [1], None),
        (torch.randint(3, VOCAB, (9,)).tolist(), [1], None),
    ]
    failures = 0
    for dtype in (torch.float32, torch.bfloat16):
        model_dt = build_model().to(dtype) if dtype != torch.float32 else model
        for block_size in (16, 32, 64):
            for beams in (1, 2):
                params = SamplingParams(max_tokens=10, num_beams=beams, temperature=0.0)
                ref = decode(model_dt, "sdpa", requests, params, block_size, dtype)
                got = decode(model_dt, BACKEND, requests, params, block_size, dtype)
                for i, (want, have) in enumerate(zip(ref, got)):
                    status = "OK" if want == have else "MISMATCH"
                    failures += status == "MISMATCH"
                    print(
                        f"[{status}] dtype={dtype} block_size={block_size} "
                        f"beams={beams} seq={i}: {have}"
                    )
            # Tokens are degenerate on this random model; the logits are not.
            ref = first_step_logits(model_dt, "sdpa", requests, block_size, dtype)
            got = first_step_logits(model_dt, BACKEND, requests, block_size, dtype)
            diff = (ref - got).abs().max().item()
            tol = 1e-3 if dtype == torch.float32 else 0.5
            status = "OK" if diff <= tol else "MISMATCH"
            failures += status == "MISMATCH"
            print(
                f"[{status}] dtype={dtype} block_size={block_size} "
                f"logits max|diff|={diff:.2e} (tol {tol})"
            )
    failures += mixed_batch(model, requests, 32, torch.float32)
    failures += backend_switch(model, requests, 32, torch.float32)
    failures += compile_off(model, requests, 32, torch.float32)
    if failures:
        raise SystemExit(f"{failures} mismatches")


if __name__ == "__main__":
    main()
