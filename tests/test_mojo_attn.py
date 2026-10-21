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


def flash_api():
    """The flash-attn-shaped entry points: semantics, GQA, cache writes, rejections."""
    torch.manual_seed(7)
    D, H, BLOCK, BLOCKS, B, T = 32, 4, 16, 4, 3, 12
    KVH = 2
    scale = D**-0.5
    fails = 0

    def check(tag, have, want, tol=2e-3):
        nonlocal fails
        diff = (have.float() - want.float()).abs().max().item()
        ok = diff <= tol
        fails += not ok
        print(f"[{'OK' if ok else 'MISMATCH'}] flash_api {tag}: max|diff|={diff:.2e}")

    def rejects(tag, fn):
        nonlocal fails
        try:
            fn()
        except (NotImplementedError, ValueError):
            print(f"[OK] flash_api rejects {tag}")
            return
        fails += 1
        print(f"[MISMATCH] flash_api accepted {tag}")

    def sdpa(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=scale
        ).transpose(1, 2)

    def caches(heads=H):
        pages = torch.arange(B * BLOCKS, dtype=torch.int32).reshape(B, BLOCKS)
        shape = (B * BLOCKS, heads, BLOCK, D)
        return torch.zeros(shape), torch.zeros(shape), pages

    q = torch.randn(B, 1, H, D)
    keys = torch.randn(B, T, H, D)
    vals = torch.randn(B, T, H, D)

    # 1. Incremental write+attend with flash's pre-append `cache_seqlens`.
    k_cache, v_cache, page_table = caches()
    for t in range(T):
        out = mojo_attn.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=keys[:, t : t + 1],
            v=vals[:, t : t + 1],
            cache_seqlens=t,
            page_table=page_table,
            causal=True,
            softmax_scale=scale,
        )
        check(f"write+attend t={t}", out, sdpa(q, keys[:, : t + 1], vals[:, : t + 1]))

    # 2. The `slot_mapping` fast path matches the pre-append path exactly.
    ka, va, pages = caches()
    kb, vb, _ = caches()
    out_a = mojo_attn.flash_attn_with_kvcache(
        q,
        ka,
        va,
        k=keys[:, 5:6],
        v=vals[:, 5:6],
        cache_seqlens=5,
        page_table=pages,
        causal=True,
        softmax_scale=scale,
    )
    slots = pages[:, 5 // BLOCK].long() * BLOCK + 5 % BLOCK
    out_b = mojo_attn.flash_attn_with_kvcache(
        q,
        kb,
        vb,
        k=keys[:, 5:6],
        v=vals[:, 5:6],
        cache_seqlens=6,
        slot_mapping=slots,
        page_table=pages,
        causal=True,
        softmax_scale=scale,
    )
    check("slot_mapping output", out_b, out_a)
    check("slot_mapping cache", kb, ka)

    # 3. Cross-attention over a staged cache, per-sequence int32 lengths.
    lens = torch.tensor([3, 7, 12], dtype=torch.int32)
    kc, vc, pages = caches()
    for b in range(B):
        n = int(lens[b])
        pos = torch.arange(n)
        slot = pages[b, pos // BLOCK].long() * BLOCK + pos % BLOCK
        mojo_attn.reshape_and_cache(keys[b, :n], vals[b, :n], kc, vc, slot)
    ref = torch.cat(
        [
            sdpa(
                q[b : b + 1],
                keys[b : b + 1, : int(lens[b])],
                vals[b : b + 1, : int(lens[b])],
            )
            for b in range(B)
        ]
    )
    out = mojo_attn.flash_attn_with_kvcache(
        q,
        kc,
        vc,
        cache_seqlens=lens,
        page_table=pages,
        causal=False,
        softmax_scale=scale,
    )
    check("cross, tensor lengths", out, ref)

    # 4. `cache_batch_idx` gathers the page-table rows (flash's remap semantics).
    #    Every sequence is filled to T so that a remapped row never reads slots
    #    that were never written.
    perm = torch.tensor([2, 0, 1], dtype=torch.int32)
    kcp, vcp, pagesp = caches()
    pos = torch.arange(T)
    for b in range(B):
        slot = pagesp[b, pos // BLOCK].long() * BLOCK + pos % BLOCK
        mojo_attn.reshape_and_cache(keys[b], vals[b], kcp, vcp, slot)
    out = mojo_attn.flash_attn_with_kvcache(
        q,
        kcp,
        vcp,
        cache_seqlens=lens,
        page_table=pagesp,
        cache_batch_idx=perm,
        softmax_scale=scale,
    )
    want = torch.cat(
        [
            sdpa(
                q[b : b + 1],
                keys[int(perm[b]) : int(perm[b]) + 1, : int(lens[b])],
                vals[int(perm[b]) : int(perm[b]) + 1, : int(lens[b])],
            )
            for b in range(B)
        ]
    )
    check("cache_batch_idx", out, want)

    # 5. GQA: 4 query heads over 2 KV heads.
    kg = torch.randn(B, 6, KVH, D)
    vg = torch.randn(B, 6, KVH, D)
    kq = torch.randn(B, 1, H, D)
    kcg, vcg, pagesg = caches(KVH)
    pos = torch.arange(6)
    for b in range(B):
        slot = pagesg[b, pos // BLOCK].long() * BLOCK + pos % BLOCK
        mojo_attn.reshape_and_cache(kg[b], vg[b], kcg, vcg, slot)
    out = mojo_attn.flash_attn_with_kvcache(
        kq, kcg, vcg, cache_seqlens=6, page_table=pagesg, softmax_scale=scale
    )
    check(
        "gqa",
        out,
        sdpa(
            kq,
            kg.repeat_interleave(H // KVH, dim=2),
            vg.repeat_interleave(H // KVH, dim=2),
        ),
    )

    # 6. `flash_attn_func` works for the single query token (kv_len 9, not a page multiple).
    dk = torch.randn(B, 9, H, D)
    dv = torch.randn(B, 9, H, D)
    check(
        "flash_attn_func L=1",
        mojo_attn.flash_attn_func(q, dk, dv, softmax_scale=scale),
        sdpa(q, dk, dv),
    )

    # 7. Engine aliases are the same call.
    kc1, vc1, pages1 = caches()
    kc2, vc2, _ = caches()
    ones = torch.ones(B, dtype=torch.int32)
    alias = mojo_attn.self_attention(
        q.squeeze(1),
        keys[:, 0],
        vals[:, 0],
        kc1,
        vc1,
        pages1[:, 0].long() * BLOCK,
        pages1,
        ones,
        scale,
    )
    direct = mojo_attn.flash_attn_with_kvcache(
        q,
        kc2,
        vc2,
        k=keys[:, 0:1],
        v=vals[:, 0:1],
        cache_seqlens=1,
        slot_mapping=pages1[:, 0].long() * BLOCK,
        page_table=pages1,
        causal=True,
        softmax_scale=scale,
    ).squeeze(1)
    check("self_attention alias", alias, direct)
    check(
        "cross_attention alias",
        mojo_attn.cross_attention(q.squeeze(1), kc, vc, pages, lens, scale),
        ref.squeeze(1),
    )

    # 8. Unsupported flash-attn features fail loudly, not silently.
    kc, vc, pages = caches()
    fresh = {
        "k": keys[:, :1],
        "v": vals[:, :1],
        "cache_seqlens": 0,
        "page_table": pages,
    }
    rejects(
        "seqlen=2",
        lambda: mojo_attn.flash_attn_with_kvcache(
            q.repeat(1, 2, 1, 1), kc, vc, **fresh
        ),
    )
    rejects(
        "two new tokens",
        lambda: mojo_attn.flash_attn_with_kvcache(
            q, kc, vc, k=keys[:, :2], v=vals[:, :2], cache_seqlens=0, page_table=pages
        ),
    )
    rejects(
        "window_size",
        lambda: mojo_attn.flash_attn_with_kvcache(
            q, kc, vc, cache_seqlens=lens, page_table=pages, window_size=(4, 0)
        ),
    )
    rejects(
        "softcap",
        lambda: mojo_attn.flash_attn_with_kvcache(
            q, kc, vc, cache_seqlens=lens, page_table=pages, softcap=1.0
        ),
    )
    rejects(
        "rotary_cos",
        lambda: mojo_attn.flash_attn_with_kvcache(
            q,
            kc,
            vc,
            cache_seqlens=lens,
            page_table=pages,
            rotary_cos=torch.zeros(4, 8),
        ),
    )
    rejects(
        "qv",
        lambda: mojo_attn.flash_attn_with_kvcache(
            q, kc, vc, cache_seqlens=lens, page_table=pages, qv=q
        ),
    )
    rejects(
        "cache_leftpad",
        lambda: mojo_attn.flash_attn_with_kvcache(
            q,
            kc,
            vc,
            cache_seqlens=lens,
            page_table=pages,
            cache_leftpad=torch.zeros(B, dtype=torch.int32),
        ),
    )
    rejects(
        "return_softmax_lse",
        lambda: mojo_attn.flash_attn_with_kvcache(
            q, kc, vc, cache_seqlens=lens, page_table=pages, return_softmax_lse=True
        ),
    )
    rejects(
        "page_table+block_table",
        lambda: mojo_attn.flash_attn_with_kvcache(
            q, kc, vc, cache_seqlens=lens, page_table=pages, block_table=pages
        ),
    )
    rejects(
        "slot_mapping without k/v",
        lambda: mojo_attn.flash_attn_with_kvcache(
            q,
            kc,
            vc,
            cache_seqlens=lens,
            page_table=pages,
            slot_mapping=torch.zeros(B, dtype=torch.int64),
        ),
    )
    rejects(
        "flash_attn_func seqlen=2",
        lambda: mojo_attn.flash_attn_func(q.repeat(1, 2, 1, 1), dk, dv),
    )
    rejects(
        "flash_attn_func dropout",
        lambda: mojo_attn.flash_attn_func(q, dk, dv, dropout_p=0.1),
    )
    rejects(
        "flash_attn_func alibi",
        lambda: mojo_attn.flash_attn_func(q, dk, dv, alibi_slopes=torch.zeros(H)),
    )
    rejects(
        "flash_attn_func window",
        lambda: mojo_attn.flash_attn_func(q, dk, dv, window_size=(2, 0)),
    )
    return fails


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
    failures += flash_api()
    failures += mixed_batch(model, requests, 32, torch.float32)
    failures += backend_switch(model, requests, 32, torch.float32)
    failures += compile_off(model, requests, 32, torch.float32)
    if failures:
        raise SystemExit(f"{failures} mismatches")


if __name__ == "__main__":
    main()
