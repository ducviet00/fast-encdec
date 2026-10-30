"""FlashAttention-2 backend parity on CUDA.

Two checks, both skipped when there is no CUDA GPU or the kernel cannot load:

1. paged self-attention: the flash paged-decode path and a plain SDPA reference,
   given the same page-major cache and metadata, must agree (bf16 tolerance).
2. a tiny random BART (head dim 32, supported by the kernel) must decode the
   same greedy/beam tokens through the flash and SDPA backends.

Run with:  PYTHONPATH=. python tests/test_gpu_flash.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from fastencdec.config import Config
from fastencdec.engine.llm_engine import LLMEngine
from fastencdec.layers import attention
from fastencdec.utils.context import set_context

VOCAB = 256
EOS = 2
BLOCK_SIZE = 256  # flash's paged kernel needs a multiple of 256


def _sdpa_single(query, key, value, scale):
    # query [H, D], key/value [L, H, D] -> single query attends to every key.
    out = F.scaled_dot_product_attention(
        query.unsqueeze(0).unsqueeze(2).float(),
        key.transpose(0, 1).unsqueeze(0).float(),
        value.transpose(0, 1).unsqueeze(0).float(),
        scale=scale,
    )
    return out.squeeze(0).squeeze(1)


def paged_parity(device):
    H, D = 2, 32
    lengths = [5, 1, 8, 3, 2, 6, 1, 7]
    B = len(lengths)
    scale = D**-0.5
    dtype = torch.bfloat16
    torch.manual_seed(0)

    k_cache = torch.randn(B, BLOCK_SIZE, H, D, dtype=dtype, device=device)
    v_cache = torch.randn(B, BLOCK_SIZE, H, D, dtype=dtype, device=device)
    query = torch.randn(B, H, D, dtype=dtype, device=device)
    key = torch.randn(B, H, D, dtype=dtype, device=device)
    value = torch.randn(B, H, D, dtype=dtype, device=device)

    # Reference: each sequence's cache is positions 0..L-1 of its page, with the
    # new token (written by the op) as the last cached key.
    want = []
    for i, length in enumerate(lengths):
        k_seq = k_cache[i, :length].clone()
        v_seq = v_cache[i, :length].clone()
        k_seq[length - 1] = key[i]
        v_seq[length - 1] = value[i]
        want.append(_sdpa_single(query[i], k_seq, v_seq, scale))
    want = torch.stack(want)

    set_context(
        context_lens=torch.tensor(lengths, dtype=torch.int32, device=device),
        block_tables=torch.arange(B, dtype=torch.int32, device=device).unsqueeze(1),
        slot_mapping=torch.tensor(
            [i * BLOCK_SIZE + (length - 1) for i, length in enumerate(lengths)],
            dtype=torch.int64,
            device=device,
        ),
        request_ids=list(range(B)),
    )
    attention.set_backend("flash")
    got = attention.decoder_self_attention(
        query, key, value, k_cache, v_cache, scale, BLOCK_SIZE
    )

    diff = (want.float() - got.float()).abs().max().item()
    ok = diff < 5e-2
    print(
        f"[{'OK' if ok else 'MISMATCH'}] paged self-attn flash vs sdpa: maxdiff={diff}"
    )
    return ok


def build_model():
    from transformers import BartConfig

    from fastencdec.models.bart import BartForConditionalGeneration

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


def decode(model, backend, requests, params, device, max_num_seqs=8):
    config = Config(
        num_blocks=32,
        block_size=BLOCK_SIZE,
        max_num_seqs=max_num_seqs,
        dtype=torch.bfloat16,
        attn_backend=backend,
        device=device,
    )
    engine = LLMEngine(model, config, eos_token_id=EOS)
    ids = engine.generate(requests, params)
    return [engine.results[i] for i in ids]


def model_parity(device):
    from fastencdec.sampling_params import SamplingParams

    model = build_model().to(torch.bfloat16).to(device)
    torch.manual_seed(1)
    requests = [
        (torch.randint(3, VOCAB, (17,)).tolist(), [1], None),
        (torch.randint(3, VOCAB, (31,)).tolist(), [1], None),
        (torch.randint(3, VOCAB, (9,)).tolist(), [1], None),
    ]
    ok = True
    for beams in (1, 2):
        params = SamplingParams(max_tokens=10, num_beams=beams, temperature=0.0)
        want = decode(model, "sdpa", requests, params, device)
        got = decode(model, "flash", requests, params, device)
        for i, (a, b) in enumerate(zip(want, got)):
            status = "OK" if a == b else "MISMATCH"
            ok &= a == b
            print(f"[{status}] tiny BART beams={beams} seq={i}: {b}")
    return ok


def graph_buckets(device):
    """A batch that fills a non-bucket ``max_num_seqs`` still decodes.

    The CUDA graph buckets are ``[1, 2, 4, 8, ...]``, so ``max_num_seqs=6`` used
    to leave no bucket for the full batch and ``replay`` raised ``StopIteration``.
    """
    from fastencdec.sampling_params import SamplingParams

    model = build_model().to(torch.bfloat16).to(device)
    torch.manual_seed(2)
    lengths = (17, 31, 9, 12, 5, 21)
    requests = [(torch.randint(3, VOCAB, (n,)).tolist(), [1], None) for n in lengths]
    params = SamplingParams(max_tokens=8, num_beams=1, temperature=0.0)
    want = decode(model, "sdpa", requests, params, device, max_num_seqs=6)
    got = decode(model, "flash", requests, params, device, max_num_seqs=6)
    ok = want == got
    print(f"[{'OK' if ok else 'MISMATCH'}] graph buckets, max_num_seqs=6 batch=6")
    return ok


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return
    from fastencdec.layers import flash_attn

    if flash_attn.module() is None:
        print("SKIP: flash_attn (Dao-AILab flash-attention) unavailable")
        return
    device = "cuda"
    ok = paged_parity(device)
    ok &= model_parity(device)
    ok &= graph_buckets(device)
    if not ok:
        raise SystemExit("flash parity failed")


if __name__ == "__main__":
    main()
