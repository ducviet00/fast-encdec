"""Kernel-vs-SDPA parity for the vendored vLLM CPU attention backend.

Builds a tiny BART whose ``head_dim`` the kernel supports (32) and checks that
greedy and beam decoding match the pure-PyTorch path token for token, for a
batch of mixed encoder lengths.  Skips silently when the extension cannot
build (no compiler / unsupported machine).

Run with:  PYTHONPATH=. python tests/test_attn_backend.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from fastencdec.config import Config
from fastencdec.engine.llm_engine import LLMEngine
from fastencdec.layers import cpu_attn
from fastencdec.models.bart import BartForConditionalGeneration
from fastencdec.sampling_params import SamplingParams

VOCAB = 256
EOS = 2


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


def decode(model, backend, requests, params):
    config = Config(
        num_blocks=256,
        block_size=16,
        max_num_seqs=8,
        dtype=torch.float32,
        attn_backend=backend,
    )
    engine = LLMEngine(model, config, eos_token_id=EOS)
    ids = engine.generate(requests, params)
    return [engine.results[i] for i in ids]


def main():
    if cpu_attn.select_isa(torch.float32, 16, 32) is None:
        print("SKIP: no kernel for this machine")
        return
    model = build_model()
    torch.manual_seed(1)
    requests = [
        (torch.randint(3, VOCAB, (17,)).tolist(), [1], None),
        (torch.randint(3, VOCAB, (31,)).tolist(), [1], None),
        (torch.randint(3, VOCAB, (9,)).tolist(), [1], None),
    ]
    failures = 0
    for beams in (1, 2):
        params = SamplingParams(max_tokens=10, num_beams=beams, temperature=0.0)
        ref = decode(model, "sdpa", requests, params)
        got = decode(model, "vllm", requests, params)
        for i, (want, have) in enumerate(zip(ref, got)):
            status = "OK" if want == have else "MISMATCH"
            failures += status == "MISMATCH"
            print(f"[{status}] beams={beams} seq={i}: {have}")
    if failures:
        raise SystemExit(f"{failures} mismatches")


if __name__ == "__main__":
    main()
