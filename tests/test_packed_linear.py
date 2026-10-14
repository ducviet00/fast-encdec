"""The CPU packed-linear path must pack when it can and never change output.

Run with:  PYTHONPATH=. python tests/test_packed_linear.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch import nn

from fastencdec import LLM, SamplingParams
from fastencdec.layers import packed_linear

MODEL = "hf-internal-testing/tiny-random-BartForConditionalGeneration"
TEXT = "The quick brown fox jumps over the lazy dog."
BATCH = [TEXT, "A second, noticeably longer sentence used to exercise batching."]


def linear_parity():
    """A decoder-shaped Linear packs and matches ``F.linear``."""
    for dtype in (torch.float32, torch.bfloat16):
        torch.manual_seed(0)
        x = torch.randn(8, 1024, dtype=dtype)
        module = nn.Linear(1024, 4096).to(dtype)
        reference = F.linear(x, module.weight, module.bias)
        applied = packed_linear.apply(nn.Sequential(module), 8)
        atol = 0.5 if dtype == torch.bfloat16 else 1e-3
        ok = bool(applied) and torch.allclose(module(x), reference, atol=atol)
        print(f"[{'OK' if ok else 'MISMATCH'}] {dtype}: packed={applied} matches")


def token_parity():
    """Packing must not change the engine's tokens (tiny random BART)."""
    original = packed_linear.available
    for dtype in (torch.float32, torch.bfloat16):
        runs = []
        for pack in (False, True):
            packed_linear.available = original if pack else (lambda: False)
            llm = LLM(MODEL, num_blocks=64, block_size=16, dtype=dtype)
            params = SamplingParams(max_tokens=12, temperature=0.0, min_length=13)
            runs.append([o.outputs[0].token_ids for o in llm.generate(BATCH, params)])
        ok = runs[0] == runs[1]
        print(f"[{'OK' if ok else 'MISMATCH'}] {dtype}: tokens unchanged")
    packed_linear.available = original


def fallback():
    """Without the mkldnn ops every layer keeps ``F.linear``."""
    original = packed_linear.available
    packed_linear.available = lambda: False
    try:
        applied = packed_linear.apply(nn.Sequential(nn.Linear(4, 4)), 8)
        print(f"[{'OK' if applied == 0 else 'MISMATCH'}] no ops -> F.linear")
    finally:
        packed_linear.available = original


def main():
    linear_parity()
    token_parity()
    fallback()


if __name__ == "__main__":
    main()
