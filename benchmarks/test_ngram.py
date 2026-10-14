"""Parity: the Triton no-repeat-n-gram mask vs transformers (CUDA)."""

import sys, torch

sys.path.insert(0, ".")
from fastencdec.layers.ngram import TritonNoRepeatNGramLogitsProcessor
from transformers.generation.logits_process import NoRepeatNGramLogitsProcessor

torch.manual_seed(0)
fails = 0
for ngram in (2, 3, 4):
    for length in (2, 3, 5, 8, 16, 40):
        for _ in range(3):
            batch, vocab = 6, 64
            ids = torch.randint(0, vocab, (batch, length), device="cuda")
            # plant repeats so matches actually happen
            if length > ngram:
                ids[0, ngram:] = ids[0, : length - ngram].clone()
                ids[1, -ngram + 1 :] = ids[1, : ngram - 1].clone()
                ids[2] = ids[2, 0].expand(length)
            scores = torch.randn(batch, vocab, device="cuda")
            ref = NoRepeatNGramLogitsProcessor(ngram)(ids, scores.clone())
            got = TritonNoRepeatNGramLogitsProcessor(ngram)(ids, scores.clone())
            if not torch.equal(ref, got):
                fails += 1
                print(f"MISMATCH ngram={ngram} length={length}")
                print("  ref", (ref != got).sum().item(), "cells differ")
print("ngram parity:", "exact" if fails == 0 else f"{fails} failures")
