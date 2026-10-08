"""Greedy/beam parity against HuggingFace on a tiny random BART.

Run with:  PYTHONPATH=. python tests/test_parity.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fastencdec import LLM, SamplingParams

MODEL = "hf-internal-testing/tiny-random-BartForConditionalGeneration"
TEXT = "The quick brown fox jumps over the lazy dog."
BATCH = [TEXT, "A second, noticeably longer sentence used to exercise batching."]


def main():
    llm = LLM(MODEL, num_blocks=64, block_size=16)

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForSeq2SeqLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    inputs = tokenizer(TEXT, return_tensors="pt")

    for beams in (1, 2, 4):
        params = SamplingParams(max_tokens=12, num_beams=beams, temperature=0.0)
        ours = llm.generate(TEXT, params)[0]
        with torch.no_grad():
            ref = hf.generate(**inputs, max_new_tokens=12, num_beams=beams,
                              do_sample=False, forced_bos_token_id=None,
                              forced_eos_token_id=None)
        ref_text = tokenizer.decode(ref[0], skip_special_tokens=True)
        status = "OK" if ours == ref_text else "MISMATCH"
        print(f"[{status}] beams={beams}: {ours!r}")

    # A batch with mixed encoder lengths must give the same tokens as running
    # each prompt on its own (exercises the batched encoder + cross-attn cache).
    params = SamplingParams(max_tokens=12, num_beams=1, temperature=0.0)
    batched = llm.generate(BATCH, params)
    single = [llm.generate(text, params)[0] for text in BATCH]
    for i, (b, s) in enumerate(zip(batched, single)):
        status = "OK" if b == s else "MISMATCH"
        print(f"[{status}] batch-vs-single {i}: {b!r}")


if __name__ == "__main__":
    main()
