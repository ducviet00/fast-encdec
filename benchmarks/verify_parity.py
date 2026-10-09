"""Token-by-token parity vs HuggingFace on the real checkpoints (bf16 + fp32).

For a fair speed comparison both engines must do the same work: same input,
same dtype, same number of output tokens.  This script checks the strongest
form of that on ``facebook/bart-large-cnn`` and
``florence-community/Florence-2-base``: greedy (and beam) output compared
element-by-element against ``generate``.

BART uses fixed-length decoding (EOS suppressed) so both engines emit exactly
``--max-tokens`` tokens.  Florence-2 uses neutral decoding and stops at EOS.

    PYTHONPATH=. python benchmarks/verify_parity.py
    PYTHONPATH=. python benchmarks/verify_parity.py --dtype float32
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch

from fastencdec import LLM, SamplingParams

BART = "facebook/bart-large-cnn"
FLORENCE = "florence-community/Florence-2-base"
IMAGE = "/tmp/opencode/florence2_bench.jpg"

BART_TEXTS = [
    "The quick brown fox jumps over the lazy dog. " * 6,
    (
        "Scientists at the university announced on Monday that they had "
        "discovered a new species of butterfly in the Amazon rainforest. The "
        "insect has bright blue wings and a wingspan of nearly ten centimetres."
    ),
    "The stock market rose sharply on Tuesday after the central bank kept rates unchanged.",
]

FLORENCE_PROMPTS = ["<CAPTION>", "<OD>", "<REFERRING_EXPRESSION_SEGMENTATION>the car"]


def _status(a, b):
    if a == b:
        return "OK"
    n = min(len(a), len(b))
    first = next((i for i in range(n) if a[i] != b[i]), n)
    return f"MISMATCH (len {len(a)} vs {len(b)}, first diff at {first})"


def bart(dtype, beams, max_tokens):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    llm = LLM(BART, num_blocks=2048, block_size=16, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(BART)
    hf = AutoModelForSeq2SeqLM.from_pretrained(BART, dtype=dtype).eval()

    params = SamplingParams(
        max_tokens=max_tokens,
        num_beams=beams,
        temperature=0.0,
        min_length=max_tokens + 1,  # EOS suppressed: exactly max_tokens
        length_penalty=2.0,
        no_repeat_ngram_size=3,
    )
    ours = [
        o.outputs[0].token_ids for o in llm.generate(BART_TEXTS, sampling_params=params)
    ]

    enc = tokenizer(BART_TEXTS, return_tensors="pt", padding=True)
    with torch.no_grad():
        ref = hf.generate(
            **enc,
            max_new_tokens=max_tokens,
            min_new_tokens=max_tokens,
            num_beams=beams,
            do_sample=False,
            forced_bos_token_id=None,
            forced_eos_token_id=None,
            early_stopping=False,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
        )
    ref = [row.tolist()[1:] for row in ref]  # drop decoder-start
    for i, (a, b) in enumerate(zip(ours, ref)):
        print(f"[{_status(a, b)}] bart beams={beams} seq={i}")


def florence(dtype, beams, max_tokens):
    from PIL import Image
    from transformers import AutoProcessor
    from transformers import Florence2ForConditionalGeneration as HFFlorence2

    image = Image.open(IMAGE).convert("RGB")
    llm = LLM(FLORENCE, num_blocks=4096, block_size=16, dtype=dtype)
    processor = AutoProcessor.from_pretrained(FLORENCE)
    hf = HFFlorence2.from_pretrained(FLORENCE, dtype=dtype).eval()

    prepared = []
    for prompt in FLORENCE_PROMPTS:
        inputs = llm.processor(text=prompt, images=image, return_tensors="pt")
        prepared.append((inputs["input_ids"][0].tolist(), inputs["pixel_values"][0]))

    params = SamplingParams(
        max_tokens=max_tokens,
        num_beams=beams,
        temperature=0.0,
        eos_token_id=llm.eos_token_id,
    )
    engine_requests = [
        (ids, [llm.decoder_start_token_id], pixels) for ids, pixels in prepared
    ]
    request_ids = llm.engine.generate(engine_requests, params)
    ours = [llm.engine.results[r] for r in request_ids]

    inputs = processor(
        text=FLORENCE_PROMPTS,
        images=[image] * len(FLORENCE_PROMPTS),
        padding=True,
        return_tensors="pt",
    )
    pad = processor.tokenizer.pad_token_id
    with torch.no_grad():
        out = hf.generate(
            **inputs,
            max_new_tokens=max_tokens,
            num_beams=beams,
            do_sample=False,
            forced_bos_token_id=None,
            forced_eos_token_id=None,
            early_stopping=False,
            length_penalty=1.0,
            no_repeat_ngram_size=0,
        )
    ref = []
    for row in out:
        toks = row.tolist()[1:]
        while toks and toks[-1] == pad:
            toks.pop()
        ref.append(toks)
    for i, (a, b) in enumerate(zip(ours, ref)):
        print(f"[{_status(a, b)}] florence beams={beams} seq={i}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)
    print(f"== {args.dtype} ==", flush=True)
    for beams in (1, 4):
        bart(dtype, beams, args.max_tokens)
    for beams in (1, 3):
        florence(dtype, beams, args.max_tokens)


if __name__ == "__main__":
    main()
