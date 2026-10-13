"""Summarize an article with BART."""

import time

from fastencdec import LLM, SamplingParams

ARTICLE = (
    "The quick brown fox jumps over the lazy dog. It was a sunny afternoon and the "
    "fox had been running through the meadow for hours, chasing butterflies and "
    "enjoying the warm weather before returning to its den."
)


def main():
    llm = LLM("facebook/bart-large-cnn")

    # bart-large-cnn needs these decoding heuristics; without them raw decoding
    # collapses to short, repetitive output.
    params = SamplingParams(
        max_tokens=64,
        num_beams=4,
        length_penalty=2.0,
        no_repeat_ngram_size=3,
        min_length=8,
    )

    start = time.perf_counter()
    outputs = llm.generate([ARTICLE], params)
    print(f"{time.perf_counter() - start:.2f}s  {outputs[0].outputs[0].text}")


if __name__ == "__main__":
    main()
