"""Summarize a few articles with greedy and beam search on CPU."""

import time

from fastencdec import LLM, SamplingParams

ARTICLES = [
    (
        "The quick brown fox jumps over the lazy dog. It was a sunny afternoon and "
        "the fox had been running through the meadow for hours, chasing butterflies "
        "and enjoying the warm weather."
    ),
    (
        "Scientists at the university announced on Monday that they had discovered "
        "a new species of butterfly in the Amazon rainforest. The insect, which has "
        "bright blue wings, was found during a three-week expedition."
    ),
]


def main():
    llm = LLM("facebook/bart-large-cnn", num_blocks=256, block_size=16)

    for beams in (1, 4):
        # bart-large-cnn expects these generation heuristics; without them raw
        # decoding collapses to short/repetitive output.
        params = SamplingParams(
            max_tokens=48,
            num_beams=beams,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            min_length=8,
        )
        start = time.perf_counter()
        outputs = llm.generate(ARTICLES, params)
        elapsed = time.perf_counter() - start
        print(f"\n=== num_beams={beams}  ({elapsed:.2f}s) ===")
        for article, summary in zip(ARTICLES, outputs):
            print(f"- {summary}")


if __name__ == "__main__":
    main()
