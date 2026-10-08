"""Token sampling and n-gram banning."""

import torch


def banned_tokens(token_ids: list[int], ngram_size: int) -> set[int]:
    """Tokens that would repeat the last (ngram_size - 1) tokens."""
    if ngram_size <= 1 or len(token_ids) < ngram_size:
        return set()
    prefix = tuple(token_ids[-(ngram_size - 1):])
    banned = set()
    for i in range(len(token_ids) - ngram_size + 1):
        if tuple(token_ids[i:i + ngram_size - 1]) == prefix:
            banned.add(token_ids[i + ngram_size - 1])
    return banned


@torch.no_grad()
def sample_token(logits: torch.Tensor, params, banned: set[int] = frozenset()) -> int:
    logits = logits.float().clone()
    if banned:
        logits[list(banned)] = -float("inf")
    if params.temperature <= 0:
        return int(torch.argmax(logits))
    logits = logits / params.temperature
    probs = torch.softmax(logits, dim=-1)
    if params.top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        keep = torch.cumsum(sorted_probs, dim=-1) - sorted_probs <= params.top_p
        sorted_probs = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
        probs = torch.zeros_like(probs).scatter(0, sorted_idx, sorted_probs)
    return int(torch.multinomial(probs, 1))
