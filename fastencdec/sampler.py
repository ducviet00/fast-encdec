"""Token sampling on already-processed logits."""

import torch


@torch.no_grad()
def sample_token(logits: torch.Tensor, temperature: float) -> int:
    """Greedy when ``temperature <= 0``, otherwise multinomial."""
    if temperature <= 0:
        return int(torch.argmax(logits))
    return int(torch.multinomial(torch.softmax(logits, dim=-1), 1))
