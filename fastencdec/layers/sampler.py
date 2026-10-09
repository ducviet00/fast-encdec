"""Token sampling on already-processed logits."""

import torch
from torch import nn


class Sampler(nn.Module):
    @torch.no_grad()
    def forward(self, logits: torch.Tensor, temperature: float) -> int:
        """Greedy when ``temperature <= 0``, otherwise multinomial."""
        if temperature <= 0:
            return int(torch.argmax(logits))
        return int(torch.multinomial(torch.softmax(logits, dim=-1), 1))
