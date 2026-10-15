import torch
from torch import nn


class Sampler(nn.Module):
    """Batched token sampler supporting greedy argmax and multinomial sampling."""

    @torch.no_grad()
    def forward(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Sample one token per row from logits (greedy when temperature <= 0)."""
        if temperature <= 0:
            return torch.argmax(logits, dim=-1)
        probs = torch.softmax(logits.float(), dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
