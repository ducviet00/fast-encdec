import torch
from torch import nn


class Sampler(nn.Module):
    @torch.no_grad()
    def forward(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature <= 0:
            return torch.argmax(logits, dim=-1)
        probs = torch.softmax(logits.float(), dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
