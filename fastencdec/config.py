"""Engine configuration shared by the scheduler and model runner."""

from dataclasses import dataclass

import torch


@dataclass
class Config:
    """Engine hyperparameters.

    Args:
        num_blocks: number of paged KV blocks (total KV capacity).
        block_size: tokens per KV block; the vendored kernel needs a multiple
            of 32.
        max_num_seqs: maximum sequences (beams) in flight at once.
        dtype: ``torch.float32`` or ``torch.bfloat16``.
        attn_backend: ``"auto"`` (vendored vLLM kernel when it builds, else
            SDPA), ``"vllm"`` (require the kernel) or ``"sdpa"``.
    """

    num_blocks: int = 512
    block_size: int = 16
    max_num_seqs: int = 32
    dtype: torch.dtype = torch.float32
    attn_backend: str = "auto"

    def __post_init__(self) -> None:
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if self.attn_backend not in ("auto", "vllm", "sdpa"):
            raise ValueError("attn_backend must be 'auto', 'vllm' or 'sdpa'")
