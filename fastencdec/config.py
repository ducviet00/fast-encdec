import platform
from dataclasses import dataclass

import torch

_BACKENDS = ("auto", "cpu", "flash", "sdpa")


def default_block_size(device) -> int:
    """Return the optimal KV page size for the target device's attention kernel."""
    # FlashAttention-2's paged kernel needs a multiple of 256; cpu-attention of 16.
    if str(device).startswith("cuda"):
        return 256
    return 16 if platform.machine().lower() in ("arm64", "aarch64") else 32


@dataclass
class Config:
    """Engine configuration shared by the scheduler, block manager, and runner."""

    num_blocks: int = 512
    block_size: int | None = None
    max_num_seqs: int = 32
    dtype: torch.dtype = torch.float32
    attn_backend: str = "auto"
    device: str = "cpu"
    use_cudagraph: bool = True
    compile: bool = True
    compile_mm_encoder: bool = True

    def __post_init__(self) -> None:
        if self.block_size is None:
            self.block_size = default_block_size(self.device)
        if self.num_blocks <= 0 or self.block_size <= 0 or self.max_num_seqs <= 0:
            raise ValueError("num_blocks, block_size and max_num_seqs must be positive")
        if self.attn_backend not in _BACKENDS:
            raise ValueError(f"attn_backend must be one of {_BACKENDS}")
