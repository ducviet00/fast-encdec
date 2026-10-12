"""Engine configuration shared by the scheduler and model runner."""

from dataclasses import dataclass

import torch


@dataclass
class Config:
    """Engine hyperparameters.

    Args:
        num_blocks: number of paged KV blocks (total KV capacity).
        block_size: tokens per KV block.  ``None`` picks a device-appropriate
            default: 256 on CUDA (FlashAttention-2's paged kernel needs a page
            size divisible by 256) and 16 on CPU (the vendored kernel needs a
            multiple of 32).
        max_num_seqs: maximum sequences (beams) in flight at once.
        dtype: ``torch.float32`` or ``torch.bfloat16``.
        attn_backend: ``"auto"`` (the best kernel for the device: the vendored
            vLLM CPU kernel or FlashAttention-2 on CUDA when they build, else
            SDPA), ``"vllm"`` (require the CPU kernel), ``"flash"`` (require
            FlashAttention-2) or ``"sdpa"`` (force the pure-PyTorch path).
        device: ``"cpu"`` or ``"cuda"`` (``"cuda:N"`` also works).
        use_cudagraph: capture the decoder step into a CUDA graph (CUDA +
            FlashAttention-2 only; default off).  Prototype.
    """

    num_blocks: int = 512
    block_size: int | None = None
    max_num_seqs: int = 32
    dtype: torch.dtype = torch.float32
    attn_backend: str = "auto"
    device: str = "cpu"
    use_cudagraph: bool = False

    def __post_init__(self) -> None:
        if self.block_size is None:
            cuda = str(self.device).startswith("cuda")
            self.block_size = 256 if cuda else 16
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if self.attn_backend not in ("auto", "vllm", "flash", "sdpa"):
            raise ValueError("attn_backend must be 'auto', 'vllm', 'flash' or 'sdpa'")
