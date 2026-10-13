"""Engine configuration shared by the scheduler and model runner."""

import platform
from dataclasses import dataclass

import torch


def default_block_size(device) -> int:
    """Page size the best attention kernel for ``device`` wants.

    CUDA's FlashAttention-2 needs a multiple of 256; the vendored CPU kernel
    needs 32 on x86 (its ``vec`` AVX-512 path) but only 16 on aarch64 (its
    NEON BFMMLA path), and a smaller page is a finer-grained copy-on-write.
    """
    if str(device).startswith("cuda"):
        return 256
    if platform.machine().lower() in ("arm64", "aarch64"):
        return 16
    return 32


@dataclass
class Config:
    """Engine hyperparameters.

    Args:
        num_blocks: number of paged KV blocks (total KV capacity).
        block_size: tokens per KV block.  ``None`` picks the smallest page the
            device's best attention kernel accepts, see
            :func:`default_block_size`.
        max_num_seqs: maximum sequences (beams) in flight at once.
        dtype: ``torch.float32`` or ``torch.bfloat16``.
        attn_backend: ``"auto"`` (the best kernel for the device: the vendored
            vLLM CPU kernel or FlashAttention-2 on CUDA when they build, else
            SDPA), ``"vllm"`` (require the CPU kernel), ``"flash"`` (require
            FlashAttention-2) or ``"sdpa"`` (force the pure-PyTorch path).
        device: ``"cpu"`` or ``"cuda"`` (``"cuda:N"`` also works).
        use_cudagraph: capture the decoder step into a CUDA graph.  On by
            default, but it only takes effect on CUDA with the FlashAttention-2
            backend; everywhere else it is ignored (the runner ANDs it with the
            device/backend check) and the eager path runs.
        compile_model: fuse the per-layer pointwise sequences (residual add +
            layer norm, activation + fc1) with ``torch.compile`` (default off),
            nano-vllm style.  Ignored for Florence-2's vision tower, which has
            its own ``compile_mm_encoder``.
    """

    num_blocks: int = 512
    block_size: int | None = None
    max_num_seqs: int = 32
    dtype: torch.dtype = torch.float32
    attn_backend: str = "auto"
    device: str = "cpu"
    use_cudagraph: bool = True
    compile_model: bool = False

    def __post_init__(self) -> None:
        if self.block_size is None:
            self.block_size = default_block_size(self.device)
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if self.attn_backend not in ("auto", "vllm", "flash", "sdpa"):
            raise ValueError("attn_backend must be 'auto', 'vllm', 'flash' or 'sdpa'")
