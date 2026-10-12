"""FlashAttention-2 backend (CUDA) via the HuggingFace kernels hub.

The kernel is fetched on first use with ``kernels.get_kernel`` and cached by
the HuggingFace ``kernels`` library (a prebuilt wheel, no compilation).  The
module is imported lazily so a CPU-only install never touches it, and
:func:`module` returns ``None`` when it cannot be loaded, letting
``attn_backend="auto"`` fall back to the PyTorch SDPA path.

FlashAttention-2 only runs on CUDA, for fp16/bf16, with a head dim it was
compiled for; :func:`supported` is the gate the model runner uses.
"""

import contextlib
import functools

import torch

KERNEL_ID = "kernels-community/flash-attn2"

# Head dims the CUDA kernel is instantiated for (the wrapper pads other dims up
# to a multiple of 8, but the templated kernel only dispatches these).
_HEAD_DIMS = {32, 64, 96, 128, 160, 192, 224, 256}
_FLOAT_DTYPES = {torch.float16, torch.bfloat16}


def supported(dtype: torch.dtype, head_dim: int) -> bool:
    """Whether FlashAttention-2 can run this ``(dtype, head_dim)`` pair."""
    return dtype in _FLOAT_DTYPES and head_dim in _HEAD_DIMS


@functools.lru_cache(maxsize=1)
def module():
    """The loaded kernel module, or ``None`` if it could not be loaded.

    ``kernels`` >= 0.17 requires an explicit kernel API version; try the known
    stable ones, then a version-less call for older releases.
    """
    try:
        from kernels import get_kernel
    except Exception:  # noqa: BLE001 - kernels not installed
        return None
    for version in (3, 2, None):
        kwargs = {} if version is None else {"version": version}
        with contextlib.suppress(Exception):  # try the next version on failure
            return get_kernel(KERNEL_ID, **kwargs)
    return None


def require():
    """The kernel module, raising if unavailable (for ``attn_backend="flash"``)."""
    mod = module()
    if mod is None:
        raise RuntimeError(
            "FlashAttention-2 requested but the kernels-community/flash-attn2 "
            "kernel could not be loaded; install the 'kernels' package and a "
            "CUDA GPU, or use attn_backend='auto'/'sdpa'."
        )
    return mod
