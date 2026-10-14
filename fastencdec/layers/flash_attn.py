import contextlib

import torch

KERNEL_ID = "kernels-community/flash-attn2"
_HEAD_DIMS = {32, 64, 96, 128, 160, 192, 224, 256}
_FLOAT_DTYPES = {torch.float16, torch.bfloat16}

_MODULE = None
_RESOLVED = False


def supported(dtype: torch.dtype, head_dim: int) -> bool:
    return dtype in _FLOAT_DTYPES and head_dim in _HEAD_DIMS


def _load():
    try:
        from kernels import get_kernel
    except Exception:  # noqa: BLE001
        return None
    for version in (3, 2, None):
        kwargs = {} if version is None else {"version": version}
        with contextlib.suppress(Exception):
            return get_kernel(KERNEL_ID, **kwargs)
    return None


def module():
    # Plain global rather than lru_cache so torch.compile does not inline _load.
    global _MODULE, _RESOLVED
    if not _RESOLVED:
        _MODULE = _load()
        _RESOLVED = True
    return _MODULE


def require():
    mod = module()
    if mod is None:
        raise RuntimeError(
            "FlashAttention-2 requested but the kernels-community/flash-attn2 "
            "kernel could not be loaded; install the 'kernels' package and a "
            "CUDA GPU, or use attn_backend='auto'/'sdpa'."
        )
    return mod
