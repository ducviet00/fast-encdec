"""Lazy loader for Dao-AILab's FlashAttention-2 (the ``flash_attn`` package)."""

import torch

_HEAD_DIMS = {32, 64, 96, 128, 160, 192, 224, 256}
_FLOAT_DTYPES = {torch.float16, torch.bfloat16}

_MODULE = None
_RESOLVED = False


def supported(dtype: torch.dtype, head_dim: int) -> bool:
    """Return True if FlashAttention-2 supports the given (dtype, head_dim) pair."""
    return dtype in _FLOAT_DTYPES and head_dim in _HEAD_DIMS


def _load():
    """Import ``flash_attn`` when a CUDA device is present, or return None."""
    if not torch.cuda.is_available():
        return None
    try:
        import flash_attn
    except Exception:  # noqa: BLE001
        return None
    return flash_attn


def module():
    """Return the loaded FlashAttention-2 module, or None if unavailable."""
    # Plain global rather than lru_cache so torch.compile does not inline _load.
    global _MODULE, _RESOLVED
    if not _RESOLVED:
        _MODULE = _load()
        _RESOLVED = True
    return _MODULE


def require():
    """Return the loaded FlashAttention-2 module, raising RuntimeError if unavailable."""
    mod = module()
    if mod is None:
        raise RuntimeError(
            "FlashAttention-2 requested but the flash_attn package could not be "
            "imported; install flash-attn (https://github.com/Dao-AILab/"
            "flash-attention) on a CUDA machine, or use attn_backend='auto'/'sdpa'."
        )
    return mod
