"""Prepacked CPU linear layers (oneDNN blocked weights).

``torch.ops.mkldnn._reorder_linear_weight`` rewrites a ``[N, K]`` weight into the
blocked layout oneDNN (x86) / oneDNN-ACL (aarch64) streams best, and
``_linear_pointwise`` runs the GEMM against it.  Packing once at load removes the
per-call reorder ``F.linear`` pays, which is worth ~2x on the decode-shaped GEMMs
(``M`` = batch) that dominate a step; Inductor's own weight-prepack pass emits
exactly this pair.  The ops are private but ship with torch, and every layer
keeps ``F.linear`` where they are missing.
"""

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:  # the host has not configured logging
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    logger.propagate = False

_DTYPES = (torch.float32, torch.bfloat16)


def available() -> bool:
    """Whether this torch can pack a weight and run a blocked-weight linear."""
    return (
        torch.backends.mkldnn.is_available()
        and hasattr(torch.ops.mkldnn, "_linear_pointwise")
        and hasattr(torch.ops.mkldnn, "_reorder_linear_weight")
    )


def _forward(packed: torch.Tensor, bias: torch.Tensor):
    # Inductor cannot consume the opaque blocked weight, so break out of any
    # enclosing torch.compile region and run the GEMM normally.
    @torch.compiler.disable
    def forward(x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        return torch.ops.mkldnn._linear_pointwise.default(
            x, packed, bias, "none", [-1], ""
        )

    return forward


def apply(model: nn.Module, m_hint: int) -> int:
    """Pack every supported ``nn.Linear`` in ``model``; return how many were.

    ``m_hint`` (the maximum batch) is only a hint to the packer: a weight packed
    for one batch runs correctly at any other.
    """
    if not available():
        logger.info("mkldnn prepacked linear unavailable, using F.linear")
        return 0
    packed = 0
    for module in model.modules():
        if not isinstance(module, nn.Linear) or module.weight.dtype not in _DTYPES:
            continue
        try:
            weight = torch.ops.mkldnn._reorder_linear_weight(
                module.weight.detach(), max(m_hint, 1)
            )
        except Exception:  # noqa: BLE001, S112 - this layer keeps F.linear
            continue
        module.forward = _forward(weight, module.bias)
        packed += 1
    logger.info("prepacked %d CPU linear weights for oneDNN", packed)
    return packed
