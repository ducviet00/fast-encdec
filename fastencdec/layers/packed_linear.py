import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    logger.propagate = False

_DTYPES = (torch.float32, torch.bfloat16)


def available() -> bool:
    """Return True if oneDNN blocked-weight linear ops are available in this PyTorch build."""
    return (
        torch.backends.mkldnn.is_available()
        and hasattr(torch.ops.mkldnn, "_linear_pointwise")
        and hasattr(torch.ops.mkldnn, "_reorder_linear_weight")
    )


def _forward(packed: torch.Tensor, bias: torch.Tensor):
    """Build a forward closure that runs the oneDNN pointwise linear op against a prepacked weight."""

    # Inductor cannot consume opaque blocked weights, so disable compilation here.
    @torch.compiler.disable
    def forward(x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        return torch.ops.mkldnn._linear_pointwise.default(
            x, packed, bias, "none", [-1], ""
        )

    return forward


def apply(model: nn.Module, m_hint: int, exclude: nn.Module | None = None) -> int:
    """Prepack all supported nn.Linear weights in model for oneDNN and return the packed count."""
    if not available():
        logger.info("mkldnn prepacked linear unavailable, using F.linear")
        return 0
    skip = {id(m) for m in exclude.modules()} if exclude is not None else set()
    packed = 0
    for module in model.modules():
        if id(module) in skip:
            continue
        if not isinstance(module, nn.Linear) or module.weight.dtype not in _DTYPES:
            continue
        try:
            weight = torch.ops.mkldnn._reorder_linear_weight(
                module.weight.detach(), max(m_hint, 1)
            )
        except Exception:  # noqa: BLE001, S112
            continue
        module.forward = _forward(weight, module.bias)
        packed += 1
    logger.info("prepacked %d CPU linear weights for oneDNN", packed)
    return packed
