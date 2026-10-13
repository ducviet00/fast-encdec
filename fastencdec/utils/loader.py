"""Load HuggingFace checkpoint weights into a fast-encdec model.

The fast-encdec modules deliberately mirror the HuggingFace parameter names
(``model.encoder.*`` for BART, ``model.vision_tower.*`` /
``model.language_model.*`` for Florence-2), so loading is a straight name
match.  The one exception is the fused attention projections: the model holds a
single ``qkv_proj`` (or ``kv_proj``) where HuggingFace has separate ones, so
:func:`fuse_projections` concatenates the parts into the fused target first.
Tied weights are shared storage in both models, so copying every key is
idempotent.
"""

import torch


def fuse_projections(model, state_dict: dict) -> None:
    """Stack the HuggingFace parts of every fused projection into its target.

    A module may hold one fused projection where HF has several parallel ones:
    ``q_proj``/``k_proj``/``v_proj`` -> ``qkv_proj`` (or ``k_proj``/``v_proj`` ->
    ``kv_proj``).  The module declares it as
    ``fused_projections = {target: (source, ...)}`` and the parts are
    concatenated along the output dimension, so the name match in
    :func:`load_hf_weights` still finds a target for every checkpoint tensor.
    """
    for prefix, module in model.named_modules():
        for target, sources in getattr(module, "fused_projections", {}).items():
            for suffix in ("weight", "bias"):
                parts = [f"{prefix}.{source}.{suffix}" for source in sources]
                if not all(part in state_dict for part in parts):
                    continue  # biasless checkpoint
                state_dict[f"{prefix}.{target}.{suffix}"] = torch.cat(
                    [state_dict.pop(part) for part in parts], dim=0
                )


def load_hf_weights(model, model_path: str, hf_model_cls=None):
    """Copy a HF checkpoint into ``model``.

    Args:
        model: fast-encdec model with HF-identical parameter names.
        model_path: HF repo id or local path.
        hf_model_cls: HF class to instantiate (defaults to
            ``AutoModelForSeq2SeqLM``; pass ``Florence2ForConditionalGeneration``
            for Florence-2).
    """
    if hf_model_cls is None:
        from transformers import AutoModelForSeq2SeqLM

        hf_model_cls = AutoModelForSeq2SeqLM

    hf_model = hf_model_cls.from_pretrained(model_path, dtype=torch.float32)
    state_dict = hf_model.state_dict()

    fuse_projections(model, state_dict)

    # ``state_dict`` (unlike ``named_parameters``) keeps tied aliases, so every
    # checkpoint key has a target even when weights share storage.
    targets = dict(model.state_dict())
    unexpected = []
    for name, tensor in state_dict.items():
        target = targets.get(name)
        if target is None:
            unexpected.append(name)
        else:
            target.copy_(tensor)
    del hf_model

    if unexpected:
        raise KeyError(f"unexpected checkpoint keys: {unexpected[:5]} ...")
    return model
