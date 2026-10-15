import torch


def fuse_projections(model, state_dict: dict) -> None:
    """Concatenate separate HF projections (e.g. q/k/v_proj -> qkv_proj) along dim 0 in state_dict."""
    for prefix, module in model.named_modules():
        for target, sources in getattr(module, "fused_projections", {}).items():
            for suffix in ("weight", "bias"):
                parts = [f"{prefix}.{source}.{suffix}" for source in sources]
                if all(part in state_dict for part in parts):
                    state_dict[f"{prefix}.{target}.{suffix}"] = torch.cat(
                        [state_dict.pop(part) for part in parts], dim=0
                    )


def load_hf_weights(model, model_path: str, hf_model_cls=None):
    """Load HuggingFace checkpoint weights (with projection fusion) into model."""
    if hf_model_cls is None:
        from transformers import AutoModelForSeq2SeqLM

        hf_model_cls = AutoModelForSeq2SeqLM

    hf_model = hf_model_cls.from_pretrained(model_path, dtype=torch.float32)
    state_dict = hf_model.state_dict()
    fuse_projections(model, state_dict)

    # state_dict() (unlike named_parameters()) keeps tied weight aliases.
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
