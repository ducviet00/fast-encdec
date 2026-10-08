"""Load HuggingFace BART weights into the paged-attention model."""

import torch


def load_hf_weights(model, model_path: str):
    """Copy a HF ``*ForConditionalGeneration`` checkpoint into ``model``."""
    from transformers import AutoModelForSeq2SeqLM

    hf_model = AutoModelForSeq2SeqLM.from_pretrained(model_path, dtype=torch.float32)
    state_dict = hf_model.state_dict()

    # Tied in our model (single shared embedding); loading `model.shared.weight`
    # updates all of them because they share storage.
    tied = {
        "model.encoder.embed_tokens.weight",
        "model.decoder.embed_tokens.weight",
        "lm_head.weight",
    }

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    unexpected = []
    for name, tensor in state_dict.items():
        if name in tied:
            continue
        if name in params:
            params[name].data.copy_(tensor)
        elif name in buffers:
            buffers[name].copy_(tensor)
        else:
            unexpected.append(name)
    del hf_model

    if unexpected:
        raise KeyError(f"unexpected checkpoint keys: {unexpected[:5]} ...")
    return model
