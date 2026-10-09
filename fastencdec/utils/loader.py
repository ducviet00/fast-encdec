"""Load HuggingFace checkpoint weights into a fast-encdec model.

The fast-encdec modules deliberately mirror the HuggingFace parameter names
(``model.encoder.*`` for BART, ``model.vision_tower.*`` /
``model.language_model.*`` for Florence-2), so loading is a straight name
match.  Tied weights are shared storage in both models, so copying every key is
idempotent.
"""

import torch


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
