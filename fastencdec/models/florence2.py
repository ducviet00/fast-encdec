"""Florence-2: DaViT vision encoder + multimodal projector + BART language model.

Only the language model is ours (:mod:`fastencdec.models.bart`); the DaViT
vision tower and the multimodal projector are imported verbatim from
``transformers`` so the HuggingFace checkpoint loads 1:1.  The module layout
mirrors HF (``model.vision_tower``, ``model.multi_modal_projector``,
``model.language_model``, ``lm_head``) so the same weight loader works.

The image-side graph runs once per request in the encoder; the decoder is the
standard paged BART decoder.  Visual tokens are produced by the projector and
scattered into the ``image_token_id`` slots of the encoder inputs, exactly as
in the reference implementation.
"""

import torch
from torch import nn
from transformers.models.florence2.modeling_florence2 import (
    Florence2MultiModalProjector,
    Florence2VisionBackbone,
)

from .bart import BartModel


class Florence2Model(nn.Module):
    """Vision tower, projector and BART language model (HF key layout)."""

    def __init__(self, config):
        super().__init__()
        self.vision_tower = Florence2VisionBackbone(config.vision_config)
        self.multi_modal_projector = Florence2MultiModalProjector(config)
        self.language_model = BartModel(config.text_config)


class Florence2ForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Florence2Model(config)
        self.lm_head = nn.Linear(
            config.text_config.d_model, config.text_config.vocab_size, bias=False
        )
        if config.text_config.tie_word_embeddings:
            self.lm_head.weight = self.model.language_model.shared.weight

    @property
    def decoder_layers(self):
        """Decoder self-attention layers, read by the model runner."""
        return self.model.language_model.decoder.layers

    @property
    def decoder(self):
        """The paged decoder module, read by the model runner."""
        return self.model.language_model.decoder

    @torch.no_grad()
    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``[B, 3, H, W]`` -> visual tokens ``[B, 1 + H' * W', d_model]``."""
        image_outputs = self.model.vision_tower(pixel_values)
        return self.model.multi_modal_projector(image_outputs.last_hidden_state)

    def compile_mm_encoder(self, mode: str = "default", dynamic: bool = False) -> None:
        """Opt in to ``torch.compile`` for the vision tower + projector.

        Inductor fuses/reorders the BF16 graph, so outputs can shift slightly and
        the first call compiles once per input shape (batch size).  Off by
        default; the decoder is untouched.
        """
        self.get_image_features = torch.compile(
            self.get_image_features, mode=mode, dynamic=dynamic
        )

    @torch.no_grad()
    def encode(
        self, encoder_input_ids: torch.Tensor, pixel_values: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Run the encoder with visual tokens scattered into image slots."""
        language_model = self.model.language_model
        positions = torch.arange(
            encoder_input_ids.shape[1], device=encoder_input_ids.device
        )
        positions = positions.unsqueeze(0).expand_as(encoder_input_ids)
        inputs_embeds = language_model.shared(encoder_input_ids)
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)
            image_features = image_features.to(inputs_embeds.dtype)
            image_mask = encoder_input_ids == self.config.image_token_id
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.unsqueeze(-1), image_features
            )
        return language_model.encoder(positions=positions, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
