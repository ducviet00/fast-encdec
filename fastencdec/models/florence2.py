import math

import torch
import torch.nn.functional as F
from torch import nn

from .bart import BartModel


class Florence2VisionMLP(nn.Module):
    def __init__(self, config, stage_idx: int):
        super().__init__()
        dim = config.embed_dim[stage_idx]
        hidden_dim = int(dim * config.mlp_ratio)
        self.activation_fn = F.gelu
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the two-layer GELU MLP on vision features."""
        return self.fc2(self.activation_fn(self.fc1(hidden_states)))


class Florence2VisionConvEmbed(nn.Module):
    def __init__(self, config, stage_idx: int):
        super().__init__()
        self.stage_idx = stage_idx
        self.patch_size = config.patch_size[stage_idx]
        self.in_channels = (
            config.in_channels if stage_idx == 0 else config.embed_dim[stage_idx - 1]
        )
        self.embed_dim = config.embed_dim[stage_idx]
        self.stride = config.patch_stride[stage_idx]
        self.padding = config.patch_padding[stage_idx]
        self.pre_norm = config.patch_prenorm[stage_idx]

        self.conv = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.stride,
            padding=self.padding,
        )
        self.norm = nn.LayerNorm(self.in_channels if self.pre_norm else self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply strided patch convolution with pre- or post-LayerNorm."""
        # LayerNorm normalizes over the last dim (channels-last).
        if self.pre_norm:
            hidden_states = self.norm(hidden_states.movedim(1, -1)).movedim(-1, 1)
        hidden_states = self.conv(hidden_states)
        if not self.pre_norm:
            hidden_states = self.norm(hidden_states.movedim(1, -1)).movedim(-1, 1)
        return hidden_states


class Florence2DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply stochastic depth during training (identity at eval time)."""
        if self.drop_prob == 0.0 or not self.training:
            return hidden_states
        keep_prob = 1 - self.drop_prob
        shape = (hidden_states.shape[0],) + (1,) * (hidden_states.ndim - 1)
        rand = torch.floor(
            torch.rand(shape, dtype=hidden_states.dtype, device=hidden_states.device)
            + keep_prob
        )
        return hidden_states.div(keep_prob) * rand


class Florence2VisionChannelAttention(nn.Module):
    def __init__(self, config, stage_idx: int):
        super().__init__()
        self.dim = config.embed_dim[stage_idx]
        self.groups = config.num_groups[stage_idx]
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=config.qkv_bias)
        self.proj = nn.Linear(self.dim, self.dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run grouped channel self-attention over the head_dim axis."""
        batch_size, num_tokens, hidden_size = hidden_states.shape
        head_dim = hidden_size // self.groups
        # Channel attention over head_dim (N is the key/value dim); contiguous for Flash SDPA.
        qkv = (
            self.qkv(hidden_states)
            .reshape(batch_size, num_tokens, 3, self.groups, head_dim)
            .permute(2, 0, 3, 4, 1)
            .contiguous()
        )
        query, key, value = qkv.unbind(0)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, scale=num_tokens**-0.5
        )
        hidden_states = hidden_states.permute(0, 3, 1, 2).reshape(
            batch_size, num_tokens, hidden_size
        )
        return self.proj(hidden_states)


class Florence2VisionWindowAttention(nn.Module):
    def __init__(self, config, stage_idx: int):
        super().__init__()
        self.dim = config.embed_dim[stage_idx]
        self.window_size = config.window_size
        self.num_heads = config.num_heads[stage_idx]
        self.scale = (self.dim // self.num_heads) ** -0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=config.qkv_bias)
        self.proj = nn.Linear(self.dim, self.dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Partition spatial features into local windows, run self-attention, and merge back."""
        batch_size, height, width, embed_dim = hidden_states.shape
        ws = self.window_size

        pad_right = (ws - width % ws) % ws
        pad_bottom = (ws - height % ws) % ws
        if pad_right or pad_bottom:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_right, 0, pad_bottom))
        padded_height, padded_width = height + pad_bottom, width + pad_right

        # Partition into non-overlapping [num_windows, ws*ws, C] windows.
        windows = (
            hidden_states.view(
                batch_size,
                padded_height // ws,
                ws,
                padded_width // ws,
                ws,
                embed_dim,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, ws * ws, embed_dim)
        )
        num_windows = windows.shape[0]

        # Contiguous QKV avoids the ~5.5x slower non-contiguous BF16 SDPA path on aarch64.
        qkv = (
            self.qkv(windows)
            .reshape(
                num_windows, ws * ws, 3, self.num_heads, embed_dim // self.num_heads
            )
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        query, key, value = qkv.unbind(0)
        windows = F.scaled_dot_product_attention(query, key, value, scale=self.scale)
        windows = self.proj(
            windows.transpose(1, 2).reshape(num_windows, ws * ws, embed_dim)
        )

        hidden_states = (
            windows.view(
                -1,
                padded_height // ws,
                padded_width // ws,
                ws,
                ws,
                embed_dim,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch_size, padded_height, padded_width, embed_dim)
        )
        if pad_right or pad_bottom:
            hidden_states = hidden_states[:, :height, :width]
        return hidden_states.reshape(batch_size, height * width, embed_dim)


class Florence2VisionSpatialBlock(nn.Module):
    def __init__(self, config, stage_idx: int, drop_path_rate: float):
        super().__init__()
        dim = config.embed_dim[stage_idx]
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.window_attn = Florence2VisionWindowAttention(config, stage_idx)
        self.drop_path1 = (
            Florence2DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = Florence2VisionMLP(config, stage_idx)
        self.drop_path2 = (
            Florence2DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run depthwise conv + local window attention + FFN on spatial features."""
        batch_size, embed_dim, height, width = hidden_states.shape

        hidden_states = self.conv1(hidden_states) + hidden_states
        residual = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.window_attn(
            self.norm1(residual).view(batch_size, height, width, embed_dim)
        )
        hidden_states = (
            (residual + self.drop_path1(hidden_states))
            .transpose(1, 2)
            .view(batch_size, embed_dim, height, width)
        )

        hidden_states = self.conv2(hidden_states) + hidden_states
        residual = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.ffn(self.norm2(residual))
        return (
            (residual + self.drop_path2(hidden_states))
            .transpose(1, 2)
            .view(batch_size, embed_dim, height, width)
        )


class Florence2VisionChannelBlock(nn.Module):
    def __init__(self, config, stage_idx: int, drop_path_rate: float):
        super().__init__()
        dim = config.embed_dim[stage_idx]
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.channel_attn = Florence2VisionChannelAttention(config, stage_idx)
        self.drop_path1 = (
            Florence2DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = Florence2VisionMLP(config, stage_idx)
        self.drop_path2 = (
            Florence2DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run depthwise conv + grouped channel attention + FFN on spatial features."""
        batch_size, embed_dim, height, width = hidden_states.shape

        hidden_states = self.conv1(hidden_states) + hidden_states
        residual = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.channel_attn(self.norm1(residual))
        hidden_states = (
            (residual + self.drop_path1(hidden_states))
            .transpose(1, 2)
            .view(batch_size, embed_dim, height, width)
        )

        hidden_states = self.conv2(hidden_states) + hidden_states
        residual = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.ffn(self.norm2(residual))
        return (
            (residual + self.drop_path2(hidden_states))
            .transpose(1, 2)
            .view(batch_size, embed_dim, height, width)
        )


class Florence2VisionBlock(nn.Module):
    def __init__(
        self,
        config,
        stage_idx: int,
        spatial_drop_path_rate: float,
        channel_drop_path_rate: float,
    ):
        super().__init__()
        self.spatial_block = Florence2VisionSpatialBlock(
            config, stage_idx, spatial_drop_path_rate
        )
        self.channel_block = Florence2VisionChannelBlock(
            config, stage_idx, channel_drop_path_rate
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run one DaViT block (spatial window block followed by channel block)."""
        return self.channel_block(self.spatial_block(hidden_states))


class Florence2VisionBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.num_groups = config.num_groups
        self.num_stages = len(self.embed_dim)

        dpr = torch.linspace(0, config.drop_path_rate, sum(config.depths) * 2).tolist()
        depth_offset = 0
        convs, blocks = [], []
        for stage_idx in range(self.num_stages):
            convs.append(Florence2VisionConvEmbed(config, stage_idx))
            blocks.append(
                nn.ModuleList(
                    Florence2VisionBlock(
                        config,
                        stage_idx,
                        spatial_drop_path_rate=dpr[depth_offset + block_idx * 2],
                        channel_drop_path_rate=dpr[depth_offset + block_idx * 2 + 1],
                    )
                    for block_idx in range(config.depths[stage_idx])
                )
            )
            depth_offset += config.depths[stage_idx] * 2

        self.convs = nn.ModuleList(convs)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images [B, 3, H, W] through the four DaViT stages into [B, C, H', W']."""
        hidden_states = pixel_values.to(dtype=self.convs[0].conv.weight.dtype)
        for conv, block in zip(self.convs, self.blocks):
            hidden_states = conv(hidden_states)
            for layer in block:
                hidden_states = layer(hidden_states)
        return hidden_states


class Florence2VisionLearnedAbsolutePositionEmbedding2D(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_pos = config.vision_config.max_position_embeddings
        embedding_dim = config.vision_config.embed_dim[-1]
        self.row_embeddings = nn.Embedding(num_pos, embedding_dim // 2)
        self.column_embeddings = nn.Embedding(
            num_pos, embedding_dim - (embedding_dim // 2)
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Build 2D row/column positional embeddings [1, C, H, W] for a spatial feature map."""
        height, width = pixel_values.shape[-2:]
        x_emb = self.column_embeddings(torch.arange(width, device=pixel_values.device))
        y_emb = self.row_embeddings(torch.arange(height, device=pixel_values.device))
        pos = torch.cat(
            [
                x_emb.unsqueeze(0).expand(height, -1, -1),
                y_emb.unsqueeze(1).expand(-1, width, -1),
            ],
            dim=-1,
        )
        return pos.permute(2, 0, 1).unsqueeze(0)


class Florence2VisionPositionalEmbeddingCosine1D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.vision_config.embed_dim[-1]
        self.max_seq_len = config.vision_config.max_temporal_embeddings
        pos_idx_to_embed = torch.empty((self.max_seq_len, self.embed_dim))
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
        emb = torch.arange(self.max_seq_len, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        pos_idx_to_embed[:, 0::2] = torch.sin(emb)
        pos_idx_to_embed[:, 1::2] = torch.cos(emb)
        self.register_buffer("pos_idx_to_embed", pos_idx_to_embed)

    def forward(self, seq_embeds: torch.Tensor) -> torch.Tensor:
        """Return 1D cosine temporal positional embeddings for seq_embeds."""
        return self.pos_idx_to_embed[: seq_embeds.size(1)]


class Florence2MultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_projection = nn.Linear(
            config.vision_config.embed_dim[-1],
            config.vision_config.projection_dim,
            bias=False,
        )
        self.image_proj_norm = nn.LayerNorm(config.vision_config.projection_dim)
        self.image_position_embed = Florence2VisionLearnedAbsolutePositionEmbedding2D(
            config
        )
        self.visual_temporal_embed = Florence2VisionPositionalEmbeddingCosine1D(config)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """Add 2D/temporal embeddings, pool global token, and project to language model dim."""
        position_features = (
            (image_features + self.image_position_embed(image_features))
            .flatten(2)
            .transpose(1, 2)
        )
        temporal_features = self.visual_temporal_embed(
            position_features[:, :1, :]
        ).unsqueeze(1)
        visual_token_features = (position_features + temporal_features).unsqueeze(1)
        image_features = torch.cat(
            [visual_token_features.mean(dim=2), visual_token_features.mean(dim=1)],
            dim=1,
        )
        return self.image_proj_norm(self.image_projection(image_features))


class Florence2Model(nn.Module):
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
        """Return the language model's decoder layer list."""
        return self.model.language_model.decoder.layers

    @property
    def decoder(self):
        """Return the language model's paged decoder module."""
        return self.model.language_model.decoder

    @torch.no_grad()
    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run the vision tower and multimodal projector to produce visual token embeddings."""
        return self.model.multi_modal_projector(self.model.vision_tower(pixel_values))

    @torch.no_grad()
    def encode(
        self,
        encoder_input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Scatter projected visual tokens into image_token_id slots and run the BART encoder.

        ``image_embeds`` lets the runner pass features it already computed (the
        tower compiles with static shapes, this graph with dynamic ones).
        """
        language_model = self.model.language_model
        positions = (
            torch.arange(encoder_input_ids.shape[1], device=encoder_input_ids.device)
            .unsqueeze(0)
            .expand_as(encoder_input_ids)
        )
        inputs_embeds = language_model.shared(encoder_input_ids)
        if image_embeds is None and pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values)
        if image_embeds is not None:
            image_embeds = image_embeds.to(inputs_embeds.dtype)
            image_mask = encoder_input_ids == self.config.image_token_id
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.unsqueeze(-1), image_embeds
            )
        return language_model.encoder(positions=positions, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project decoder hidden states to vocabulary logits."""
        return self.lm_head(hidden_states)
