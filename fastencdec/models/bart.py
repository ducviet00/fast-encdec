import math
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from ..layers.attention import (
    decoder_cross_attention,
    decoder_self_attention,
    encoder_self_attention,
)
from ..layers.cpu_ops import register_cross_layer

ACT2FN = {"gelu": F.gelu}


def _add_layer_norm(norm, residual, x):
    """Apply residual addition followed by layer normalization."""
    return norm(residual + x)


def _activation_linear(linear, activation, x):
    """Apply a linear projection followed by an activation function."""
    return activation(linear(x))


class BartLearnedPositionalEmbedding(nn.Embedding):
    """Learned positional embedding that applies BART's +2 position offset."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input: torch.Tensor):
        """Look up positional embeddings with +2 offset."""
        return super().forward(input + self.offset)


class BartScaledWordEmbedding(nn.Embedding):
    """Token embedding layer with optional sqrt(d_model) scaling."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float = 1.0,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input: torch.Tensor):
        """Look up token embeddings and multiply by embed_scale."""
        return super().forward(input) * self.embed_scale


class _BartAttentionBase(nn.Module):
    """Shared head-dimension validation and scaling for BART attention modules."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5


class BartEncoderSelfAttention(_BartAttentionBase):
    """Dense non-causal multi-head self-attention with fused QKV projection."""

    fused_projections: ClassVar[dict[str, tuple[str, ...]]] = {
        "qkv_proj": ("q_proj", "k_proj", "v_proj")
    }

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        config=None,
        layer_idx: int | None = None,
    ):
        super().__init__(embed_dim, num_heads)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project QKV, run dense encoder self-attention, and project output."""
        shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        # .contiguous() avoids the ~20x slower non-contiguous BF16 SDPA path on aarch64.
        q, k, v = (
            t.reshape(shape).transpose(1, 2).contiguous()
            for t in self.qkv_proj(hidden_states).chunk(3, dim=-1)
        )
        out = encoder_self_attention(q, k, v)
        out = out.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1).contiguous()
        return self.out_proj(out)


class BartDecoderSelfAttention(_BartAttentionBase):
    """Paged causal multi-head self-attention for the decoder."""

    fused_projections: ClassVar[dict[str, tuple[str, ...]]] = {
        "qkv_proj": ("q_proj", "k_proj", "v_proj")
    }

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        config=None,
        layer_idx: int | None = None,
    ):
        super().__init__(embed_dim, num_heads)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_cache = torch.empty([])
        self.v_cache = torch.empty([])
        self.block_size = 16

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project QKV, update paged KV cache, and run causal self-attention."""
        num_tokens = hidden_states.shape[0]
        shape = (num_tokens, self.num_heads, self.head_dim)
        q, k, v = (
            t.reshape(shape) for t in self.qkv_proj(hidden_states).chunk(3, dim=-1)
        )
        out = decoder_self_attention(
            q, k, v, self.k_cache, self.v_cache, self.scaling, self.block_size
        )
        return self.out_proj(out.reshape(num_tokens, self.embed_dim))


class BartDecoderCrossAttention(_BartAttentionBase):
    """Decoder cross-attention using per-request cached encoder K/V."""

    fused_projections: ClassVar[dict[str, tuple[str, ...]]] = {
        "kv_proj": ("k_proj", "v_proj")
    }

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        config=None,
        layer_idx: int | None = None,
    ):
        super().__init__(embed_dim, num_heads)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # request_id -> (key, value) [num_heads, enc_len, head_dim]
        self.encoder_kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.cross_cache = None
        self.block_size = 16
        register_cross_layer(self)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project decoder queries and attend over cached encoder K/V."""
        num_tokens = hidden_states.shape[0]
        query = self.q_proj(hidden_states).view(
            num_tokens, self.num_heads, self.head_dim
        )
        out = decoder_cross_attention(query, self, self.scaling, self.block_size)
        return self.out_proj(out.reshape(num_tokens, self.embed_dim))


class BartEncoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int | None = None):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BartEncoderSelfAttention(
            self.embed_dim,
            config.encoder_attention_heads,
            config=config,
            layer_idx=layer_idx,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = ACT2FN.get(config.activation_function, F.gelu)
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run one BART encoder layer (self-attention + FFN with post-norm residuals)."""
        hidden_states = _add_layer_norm(
            self.self_attn_layer_norm, hidden_states, self.self_attn(hidden_states)
        )
        ffn = self.fc2(_activation_linear(self.fc1, self.activation_fn, hidden_states))
        return _add_layer_norm(self.final_layer_norm, hidden_states, ffn)


class BartDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int | None = None):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BartDecoderSelfAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            config=config,
            layer_idx=layer_idx,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = ACT2FN.get(config.activation_function, F.gelu)
        self.encoder_attn = BartDecoderCrossAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            config=config,
            layer_idx=layer_idx,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run one BART decoder layer (self-attention + cross-attention + FFN)."""
        hidden_states = _add_layer_norm(
            self.self_attn_layer_norm, hidden_states, self.self_attn(hidden_states)
        )
        hidden_states = _add_layer_norm(
            self.encoder_attn_layer_norm,
            hidden_states,
            self.encoder_attn(hidden_states),
        )
        ffn = self.fc2(_activation_linear(self.fc1, self.activation_fn, hidden_states))
        return _add_layer_norm(self.final_layer_norm, hidden_states, ffn)


class BartEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.embed_tokens = BartScaledWordEmbedding(
            config.vocab_size, embed_dim, self.padding_idx, embed_scale=self.embed_scale
        )
        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings, embed_dim
        )
        self.layers = nn.ModuleList(
            [
                BartEncoderLayer(config, layer_idx=i)
                for i in range(config.encoder_layers)
            ]
        )
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode token IDs or precomputed embeddings into encoder hidden states."""
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = self.layernorm_embedding(
            inputs_embeds + self.embed_positions(positions)
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class BartDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = BartScaledWordEmbedding(
            config.vocab_size,
            config.d_model,
            self.padding_idx,
            embed_scale=self.embed_scale,
        )
        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings, config.d_model
        )
        self.layers = nn.ModuleList(
            [
                BartDecoderLayer(config, layer_idx=i)
                for i in range(config.decoder_layers)
            ]
        )
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Run the paged decoder stack on flattened step input_ids and positions."""
        hidden_states = self.layernorm_embedding(
            self.embed_tokens(input_ids) + self.embed_positions(positions)
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class BartModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.shared = BartScaledWordEmbedding(
            config.vocab_size,
            config.d_model,
            config.pad_token_id,
            embed_scale=math.sqrt(config.d_model) if config.scale_embedding else 1.0,
        )
        self.encoder = BartEncoder(config)
        self.decoder = BartDecoder(config)

        if config.tie_word_embeddings:
            self.encoder.embed_tokens.weight = self.shared.weight
            self.decoder.embed_tokens.weight = self.shared.weight


class BartForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = BartModel(config)
        self.register_buffer(
            "final_logits_bias", torch.zeros(1, self.model.shared.num_embeddings)
        )
        self.lm_head = nn.Linear(
            config.d_model, self.model.shared.num_embeddings, bias=False
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.shared.weight

    @torch.no_grad()
    def encode(self, encoder_input_ids: torch.Tensor) -> torch.Tensor:
        """Run the encoder on a batch of token IDs [B, S] and return hidden states [B, S, D]."""
        positions = (
            torch.arange(encoder_input_ids.shape[1], device=encoder_input_ids.device)
            .unsqueeze(0)
            .expand_as(encoder_input_ids)
        )
        return self.model.encoder(encoder_input_ids, positions)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Run one decoder step and return logits [B, vocab] (the compiled decode graph)."""
        return self.compute_logits(self.decoder(input_ids, positions))

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project decoder hidden states to vocabulary logits plus final_logits_bias."""
        return self.lm_head(hidden_states) + self.final_logits_bias

    @property
    def decoder_layers(self):
        """Return the decoder layer list."""
        return self.model.decoder.layers

    @property
    def decoder(self):
        """Return the paged decoder module."""
        return self.model.decoder
