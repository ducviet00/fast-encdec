"""BART model with CPU paged attention.

Adapted from the HuggingFace BART reference.  Compared to the GPU version this
only swaps the attention kernels:

* encoder self-attention stays a plain non-causal SDPA (it runs once and holds
  no cache),
* decoder self-attention reads/writes the paged KV cache through the global
  :class:`~fastencdec.context.Context`,
* decoder cross-attention uses encoder K/V cached per request on the layer.

Everything else (projections, LayerNorms, embeddings, residual layout, the
``+2`` positional-embedding offset and the logits bias) is unchanged so the
HuggingFace weights load 1:1.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..attention import cached_cross_attention, paged_attention, store_kvcache
from ..context import get_context

ACT2FN = {"gelu": F.gelu}


class BartLearnedPositionalEmbedding(nn.Embedding):
    """Learned positional embeddings; BART offsets ids by 2 (padding hack)."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input: torch.Tensor):
        return super().forward(input + self.offset)


class BartScaledWordEmbedding(nn.Embedding):
    """Embeddings scaled by ``sqrt(d_model)`` when ``scale_embedding`` is set."""

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
        return super().forward(input) * self.embed_scale


class BartEncoderSelfAttention(nn.Module):
    """Non-causal multi-head self-attention for the encoder (no cache)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        config=None,
        layer_idx: int | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [batch, seq, dim] -> [batch, heads, seq, head_dim]
        shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        query = self.q_proj(hidden_states).view(shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        out = F.scaled_dot_product_attention(query, key, value, is_causal=False)
        out = out.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1).contiguous()
        return self.out_proj(out)


class BartDecoderSelfAttention(nn.Module):
    """Causal multi-head self-attention backed by the paged KV cache."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        config=None,
        layer_idx: int | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Assigned by the model runner: [num_blocks, block_size, heads, head_dim].
        self.k_cache = None
        self.v_cache = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_context()
        num_tokens = hidden_states.shape[0]
        head = (num_tokens, self.num_heads, self.head_dim)

        query = self.q_proj(hidden_states).view(head)
        key = self.k_proj(hidden_states).view(head)
        value = self.v_proj(hidden_states).view(head)

        store_kvcache(key, value, self.k_cache, self.v_cache, context.slot_mapping)
        out = paged_attention(query, self.k_cache, self.v_cache, self.scaling)
        return self.out_proj(out.reshape(num_tokens, self.embed_dim))


class BartDecoderCrossAttention(nn.Module):
    """Cross-attention using encoder K/V cached once per request."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        config=None,
        layer_idx: int | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # request_id -> (key, value), each [num_heads, enc_len, head_dim].
        # Dense per request (not paged); shared by all beams of the request.
        # Stored head-first so the flash kernel sees a contiguous key tensor.
        self.encoder_kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        head = (num_tokens, self.num_heads, self.head_dim)
        query = self.q_proj(hidden_states).view(head)
        out = cached_cross_attention(query, self, self.scaling)
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
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states)
        hidden_states = self.self_attn_layer_norm(residual + hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        return self.final_layer_norm(residual + hidden_states)


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
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states)
        hidden_states = self.self_attn_layer_norm(residual + hidden_states)

        residual = hidden_states
        hidden_states = self.encoder_attn(hidden_states)
        hidden_states = self.encoder_attn_layer_norm(residual + hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        return self.final_layer_norm(residual + hidden_states)


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
        # ``inputs_embeds`` is used by multimodal models (Florence-2) that mix
        # projected image features into the token embeddings before encoding.
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds + self.embed_positions(positions)
        hidden_states = self.layernorm_embedding(hidden_states)
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
        # input_ids/positions are flattened [num_tokens] across the batch.
        hidden_states = self.embed_tokens(input_ids) + self.embed_positions(positions)
        hidden_states = self.layernorm_embedding(hidden_states)
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
            # Share storage so a single load updates all three.
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
        """Run the encoder for a batch of ids [B, S] -> hidden [B, S, D]."""
        positions = torch.arange(
            encoder_input_ids.shape[1], device=encoder_input_ids.device
        )
        positions = positions.unsqueeze(0).expand_as(encoder_input_ids)
        return self.model.encoder(encoder_input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states) + self.final_logits_bias

    @property
    def decoder_layers(self):
        """Decoder self-attention layers, read by the model runner."""
        return self.model.decoder.layers

    @property
    def decoder(self):
        """The paged decoder module, read by the model runner."""
        return self.model.decoder
