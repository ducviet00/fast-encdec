from collections import defaultdict

import torch

from ..layers import attention
from ..models import bart
from ..utils.context import set_context
from .cudagraph import DecodeGraphRunner


class ModelRunner:
    def __init__(
        self,
        model,
        block_manager,
        dtype=torch.float32,
        attn_backend="auto",
        device="cpu",
        max_num_seqs=32,
        use_cudagraph=True,
        compile_model=False,
    ):
        self.device = torch.device(device)
        self.model = model.eval()
        self.block_manager = block_manager
        self.block_size = block_manager.block_size
        self.dtype = dtype
        self.decoder_layers = list(model.decoder_layers)
        # Fuse per-layer pointwise ops (add+norm, act+fc1) when compile_model=True.
        bart.enable_compile(compile_model)
        self.decoder = model.decoder

        attention.set_backend(attn_backend)

        self.cache_layers = [layer.self_attn for layer in self.decoder_layers]
        self.num_heads = self.cache_layers[0].num_heads
        self.head_dim = self.cache_layers[0].head_dim
        self.use_cudagraph = use_cudagraph and (
            self.device.type == "cuda"
            and attention.backend(self.device, dtype, self.head_dim, self.block_size)
            == "flash"
        )
        # One extra cache row serves as scratch for padded CUDA-graph rows.
        rows = block_manager.num_blocks + int(self.use_cudagraph)
        shape = attention.self_cache_shape(
            rows, self.block_size, self.num_heads, self.head_dim, self.device, dtype
        )
        for layer in self.cache_layers:
            layer.k_cache = torch.zeros(shape, dtype=dtype, device=self.device)
            layer.v_cache = torch.zeros(shape, dtype=dtype, device=self.device)
            layer.block_size = self.block_size
        for layer in self.decoder_layers:
            layer.encoder_attn.block_size = self.block_size
        self.graph = (
            DecodeGraphRunner(self, max_num_seqs) if self.use_cudagraph else None
        )

    def _copy_block(self, src: int, dst: int) -> None:
        for layer in self.cache_layers:
            layer.k_cache[dst].copy_(layer.k_cache[src])
            layer.v_cache[dst].copy_(layer.v_cache[src])

    @torch.no_grad()
    def _encode(self, seqs) -> None:
        # Group fresh sequences by (encoder_len, image_shape) and cache cross-attn K/V.
        groups = defaultdict(list)
        for seq in seqs:
            px_shape = (
                None if seq.pixel_values is None else tuple(seq.pixel_values.shape)
            )
            groups[(len(seq.encoder_token_ids), px_shape)].append(seq)
        for group in groups.values():
            ids = torch.tensor(
                [s.encoder_token_ids for s in group],
                dtype=torch.long,
                device=self.device,
            )
            if group[0].pixel_values is None:
                hidden = self.model.encode(ids)
            else:
                px = torch.stack([s.pixel_values for s in group]).to(self.device)
                hidden = self.model.encode(ids, pixel_values=px)
            for layer in self.decoder_layers:
                attn = layer.encoder_attn
                shape = (len(group), ids.shape[1], attn.num_heads, attn.head_dim)
                key, value = attn.kv_proj(hidden).chunk(2, dim=-1)
                key, value = key.reshape(shape), value.reshape(shape)
                for j, seq in enumerate(group):
                    attn.encoder_kv_cache[seq.request_id] = (
                        key[j].transpose(0, 1).contiguous(),
                        value[j].transpose(0, 1).contiguous(),
                    )

    def clear_request(self, request_id: int) -> None:
        if self.graph is not None:
            self.graph.req_blocks.pop(request_id, None)
        for layer in self.decoder_layers:
            attn = layer.encoder_attn
            attn.encoder_kv_cache.pop(request_id, None)
            attn.cross_staged = None

    def _prepare(self, seqs):
        bs = self.block_size
        # Check block budget upfront so a full cache raises before mutating state.
        required = sum(
            self.block_manager.required_new_blocks(
                seq, len(seq.token_ids) - seq.num_cached_tokens
            )
            for seq in seqs
        )
        if required > self.block_manager.num_free_blocks:
            raise RuntimeError("KV cache is full; increase num_blocks")

        input_ids, positions, new_slots = [], [], []
        context_lens, seq_block_tables = [], []
        is_prefill = True
        for seq in seqs:
            num_new = len(seq.token_ids) - seq.num_cached_tokens
            is_prefill &= seq.num_cached_tokens == 0
            self.block_manager.ensure_capacity(seq, num_new, self._copy_block)
            start = seq.num_cached_tokens
            for pos in range(start, start + num_new):
                input_ids.append(seq.token_ids[pos])
                positions.append(pos)
                new_slots.append(seq.block_table[pos // bs] * bs + pos % bs)
            seq.num_cached_tokens += num_new
            context_lens.append(seq.num_cached_tokens)
            seq_block_tables.append(seq.block_table)

        max_blocks = max(len(t) for t in seq_block_tables)
        padded = [t + [0] * (max_blocks - len(t)) for t in seq_block_tables]
        # Pinned host memory + non_blocking avoids stream syncs on CUDA.
        pin = self.device.type == "cuda"
        to_dev = lambda x, dtype: torch.tensor(x, dtype=dtype, pin_memory=pin).to(
            self.device, non_blocking=True
        )
        fields = {
            "is_prefill": is_prefill,
            "context_lens": to_dev(context_lens, torch.int32),
            "block_tables": to_dev(padded, torch.int32),
            "slot_mapping": to_dev(new_slots, torch.int64),
            "request_ids": [seq.request_id for seq in seqs],
        }
        return to_dev(input_ids, torch.long), to_dev(positions, torch.long), fields

    @torch.no_grad()
    def run(self, seqs):
        fresh = [seq for seq in seqs if seq.num_cached_tokens == 0]
        if fresh:
            self._encode(fresh)

        input_ids, positions, fields = self._prepare(seqs)

        if self.graph is not None:
            self.graph.stage(seqs)
            if len(seqs) <= self.graph.max_bs and self.graph.ready():
                self.graph.capture()
                hidden = self.graph.replay(len(seqs), input_ids, positions, fields)
                return self.model.compute_logits(hidden)

        set_context(**fields)
        hidden = self.decoder(input_ids, positions)
        return self.model.compute_logits(hidden)
