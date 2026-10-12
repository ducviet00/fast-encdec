"""Batching + forward pass for the paged engine.

The runner only builds the per-step :class:`~fastencdec.utils.context.Context`
and the KV caches; every attention op picks its own kernel (see
:mod:`fastencdec.layers.attention`).
"""

import torch

from ..layers import attention
from ..utils.context import set_context
from .cudagraph import DecodeGraphRunner


class ModelRunner:
    """Turns a list of :class:`Sequence` into logits, using the paged KV cache."""

    def __init__(
        self,
        model,
        block_manager,
        dtype=torch.float32,
        attn_backend="auto",
        device="cpu",
        max_num_seqs=32,
        use_cudagraph=False,
        compile_model=False,
    ):
        self.device = torch.device(device)
        self.model = model.eval()
        self.block_manager = block_manager
        self.block_size = block_manager.block_size
        self.dtype = dtype
        self.decoder_layers = list(model.decoder_layers)
        # torch.compile the decoder (nano-vllm style); the paged attention ops
        # graph-break, so this fuses the model's pointwise work around them.
        self.decoder = torch.compile(model.decoder) if compile_model else model.decoder

        attention.set_backend(attn_backend)

        # One paged cache per decoder self-attention layer; the layout is the one
        # the kernel the ops will pick expects.
        self.cache_layers = [layer.self_attn for layer in self.decoder_layers]
        self.num_heads = self.cache_layers[0].num_heads
        self.head_dim = self.cache_layers[0].head_dim
        # CUDA graphs capture the decoder step; only meaningful on the flash path.
        self.use_cudagraph = use_cudagraph and (
            self.device.type == "cuda"
            and attention.backend(self.device, dtype, self.head_dim, self.block_size)
            == "flash"
        )
        # The graph path reserves one extra cache row as a scratch block for the
        # padded rows of an undersized batch.
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

    # --------------------------------------------------------------- encode
    @torch.no_grad()
    def _encode(self, seqs) -> None:
        """Run the encoder once per distinct shape and cache cross-attn K/V."""
        groups: dict[tuple, list] = {}
        for seq in seqs:
            pixel_shape = (
                None if seq.pixel_values is None else tuple(seq.pixel_values.shape)
            )
            groups.setdefault((len(seq.encoder_token_ids), pixel_shape), []).append(seq)
        for group in groups.values():
            ids = torch.tensor(
                [s.encoder_token_ids for s in group],
                dtype=torch.long,
                device=self.device,
            )
            if group[0].pixel_values is None:
                hidden = self.model.encode(ids)
            else:
                pixel_values = torch.stack([s.pixel_values for s in group]).to(
                    self.device
                )
                hidden = self.model.encode(ids, pixel_values=pixel_values)
            for layer in self.decoder_layers:
                attn = layer.encoder_attn
                head = (len(group), ids.shape[1], attn.num_heads, attn.head_dim)
                key = attn.k_proj(hidden).view(head)
                value = attn.v_proj(hidden).view(head)
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

    # -------------------------------------------------------------- prepare
    def _prepare(self, seqs):
        bs = self.block_size
        # Reserve the whole batch's blocks before mutating anything so a full
        # cache raises cleanly instead of leaving sequences half-prepared.
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

        max_blocks = max(len(table) for table in seq_block_tables)
        # Pad the block tables into a rectangle so the whole batch is one copy.
        padded = [table + [0] * (max_blocks - len(table)) for table in seq_block_tables]
        # nano-vllm style: build the host tensor pinned, then copy with
        # non_blocking (a pageable copy would synchronise the stream).  The
        # pinned allocator is CUDA-only, so CPU builds the tensor directly.
        pin = self.device.type == "cuda"
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=pin).to(
            self.device, non_blocking=True
        )
        block_tables = torch.tensor(padded, dtype=torch.int32, pin_memory=pin).to(
            self.device, non_blocking=True
        )
        slot_mapping = torch.tensor(new_slots, dtype=torch.int64, pin_memory=pin).to(
            self.device, non_blocking=True
        )
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=pin).to(
            self.device, non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.long, pin_memory=pin).to(
            self.device, non_blocking=True
        )
        fields = {
            "is_prefill": is_prefill,
            "context_lens": context_lens,
            "block_tables": block_tables,
            "slot_mapping": slot_mapping,
            "request_ids": [seq.request_id for seq in seqs],
        }
        return input_ids, positions, fields

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
        # One new token per sequence (the decoder prompt is a single token), so
        # the logits are just the rows in sequence order.
        return self.model.compute_logits(hidden)
