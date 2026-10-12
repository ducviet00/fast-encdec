"""Batching + forward pass for the paged engine.

The runner only builds the per-step :class:`~fastencdec.utils.context.Context`
and the KV caches; every attention op picks its own kernel (see
:mod:`fastencdec.layers.attention`).
"""

import torch

from ..layers import attention
from ..utils.context import set_context


class ModelRunner:
    """Turns a list of :class:`Sequence` into logits, using the paged KV cache."""

    def __init__(
        self,
        model,
        block_manager,
        dtype=torch.float32,
        attn_backend="auto",
        device="cpu",
    ):
        self.device = torch.device(device)
        self.model = model.eval()
        self.block_manager = block_manager
        self.block_size = block_manager.block_size
        self.dtype = dtype
        self.decoder_layers = list(model.decoder_layers)

        attention.set_backend(attn_backend)

        # One paged cache per decoder self-attention layer; the layout is the one
        # the kernel the ops will pick expects.
        self.cache_layers = [layer.self_attn for layer in self.decoder_layers]
        self.num_heads = self.cache_layers[0].num_heads
        self.head_dim = self.cache_layers[0].head_dim
        shape = attention.self_cache_shape(
            block_manager.num_blocks,
            self.block_size,
            self.num_heads,
            self.head_dim,
            self.device,
            dtype,
        )
        for layer in self.cache_layers:
            layer.k_cache = torch.zeros(shape, dtype=dtype, device=self.device)
            layer.v_cache = torch.zeros(shape, dtype=dtype, device=self.device)
            layer.block_size = self.block_size
        for layer in self.decoder_layers:
            layer.encoder_attn.block_size = self.block_size

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
        block_tables = torch.zeros(
            (len(seqs), max_blocks), dtype=torch.int32, device=self.device
        )
        for i, table in enumerate(seq_block_tables):
            block_tables[i, : len(table)] = torch.tensor(
                table, dtype=torch.int32, device=self.device
            )

        fields = {
            "is_prefill": is_prefill,
            "context_lens": torch.tensor(
                context_lens, dtype=torch.int32, device=self.device
            ),
            "block_tables": block_tables,
            "slot_mapping": torch.tensor(
                new_slots, dtype=torch.int64, device=self.device
            ),
            "request_ids": [seq.request_id for seq in seqs],
        }
        return (
            torch.tensor(input_ids, dtype=torch.long, device=self.device),
            torch.tensor(positions, dtype=torch.long, device=self.device),
            fields,
        )

    @torch.no_grad()
    def run(self, seqs):
        fresh = [seq for seq in seqs if seq.num_cached_tokens == 0]
        if fresh:
            self._encode(fresh)

        input_ids, positions, fields = self._prepare(seqs)
        set_context(**fields)

        hidden = self.model.decoder(input_ids, positions)
        # One new token per sequence (the decoder prompt is a single token), so
        # the logits are just the rows in sequence order.
        return self.model.compute_logits(hidden)
