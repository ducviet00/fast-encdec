"""Batching + forward pass for the paged engine."""

import torch

from .context import Context, set_context


class ModelRunner:
    """Turns a list of :class:`Sequence` into logits, using the paged KV cache."""

    def __init__(self, model, block_manager, dtype=torch.float32):
        self.model = model.eval()
        self.block_manager = block_manager

        # One paged cache per decoder self-attention layer:
        # [num_blocks, block_size, num_heads, head_dim].
        self.cache_layers = [layer.self_attn for layer in model.model.decoder.layers]
        for layer in self.cache_layers:
            shape = (block_manager.num_blocks, block_manager.block_size,
                     layer.num_heads, layer.head_dim)
            layer.k_cache = torch.zeros(shape, dtype=dtype)
            layer.v_cache = torch.zeros(shape, dtype=dtype)

    def _copy_block(self, src: int, dst: int) -> None:
        for layer in self.cache_layers:
            layer.k_cache[dst].copy_(layer.k_cache[src])
            layer.v_cache[dst].copy_(layer.v_cache[src])

    def _encode(self, seqs) -> None:
        """Run the encoder once per distinct encoder length and cache cross-attn K/V.

        Grouping by length lets every sequence in a group share one batched
        forward (bigger, more efficient GEMMs) without any padding.
        """
        groups: dict[int, list] = {}
        for seq in seqs:
            groups.setdefault(len(seq.encoder_token_ids), []).append(seq)
        for group in groups.values():
            ids = torch.tensor([s.encoder_token_ids for s in group], dtype=torch.long)
            hidden = self.model.encode(ids)  # [G, enc_len, dim]
            for layer in self.model.model.decoder.layers:
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
        for layer in self.model.model.decoder.layers:
            layer.encoder_attn.encoder_kv_cache.pop(request_id, None)

    def _prepare(self, seqs):
        bs = self.block_manager.block_size
        input_ids, positions, slot_mapping = [], [], []
        query_start_loc = [0]
        context_lens, block_tables = [], []

        for seq in seqs:
            num_new = len(seq.token_ids) - seq.num_cached_tokens
            self.block_manager.ensure_capacity(seq, num_new, self._copy_block)
            start = seq.num_cached_tokens
            for pos in range(start, start + num_new):
                input_ids.append(seq.token_ids[pos])
                positions.append(pos)
                slot_mapping.append(seq.block_table[pos // bs] * bs + pos % bs)
            seq.num_cached_tokens += num_new

            query_start_loc.append(len(input_ids))
            context_lens.append(seq.num_cached_tokens)
            block_tables.append(seq.block_table)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(positions, dtype=torch.long),
            slot_mapping,
            query_start_loc,
            context_lens,
            block_tables,
        )

    @torch.no_grad()
    def run(self, seqs):
        fresh = [seq for seq in seqs if seq.num_cached_tokens == 0]
        if fresh:
            self._encode(fresh)

        input_ids, positions, slot_mapping, qsl, context_lens, block_tables = (
            self._prepare(seqs))

        context = Context(
            slot_mapping=slot_mapping,
            query_start_loc=qsl,
            context_lens=context_lens,
            block_tables=block_tables,
            request_ids=[seq.request_id for seq in seqs],
            num_seqs=len(seqs),
        )
        set_context(context)

        hidden = self.model.model.decoder(input_ids, positions)
        last = torch.tensor([qsl[i + 1] - 1 for i in range(len(seqs))],
                            dtype=torch.long)
        return self.model.compute_logits(hidden.index_select(0, last))
