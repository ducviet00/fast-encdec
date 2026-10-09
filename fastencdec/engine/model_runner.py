"""Batching + forward pass for the paged engine."""

import torch

from ..layers import cpu_attn
from ..utils.context import set_context


class ModelRunner:
    """Turns a list of :class:`Sequence` into logits, using the paged KV cache."""

    def __init__(self, model, block_manager, dtype=torch.float32, attn_backend="auto"):
        self.model = model.eval()
        self.block_manager = block_manager
        self.block_size = block_manager.block_size
        self.dtype = dtype
        self.decoder_layers = list(model.decoder_layers)

        # One paged cache per decoder self-attention layer.  Layout is vLLM's
        # [num_blocks, num_heads, block_size, head_dim] so the vendored kernel
        # reads it directly.
        self.cache_layers = [layer.self_attn for layer in self.decoder_layers]
        self.num_heads = self.cache_layers[0].num_heads
        self.head_dim = self.cache_layers[0].head_dim

        self.attn_isa = None
        if attn_backend != "sdpa":
            isa = cpu_attn.select_isa(self.dtype, self.block_size, self.head_dim)
            if isa is not None:
                built = (
                    cpu_attn.require() if attn_backend == "vllm" else cpu_attn.module()
                )
                if built is not None:
                    self.attn_isa = isa
            elif attn_backend == "vllm":
                raise RuntimeError(
                    f"vLLM CPU attention does not support head_dim={self.head_dim} "
                    f"block_size={self.block_size} dtype={self.dtype}"
                )

        shape = (
            block_manager.num_blocks,
            self.num_heads,
            self.block_size,
            self.head_dim,
        )
        for layer in self.cache_layers:
            layer.k_cache = torch.zeros(shape, dtype=dtype)
            layer.v_cache = torch.zeros(shape, dtype=dtype)

        # Encoder (cross-attention) paged cache, built once per offline batch.
        self._enc_block_table = None
        self._enc_seq_lens = None
        self._enc_row = {}

    def _copy_block(self, src: int, dst: int) -> None:
        for layer in self.cache_layers:
            layer.k_cache[dst].copy_(layer.k_cache[src])
            layer.v_cache[dst].copy_(layer.v_cache[src])

    # --------------------------------------------------------------- encode
    def _build_encoder_paged(self, seqs) -> None:
        """Stage every active request's encoder K/V into the kernel's paged cache.

        Beams share a request's encoder K/V, so there is one row per request.
        Called whenever new requests are admitted (the active set changed).
        """
        bs = self.block_size
        seen, uniq = set(), []
        for seq in seqs:
            if seq.request_id not in seen:
                seen.add(seq.request_id)
                uniq.append(seq)
        enc_lens = [len(seq.encoder_token_ids) for seq in uniq]
        pages = [(length + bs - 1) // bs for length in enc_lens]
        max_pages = max(pages)
        total = sum(pages)
        self._enc_seq_lens = torch.tensor(enc_lens, dtype=torch.int32)
        self._enc_block_table = torch.zeros(len(uniq), max_pages, dtype=torch.int32)
        self._enc_row = {}
        pid = 0
        for i, (seq, page_num) in enumerate(zip(uniq, pages)):
            self._enc_block_table[i, :page_num] = torch.arange(
                pid, pid + page_num, dtype=torch.int32
            )
            self._enc_row[seq.request_id] = i
            pid += page_num
        mod = cpu_attn.module()
        for layer in self.decoder_layers:
            attn = layer.encoder_attn
            shape = (total, attn.num_heads, bs, attn.head_dim)
            attn.encoder_paged_cache = (
                torch.zeros(shape, dtype=self.dtype),
                torch.zeros(shape, dtype=self.dtype),
            )
            k_cache, v_cache = attn.encoder_paged_cache
            for i, seq in enumerate(uniq):
                key, value = attn.encoder_kv_cache[seq.request_id]  # [H, L, D]
                length = key.shape[1]
                slots = self._encoder_slots(i, length)
                mod.reshape_and_cache(
                    key.transpose(0, 1).contiguous(),
                    value.transpose(0, 1).contiguous(),
                    k_cache,
                    v_cache,
                    slots,
                    self.attn_isa,
                )

    def _encoder_slots(self, row: int, length: int) -> torch.Tensor:
        bs = self.block_size
        pos = torch.arange(length)
        blocks = self._enc_block_table[row, pos // bs].long()
        return blocks * bs + (pos % bs)

    def _encode(self, seqs) -> None:
        """Run the encoder once per distinct shape and cache cross-attn K/V."""
        groups: dict[tuple, list] = {}
        for seq in seqs:
            pixel_shape = (
                None if seq.pixel_values is None else tuple(seq.pixel_values.shape)
            )
            groups.setdefault((len(seq.encoder_token_ids), pixel_shape), []).append(seq)
        for group in groups.values():
            ids = torch.tensor([s.encoder_token_ids for s in group], dtype=torch.long)
            if group[0].pixel_values is None:
                hidden = self.model.encode(ids)
            else:
                pixel_values = torch.stack([s.pixel_values for s in group])
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
            layer.encoder_attn.encoder_kv_cache.pop(request_id, None)

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

        input_ids, positions = [], []
        new_slots = []
        query_start_loc = [0]
        context_lens = []
        seq_block_tables = []
        dynamic_causal = []

        for seq in seqs:
            num_new = len(seq.token_ids) - seq.num_cached_tokens
            fresh = seq.num_cached_tokens == 0
            self.block_manager.ensure_capacity(seq, num_new, self._copy_block)
            start = seq.num_cached_tokens
            for pos in range(start, start + num_new):
                input_ids.append(seq.token_ids[pos])
                positions.append(pos)
                new_slots.append(seq.block_table[pos // bs] * bs + pos % bs)
            seq.num_cached_tokens += num_new

            query_start_loc.append(len(input_ids))
            context_lens.append(seq.num_cached_tokens)
            seq_block_tables.append(seq.block_table)
            dynamic_causal.append(1 if fresh else 0)

        max_blocks = max(len(table) for table in seq_block_tables)
        block_table = torch.zeros((len(seqs), max_blocks), dtype=torch.int32)
        for i, table in enumerate(seq_block_tables):
            block_table[i, : len(table)] = torch.tensor(table, dtype=torch.int32)

        key_slot_ids = self._key_slots(block_table, context_lens)

        fields = {
            "slot_mapping": torch.tensor(new_slots, dtype=torch.int64),
            "query_start_loc": query_start_loc,
            "context_lens": context_lens,
            "request_ids": [seq.request_id for seq in seqs],
            "num_seqs": len(seqs),
            "attn_isa": self.attn_isa,
            "block_table": block_table,
            "key_slot_ids": key_slot_ids,
        }
        if self.attn_isa:
            fields.update(
                self._kernel_fields(
                    query_start_loc,
                    context_lens,
                    dynamic_causal,
                    fields["request_ids"],
                )
            )

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(positions, dtype=torch.long),
            fields,
        )

    def _key_slots(self, block_table, context_lens):
        """(block_ids, pos_in_block) for every cached key, ordered by sequence."""
        lens = torch.tensor(context_lens, dtype=torch.long)
        max_len = int(lens.max())
        pos = torch.arange(max_len)
        blocks = block_table.long()[:, pos // self.block_size]
        pos_in_block = pos % self.block_size
        mask = pos.unsqueeze(0) < lens.unsqueeze(1)
        return blocks[mask], pos_in_block.expand(len(context_lens), -1)[mask]

    def _kernel_fields(
        self, query_start_loc, context_lens, dynamic_causal, request_ids
    ):
        mod = cpu_attn.module()
        cpu_qsl = torch.tensor(query_start_loc, dtype=torch.int32)
        cpu_seq_lens = torch.tensor(context_lens, dtype=torch.int32)
        dyn = torch.tensor(dynamic_causal, dtype=torch.int32)
        metadata = mod.build_metadata(
            len(context_lens),
            self.num_heads,
            self.num_heads,
            self.head_dim,
            cpu_seq_lens,
            self.dtype,
            cpu_qsl,
            False,
            -1,
            self.attn_isa,
            True,
            dyn,
        )
        fields = {
            "cpu_query_start_loc": cpu_qsl,
            "cpu_seq_lens": cpu_seq_lens,
            "dynamic_causal": dyn,
            "attn_metadata": metadata,
        }
        cross = self._cross_fields(query_start_loc, request_ids)
        if cross is not None:
            fields.update(cross)
        return fields

    def _cross_fields(self, query_start_loc, request_ids):
        if self._enc_block_table is None:
            return None
        rows = torch.tensor(
            [self._enc_row[request_id] for request_id in request_ids],
            dtype=torch.long,
        )
        cross_block_table = self._enc_block_table[rows]
        cross_seq_lens = self._enc_seq_lens[rows].contiguous()
        cpu_qsl = torch.tensor(query_start_loc, dtype=torch.int32)
        metadata = cpu_attn.module().build_metadata(
            rows.numel(),
            self.num_heads,
            self.num_heads,
            self.head_dim,
            cross_seq_lens,
            self.dtype,
            cpu_qsl,
            False,
            -1,
            self.attn_isa,
            True,
            None,
        )
        return {
            "cross_block_table": cross_block_table,
            "cross_seq_lens": cross_seq_lens,
            "cross_metadata": metadata,
        }

    @torch.no_grad()
    def run(self, seqs):
        fresh = [seq for seq in seqs if seq.num_cached_tokens == 0]
        if fresh:
            self._encode(fresh)
            if self.attn_isa:
                # New requests were admitted: restage the encoder paged cache
                # for every active request (fresh + still-running).
                self._build_encoder_paged(seqs)

        input_ids, positions, fields = self._prepare(seqs)
        set_context(**fields)

        hidden = self.model.decoder(input_ids, positions)
        last = torch.tensor(
            [fields["query_start_loc"][i + 1] - 1 for i in range(len(seqs))],
            dtype=torch.long,
        )
        return self.model.compute_logits(hidden.index_select(0, last))
