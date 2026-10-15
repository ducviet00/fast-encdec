import torch

from ..layers.attention import store_kvcache
from ..utils.context import set_context


class DecodeGraphRunner:
    """Captures and replays bucketed CUDA graphs for the FlashAttention-2 decode step."""

    def __init__(self, runner, max_num_seqs: int):
        self.runner = runner
        self.device = runner.device
        self.dtype = runner.dtype
        self.block_size = runner.block_size
        self.max_bs = max(1, min(max_num_seqs, 512))
        # Small powers of 2, multiples of 16, plus max_bs so every batch size fits a bucket.
        buckets = [1, 2, 4, 8, *range(16, self.max_bs + 1, 16), self.max_bs]
        self.graph_bs = sorted({b for b in buckets if b <= self.max_bs})
        # Reserved self-attn cache row that padded rows write to.
        self.scratch_block = runner.block_manager.num_blocks
        self.scratch_slot = self.scratch_block * self.block_size

        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.pool = None
        self.bufs: dict | None = None

        # request_id -> (first_block, pages, enc_len)
        self.req_blocks: dict[int, tuple[int, int, int]] = {}
        self.next_block = 0
        self.cross_capacity = 0
        self.cross_scratch = 0
        self.cross_bt = None
        self.cross_lens = None
        self._staged_ids = None

    def stage(self, seqs) -> None:
        """Stage per-request encoder K/V and block tables into static buffers when active requests change."""
        bs = self.block_size
        if not self.req_blocks:
            self.next_block = 0
        attn0 = self.runner.decoder_layers[0].encoder_attn
        added = False
        for seq in seqs:
            if seq.request_id in self.req_blocks:
                continue
            key = attn0.encoder_kv_cache[seq.request_id][0]
            pages = max(1, (key.shape[1] + bs - 1) // bs)
            self.req_blocks[seq.request_id] = (self.next_block, pages, key.shape[1])
            self.next_block += pages
            added = True
        if self.next_block + 1 > self.cross_capacity:
            self._alloc_cross(max(2 * self.cross_capacity, self.next_block + 1))
            added = True
        if added:
            self._write_cross()
            # Only rebuild static buffers/graphs when cross-cache capacity grows.
            if self.bufs is None:
                self._alloc_bufs()
        ids = tuple(seq.request_id for seq in seqs)
        if added or ids != self._staged_ids:
            self._rows(seqs)
            self._staged_ids = ids

    def _alloc_cross(self, capacity: int) -> None:
        """Allocate fixed page-major cross-attention KV caches and invalidate captured graphs."""
        self.cross_capacity = capacity
        self.cross_scratch = capacity - 1
        for layer in self.runner.decoder_layers:
            attn = layer.encoder_attn
            shape = (capacity, self.block_size, attn.num_heads, attn.head_dim)
            attn.cross_cache = (
                torch.zeros(shape, dtype=self.dtype, device=self.device),
                torch.zeros(shape, dtype=self.dtype, device=self.device),
            )
        self.bufs = None
        self.graphs.clear()
        self.pool = None

    def _write_cross(self) -> None:
        """Scatter each active request's dense encoder K/V into the static paged cross_cache."""
        bs = self.block_size
        for layer in self.runner.decoder_layers:
            attn = layer.encoder_attn
            k_cache, v_cache = attn.cross_cache
            for request_id, (first, _, length) in self.req_blocks.items():
                pair = attn.encoder_kv_cache.get(request_id)
                if pair is None:
                    continue
                key, value = pair
                pos = torch.arange(length, device=self.device)
                slots = (first + pos // bs) * bs + (pos % bs)
                store_kvcache(
                    key.transpose(0, 1).contiguous(),
                    value.transpose(0, 1).contiguous(),
                    k_cache,
                    v_cache,
                    slots,
                    bs,
                    page_major=True,
                )

    def _rows(self, seqs) -> None:
        """Build per-sequence cross-attention block tables and sequence lengths for seqs."""
        width = max((p for _, p, _ in self.req_blocks.values()), default=1)
        table = torch.zeros(len(seqs), width, dtype=torch.int32, device=self.device)
        for i, seq in enumerate(seqs):
            first, pages, _ = self.req_blocks[seq.request_id]
            table[i, :pages] = torch.arange(
                first, first + pages, dtype=torch.int32, device=self.device
            )
        self.cross_bt = table
        self.cross_lens = torch.tensor(
            [self.req_blocks[s.request_id][2] for s in seqs],
            dtype=torch.int32,
            device=self.device,
        )

    def _alloc_bufs(self) -> None:
        """Preallocate static input, metadata, and output tensors for graph capture and replay."""
        dev = self.device
        num_blocks = self.runner.block_manager.num_blocks
        width = self.cross_scratch + 1
        self.bufs = {
            "input_ids": torch.zeros(self.max_bs, dtype=torch.long, device=dev),
            "positions": torch.zeros(self.max_bs, dtype=torch.long, device=dev),
            "slot_mapping": torch.zeros(self.max_bs, dtype=torch.int64, device=dev),
            "context_lens": torch.zeros(self.max_bs, dtype=torch.int32, device=dev),
            "block_tables": torch.zeros(
                self.max_bs, num_blocks + 1, dtype=torch.int32, device=dev
            ),
            "cross_lens": torch.zeros(self.max_bs, dtype=torch.int32, device=dev),
            "cross_bt": torch.zeros(self.max_bs, width, dtype=torch.int32, device=dev),
            "outputs": torch.zeros(
                self.max_bs,
                self.runner.model.decoder.embed_tokens.embedding_dim,
                dtype=self.dtype,
                device=dev,
            ),
            "cross_w": width,
        }
        self.graphs.clear()
        self.pool = None

    def ready(self) -> bool:
        """Return True when static graph buffers are allocated."""
        return self.bufs is not None

    def _fill_sentinels(self) -> None:
        """Fill static metadata buffers with scratch sentinels so padded rows are no-ops."""
        b = self.bufs
        b["slot_mapping"].fill_(self.scratch_slot)
        b["context_lens"].fill_(1)
        b["block_tables"].fill_(self.scratch_block)
        b["cross_lens"].fill_(1)
        b["cross_bt"].fill_(self.cross_scratch)

    def _set_context(self, bs: int) -> None:
        """Bind the global Context to the first bs rows of the static graph buffers."""
        b = self.bufs
        set_context(
            is_prefill=False,
            context_lens=b["context_lens"][:bs],
            block_tables=b["block_tables"][:bs],
            slot_mapping=b["slot_mapping"][:bs],
            request_ids=None,
            cross_cache_seqlens=b["cross_lens"][:bs],
            cross_block_tables=b["cross_bt"][:bs],
        )

    def capture(self) -> None:
        """Capture one CUDA graph per batch-size bucket into a shared memory pool."""
        if self.graphs:
            return
        b = self.bufs
        self._fill_sentinels()
        for bs in reversed(self.graph_bs):
            self._set_context(bs)
            b["outputs"][:bs] = self.runner.decoder(
                b["input_ids"][:bs], b["positions"][:bs]
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, self.pool):
                b["outputs"][:bs] = self.runner.decoder(
                    b["input_ids"][:bs], b["positions"][:bs]
                )
            if self.pool is None:
                self.pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
        self._fill_sentinels()

    def replay(self, bs: int, input_ids, positions, fields):
        """Copy step inputs into the static buffers, replay the bucket graph, and return hidden states."""
        b = self.bufs
        bucket = next(x for x in self.graph_bs if x >= bs)
        self._fill_sentinels()
        b["input_ids"][:bs].copy_(input_ids)
        b["positions"][:bs].copy_(positions)
        b["slot_mapping"][:bs].copy_(fields["slot_mapping"])
        b["context_lens"][:bs].copy_(fields["context_lens"])
        b["block_tables"][:bs, : fields["block_tables"].shape[1]].copy_(
            fields["block_tables"]
        )
        b["cross_lens"][:bs].copy_(self.cross_lens)
        b["cross_bt"][:bs, : self.cross_bt.shape[1]].copy_(self.cross_bt)
        self.graphs[bucket].replay()
        return b["outputs"][:bs]
