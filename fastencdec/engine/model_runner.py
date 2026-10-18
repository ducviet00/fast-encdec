from collections import defaultdict
from contextlib import nullcontext

import torch

from ..layers import attention
from ..utils.context import set_context
from .cudagraph import DecodeGraphRunner

# Number of compiled runners in this process; each one costs Dynamo cache entries
# (a new weight set and the batch-1 specialization), and the limit is per frame.
_COMPILED_RUNNERS = 0


class ModelRunner:
    """Manages KV caches, encoder caching, step preparation, and model execution."""

    def __init__(
        self,
        model,
        block_manager,
        dtype=torch.float32,
        attn_backend="auto",
        device="cpu",
        max_num_seqs=32,
        use_cudagraph=True,
        compile=True,
        compile_mm_encoder=True,
    ):
        self.device = torch.device(device)
        self.model = model.eval()
        self.block_manager = block_manager
        self.block_size = block_manager.block_size
        self.dtype = dtype
        self.decoder_layers = list(model.decoder_layers)

        attention.set_backend(attn_backend)

        self.cache_layers = [layer.self_attn for layer in self.decoder_layers]
        self.num_heads = self.cache_layers[0].num_heads
        self.head_dim = self.cache_layers[0].head_dim
        self.attn_backend = attention.backend(
            self.device, dtype, self.head_dim, self.block_size
        )
        self.decoder = model.decoder
        self._setup_compile(compile, compile_mm_encoder)
        self.use_cudagraph = use_cudagraph and (
            self.device.type == "cuda" and self.attn_backend == "flash"
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

    def _setup_compile(self, compile: bool, compile_mm_encoder: bool) -> None:
        """Build the frozen CPU graphs (and the vision tower) when enabled.

        The paged attention ops are opaque custom ops, so Inductor traces the
        decoder + lm head as one graph and freezing packs every GEMM. The SDPA
        fallback reads the per-step Context and must not be compiled, hence the
        ``vllm`` gate. See ``TORCH_COMPILE_FREEZING.md``.
        """
        self.compile_decoder = (
            compile
            and self.device.type == "cpu"
            and self.dtype in (torch.float32, torch.bfloat16)
            and self.attn_backend == "vllm"
        )
        self.decode_step = None
        self.encode_fn = self.model.encode
        self.mm_step = None
        if self.compile_decoder:
            global _COMPILED_RUNNERS
            _COMPILED_RUNNERS += 1
            # Every engine adds Dynamo cache entries (its weights and the batch-1
            # specialization); grow the per-frame budget with the engine count.
            torch._dynamo.config.recompile_limit = max(
                torch._dynamo.config.recompile_limit, 2 * _COMPILED_RUNNERS + 4
            )
            # Four workers compile fast without the GB-scale memory spike of the
            # default thread-count pool on small boxes.
            torch._inductor.config.compile_threads = min(
                4, torch._inductor.config.compile_threads
            )
            # Symbolic shapes keep one artifact per frame: freezing duplicates the
            # packed weights per static shape. The Florence-2 tower is separate.
            self.decode_step = torch.compile(self.model, dynamic=True)
            self.encode_fn = torch.compile(self.model.encode, dynamic=True)
        if hasattr(self.model, "get_image_features") and compile_mm_encoder:
            # Dynamic convs mis-stride DaViT's windowed maps, so the tower keeps
            # static shapes (one graph per image shape, fixed in practice).
            self.mm_step = torch.compile(self.model.get_image_features, dynamic=False)

    def _copy_block(self, src: int, dst: int) -> None:
        """Copy KV cache contents from block src to dst across all decoder layers."""
        for layer in self.cache_layers:
            layer.k_cache[dst].copy_(layer.k_cache[src])
            layer.v_cache[dst].copy_(layer.v_cache[src])

    @torch.no_grad()
    def _encode(self, seqs) -> None:
        """Batch fresh sequences by (encoder_len, image_shape) and cache cross-attn K/V."""
        groups = defaultdict(list)
        for seq in seqs:
            px_shape = (
                None if seq.pixel_values is None else tuple(seq.pixel_values.shape)
            )
            groups[(len(seq.encoder_token_ids), px_shape)].append(seq)
        encode = self.encode_fn
        for group in groups.values():
            ids = torch.tensor(
                [s.encoder_token_ids for s in group],
                dtype=torch.long,
                device=self.device,
            )
            if group[0].pixel_values is None:
                with self._inductor_ctx():
                    hidden = encode(ids)
            else:
                px = torch.stack([s.pixel_values for s in group]).to(self.device)
                with self._inductor_ctx():
                    if self.mm_step is None:
                        embeds = self.model.get_image_features(px)
                    else:
                        embeds = self.mm_step(px)
                    hidden = encode(ids, image_embeds=embeds)
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
        """Release cached encoder K/V and staged cross-attention state for a finished request."""
        if self.graph is not None:
            self.graph.req_blocks.pop(request_id, None)
        for layer in self.decoder_layers:
            attn = layer.encoder_attn
            attn.encoder_kv_cache.pop(request_id, None)
            attn.cross_staged = None
            attn.cross_staged_key = None

    def _prepare(self, seqs):
        """Allocate KV blocks and build input_ids, positions, and Context tensors for the step."""
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
        for seq in seqs:
            num_new = len(seq.token_ids) - seq.num_cached_tokens
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
            "context_lens": to_dev(context_lens, torch.int32),
            "block_tables": to_dev(padded, torch.int32),
            "slot_mapping": to_dev(new_slots, torch.int64),
            "request_ids": [seq.request_id for seq in seqs],
        }
        return to_dev(input_ids, torch.long), to_dev(positions, torch.long), fields

    @torch.no_grad()
    def run(self, seqs):
        """Encode any fresh sequences, run one decoder step, and return next-token logits."""
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

        if self.attn_backend == "vllm":
            # Stage the paged encoder K/V outside the compiled graph; the opaque
            # cross-attention op reads it from the Context at call time.
            fields["cross_staged"] = attention.stage_cross(
                self.decoder_layers, fields["request_ids"]
            )
        set_context(**fields)
        if self.decode_step is not None and attention.preferred() not in (
            "auto",
            "vllm",
        ):
            # The backend was overridden after this runner compiled (a benchmark
            # comparing backends, or a second engine setting a new preference).
            # The SDPA path reads the per-step Context, which must stay out of
            # the graph, so run this step eagerly instead of re-tracing.
            return self.model(input_ids, positions)
        with self._inductor_ctx():
            # model.forward is decoder + lm head, so freezing packs both.
            if self.decode_step is None:
                return self.model(input_ids, positions)
            return self.decode_step(input_ids, positions)

    def _inductor_ctx(self):
        """Patch Inductor freezing on for compiled calls (packing at compile)."""
        if self.compile_decoder:
            return torch._inductor.config.patch(freezing=True)
        return nullcontext()
