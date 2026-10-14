"""Continuous-batching engine with greedy/sampling and beam search.

The engine is offline: :meth:`LLMEngine.generate` adds a whole batch with a
single :class:`SamplingParams` and runs to completion.  Beam search is built on
the same paged cache: each beam is a normal :class:`Sequence`, blocks are shared
by reference count and copy-on-write happens on the next append (see
:mod:`fastencdec.engine.block_manager`).
"""

from collections import Counter, defaultdict
from dataclasses import replace

import torch

from ..config import Config
from ..layers.logits_processor import _ngram_processor, build_logits_processor
from ..layers.ngram import triton_available
from ..layers.sampler import Sampler
from .block_manager import BlockManager
from .model_runner import ModelRunner
from .scheduler import Scheduler
from .sequence import Sequence


def _length_penalized(seq: Sequence, length_penalty: float) -> float:
    length = max(seq.num_generated, 1)
    return seq.cum_logprob / (length**length_penalty)


def select_beam_candidates(
    stopped: list[bool], num_beams: int
) -> tuple[list[int], list[int]]:
    """Split a descending-ranked candidate list into finished and live ranks.

    ``stopped`` marks candidates that hit a stopping criterion (EOS or max
    length); the lists are parallel to that ranking.  A stopped candidate may
    only finish if it ranks within the top ``num_beams`` — otherwise it is
    dropped, as in ``generate`` — and the remaining slots go to the best
    non-stopped continuations.
    """
    finished = [r for r, stop in enumerate(stopped) if stop and r < num_beams]
    live = [r for r, stop in enumerate(stopped) if not stop][:num_beams]
    return finished, live


class LLMEngine:
    def __init__(
        self,
        model,
        config: Config,
        eos_token_id: int | None = None,
    ):
        self.block_manager = BlockManager(config.num_blocks, config.block_size)
        self.runner = ModelRunner(
            model,
            self.block_manager,
            config.dtype,
            attn_backend=config.attn_backend,
            device=config.device,
            max_num_seqs=config.max_num_seqs,
            use_cudagraph=config.use_cudagraph,
            compile_model=config.compile_model,
        )
        self.scheduler = Scheduler(config)
        self.sampler = Sampler()
        self.eos_token_id = eos_token_id
        self.device = torch.device(config.device)
        self.results: dict[int, list[int]] = {}
        self._finished_beams: dict[int, list[Sequence]] = defaultdict(list)
        self._logits_processor = None
        self._ngram = None
        self._next_request_id = 0

    def generate(self, requests, params) -> list[int]:
        """Offline entrypoint: add every request, run, return the request ids.

        ``requests`` is an iterable of ``(encoder_token_ids, decoder_token_ids,
        pixel_values)``; ``params`` is a single :class:`SamplingParams` shared by
        the whole batch.  The decoder prompt length must be uniform (the facades
        always use a single decoder-start token).
        """
        requests = list(requests)
        if not requests:
            return []
        if params.eos_token_id is None and self.eos_token_id is not None:
            params = replace(params, eos_token_id=self.eos_token_id)

        prompt_len = len(requests[0][1])
        if any(len(decoder_ids) != prompt_len for _, decoder_ids, _ in requests):
            raise ValueError("decoder prompt length must be uniform across the batch")
        if prompt_len != 1:
            # The attention ops feed one token per sequence per step.
            raise ValueError("the decoder prompt must be a single token")
        begin_index = prompt_len + int(
            params.forced_bos_token_id is not None and prompt_len == 1
        )
        self._logits_processor = build_logits_processor(
            params,
            max_length=prompt_len + params.max_tokens,
            begin_index=begin_index,
            is_beam=params.num_beams > 1,
            device=self.device,
        )

        request_ids = []
        for encoder_token_ids, decoder_token_ids, pixel_values in requests:
            request_id = self._next_request_id
            self._next_request_id += 1
            self.scheduler.add(
                Sequence(
                    request_id=request_id,
                    decoder_prompt_ids=decoder_token_ids,
                    encoder_token_ids=encoder_token_ids,
                    sampling=params,
                    pixel_values=pixel_values,
                )
            )
            request_ids.append(request_id)
        self.run()
        return request_ids

    # ------------------------------------------------------------------ run
    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def step(self) -> None:
        seqs = self.scheduler.schedule()
        logits = self.runner.run(seqs)
        self._postprocess(seqs, logits)

    def run(self) -> None:
        mode = self._pipeline_mode()
        if mode == "greedy":
            self._run_pipelined()
            return
        if mode == "beam":
            self._run_pipelined_beam()
            return
        while not self.is_finished():
            self.step()

    # ---------------------------------------------------- pipelined decode
    def _pipeline_mode(self):
        """``"greedy"``, ``"beam"`` or ``None`` for the synchronous loop.

        Only CUDA batches whose processors run on the device without a host
        round-trip qualify.  Greedy handles early EOS by truncation, so it only
        needs greedy + device processors; the pipelined beam keeps every beam
        live for the whole run, so it additionally needs EOS to be impossible
        (``min_length > max_tokens``).
        """
        if self.device.type != "cuda":
            return None
        seqs = list(self.scheduler.waiting) + list(self.scheduler.running)
        if not seqs:
            return None
        params = seqs[0].sampling
        if any(seq.sampling is not params for seq in seqs):
            return None
        if params.temperature > 0:
            return None
        if (
            params.forced_bos_token_id is not None
            or params.forced_eos_token_id is not None
        ):
            return None
        if params.no_repeat_ngram_size and not triton_available():
            return None
        if params.num_beams == 1:
            return "greedy"
        if params.min_length > params.max_tokens:
            return "beam"
        return None

    @staticmethod
    def _order_slots(seqs):
        """Group sequences request-major / beam-minor (the ranking's row order)."""
        request_ids = []
        for seq in seqs:
            if seq.request_id not in request_ids:
                request_ids.append(seq.request_id)
        slots = [s for rid in request_ids for s in seqs if s.request_id == rid]
        return slots, request_ids

    def _run_pipelined_beam(self) -> None:
        """Prefill + beam expansion on the host, then decode without per-step syncs."""
        self.step()  # prefill: encodes, captures the graph, expands to num_beams
        seqs = self.scheduler.schedule()
        params = seqs[0].sampling if seqs else None
        if (
            self.is_finished()
            or params is None
            or not self._lockstep_beam(seqs, params)
        ):
            while not self.is_finished():
                self.step()
            return
        if self.runner.graph is None or not self.runner.graph.ready():
            while not self.is_finished():
                self.step()
            return
        self._pipelined_beam(seqs, params)
        while not self.is_finished():
            self.step()

    def _lockstep_beam(self, seqs, params) -> bool:
        if not seqs:
            return False
        counts = Counter(seq.request_id for seq in seqs)
        if any(count != params.num_beams for count in counts.values()):
            return False
        if len({seq.num_cached_tokens for seq in seqs}) != 1:
            return False
        return len({len(seq.token_ids) for seq in seqs}) == 1

    @torch.no_grad()
    def _pipelined_beam(self, seqs, params) -> None:
        """Beam decode with the ranking, the fork and the tokens on the device.

        Only the copy-on-write allocation stays on the host (one sync per step):
        the allocator is host-side and the number of copies is data dependent.
        Mirrors ``_beam_group`` + ``_beam_advance`` for the fixed-length case,
        where no beam finishes early and the live count stays ``num_beams``.
        """
        runner = self.runner
        graph = runner.graph
        model = runner.model
        block_manager = self.block_manager
        block_size = runner.block_size
        device = runner.device
        cache_layers = runner.cache_layers
        beams = params.num_beams

        slots, request_ids = self._order_slots(seqs)
        n_requests = len(request_ids)
        batch = len(slots)
        remaining = params.max_tokens - slots[0].num_completion_tokens
        if remaining <= 0:
            return

        # Reserve the whole run up front so the block table stays static.
        for seq in slots:
            block_manager.ensure_capacity(seq, remaining, runner._copy_block)
        width = max(len(seq.block_table) for seq in slots)
        block_tables = torch.tensor(
            [seq.block_table + [0] * (width - len(seq.block_table)) for seq in slots],
            dtype=torch.int32,
            device=device,
        )
        graph.stage(slots)
        graph.capture()

        history = torch.zeros(
            batch, 1 + params.max_tokens, dtype=torch.long, device=device
        )
        for row, seq in enumerate(slots):
            history[row, : len(seq.token_ids)] = torch.tensor(
                seq.token_ids, device=device
            )
        ids = torch.tensor(
            [seq.token_ids[-1] for seq in slots], dtype=torch.long, device=device
        )
        cum = torch.tensor(
            [seq.cum_logprob for seq in slots], dtype=torch.float32, device=device
        )
        position = slots[0].num_cached_tokens

        live = Counter(int(b) for seq in slots for b in seq.block_table)
        for block, count in live.items():
            block_manager.blocks[block].ref_count = count
        host_bt = [list(seq.block_table) for seq in slots]

        def cow(write_block: int) -> None:
            """Copy-on-write the block this step writes, before it is written."""
            groups: dict[int, list[int]] = {}
            for slot, table in enumerate(host_bt):
                groups.setdefault(table[write_block], []).append(slot)
            moved = []
            for block, sharing in groups.items():
                for slot in sharing[1:]:
                    new_block = block_manager._allocate_block()
                    host_bt[slot][write_block] = new_block
                    moved.append((slot, new_block))
                    live[new_block] = 1  # freed below if no child inherits it
                    for layer in cache_layers:
                        layer.k_cache[new_block].copy_(layer.k_cache[block])
                        layer.v_cache[new_block].copy_(layer.v_cache[block])
            if moved:
                rows = torch.tensor(
                    [m[0] for m in moved], dtype=torch.long, device=device
                )
                vals = torch.tensor(
                    [m[1] for m in moved], dtype=torch.int32, device=device
                )
                block_tables[rows, write_block] = vals

        for step in range(remaining):
            cow(position // block_size)
            fields = {
                "is_prefill": False,
                "context_lens": torch.full(
                    (batch,), position + 1, dtype=torch.int32, device=device
                ),
                "slot_mapping": (
                    block_tables[:, position // block_size] * block_size
                    + position % block_size
                ).to(torch.int64),
                "block_tables": block_tables,
                "request_ids": None,
            }
            positions = torch.full((batch,), position, dtype=torch.long, device=device)
            hidden = graph.replay(batch, ids, positions, fields)
            logits = model.compute_logits(hidden).float()
            vocab = logits.shape[-1]
            logprobs = self._device_mask(
                torch.log_softmax(logits, dim=-1), history[:, : position + 1], params
            )
            scores = logprobs.reshape(n_requests, beams, vocab) + cum.reshape(
                n_requests, beams, 1
            )
            top_values, top_indices = torch.topk(
                scores.reshape(n_requests, -1), 2 * beams
            )

            parents = (top_indices[:, :beams] // vocab).tolist()
            parent_slot_idx = [
                [r * beams + parents[r][b] for b in range(beams)]
                for r in range(n_requests)
            ]
            parent_slots = torch.tensor(
                parent_slot_idx, dtype=torch.long, device=device
            ).reshape(-1)

            block_tables = block_tables[parent_slots]  # device fork
            host_bt = [list(host_bt[ps]) for row in parent_slot_idx for ps in row]
            refs = Counter(b for table in host_bt for b in table)
            for block in list(live):
                if block not in refs:
                    block_manager._deallocate_block(block)
                    live.pop(block, None)
            for block, count in refs.items():
                block_manager.blocks[block].ref_count = count
                live[block] = count

            ids = (top_indices[:, :beams] % vocab).reshape(-1)
            cum = top_values[:, :beams].reshape(-1)
            history_next = history[parent_slots]
            history_next[:, position + 1] = ids
            history = history_next
            position += 1

        # Every hypothesis has the same length, so the best is the highest
        # cumulative score (the synchronous path's length-normalized ranking).
        best = cum.reshape(n_requests, beams).argmax(dim=-1)
        for r, request_id in enumerate(request_ids):
            seq = slots[r * beams + int(best[r])]
            completion = history[r * beams + int(best[r]), 1 : 1 + params.max_tokens]
            seq.token_ids = list(seq.decoder_prompt_ids) + completion.tolist()
            seq.num_cached_tokens = len(seq.token_ids)
            self.results[request_id] = seq.completion_token_ids
        for block in live:
            block_manager._deallocate_block(block)
        for seq in slots:
            self.scheduler.remove(seq)
        for request_id in request_ids:
            self._release(request_id)

    def _run_pipelined(self) -> None:
        """Prefill with the ordinary path, then decode without per-step syncs.

        Falls back to :meth:`step` whenever the pipelined decode does not apply
        (no CUDA graph, an uneven batch, ...), so the result is the same.
        """
        self.step()  # prefill: encodes, captures the graph, samples token 1
        seqs = self.scheduler.schedule()
        params = seqs[0].sampling if seqs else None
        uneven = (
            len({seq.num_cached_tokens for seq in seqs}) != 1
            or len({len(seq.token_ids) for seq in seqs}) != 1
        )
        if (
            self.is_finished()
            or params is None
            or self.runner.graph is None
            or not self.runner.graph.ready()
            or uneven
        ):
            while not self.is_finished():
                self.step()
            return
        self._pipelined_greedy(seqs, params)
        while not self.is_finished():
            self.step()

    @torch.no_grad()
    def _pipelined_greedy(self, seqs, params) -> None:
        """Greedy decode with the token kept on the device.

        The step metadata (positions / context lengths / slots / block tables)
        and the sampling live on the device, so the host never blocks on a
        per-step ``.tolist()``.  The logits processors that greedy uses are
        reproduced on the device (see :meth:`_device_mask`).
        """
        runner = self.runner
        graph = runner.graph
        model = runner.model
        block_manager = self.block_manager
        block_size = runner.block_size
        device = runner.device
        batch = len(seqs)
        eos_ids = params.eos_ids()

        # Reserve the whole run up front so the block table stays static.
        remaining = params.max_tokens - seqs[0].num_completion_tokens
        if remaining <= 0:
            return
        for seq in seqs:
            block_manager.ensure_capacity(seq, remaining, runner._copy_block)
        width = max(len(seq.block_table) for seq in seqs)
        block_tables = torch.tensor(
            [seq.block_table + [0] * (width - len(seq.block_table)) for seq in seqs],
            dtype=torch.int32,
            device=device,
        )
        graph.stage(seqs)
        graph.capture()

        # Token history the device processors read, one row per sequence.
        history = torch.zeros(
            batch, 1 + params.max_tokens, dtype=torch.long, device=device
        )
        for row, seq in enumerate(seqs):
            history[row, : len(seq.token_ids)] = torch.tensor(
                seq.token_ids, device=device
            )

        ids = torch.tensor(
            [seq.token_ids[-1] for seq in seqs], dtype=torch.long, device=device
        )
        position = seqs[0].num_cached_tokens
        sampled = torch.zeros(batch, remaining, dtype=torch.long, device=device)

        for step in range(remaining):
            fields = {
                "is_prefill": False,
                "context_lens": torch.full(
                    (batch,), position + 1, dtype=torch.int32, device=device
                ),
                "slot_mapping": (
                    block_tables[:, position // block_size] * block_size
                    + position % block_size
                ).to(torch.int64),
                "block_tables": block_tables,
                "request_ids": None,
            }
            positions = torch.full((batch,), position, dtype=torch.long, device=device)
            hidden = graph.replay(batch, ids, positions, fields)
            scores = model.compute_logits(hidden).float()
            scores = self._device_mask(scores, history[:, : position + 1], params)
            ids = torch.argmax(scores, dim=-1)
            sampled[:, step] = ids
            history[:, position + 1] = ids
            position += 1

        # A sequence that hit EOS is truncated at it; the extra steps it ran do
        # not affect the other rows (their caches and processors are separate).
        for row, seq in enumerate(seqs):
            for token in sampled[row].tolist():
                seq.append_token(token)
                if token in eos_ids:
                    break
            seq.num_cached_tokens = len(seq.token_ids)
            self.scheduler.remove(seq)
            block_manager.deallocate(seq)
            self._release(seq.request_id)
            self.results[seq.request_id] = seq.completion_token_ids

    def _device_mask(self, scores, history, params):
        """Logits processors on the device, in ``build_logits_processor`` order.

        Greedy applies them to the logits and beam search to the log-probs, as
        ``_sample`` / ``_beam_group`` do; the forced-token processors are
        excluded by the pipeline guard.
        """
        length = history.shape[-1]
        if params.repetition_penalty != 1.0:
            gathered = torch.gather(scores, -1, history)
            gathered = torch.where(
                gathered < 0,
                gathered * params.repetition_penalty,
                gathered / params.repetition_penalty,
            )
            scores.scatter_(-1, history, gathered)
        if params.no_repeat_ngram_size:
            if self._ngram is None:
                self._ngram = _ngram_processor(params.no_repeat_ngram_size, self.device)
            self._ngram(history, scores)
        if params.min_length and length < params.min_length:
            for token in params.eos_ids():
                scores[:, token] = float("-inf")
        for token in params.suppress_tokens or ():
            scores[:, token] = float("-inf")
        if params.begin_suppress_tokens and length == 1:
            for token in params.begin_suppress_tokens:
                scores[:, token] = float("-inf")
        return scores

    def _postprocess(self, seqs, logits) -> None:
        processor = self._logits_processor
        # Group by request (a request's beams move together), then by beam width
        # and decoder length: one log_softmax + processor + topk can serve every
        # request that shares them.
        by_request: dict[int, list[int]] = defaultdict(list)
        for index, seq in enumerate(seqs):
            by_request[seq.request_id].append(index)

        device = logits.device
        pin = device.type == "cuda"
        beam_groups: dict[tuple[int, int, int], list[tuple[int, list[int]]]] = {}
        buckets: dict[int, list[int]] = {}
        for request_id, indices in by_request.items():
            params = seqs[indices[0]].sampling
            length = len(seqs[indices[0]].token_ids)
            if params.num_beams > 1:
                # A request does not always have ``num_beams`` live beams (the
                # seed is one, and beams can finish early), so the group key
                # carries the actual count -- every row must pack the same way.
                beam_groups.setdefault(
                    (params.num_beams, len(indices), length), []
                ).append((request_id, indices))
            else:
                buckets.setdefault(length, []).extend(indices)

        for (num_beams, beam_count, _), group in beam_groups.items():
            self._beam_group(
                seqs, group, num_beams, beam_count, processor, logits, device, pin
            )

        for indices in buckets.values():
            # pin_memory + non_blocking: a pageable copy would sync the stream.
            input_ids = torch.tensor(
                [seqs[i].token_ids for i in indices],
                dtype=torch.long,
                pin_memory=pin,
            ).to(device, non_blocking=True)
            scores = processor(input_ids, logits[indices].float())
            # One batched sample (one device sync) for the whole bucket.
            tokens = self.sampler(
                scores, seqs[indices[0]].sampling.temperature
            ).tolist()
            for row, i in enumerate(indices):
                self._sample(seqs[i], seqs[i].sampling, tokens[row])

    # -------------------------------------------------------------- sampling
    def _release(self, request_id: int) -> None:
        self.runner.clear_request(request_id)

    def _sample(self, seq: Sequence, params, token: int) -> None:
        seq.append_token(token)
        if token in params.eos_ids() or seq.num_completion_tokens >= params.max_tokens:
            self.scheduler.remove(seq)
            self.block_manager.deallocate(seq)
            self._release(seq.request_id)
            self.results[seq.request_id] = seq.completion_token_ids

    # ------------------------------------------------------------ beam search
    def _fork(self, parent: Sequence, token: int, score: float) -> Sequence:
        child = Sequence(
            request_id=parent.request_id,
            decoder_prompt_ids=parent.decoder_prompt_ids,
            encoder_token_ids=parent.encoder_token_ids,
            sampling=parent.sampling,
            pixel_values=parent.pixel_values,
        )
        child.token_ids = parent.token_ids + [token]
        child.num_cached_tokens = parent.num_cached_tokens
        child.cum_logprob = score
        self.block_manager.fork(parent, child)
        return child

    def _beam_group(
        self, seqs, group, num_beams, beam_count, processor, logits, device, pin
    ) -> None:
        """Rank all continuations for a batch of same-shape beam requests.

        ``group`` is ``[(request_id, seq_indices)]`` and every request has
        ``beam_count`` live beams and the same decoder length.  ``logits[rows]``
        is laid out request-major / beam-minor, so one ``log_softmax``, one
        processor call and one ``topk`` over ``[num_requests, beam_count *
        vocab]`` replace the old per-request loop -- the dominant CUDA
        beam-decode cost (8x fewer kernel launches, and the two ``tolist`` calls
        below cut the per-step device syncs from ~3/request to 2/group).
        """
        vocab = logits.shape[-1]
        rows = [i for _, seq_indices in group for i in seq_indices]
        input_ids = torch.tensor(
            [seqs[i].token_ids for i in rows], dtype=torch.long, pin_memory=pin
        ).to(device, non_blocking=True)
        logprobs = processor(input_ids, torch.log_softmax(logits[rows].float(), dim=-1))
        cumulative = torch.tensor(
            [[seqs[i].cum_logprob for i in seq_indices] for _, seq_indices in group],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(-1)
        scores = (logprobs.reshape(len(group), beam_count, vocab) + cumulative).reshape(
            len(group), -1
        )
        params = seqs[group[0][1][0]].sampling
        keep = min(max(2, 1 + len(params.eos_ids())) * num_beams, scores.shape[-1])
        top_values, top_indices = torch.topk(scores, keep, dim=-1)
        values = top_values.tolist()  # one sync for the whole group
        indices = top_indices.tolist()
        for row, (request_id, seq_indices) in enumerate(group):
            self._beam_advance(
                request_id,
                [seqs[i] for i in seq_indices],
                params,
                [k // vocab for k in indices[row]],
                [k % vocab for k in indices[row]],
                values[row],
            )

    def _beam_advance(self, request_id, beams, params, parents, tokens, top_values):
        num_beams = params.num_beams
        eos_ids = params.eos_ids()

        stopped = [
            token in eos_ids or beams[parent].num_generated + 1 >= params.max_tokens
            for parent, token in zip(parents, tokens)
        ]
        finished_ranks, live_ranks = select_beam_candidates(stopped, num_beams)

        # Build children before freeing parents (children share parent blocks).
        new_finished = [
            self._fork(beams[parents[r]], tokens[r], top_values[r])
            for r in finished_ranks
        ]
        live = [
            self._fork(beams[parents[r]], tokens[r], top_values[r]) for r in live_ranks
        ]

        for beam in beams:
            self.scheduler.remove(beam)
            self.block_manager.deallocate(beam)

        self._finished_beams[request_id].extend(new_finished)
        self._prune_finished(request_id, params)
        for child in live:
            self.scheduler.add_running(child)

        if not live or self._should_stop(request_id, params, live):
            self._finish_beam_request(request_id, params)

    def _prune_finished(self, request_id: int, params) -> None:
        """Keep only the best ``num_beams`` finished hypotheses."""
        finished = self._finished_beams[request_id]
        if len(finished) <= params.num_beams:
            return
        finished.sort(
            key=lambda seq: _length_penalized(seq, params.length_penalty), reverse=True
        )
        for seq in finished[params.num_beams :]:
            self.block_manager.deallocate(seq)
        del finished[params.num_beams :]

    def _should_stop(self, request_id: int, params, live) -> bool:
        """HF ``early_stopping=False``: stop when no live beam can improve.

        Once ``num_beams`` hypotheses are finished, compare the worst finished
        (length-normalized) score against the best possible live score at the
        current length.
        """
        finished = self._finished_beams[request_id]
        if len(finished) < params.num_beams:
            return False
        worst_finished = min(
            _length_penalized(seq, params.length_penalty) for seq in finished
        )
        best_possible = max(seq.cum_logprob for seq in live) / (
            max(live[0].num_generated, 1) ** params.length_penalty
        )
        return worst_finished >= best_possible

    def _finish_beam_request(self, request_id: int, params) -> None:
        finished = self._finished_beams.pop(request_id, [])
        live = [s for s in list(self.scheduler.running) if s.request_id == request_id]

        best = max(
            finished,
            key=lambda seq: _length_penalized(seq, params.length_penalty),
            default=None,
        )

        for seq in live:
            self.scheduler.remove(seq)
        for seq in finished + live:
            self.block_manager.deallocate(seq)
        self._release(request_id)
        self.results[request_id] = best.completion_token_ids if best else []
