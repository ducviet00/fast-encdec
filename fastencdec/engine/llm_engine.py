"""Continuous-batching engine with greedy/sampling and beam search.

The engine is offline: :meth:`LLMEngine.generate` adds a whole batch with a
single :class:`SamplingParams` and runs to completion.  Beam search is built on
the same paged cache: each beam is a normal :class:`Sequence`, blocks are shared
by reference count and copy-on-write happens on the next append (see
:mod:`fastencdec.engine.block_manager`).
"""

from collections import defaultdict
from dataclasses import replace

import torch

from ..config import Config
from ..layers.logits_processor import build_logits_processor
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
        )
        self.scheduler = Scheduler(config)
        self.sampler = Sampler()
        self.eos_token_id = eos_token_id
        self.results: dict[int, list[int]] = {}
        self._finished_beams: dict[int, list[Sequence]] = defaultdict(list)
        self._logits_processor = None
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
        begin_index = prompt_len + int(
            params.forced_bos_token_id is not None and prompt_len == 1
        )
        self._logits_processor = build_logits_processor(
            params,
            max_length=prompt_len + params.max_tokens,
            begin_index=begin_index,
            is_beam=params.num_beams > 1,
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
        while not self.is_finished():
            self.step()

    def _postprocess(self, seqs, logits) -> None:
        processor = self._logits_processor
        groups = defaultdict(list)
        for index, seq in enumerate(seqs):
            groups[seq.request_id].append(index)

        # Greedy rows are independent, so requests at the same decoder length
        # are served by one processor call; beam needs the per-request ranking.
        buckets: dict[int, list[int]] = {}
        for request_id, indices in groups.items():
            params = seqs[indices[0]].sampling
            if params.num_beams > 1:
                input_ids = torch.tensor(
                    [seqs[i].token_ids for i in indices], dtype=torch.long
                )
                logprobs = processor(
                    input_ids, torch.log_softmax(logits[indices].float(), dim=-1)
                )
                self._beam_step(
                    request_id, [seqs[i] for i in indices], params, logprobs
                )
            else:
                length = len(seqs[indices[0]].token_ids)
                buckets.setdefault(length, []).extend(indices)

        for indices in buckets.values():
            input_ids = torch.tensor(
                [seqs[i].token_ids for i in indices], dtype=torch.long
            )
            scores = processor(input_ids, logits[indices].float())
            for row, i in enumerate(indices):
                self._sample(seqs[i], seqs[i].sampling, scores[row])

    # -------------------------------------------------------------- sampling
    def _release(self, request_id: int) -> None:
        self.runner.clear_request(request_id)

    def _sample(self, seq: Sequence, params, logits) -> None:
        token = self.sampler(logits, params.temperature)
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

    def _beam_step(self, request_id: int, beams, params, logprobs) -> None:
        num_beams = params.num_beams
        eos_ids = params.eos_ids()
        vocab = logprobs.shape[-1]

        # Rank every continuation globally, like HF's beam search.
        cumulative = torch.tensor([beam.cum_logprob for beam in beams]).unsqueeze(1)
        scores = (logprobs + cumulative).reshape(-1)
        keep = min(max(2, 1 + len(eos_ids)) * num_beams, scores.numel())
        top_scores, top_indices = torch.topk(scores, keep)
        parents = (top_indices // vocab).tolist()
        tokens = (top_indices % vocab).tolist()

        stopped = [
            token in eos_ids or beams[parent].num_generated + 1 >= params.max_tokens
            for parent, token in zip(parents, tokens)
        ]
        finished_ranks, live_ranks = select_beam_candidates(stopped, num_beams)

        # Build children before freeing parents (children share parent blocks).
        new_finished = [
            self._fork(beams[parents[r]], tokens[r], top_scores[r].item())
            for r in finished_ranks
        ]
        live = [
            self._fork(beams[parents[r]], tokens[r], top_scores[r].item())
            for r in live_ranks
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
