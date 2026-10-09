"""Continuous-batching engine with greedy/sampling and beam search.

The scheduler keeps a ``running`` batch that requests join as soon as they
arrive (continuous batching) and leave when they finish.  Beam search is
implemented on top of the same paged cache: each beam is a normal
:class:`Sequence`, blocks are shared by reference count and copy-on-write
happens on the next append (see :mod:`fastencdec.block_manager`).
"""

from collections import defaultdict, deque
from dataclasses import replace

import torch

from .logits_processor import build_logits_processor
from .model_runner import ModelRunner
from .sampler import sample_token
from .sequence import Sequence


class Scheduler:
    def __init__(self, max_num_seqs: int = 32):
        self.waiting = deque()
        self.running: list[Sequence] = []
        self.max_num_seqs = max_num_seqs

    def add(self, seq: Sequence) -> None:
        if seq.sampling.num_beams > self.max_num_seqs:
            raise ValueError(
                f"num_beams ({seq.sampling.num_beams}) exceeds "
                f"max_num_seqs ({self.max_num_seqs})"
            )
        self.waiting.append(seq)

    def add_running(self, seq: Sequence) -> None:
        self.running.append(seq)

    def remove(self, seq: Sequence) -> None:
        self.running.remove(seq)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> list[Sequence]:
        # A request expands to ``num_beams`` sequences once beam search starts,
        # so charge every in-flight request its full beam width.  Counting
        # sequences directly would admit ``max_num_seqs`` seeds and then
        # overflow the batch on the following step.
        widths = {seq.request_id: seq.sampling.num_beams for seq in self.running}
        used = sum(widths.values())
        while self.waiting:
            seq = self.waiting[0]
            if used + seq.sampling.num_beams > self.max_num_seqs:
                break
            self.waiting.popleft()
            self.running.append(seq)
            used += seq.sampling.num_beams
        return list(self.running)


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
        block_manager,
        max_num_seqs: int = 32,
        dtype=torch.float32,
        eos_token_id: int | None = None,
    ):
        self.runner = ModelRunner(model, block_manager, dtype)
        self.scheduler = Scheduler(max_num_seqs)
        self.eos_token_id = eos_token_id
        self.results: dict[int, list[int]] = {}
        self._finished_beams: dict[int, list[Sequence]] = defaultdict(list)
        self._logits_processors: dict[int, object] = {}
        self._next_request_id = 0

    def add_request(
        self,
        encoder_token_ids: list[int],
        decoder_token_ids: list[int],
        params,
        pixel_values=None,
    ) -> int:
        request_id = self._next_request_id
        self._next_request_id += 1
        if params.eos_token_id is None and self.eos_token_id is not None:
            params = replace(params, eos_token_id=self.eos_token_id)
        seq = Sequence(
            request_id=request_id,
            decoder_prompt_ids=decoder_token_ids,
            encoder_token_ids=encoder_token_ids,
            sampling=params,
            pixel_values=pixel_values,
        )
        begin_index = len(decoder_token_ids) + int(
            params.forced_bos_token_id is not None and len(decoder_token_ids) == 1
        )
        self._logits_processors[request_id] = build_logits_processor(
            params,
            max_length=len(decoder_token_ids) + params.max_tokens,
            begin_index=begin_index,
            is_beam=params.num_beams > 1,
        )
        self.scheduler.add(seq)
        return request_id

    # ------------------------------------------------------------------ run
    def run(self) -> None:
        while self.scheduler.has_work():
            seqs = self.scheduler.schedule()
            logits = self.runner.run(seqs)

            groups = defaultdict(list)
            for index, seq in enumerate(seqs):
                groups[seq.request_id].append(index)

            for request_id, indices in groups.items():
                params = seqs[indices[0]].sampling
                input_ids = torch.tensor(
                    [seqs[i].token_ids for i in indices], dtype=torch.long
                )
                logits_batch = logits[indices].float()
                processor = self._logits_processors[request_id]
                if params.num_beams > 1:
                    # generate's beam search applies the processors to
                    # log-probs, so a masking processor does not renormalize
                    # the surviving scores (unlike the greedy path below).
                    logprobs = processor(
                        input_ids, torch.log_softmax(logits_batch, dim=-1)
                    )
                    beams = [seqs[i] for i in indices]
                    self._beam_step(request_id, beams, params, logprobs)
                else:
                    scores = processor(input_ids, logits_batch)
                    for i, score in zip(indices, scores):
                        self._sample_step(seqs[i], params, score)

    # -------------------------------------------------------------- sampling
    def _release(self, request_id: int) -> None:
        self.runner.clear_request(request_id)
        self._logits_processors.pop(request_id, None)

    def _sample_step(self, seq: Sequence, params, logits) -> None:
        token = sample_token(logits, params.temperature)
        seq.token_ids.append(token)
        if token in params.eos_ids() or seq.num_generated >= params.max_tokens:
            self.scheduler.remove(seq)
            self.runner.block_manager.free(seq)
            self._release(seq.request_id)
            self.results[seq.request_id] = seq.generated_ids()

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
        self.runner.block_manager.fork(parent, child)
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
            self.runner.block_manager.free(beam)

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
            self.runner.block_manager.free(seq)
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
            self.runner.block_manager.free(seq)
        self._release(request_id)
        self.results[request_id] = best.generated_ids() if best else []
