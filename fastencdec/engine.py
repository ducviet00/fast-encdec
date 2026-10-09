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
                scores = self._logits_processors[request_id](
                    input_ids, logits[indices].float()
                )
                if params.num_beams > 1:
                    beams = [seqs[i] for i in indices]
                    self._beam_step(request_id, beams, params, scores)
                else:
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

    def _beam_step(self, request_id: int, beams, params, logits) -> None:
        num_beams = params.num_beams
        eos_ids = params.eos_ids()
        logprobs = torch.log_softmax(logits, dim=-1)

        candidates = []  # (score, parent_index, token)
        for index, beam in enumerate(beams):
            topk = min(2 * num_beams, logprobs.shape[-1])
            values, tokens = torch.topk(logprobs[index], topk)
            for value, token in zip(values.tolist(), tokens.tolist()):
                candidates.append((beam.cum_logprob + value, index, token))

        eos_candidates = sorted(
            (c for c in candidates if c[2] in eos_ids), reverse=True
        )
        live_candidates = sorted(
            (c for c in candidates if c[2] not in eos_ids), reverse=True
        )

        # Build children before freeing parents (children share parent blocks).
        finished = [
            self._fork(beams[i], tok, score)
            for score, i, tok in eos_candidates[:num_beams]
        ]
        live = []
        for score, i, tok in live_candidates[:num_beams]:
            child = self._fork(beams[i], tok, score)
            if child.num_generated >= params.max_tokens:
                finished.append(child)
            else:
                live.append(child)

        for beam in beams:
            self.scheduler.remove(beam)
            self.runner.block_manager.free(beam)

        self._finished_beams[request_id].extend(finished)
        for child in live:
            self.scheduler.add_running(child)

        if not live or len(self._finished_beams[request_id]) >= num_beams:
            self._finish_beam_request(request_id, params)

    def _finish_beam_request(self, request_id: int, params) -> None:
        done = self._finished_beams.pop(request_id, [])
        live = [s for s in list(self.scheduler.running) if s.request_id == request_id]
        candidates = done + live

        def length_penalized(seq: Sequence) -> float:
            length = max(seq.num_generated, 1)
            return seq.cum_logprob / (length**params.length_penalty)

        best = max(candidates, key=length_penalized) if candidates else None

        for seq in live:
            self.scheduler.remove(seq)
        for seq in candidates:
            self.runner.block_manager.free(seq)
        self._release(request_id)
        self.results[request_id] = best.generated_ids() if best else []
