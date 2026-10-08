"""Continuous-batching engine with greedy/sampling and beam search.

The scheduler keeps a ``running`` batch that requests join as soon as they
arrive (continuous batching) and leave when they finish.  Beam search is
implemented on top of the same paged cache: each beam is a normal
:class:`Sequence`, blocks are shared by reference count and copy-on-write
happens on the next append (see :mod:`fastencdec.block_manager`).
"""

from collections import defaultdict, deque

import torch

from .model_runner import ModelRunner
from .sampler import banned_tokens, sample_token
from .sequence import Sequence


class Scheduler:
    def __init__(self, max_num_seqs: int = 32):
        self.waiting = deque()
        self.running: list[Sequence] = []
        self.max_num_seqs = max_num_seqs

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def add_running(self, seq: Sequence) -> None:
        self.running.append(seq)

    def remove(self, seq: Sequence) -> None:
        self.running.remove(seq)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> list[Sequence]:
        while self.waiting and len(self.running) < self.max_num_seqs:
            self.running.append(self.waiting.popleft())
        return list(self.running)


class LLMEngine:
    def __init__(self, model, block_manager, max_num_seqs: int = 32,
                 dtype=torch.float32):
        self.runner = ModelRunner(model, block_manager, dtype)
        self.scheduler = Scheduler(max_num_seqs)
        self.results: dict[int, list[int]] = {}
        self._finished_beams: dict[int, list[Sequence]] = defaultdict(list)
        self._next_request_id = 0

    def add_request(self, encoder_token_ids: list[int], decoder_token_ids: list[int],
                    params) -> int:
        request_id = self._next_request_id
        self._next_request_id += 1
        seq = Sequence(
            request_id=request_id,
            decoder_prompt_ids=decoder_token_ids,
            encoder_token_ids=encoder_token_ids,
            sampling=params,
        )
        self.scheduler.add(seq)
        return request_id

    # ------------------------------------------------------------------ run
    def run(self) -> None:
        while self.scheduler.has_work():
            seqs = self.scheduler.schedule()
            logits = self.runner.run(seqs)

            groups = defaultdict(list)
            for seq, logit in zip(seqs, logits):
                groups[seq.request_id].append((seq, logit))

            for request_id, items in groups.items():
                params = items[0][0].sampling
                if params.num_beams > 1:
                    self._beam_step(request_id, items, params)
                else:
                    for seq, logit in items:
                        self._sample_step(seq, logit, params)

    # -------------------------------------------------------------- sampling
    def _banned(self, seq: Sequence, params) -> set[int]:
        banned = set()
        if seq.num_generated < params.min_length:
            banned |= params.eos_ids()
        banned |= banned_tokens(seq.token_ids, params.no_repeat_ngram_size)
        return banned

    def _sample_step(self, seq: Sequence, logits, params) -> None:
        token = sample_token(logits, params, self._banned(seq, params))
        seq.token_ids.append(token)
        if token in params.eos_ids() or seq.num_generated >= params.max_tokens:
            self.scheduler.remove(seq)
            self.runner.block_manager.free(seq)
            self.runner.clear_request(seq.request_id)
            self.results[seq.request_id] = seq.generated_ids()

    # ------------------------------------------------------------ beam search
    def _fork(self, parent: Sequence, token: int, score: float) -> Sequence:
        child = Sequence(
            request_id=parent.request_id,
            decoder_prompt_ids=parent.decoder_prompt_ids,
            encoder_token_ids=parent.encoder_token_ids,
            sampling=parent.sampling,
        )
        child.token_ids = parent.token_ids + [token]
        child.num_cached_tokens = parent.num_cached_tokens
        child.cum_logprob = score
        self.runner.block_manager.fork(parent, child)
        return child

    def _beam_step(self, request_id: int, items, params) -> None:
        num_beams = params.num_beams
        eos_ids = params.eos_ids()

        candidates = []  # (score, parent_index, token)
        for index, (beam, logits) in enumerate(items):
            masked = logits.float().clone()
            banned = self._banned(beam, params)
            if banned:
                masked[list(banned)] = -float("inf")
            logprobs = torch.log_softmax(masked, dim=-1)
            topk = min(2 * num_beams, logprobs.numel())
            values, tokens = torch.topk(logprobs, topk)
            for value, token in zip(values.tolist(), tokens.tolist()):
                candidates.append((beam.cum_logprob + value, index, token))

        eos_candidates = sorted(
            (c for c in candidates if c[2] in eos_ids), reverse=True)
        live_candidates = sorted(
            (c for c in candidates if c[2] not in eos_ids), reverse=True)

        beams = [seq for seq, _ in items]

        # Build children before freeing parents (children share parent blocks).
        finished = [self._fork(beams[i], tok, score)
                    for score, i, tok in eos_candidates[:num_beams]]
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
        live = [s for s in list(self.scheduler.running)
                if s.request_id == request_id]
        candidates = done + live

        def length_penalized(seq: Sequence) -> float:
            length = max(seq.num_generated, 1)
            return seq.cum_logprob / (length ** params.length_penalty)

        best = max(candidates, key=length_penalized) if candidates else None

        for seq in live:
            self.scheduler.remove(seq)
        for seq in candidates:
            self.runner.block_manager.free(seq)
        self.runner.clear_request(request_id)
        self.results[request_id] = best.generated_ids() if best else []
