"""Continuous-batching scheduler.

The scheduler keeps a ``running`` batch that requests join as soon as they
arrive and leave when they finish.  Beam search is implemented on top of the
same paged cache: each beam is a normal :class:`Sequence`, blocks are shared by
reference count and copy-on-write happens on the next append (see
:mod:`fastencdec.engine.block_manager`).
"""

from collections import deque

from ..config import Config
from .sequence import Sequence


class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

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
