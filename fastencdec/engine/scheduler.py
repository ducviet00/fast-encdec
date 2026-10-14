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
        # Reserve full beam width per active request so beam expansion cannot overflow.
        used = sum({s.request_id: s.sampling.num_beams for s in self.running}.values())
        while (
            self.waiting
            and used + self.waiting[0].sampling.num_beams <= self.max_num_seqs
        ):
            seq = self.waiting.popleft()
            self.running.append(seq)
            used += seq.sampling.num_beams
        return list(self.running)
