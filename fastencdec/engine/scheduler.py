from collections import deque

from ..config import Config
from .sequence import Sequence


class Scheduler:
    """Continuous-batching scheduler that admits requests up to max_num_seqs."""

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []

    def is_finished(self) -> bool:
        """Return True when both waiting and running queues are empty."""
        return not self.waiting and not self.running

    def add(self, seq: Sequence) -> None:
        """Enqueue a new request sequence into the waiting queue."""
        if seq.sampling.num_beams > self.max_num_seqs:
            raise ValueError(
                f"num_beams ({seq.sampling.num_beams}) exceeds "
                f"max_num_seqs ({self.max_num_seqs})"
            )
        self.waiting.append(seq)

    def add_running(self, seq: Sequence) -> None:
        """Append a live beam hypothesis directly to the running batch."""
        self.running.append(seq)

    def remove(self, seq: Sequence) -> None:
        """Remove a finished or pruned sequence from the running batch."""
        self.running.remove(seq)

    def schedule(self) -> list[Sequence]:
        """Admit waiting sequences up to max_num_seqs (reserving full beam width) and return running."""
        used = sum({s.request_id: s.sampling.num_beams for s in self.running}.values())
        while (
            self.waiting
            and used + self.waiting[0].sampling.num_beams <= self.max_num_seqs
        ):
            seq = self.waiting.popleft()
            self.running.append(seq)
            used += seq.sampling.num_beams
        return list(self.running)
