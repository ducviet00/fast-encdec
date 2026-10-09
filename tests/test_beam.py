"""Beam-search candidate selection: the EOS/rank gate from ``generate``.

A stopped candidate (EOS or max length) may only become a finished hypothesis
if it ranks within the top ``num_beams`` of the combined ranking; weaker ones
are dropped rather than filling the finished set and triggering an early stop.

Run with:  PYTHONPATH=. python tests/test_beam.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastencdec.engine import select_beam_candidates

# Each case is (stopped flags best-first, num_beams, finished ranks, live ranks).
CASES = [
    # EOS in the top rows finish; the rest continue.
    ([True, True, False, False], 2, [0, 1], [2, 3]),
    # EOS ranked below num_beams must not finish, and does not fill a slot.
    ([False, False, True, True], 2, [], [0, 1]),
    # Only the first EOS qualifies; the weaker EOS is dropped, not finished.
    ([False, True, True, False], 2, [1], [0, 3]),
    ([True, False, True, False], 2, [0], [1, 3]),
    # More stopped candidates than beams: extras are dropped, no live beams.
    ([True, True, True], 2, [0, 1], []),
]


def main():
    for stopped, num_beams, want_finished, want_live in CASES:
        finished, live = select_beam_candidates(stopped, num_beams)
        ok = finished == want_finished and live == want_live
        print(
            f"[{'OK' if ok else 'FAIL'}] stopped={stopped} beams={num_beams} "
            f"-> finished={finished} live={live}"
        )
        assert ok, (stopped, num_beams, finished, live, want_finished, want_live)


if __name__ == "__main__":
    main()
