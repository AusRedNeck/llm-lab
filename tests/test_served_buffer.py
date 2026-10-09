"""The served-metric buffer must ROLL, not freeze after its first fill.

The bug this covers (2026-10-08): the training loop guarded the append with
`if len(buffer) < SERVED_BUFFER_SIZE`, so the buffer froze at ~step 6 of every
process and `served()` scored the FIRST 200 micro-batches forever. Because a
resume replays seed 0, every relaunch re-served byte-identical batches; by the
third exposure served beat random_train by 0.16-0.22 bpb, the recycling
tripwire fired twice running, and a healthy run was stopped at step 9400 of
24,694 -- on a loader (`get_batch` -> `torch.randint` over the whole corpus)
that cannot recycle at all.

The metric decides whether a run lives, so the buffer it reads has to mean what
its comment says.

Run: pytest tests/test_served_buffer.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.train import push_served  # noqa: E402

SIZE = 200


def test_buffer_fills_then_rolls():
    buf = []
    for i in range(SIZE + 50):
        push_served(buf, i, SIZE)
    assert len(buf) == SIZE, "must never exceed the cap"
    assert buf[-1] == SIZE + 49, "newest item must be kept"
    assert buf[0] == 50, "oldest item must be evicted once the cap is reached"


def test_buffer_keeps_the_last_n_not_the_first_n():
    """The exact defect: fill-once keeps items 0..199, rolling keeps the last 200."""
    rolling, frozen = [], []
    for i in range(1000):
        push_served(rolling, i, SIZE)
        if len(frozen) < SIZE:            # the original guard
            frozen.append(i)
    assert frozen[0] == 0 and frozen[-1] == 199, "control: the old code froze here"
    assert rolling[0] == 800 and rolling[-1] == 999, "rolling must track the head"
    assert rolling != frozen


def test_buffer_starts_empty_and_never_exceeds_before_the_first_fill():
    buf = []
    push_served(buf, "only", SIZE)
    assert buf == ["only"]
    for i in range(10):
        push_served(buf, i, SIZE)
    assert len(buf) == 11 < SIZE


def test_size_one_degenerates_to_the_latest_item():
    buf = []
    for i in range(5):
        push_served(buf, i, 1)
    assert buf == [4]
