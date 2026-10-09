"""The data-recycling tripwire's margin must be a measured number, not a guess.

A false positive cost this lab 8.5 hours of GPU on 2026-10-08: the margin was
hardcoded at 0.15 while the HEALTHY baseline -- what a model scores on the tokens
it was just gradient-descended on, versus a uniform crop of the same train split
-- is about 0.13 bpb. Measured on p160-pile-parity-full once the served buffer was
made to roll: -0.1230 @9100, -0.1330 @9200, -0.1527 @9300, -0.1640 @9400. Two
consecutive breaches stop the run, so at 0.15 it stopped at step 9400 declaring
"the loader is recycling data" -- on a loader (get_batch -> torch.randint over all
3.23B tokens) that cannot recycle at all.

So the predicate is factored out here and the margin is a CLI flag
(--recycle-margin, default 0.15 to preserve existing behaviour; this arm runs at
0.30, twice the measured healthy max). 0.30 is deliberately NOT claimed to be
calibrated -- no run in this lab can produce a positive control while the loader
draws uniformly.

Run: pytest tests/test_recycle_tripwire.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.train import recycle_gap_trips  # noqa: E402

DEFAULT = 0.15
THIS_ARM = 0.30

# served / random_train bpb pairs actually logged by the run after the buffer fix
MEASURED = [
    (1.1724, 1.2954),   # step 9100, gap -0.1230
    (1.2097, 1.3427),   # step 9200, gap -0.1330
    (1.1762, 1.3289),   # step 9300, gap -0.1527  <- first hit
    (1.1718, 1.3358),   # step 9400, gap -0.1640  <- second hit, run stopped
]


def test_measured_healthy_run_trips_at_the_old_default():
    """The false positive, reproduced: two consecutive breaches, run stopped."""
    hits = [recycle_gap_trips(s, r, DEFAULT) for s, r in MEASURED]
    assert hits == [False, False, True, True]


def test_the_shipped_arm_margin_survives_every_measurement():
    for served, random_train in MEASURED:
        assert recycle_gap_trips(served, random_train, THIS_ARM) is False, \
            f"gap {served - random_train:.4f} must not stop this arm at 0.30"


def test_default_still_fires_on_a_real_loop():
    """A looping loader hands back the same tokens: far below any healthy reading."""
    assert recycle_gap_trips(1.00, 1.60, DEFAULT) is True


def test_margin_zero_disables_the_tripwire():
    """0 means OFF, not 'trip whenever served is marginally better' (which is always)."""
    assert recycle_gap_trips(0.01, 3.00, 0) is False
    assert recycle_gap_trips(0.01, 3.00, -1.0) is False


def test_missing_measurements_never_count_as_a_hit():
    assert recycle_gap_trips(None, 1.5, DEFAULT) is False
    assert recycle_gap_trips(1.5, None, DEFAULT) is False
    assert recycle_gap_trips(None, None, DEFAULT) is False


def test_boundary_is_strict():
    """Exactly at the margin is not a breach -- the run must not die on rounding."""
    assert recycle_gap_trips(1.35, 1.50, DEFAULT) is False
    assert recycle_gap_trips(1.3499, 1.50, DEFAULT) is True
