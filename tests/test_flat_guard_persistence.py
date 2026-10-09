"""Regression tests for the saturation guard's PERSISTENCE rule.

The guard (train/train.flat_envelope_stop) stopped a healthy run once:

  2026-10-08, p160-pile-parity-full, step 8900/24694 (36%). Four straight improving
  evals (1.5566 -> 1.5426 -> 1.5095 -> 1.4933) were followed by one spike, 1.5856,
  which happened to be the worst point of the trailing 62-eval window (prior max
  1.5751). "No prior round was worse than the newest" is true of a spike and of no
  state the curve was in -- so the run was stopped by one point, not by a stall.

The fix: the statistic still answers for the newest point, but the caller requires
it to hold for TWO consecutive evals. These tests are synthetic on purpose: the real
curves live in runs/ (gitignored), so tests/test_saturation_guard.py replays them
locally while this file keeps the rule covered by the suite on any machine.

Run: pytest tests/test_flat_guard_persistence.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.train import flat_envelope_stop  # noqa: E402

# The parity run's own numbers: 24,694 steps, eval every 100, flat_frac 0.25,
# flat_delta 0.005, no min_step floor (min_steps_frac 0.0 in that job spec).
STEPS, VAL_EVERY, FLAT_FRAC, DELTA, WINDOW = 24694, 100, 0.25, 0.005, 4


def stop(hist, step=None):
    step = STEPS if step is None else step
    return flat_envelope_stop(list(hist), WINDOW, DELTA, step, STEPS,
                              FLAT_FRAC, VAL_EVERY, min_step=0)


def test_spiky_but_still_learning_run_is_not_stopped():
    """The 8900 regression: a sawtooth whose newest point is the window's worst."""
    # 60 evals of a healthy sawtooth (peaks under 1.58, troughs deepening) ...
    hist = [1.45 + (0.10 if i % 5 == 4 else -0.01 * (i % 5)) for i in range(60)]
    hist += [1.5566, 1.5426, 1.5095, 1.4933]      # four straight improvements
    hist += [1.5856]                              # then one spike
    assert max(hist[:-1]) < hist[-1] + DELTA, "spike must be the window's worst"
    assert stop(hist) is False, "one spike must not stop a run"


def test_a_genuinely_flat_curve_is_stopped():
    """What the guard exists for: pinned level, no descent, over the whole budget."""
    hist = [1.4432 + (0.001 if i % 2 else -0.001) for i in range(80)]
    assert stop(hist) is True, "a flat curve must be stopped"


def test_a_still_descending_curve_is_not_stopped():
    hist = [2.0 - 0.02 * i for i in range(80)]
    assert stop(hist) is False, "a learning curve must be left alone"


def test_needs_two_consecutive_stalled_evals():
    """Persistence: stalled now but still learning one eval ago -> keep running.

    Flat at 1.50, then a fresh low of 1.40 (an improvement, so the guard must not
    fire), then a spike to 1.55. The spike alone says 'stalled'; one eval ago the
    curve was still descending, so the run lives. Stop only when two evals agree.
    """
    flat = [1.4432] * 80
    improved_then_spike = [1.50] * 60 + [1.40] + [1.55]
    assert stop(flat) is True
    assert stop(improved_then_spike) is False, "the stall must be visible on two evals in a row"


def test_guard_is_inert_before_it_has_a_budget_to_judge():
    assert stop([1.5] * 5) is False, "not enough history to judge a stall"
    assert flat_envelope_stop([1.4432] * 200, WINDOW, DELTA, 100, STEPS,
                              FLAT_FRAC, VAL_EVERY, min_step=STEPS) is False, \
        "never fires before min_step"


def test_disabled_guard_never_fires():
    hist = [1.4432] * 100
    assert flat_envelope_stop(hist, WINDOW, DELTA, 20000, STEPS, 0.0, VAL_EVERY) is False
