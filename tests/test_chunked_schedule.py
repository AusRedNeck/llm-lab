import pytest

from train.chunking import (
    chunk_is_final,
    resolve_stop_at_step,
    validate_schedule_horizon,
)
from train.train import lr_schedule


def test_stop_defaults_to_schedule_horizon():
    assert resolve_stop_at_step(16_000, None, 0) == 16_000


def test_four_k_chunk_boundary_does_not_reset_global_schedule():
    total, warmup, peak = 16_000, 160, 3e-4
    assert resolve_stop_at_step(total, 4_000, 0) == 4_000
    before, boundary, after = (lr_schedule(s, warmup, total, peak)
                               for s in (3_999, 4_000, 4_001))
    assert before > boundary > after > 0.1 * peak
    assert chunk_is_final(4_000, total) is False
    assert chunk_is_final(total, total) is True


@pytest.mark.parametrize("start,stop", [(4_000, 4_000), (8_000, 4_000), (0, 16_001)])
def test_invalid_chunk_endpoint_is_rejected(start, stop):
    with pytest.raises(ValueError):
        resolve_stop_at_step(16_000, stop, start)


def test_resume_must_keep_the_original_schedule_horizon():
    with pytest.raises(ValueError, match="schedule horizon"):
        validate_schedule_horizon(saved=16_000, requested=8_000)
    validate_schedule_horizon(saved=16_000, requested=16_000)
