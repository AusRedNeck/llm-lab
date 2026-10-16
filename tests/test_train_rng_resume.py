import pytest
import torch

from train.rng_state import capture_rng_state, make_generator, restore_rng_state


def test_rng_checkpoint_restores_the_next_training_draw():
    torch.manual_seed(1204)
    saved = capture_rng_state(torch)
    expected = torch.randint(0, 10_000, (32,))
    restore_rng_state(torch, saved)
    actual = torch.randint(0, 10_000, (32,))
    assert torch.equal(actual, expected)


def test_dedicated_sample_generator_does_not_advance_training_rng():
    torch.manual_seed(77)
    before = torch.get_rng_state().clone()
    generator = make_generator(torch, torch.device("cpu"), 0)
    torch.multinomial(torch.tensor([0.2, 0.8]), 20, replacement=True,
                      generator=generator)
    assert torch.equal(torch.get_rng_state(), before)


def test_rng_restore_without_checkpoint_state_is_legacy_noop():
    assert restore_rng_state(torch, None) is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_rng_checkpoint_restores_all_device_states():
    torch.cuda.manual_seed_all(91)
    saved = capture_rng_state(torch)
    expected = torch.randint(0, 10_000, (32,), device="cuda")
    restore_rng_state(torch, saved)
    actual = torch.randint(0, 10_000, (32,), device="cuda")
    assert torch.equal(actual, expected)
