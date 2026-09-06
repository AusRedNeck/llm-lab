import torch

from model.transformer import Transformer
from train.train import early_stop_update


def test_dropout_zero_is_identity_in_train():
    # p=0 drops nothing, so train mode still deterministic.
    m = Transformer(vocab_size=32, context_length=8, embedding_dim=16,
                    num_heads=4, num_layers=2, dropout=0.0)
    m.train()
    x = torch.randint(0, 32, (2, 8))
    assert torch.equal(m(x), m(x))


def test_dropout_shakes_train_not_eval():
    # Regularizer on in train, off in eval — that's the whole deal.
    m = Transformer(vocab_size=32, context_length=8, embedding_dim=16,
                    num_heads=4, num_layers=2, dropout=0.5)
    x = torch.randint(0, 32, (2, 8))
    m.train()
    a, b = m(x), m(x)
    assert not torch.equal(a, b)  # dropout noise present
    m.eval()
    assert torch.equal(m(x), m(x))  # eval is deterministic


def test_dropout_keeps_shapes_and_capture_agrees():
    # Dropout must not change tensor shapes or break the X-ray path.
    m = Transformer(vocab_size=32, context_length=8, embedding_dim=16,
                    num_heads=4, num_layers=2, dropout=0.1)
    m.eval()
    x = torch.randint(0, 32, (2, 8))
    logits = m(x)
    assert logits.shape == (2, 8, 32)
    out_logits, out = m.forward_with_capture(x)
    assert out_logits.shape == (2, 8, 32)
    assert torch.allclose(out["logits"], logits)


def test_early_stop_improvement_resets_bad():
    # Lower val = progress, bad counter resets, keep going.
    best, bad, stop = early_stop_update(2.0, 1.9, 2, patience=3)
    assert best == 1.9 and bad == 0 and stop is False


def test_early_stop_plateau_counts_then_trips():
    # Flat val burns patience, third strike stops the run.
    best, bad, stop = early_stop_update(2.0, 2.0, 0, patience=3)
    assert (best, bad, stop) == (2.0, 1, False)
    _, bad, stop = early_stop_update(2.0, 2.0, 2, patience=3)
    assert (bad, stop) == (3, True)


def test_early_stop_patience_zero_never_stops():
    # Patience 0 = old behavior, run the full schedule.
    _, _, stop = early_stop_update(2.0, 5.0, 999, patience=0)
    assert stop is False
