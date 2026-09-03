import torch
from model.loss import simple_mse_loss, cross_entropy_loss

def test_perfect_prediction_is_near_zero():
    # batch 1, seq 2, vocab 4 — logits that strongly favor the target
    logits = torch.tensor([[[10, -10, -10, -10], [-10, 10, -10, -10]]], dtype=torch.float32)
    targets = torch.tensor([[0, 1]])
    loss = simple_mse_loss(logits, targets)
    assert loss.item() < 0.01

def test_wrong_prediction_is_high():
    logits = torch.tensor([[[ -10, 10, -10, -10]]], dtype=torch.float32) # predicts 1
    targets = torch.tensor([[0]]) # wants 0
    loss = simple_mse_loss(logits, targets)
    assert loss.item() > 0.3

def test_shape_and_grad():
    logits = torch.randn(2, 4, 8, requires_grad=True)
    targets = torch.randint(0, 8, (2, 4))
    loss = simple_mse_loss(logits, targets)
    assert loss.dim() == 0
    loss.backward()
    assert logits.grad is not None

    from model.loss import cross_entropy_loss

def test_ce_lower_when_correct():
    logits = torch.tensor([[[10,-10,-10],[-10,10,-10]]], dtype=torch.float32)
    targets = torch.tensor([[0,1]])
    assert cross_entropy_loss(logits, targets).item() < 0.1

def test_ce_higher_when_wrong():
    logits = torch.tensor([[[ -10,10,-10]]], dtype=torch.float32)
    targets = torch.tensor([[0]])
    assert cross_entropy_loss(logits, targets).item() > 5.0