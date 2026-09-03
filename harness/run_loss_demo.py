import torch
from model.loss import simple_mse_loss

logits = torch.randn(2,4,8, requires_grad=True)
targets = torch.randint(0,8,(2,4))
loss = simple_mse_loss(logits, targets)
print(f"loss={loss.item():.4f}")

loss.backward()
print("grad ok:", logits.grad is not None)
print("grad shape:", logits.grad.shape)
print("mse:", simple_mse_loss(logits, targets).item())
print("ce: ", cross_entropy_loss(logits, targets).item())