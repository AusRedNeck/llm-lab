import torch
import torch.nn.functional as F

def simple_mse_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    
    # Simplest loss before cross-entropy:
    # softmax(logits) → one_hot(targets) → mean squared error
    # logits: [B, T, V], targets: [B, T]
    
    probs = torch.softmax(logits, dim=-1)
    # one_hot needs vocab size
    vocab_size = logits.size(-1)
    target_one_hot = F.one_hot(targets, num_classes=vocab_size).float()
    return torch.mean((probs - target_one_hot) ** 2)