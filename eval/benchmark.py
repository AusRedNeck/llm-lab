import math
import torch

from model.loss import cross_entropy_loss

def perplexity(logits: torch.Tensor, targets: torch.Tensor) -> float:
    # ppl = exp(avg CE). 0.73 val loss -> ~2.08 ppl. Lower = less surprised.
    return math.exp(cross_entropy_loss(logits, targets).item())

def token_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    # fraction of positions where argmax == target
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()