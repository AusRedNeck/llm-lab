import math
import torch
import pytest

from eval.benchmark import perplexity, token_accuracy

def test_perplexity_is_exp_of_ce():
    # CE of ln(4) per token -> ppl should be 4.0
    logits = torch.zeros(1, 3, 4) # uniform -> CE = ln(4)
    targets = torch.zeros(1, 3, dtype=torch.long)
    assert perplexity(logits, targets) == pytest.approx(4.0)