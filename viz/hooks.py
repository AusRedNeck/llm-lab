"""Glass-box capture: run the model WITHOUT changing it, get the X-ray data."""
import torch


@torch.no_grad()
def capture_forward(model, token_ids: torch.Tensor) -> dict:
    """Single connector every viz view reads from.

    Input: model + [1, T] token IDs.
    Output: dict with embeddings, per-layer hiddens, per-layer
    attention [layers, heads, T, T], logits, probs.
    The model file never imports viz — viz reads this dict.
    """
    model.eval()
    logits, captures = model.forward_with_capture(token_ids)
    return {"logits": logits, **captures}
