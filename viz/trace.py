"""Bounded JSON summaries of one selected-token model trace."""
from __future__ import annotations

DEFAULT_MAX_CONTEXT = 64
MAX_TRACE_CONTEXT = 128


def build_trace(captures, token_ids, token_labels, *, selected_index,
                max_context=DEFAULT_MAX_CONTEXT):
    """Reduce opt-in captures to small per-token/per-layer inspection values."""
    if isinstance(max_context, bool) or not isinstance(max_context, int) or not 1 <= max_context <= MAX_TRACE_CONTEXT:
        raise ValueError(f"max_context must be between 1 and {MAX_TRACE_CONTEXT}")
    if len(token_ids) != len(token_labels) or not token_ids:
        raise ValueError("token_ids and token_labels must have equal non-zero length")
    if isinstance(selected_index, bool) or not isinstance(selected_index, int) or not 0 <= selected_index < len(token_ids):
        raise ValueError("selected_index must identify a prompt token")
    if len(token_ids) > max_context:
        raise ValueError(f"trace exceeds max_context ({max_context})")
    import torch
    def floats(tensor):
        return tensor.detach().to(device="cpu", dtype=torch.float32).tolist()

    weights_by_layer = captures.get("layer_attention") or captures.get("attentions")
    if not weights_by_layer:
        raise ValueError("capture does not include attention weights")
    details_by_layer = captures.get("layer_details")
    hidden_by_layer = captures.get("layer_hiddens")
    if details_by_layer is None and hidden_by_layer is None:
        states = captures.get("hidden_states")
        hidden_by_layer = list(states[1:]) if states else []
    layers = []
    for index, weights in enumerate(weights_by_layer):
        attention = weights[0, :, selected_index, :]
        hidden = (details_by_layer[index]["hidden"] if details_by_layer is not None
                  else hidden_by_layer[index])
        detail = details_by_layer[index] if details_by_layer is not None else {}
        def norm_series(key):
            value = detail.get(key)
            return floats(value.float().norm(dim=-1)[0]) if value is not None else None
        activation = detail.get("ffn_activation")
        layers.append({
            "attention_by_head": floats(attention),
            "hidden_norms": floats(hidden.float().norm(dim=-1)[0]),
            "attention_delta_norms": norm_series("attention_delta"),
            "ffn_delta_norms": norm_series("ffn_delta"),
            "ffn_activation_rms": (floats(activation.float().square().mean(dim=-1).sqrt()[0])
                                   if activation is not None else None),
        })
    return {"token_ids": [int(x) for x in token_ids],
            "tokens": [str(x) for x in token_labels],
            "selected_index": selected_index,
            "layers": layers,
            "capture_kind": captures.get("capture_kind",
                                         "native-full" if details_by_layer is not None else "attention-hidden")}
