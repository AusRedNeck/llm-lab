import torch

from model.transformer import Transformer
from viz.trace import build_trace


def test_native_trace_exposes_selected_token_pathway_and_is_bounded():
    torch.manual_seed(4)
    model = Transformer(vocab_size=32, context_length=16, embedding_dim=16,
                        num_heads=4, num_layers=2)
    model.eval()
    token_ids = [2, 5, 7, 11, 13]
    x = torch.tensor([token_ids])
    with torch.inference_mode():
        _, captures = model.forward_with_capture(x)
    trace = build_trace(captures, token_ids, [f"t{i}" for i in token_ids],
                        selected_index=3, max_context=8)
    assert trace["selected_index"] == 3
    assert trace["tokens"] == ["t2", "t5", "t7", "t11", "t13"]
    assert len(trace["layers"]) == 2
    assert len(trace["layers"][0]["attention_by_head"]) == 4
    assert len(trace["layers"][0]["attention_by_head"][0]) == 5
    assert len(trace["layers"][0]["hidden_norms"]) == 5
    assert len(trace["layers"][0]["attention_delta_norms"]) == 5
    assert len(trace["layers"][0]["ffn_delta_norms"]) == 5
    assert len(trace["layers"][0]["ffn_activation_rms"]) == 5
    assert trace["layers"][0]["attention_by_head"][0][4] == 0


def test_open_weight_style_capture_marks_branch_details_unavailable():
    torch.manual_seed(5)
    model = Transformer(vocab_size=16, context_length=8, embedding_dim=8,
                        num_heads=2, num_layers=1)
    model.eval()
    tokens = torch.tensor([[2, 4, 6]])
    with torch.inference_mode():
        _, full = model.forward_with_capture(tokens)
    partial = {"attentions": tuple(full["layer_attention"]),
               "hidden_states": (full["embeddings"], *full["layer_hiddens"]),
               "capture_kind": "open-weights-attention-hidden"}
    trace = build_trace(partial, [2, 4, 6], ["a", "b", "c"],
                        selected_index=2, max_context=8)
    assert trace["capture_kind"] == "open-weights-attention-hidden"
    assert trace["layers"][0]["hidden_norms"]
    assert trace["layers"][0]["ffn_delta_norms"] is None


def test_trace_rejects_context_and_position_overflow():
    with torch.inference_mode():
        model = Transformer(vocab_size=16, context_length=8, embedding_dim=8,
                            num_heads=2, num_layers=1)
        _, captures = model.forward_with_capture(torch.tensor([[1, 2, 3, 4, 5]]))
    ids = [1, 2, 3, 4, 5]
    labels = [str(i) for i in ids]
    try:
        build_trace(captures, ids, labels, selected_index=5, max_context=4)
    except ValueError as exc:
        assert "selected_index" in str(exc)
    else:
        raise AssertionError("out-of-range token position must be rejected")

    try:
        build_trace(captures, ids, labels, selected_index=0, max_context=4)
    except ValueError as exc:
        assert "max_context" in str(exc)
    else:
        raise AssertionError("oversized trace must be rejected, not silently truncated")
