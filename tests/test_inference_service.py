from pathlib import Path

import pytest

from viz.inference_service import (InferenceService, checkpoint_catalog,
                                   open_weight_catalog, resolve_checkpoint,
                                   resolve_open_weight, validate_options)


def test_catalog_lists_only_checkpoint_files(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    (root / "demo_step10.pt").write_bytes(b"tiny")
    (root / "notes.txt").write_text("ignore", encoding="utf-8")
    items = checkpoint_catalog(root)
    assert [item["id"] for item in items] == ["demo_step10.pt"]
    assert items[0]["size_bytes"] == 4


def test_checkpoint_resolution_rejects_paths_outside_catalog(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    outside = tmp_path / "secret.pt"
    outside.write_bytes(b"weights")
    with pytest.raises(ValueError):
        resolve_checkpoint("../secret.pt", root)


def test_local_open_weights_are_catalogued_by_opaque_id(tmp_path):
    model_dir = tmp_path / "data" / "incoming" / "pythia70m_step1000"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    items = open_weight_catalog(tmp_path)
    item = next(entry for entry in items if entry["id"] == "hf:pythia70m-step1000")
    assert item["kind"] == "huggingface"
    assert item["size_bytes"] == 7
    service = InferenceService(project_root=tmp_path,
                               checkpoint_root=tmp_path / "checkpoints",
                               train_lock_path=tmp_path / ".train.lock")
    assert item["id"] in [entry["id"] for entry in service.checkpoints()]
    assert resolve_open_weight(item["id"], tmp_path) == model_dir.resolve()
    with pytest.raises(ValueError):
        resolve_open_weight(str(model_dir), tmp_path)


def test_generation_options_are_bounded():
    opts = validate_options({"prompt": "hello", "max_new_tokens": 12,
                             "temperature": 0.7, "top_k": 20, "seed": 9})
    assert opts["max_new_tokens"] == 12
    with pytest.raises(ValueError):
        validate_options({"prompt": "hello", "max_new_tokens": 10000})


def test_cpu_checkpoint_can_generate_a_token_trace(tmp_path):
    import time
    import torch
    from model.transformer import Transformer

    root = tmp_path / "checkpoints"
    root.mkdir()
    cfg = {"vocab_size": 256, "context_length": 8, "embedding_dim": 8,
           "num_heads": 2, "num_layers": 1, "use_rope": False,
           "parallel_residual": True, "tokenizer": None}
    torch.manual_seed(4)
    model = Transformer(vocab_size=256, context_length=8, embedding_dim=8,
                        num_heads=2, num_layers=1)
    torch.save({"cfg": cfg, "model": model.state_dict()}, root / "toy.pt")
    service = InferenceService(project_root=tmp_path, checkpoint_root=root,
                               train_lock_path=tmp_path / ".train.lock")
    loaded = service.load("toy.pt", device="cpu")
    assert loaded["loaded"] is True
    trace = service.inspect({"prompt":"abcd", "selected_index":2, "max_context":8})
    assert trace["capture_kind"] == "native-full"
    assert trace["tokens"][2] == "c"
    assert len(trace["layers"]) == 1
    assert len(trace["layers"][0]["attention_by_head"]) == 2
    task = service.start_generation({"prompt": "a", "max_new_tokens": 2,
                                     "temperature": 0, "top_k": 0, "seed": 3})
    deadline = time.monotonic() + 15
    while service.generation(task["id"])["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    result = service.generation(task["id"])
    assert result["status"] == "completed"
    assert len(result["events"]) == 2
    assert result["text"].startswith("a")
    service.unload()


def test_cuda_load_cannot_overlap_the_train_singleton(tmp_path):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from train.runtime_lock import acquire_train_lock

    root=tmp_path / "checkpoints"
    root.mkdir()
    (root / "demo.pt").write_bytes(b"not reached")
    lock_path=tmp_path / ".train.lock"
    trainer=acquire_train_lock(path=lock_path, owner={"role":"test-trainer"})
    service=InferenceService(project_root=tmp_path, checkpoint_root=root,
                             train_lock_path=lock_path)
    try:
        with pytest.raises(RuntimeError, match="already held"):
            service.load("demo.pt", device="cuda")
    finally:
        trainer.release()


def test_two_checkpoint_compare_restores_the_original_model(tmp_path):
    import torch
    from model.transformer import Transformer
    root=tmp_path / "checkpoints"
    root.mkdir()
    cfg={"vocab_size":256,"context_length":8,"embedding_dim":8,
         "num_heads":2,"num_layers":1,"use_rope":False,
         "parallel_residual":True,"tokenizer":None}
    torch.manual_seed(10)
    first=Transformer(vocab_size=256,context_length=8,embedding_dim=8,num_heads=2,num_layers=1)
    torch.save({"cfg":cfg,"model":first.state_dict()},root/"first.pt")
    torch.manual_seed(11)
    second=Transformer(vocab_size=256,context_length=8,embedding_dim=8,num_heads=2,num_layers=1)
    torch.save({"cfg":cfg,"model":second.state_dict()},root/"second.pt")
    service=InferenceService(project_root=tmp_path,checkpoint_root=root,
                             train_lock_path=tmp_path/".train.lock")
    service.load("first.pt","cpu")
    result=service.compare_checkpoints({"checkpoint_ids":["first.pt","second.pt"],
                                        "prompt":"test"})
    assert result["token_ids_match"] is True
    assert [item["checkpoint_id"] for item in result["models"]]==["first.pt","second.pt"]
    assert len(result["models"][0]["top"])==5
    assert service.status()["checkpoint_id"]=="first.pt"
    service.unload()
