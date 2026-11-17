from pathlib import Path
from types import SimpleNamespace

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


def _write_sharded_qwen(root, *, shard_names=None):
    import json
    model_dir = root / "data" / "incoming" / "qwen3-1.7b"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
        "vocab_size": 151936, "hidden_size": 2048,
        "intermediate_size": 6144, "num_hidden_layers": 28,
        "num_attention_heads": 16, "num_key_value_heads": 8,
        "max_position_embeddings": 40960, "torch_dtype": "bfloat16",
    }), encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    names = shard_names or ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    (model_dir / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 8},
        "weight_map": {"a.weight": names[0], "b.weight": names[1]},
    }), encoding="utf-8")
    for name in names:
        if ".." not in name and not Path(name).is_absolute():
            (model_dir / name).write_bytes(b"weights")
    return model_dir


def test_sharded_qwen_snapshot_is_catalogued_and_resolved(tmp_path):
    model_dir = _write_sharded_qwen(tmp_path)
    items = open_weight_catalog(tmp_path)
    item = next(entry for entry in items if entry["id"] == "hf:qwen3-1.7b")
    assert item["kind"] == "huggingface"
    assert resolve_open_weight(item["id"], tmp_path) == model_dir.resolve()


def test_sharded_qwen_catalog_rejects_missing_shard(tmp_path):
    model_dir = _write_sharded_qwen(tmp_path)
    (model_dir / "model-00002-of-00002.safetensors").unlink()
    assert "hf:qwen3-1.7b" not in [item["id"] for item in open_weight_catalog(tmp_path)]
    with pytest.raises(ValueError, match="shard"):
        resolve_open_weight("hf:qwen3-1.7b", tmp_path)


def test_sharded_qwen_catalog_rejects_escaping_symlink(tmp_path):
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside")
    model_dir = _write_sharded_qwen(tmp_path)
    shard = model_dir / "model-00002-of-00002.safetensors"
    shard.unlink()
    shard.symlink_to(outside)
    assert "hf:qwen3-1.7b" not in [item["id"] for item in open_weight_catalog(tmp_path)]
    with pytest.raises(ValueError, match="shard"):
        resolve_open_weight("hf:qwen3-1.7b", tmp_path)


def test_sharded_qwen_catalog_rejects_escaping_index_symlink(tmp_path):
    import json
    model_dir = _write_sharded_qwen(tmp_path)
    external = tmp_path / "outside-index.json"
    external.write_text(json.dumps({"weight_map": {
        "a.weight": "model-00001-of-00002.safetensors",
    }}), encoding="utf-8")
    index = model_dir / "model.safetensors.index.json"
    index.unlink()
    index.symlink_to(external)
    assert "hf:qwen3-1.7b" not in [item["id"] for item in open_weight_catalog(tmp_path)]
    with pytest.raises(ValueError, match="shard index"):
        resolve_open_weight("hf:qwen3-1.7b", tmp_path)


def test_sharded_qwen_catalog_rejects_path_traversal(tmp_path):
    import json
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside")
    model_dir = _write_sharded_qwen(tmp_path)
    index = model_dir / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a.weight": "../outside.safetensors"}}),
                     encoding="utf-8")
    assert "hf:qwen3-1.7b" not in [item["id"] for item in open_weight_catalog(tmp_path)]
    with pytest.raises(ValueError, match="shard"):
        resolve_open_weight("hf:qwen3-1.7b", tmp_path)


def test_sharded_qwen_catalog_rejects_wrong_architecture(tmp_path):
    import json
    model_dir = _write_sharded_qwen(tmp_path)
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["architectures"] = ["NotQwen3"]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert "hf:qwen3-1.7b" not in [item["id"] for item in open_weight_catalog(tmp_path)]
    with pytest.raises(ValueError, match="architecture"):
        resolve_open_weight("hf:qwen3-1.7b", tmp_path)


def test_devices_reports_mps_availability(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", SimpleNamespace(is_available=lambda: True), raising=False)
    service = InferenceService(project_root=tmp_path,
                               checkpoint_root=tmp_path / "checkpoints",
                               train_lock_path=tmp_path / ".train.lock")
    devices = service.devices()
    assert devices["mps_available"] is True
    assert devices["mps_held_by_visualizer"] is False


def test_mps_status_exposes_current_driver_and_limit_bytes(monkeypatch, tmp_path):
    import torch
    mps = SimpleNamespace(current_allocated_memory=lambda: 100,
                          driver_allocated_memory=lambda: 200,
                          recommended_max_memory=lambda: 300)
    monkeypatch.setattr(torch, "mps", mps, raising=False)
    service = InferenceService(train_lock_path=tmp_path / ".train.lock")
    service._model = torch.nn.Linear(1, 1)
    service._device = "mps"
    status = service.status()
    assert status["mps_memory"] == {"current_allocated_bytes": 100,
                                    "driver_allocated_bytes": 200,
                                    "recommended_max_bytes": 300}


def test_idle_mps_status_exposes_allocator_baseline(monkeypatch, tmp_path):
    import torch
    from types import SimpleNamespace

    stats = {"current_allocated_bytes": 64, "driver_allocated_bytes": 1024,
             "recommended_max_bytes": 4096}
    monkeypatch.setattr(torch, "mps", SimpleNamespace(
        current_allocated_memory=lambda: stats["current_allocated_bytes"],
        driver_allocated_memory=lambda: stats["driver_allocated_bytes"],
        recommended_max_memory=lambda: stats["recommended_max_bytes"]), raising=False)
    monkeypatch.setattr(torch.backends, "mps", SimpleNamespace(is_available=lambda: True), raising=False)
    service = InferenceService(train_lock_path=tmp_path / ".train.lock")
    status = service.status()
    assert status["loaded"] is False
    assert status["mps_memory"] == stats


def test_qwen_prompt_uses_chat_template_when_available():
    from viz.inference_service import _encode_prompt
    class FakeTokenizer:
        chat_template = "configured"
        def __init__(self):
            self.calls = []
        def apply_chat_template(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return {"input_ids": [10, 11, 12], "attention_mask": [1, 1, 1]}
        def encode(self, prompt):
            raise AssertionError("Qwen prompt must use its chat template")

    tokenizer = FakeTokenizer()
    ids = _encode_prompt("Hi", tokenizer, {"use_chat_template": True})
    assert ids == [10, 11, 12]
    assert tokenizer.calls == [([{"role": "user", "content": "Hi"}], {
        "tokenize": True, "add_generation_prompt": True, "enable_thinking": False})]


def test_mps_loader_sets_verified_dtype_and_separate_context_caps(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import torch
    import viz.inference_service as inference_service

    captured = {}
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.config = SimpleNamespace(
                vocab_size=151936, max_position_embeddings=40960, hidden_size=2048,
                num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8)
        def eval(self):
            return super().eval()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            captured.update(kwargs)
            return FakeModel()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return "fake-tokenizer"

    monkeypatch.setattr(inference_service, "_probe_mps_dtype", lambda _torch: torch.bfloat16)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=FakeAutoModel, AutoTokenizer=FakeAutoTokenizer))
    model, config, tokenizer = inference_service.load_open_weight_model(tmp_path, "mps")
    assert captured["dtype"] is torch.bfloat16
    assert captured["device_map"] == {"": "mps"}
    assert captured["attn_implementation"] == "eager"
    assert captured["local_files_only"] is True
    assert captured["trust_remote_code"] is False
    assert config["context_length"] == 4096
    assert config["native_context_length"] == 40960
    assert config["num_layers"] == 28 and config["num_heads"] == 16
    assert config["num_key_value_heads"] == 8
    assert config["dtype"] == "torch.bfloat16"
    assert tokenizer == "fake-tokenizer"
    assert model.training is False


def test_mps_dtype_probe_supports_qwen_sized_ops():
    import torch
    from viz.inference_service import _probe_mps_dtype
    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    dtype = _probe_mps_dtype(torch)
    assert dtype in (torch.bfloat16, torch.float16)


def test_mps_model_holds_and_releases_training_singleton(tmp_path, monkeypatch):
    import torch
    import viz.inference_service as inference_service
    from train.runtime_lock import TrainLockError, acquire_train_lock
    _write_sharded_qwen(tmp_path)
    monkeypatch.setattr(torch.backends, "mps", SimpleNamespace(is_available=lambda: True), raising=False)
    monkeypatch.setattr(inference_service, "load_open_weight_model", lambda *_: (
        torch.nn.Linear(2, 2), {"context_length": 4096}, None))
    lock_path = tmp_path / ".train.lock"
    service = InferenceService(project_root=tmp_path,
                               checkpoint_root=tmp_path / "checkpoints",
                               train_lock_path=lock_path)
    state = service.load("hf:qwen3-1.7b", device="mps")
    assert state["device"] == "mps"
    assert service.devices()["mps_held_by_visualizer"] is True
    with pytest.raises(TrainLockError, match="already held"):
        acquire_train_lock(path=lock_path, owner={"role": "test-trainer"})
    service.unload()
    trainer = acquire_train_lock(path=lock_path, owner={"role": "test-trainer"})
    trainer.release()


def test_mps_load_failure_releases_training_singleton(tmp_path, monkeypatch):
    import torch
    import viz.inference_service as inference_service
    from train.runtime_lock import acquire_train_lock
    _write_sharded_qwen(tmp_path)
    monkeypatch.setattr(torch.backends, "mps", SimpleNamespace(is_available=lambda: True), raising=False)
    monkeypatch.setattr(inference_service, "load_open_weight_model",
                        lambda *_: (_ for _ in ()).throw(RuntimeError("bad MPS kernel")))
    lock_path = tmp_path / ".train.lock"
    service = InferenceService(project_root=tmp_path,
                               checkpoint_root=tmp_path / "checkpoints",
                               train_lock_path=lock_path)
    with pytest.raises(RuntimeError, match="bad MPS kernel"):
        service.load("hf:qwen3-1.7b", device="mps")
    trainer = acquire_train_lock(path=lock_path, owner={"role": "test-trainer"})
    trainer.release()


def test_mps_load_rejects_unavailable_backend_without_lock(tmp_path, monkeypatch):
    import torch
    from train.runtime_lock import acquire_train_lock
    _write_sharded_qwen(tmp_path)
    monkeypatch.setattr(torch.backends, "mps", SimpleNamespace(is_available=lambda: False), raising=False)
    lock_path = tmp_path / ".train.lock"
    service = InferenceService(project_root=tmp_path,
                               checkpoint_root=tmp_path / "checkpoints",
                               train_lock_path=lock_path)
    with pytest.raises(RuntimeError, match="MPS is not available"):
        service.load("hf:qwen3-1.7b", device="mps")
    trainer = acquire_train_lock(path=lock_path, owner={"role": "test-trainer"})
    trainer.release()


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
    generated = bytes(event["token_id"] % 256 for event in result["events"]).decode("utf-8", errors="replace")
    assert result["text"] == generated
    assert result["prompt_tokens"] == 1
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
