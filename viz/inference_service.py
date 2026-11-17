"""Checkpoint catalog and bounded generation helpers for the Think workspace."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import math
import threading
import uuid

LAB_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = LAB_ROOT / "checkpoints"
MAX_PROMPT_CHARS = 4000
MAX_NEW_TOKENS = 128
OPEN_WEIGHT_SOURCES = {
    "hf:pythia70m-step1000": ("data/incoming/pythia70m_step1000", "EleutherAI Pythia-70M / step 1000"),
    "hf:pythia70m-reference": ("data/incoming/pythia70m_weights", "EleutherAI Pythia-70M / reference weights"),
    "hf:qwen3-1.7b": ("data/incoming/qwen3-1.7b", "Qwen3-1.7B / instruction tuned"),
}
OPEN_WEIGHT_RUNTIME_CONTEXT = 4096
QWEN3_17B_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
    "vocab_size": 151936, "hidden_size": 2048, "intermediate_size": 6144,
    "num_hidden_layers": 28, "num_attention_heads": 16,
    "num_key_value_heads": 8, "max_position_embeddings": 40960,
    "torch_dtype": "bfloat16",
}


def _contained_model_file(folder, relative, description):
    if not isinstance(relative, str) or not relative or chr(92) in relative:
        raise ValueError(f"open-weight {description} path is invalid")
    candidate = (folder / relative).resolve()
    if folder not in candidate.parents or not candidate.is_file():
        raise ValueError(f"open-weight {description} is missing or outside the model directory: {relative}")
    return candidate


def _snapshot_weight_files(folder):
    index = folder / "model.safetensors.index.json"
    if index.exists() or index.is_symlink():
        index = _contained_model_file(folder, index.name, "shard index")
        try:
            manifest = json.loads(index.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("open-weight shard index is unreadable") from exc
        weight_map = manifest.get("weight_map") if isinstance(manifest, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("open-weight shard index has no weight_map")
        names = set()
        for name in weight_map.values():
            if not isinstance(name, str) or not name.endswith(".safetensors"):
                raise ValueError("open-weight shard index contains an invalid shard name")
            names.add(name)
        return [_contained_model_file(folder, name, "shard") for name in sorted(names)]
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = folder / name
        if candidate.is_file():
            return [_contained_model_file(folder, name, "weights")]
    raise ValueError("open-weight model weights or shard index are missing")


def _validate_open_weight_snapshot(folder, model_id=None):
    if not folder.is_dir():
        raise ValueError("open-weight model is not available locally")
    config_path = _contained_model_file(folder, "config.json", "config")
    _contained_model_file(folder, "tokenizer.json", "tokenizer")
    if model_id == "hf:qwen3-1.7b":
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Qwen3-1.7B config is unreadable") from exc
        for key, expected in QWEN3_17B_CONFIG.items():
            if config.get(key) != expected:
                raise ValueError(f"Qwen3-1.7B config mismatch for {key}: expected {expected!r}")
    return _snapshot_weight_files(folder)


def open_weight_catalog(project_root=LAB_ROOT):
    """List only explicitly approved, complete local Hugging Face snapshots."""
    root = Path(project_root).resolve()
    items = []
    for model_id, (relative, label) in OPEN_WEIGHT_SOURCES.items():
        folder = (root / relative).resolve()
        if root not in folder.parents:
            continue
        try:
            weights = _validate_open_weight_snapshot(folder, model_id)
            size_bytes = sum(path.stat().st_size for path in weights)
        except (OSError, ValueError):
            continue
        modified = max(path.stat().st_mtime for path in weights)
        items.append({"id": model_id, "label": label, "kind": "huggingface",
                      "size_bytes": size_bytes,
                      "modified_at": datetime.fromtimestamp(modified).astimezone().isoformat(timespec="seconds")})
    return items


def resolve_open_weight(model_id, project_root=LAB_ROOT):
    """Resolve an approved local HF model after validating every indexed shard."""
    if model_id not in OPEN_WEIGHT_SOURCES:
        raise ValueError("choose an open-weight model from the local catalog")
    root = Path(project_root).resolve()
    folder = (root / OPEN_WEIGHT_SOURCES[model_id][0]).resolve()
    if root not in folder.parents:
        raise ValueError("open-weight model path escapes the project root")
    _validate_open_weight_snapshot(folder, model_id)
    return folder


def checkpoint_catalog(root=CHECKPOINT_ROOT):
    """List trusted project checkpoint files without deserializing their weights."""
    root = Path(root).resolve()
    if not root.is_dir():
        return []
    items = []
    for path in sorted(root.glob("*.pt"), key=lambda p: p.name.lower()):
        try:
            stat = path.stat()
            if path.is_file() and path.resolve().parent == root:
                items.append({"id": path.name, "size_bytes": stat.st_size,
                              "modified_at": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds")})
        except OSError:
            continue
    return items


def resolve_checkpoint(checkpoint_id, root=CHECKPOINT_ROOT):
    """Resolve an opaque checkpoint ID; reject traversal and non-PT assets."""
    if not isinstance(checkpoint_id, str) or not checkpoint_id or Path(checkpoint_id).name != checkpoint_id:
        raise ValueError("choose a checkpoint from the local catalog")
    root = Path(root).resolve()
    path = (root / checkpoint_id).resolve()
    if path.parent != root or path.suffix.lower() != ".pt" or not path.is_file():
        raise ValueError("checkpoint is not in the local catalog")
    return path


def _encode_prompt(prompt, tokenizer, config):
    if tokenizer is None:
        return list(prompt.encode("utf-8"))
    if config.get("use_chat_template") and callable(getattr(tokenizer, "apply_chat_template", None)):
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False)
        token_ids = encoded.get("input_ids") if isinstance(encoded, dict) else getattr(encoded, "input_ids", encoded)
        if token_ids is None:
            raise ValueError("tokenizer chat template returned no input_ids")
        if len(token_ids) and isinstance(token_ids[0], (list, tuple)):
            token_ids = token_ids[0]
        return list(token_ids)
    return list(tokenizer.encode(prompt))


def _probe_mps_dtype(torch):
    """Choose the first low-precision dtype that supports Qwen3-1.7B core ops."""
    failures = []
    for dtype in (torch.bfloat16, torch.float16):
        x = projection = hidden = q = k = weights = None
        try:
            x = torch.randn((1, 2048), device="mps", dtype=dtype)
            projection = torch.randn((6144, 2048), device="mps", dtype=dtype)
            hidden = torch.nn.functional.layer_norm(
                torch.nn.functional.linear(x, projection), (6144,))
            q = torch.randn((1, 16, 128, 128), device="mps", dtype=dtype)
            k = torch.randn((1, 16, 128, 128), device="mps", dtype=dtype)
            weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(128), dim=-1)
            torch.mps.synchronize()
            if not bool(torch.isfinite(hidden).all().item() and torch.isfinite(weights).all().item()):
                raise RuntimeError("MPS produced non-finite values")
            return dtype
        except Exception as exc:
            failures.append(f"{dtype}: {exc}")
        finally:
            del x, projection, hidden, q, k, weights
            torch.mps.empty_cache()
    raise RuntimeError("MPS has no working BF16/FP16 path for Qwen3 ops: " + "; ".join(failures))


def load_open_weight_model(folder, device):
    """Load one complete local HF snapshot; MPS uses verified low precision."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs = {"local_files_only": True, "trust_remote_code": False,
                   "low_cpu_mem_usage": True, "attn_implementation": "eager"}
    selected_dtype = None
    if device == "mps":
        selected_dtype = _probe_mps_dtype(torch)
        load_kwargs["dtype"] = selected_dtype
        load_kwargs["device_map"] = {"": "mps"}
    model = AutoModelForCausalLM.from_pretrained(str(folder), **load_kwargs)
    if device == "cpu":
        model = model.to(dtype=torch.float32)
    elif device == "cuda":
        model = model.to(torch.device(device))
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        str(folder), local_files_only=True, use_fast=True, trust_remote_code=False)
    cfg = model.config
    native_context = int(getattr(cfg, "max_position_embeddings", 512))
    num_heads = int(cfg.num_attention_heads)
    config = {"vocab_size": int(cfg.vocab_size),
              "context_length": min(native_context, OPEN_WEIGHT_RUNTIME_CONTEXT),
              "native_context_length": native_context,
              "embedding_dim": int(cfg.hidden_size),
              "num_layers": int(cfg.num_hidden_layers),
              "num_heads": num_heads,
              "num_key_value_heads": int(getattr(cfg, "num_key_value_heads", num_heads) or num_heads),
              "use_rope": True,
              "tokenizer": str(Path(folder) / "tokenizer.json"),
              "source": "huggingface-open-weights",
              "use_chat_template": bool(getattr(tokenizer, "chat_template", None)),
              "dtype": str(selected_dtype or next(model.parameters()).dtype)}
    return model, config, tokenizer


def validate_options(value):
    """Validate and normalize the bounded prompt-generation payload."""
    if not isinstance(value, dict):
        raise ValueError("request body must be a JSON object")
    prompt = value.get("prompt", "")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f"prompt must contain 1-{MAX_PROMPT_CHARS} characters")
    count = value.get("max_new_tokens", 32)
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must be between 1 and {MAX_NEW_TOKENS}")
    temperature = value.get("temperature", 0.8)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError("temperature must be finite and between 0 and 2")
    top_k = value.get("top_k", 40)
    seed = value.get("seed", 0)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 0 <= top_k <= 1000:
        raise ValueError("top_k must be between 0 and 1000")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 4294967295:
        raise ValueError("seed must be an integer between 0 and 4294967295")
    return {"prompt": prompt, "max_new_tokens": count, "temperature": float(temperature),
            "top_k": top_k, "seed": seed}


class InferenceService:
    """One local checkpoint session and one cancellable-by-exit generation worker."""
    def __init__(self, project_root=LAB_ROOT, checkpoint_root=CHECKPOINT_ROOT,
                 train_lock_path=None):
        self.project_root = Path(project_root).resolve()
        self.checkpoint_root = Path(checkpoint_root).resolve()
        if train_lock_path is None:
            from train.runtime_lock import DEFAULT_LOCK_PATH
            train_lock_path = DEFAULT_LOCK_PATH
        self.train_lock_path = Path(train_lock_path).resolve()
        self._state_lock = threading.RLock()
        self._model = self._tokenizer = self._config = None
        self._checkpoint_id = self._device = self._kind = None
        self._gpu_lock = None
        self._generations = {}
        self._active_generation = None

    def checkpoints(self):
        items = checkpoint_catalog(self.checkpoint_root)
        items.extend(open_weight_catalog(self.project_root))
        return sorted(items, key=lambda item: item["id"].lower())

    def status(self):
        with self._state_lock:
            model_dtype = None
            if self._model is not None:
                try:
                    model_dtype = str(next(self._model.parameters()).dtype)
                except (AttributeError, StopIteration):
                    model_dtype = self._config.get("dtype") if self._config else None
            import torch
            mps_available = (hasattr(torch.backends, "mps")
                             and torch.backends.mps.is_available())
            mps_memory = None
            if self._device == "mps" or mps_available:
                mps_memory = {"current_allocated_bytes": torch.mps.current_allocated_memory(),
                              "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
                              "recommended_max_bytes": torch.mps.recommended_max_memory()}
            return {"loaded": self._model is not None,
                    "checkpoint_id": self._checkpoint_id, "device": self._device,
                    "dtype": model_dtype, "kind": self._kind, "mps_memory": mps_memory,
                    "config": self._public_config(self._config) if self._config else None,
                    "generating": self._active_generation is not None,
                    "generation_id": self._active_generation}

    @staticmethod
    def _public_config(config):
        keys = ("vocab_size", "context_length", "native_context_length",
                "embedding_dim", "num_layers", "num_heads", "num_key_value_heads",
                "use_rope", "rotary_pct", "parallel_residual", "tokenizer")
        return {key: config[key] for key in keys if key in config}

    def devices(self):
        import torch
        busy = False
        held_by_visualizer = self._gpu_lock is not None
        if not held_by_visualizer:
            from train.runtime_lock import TrainLockError, acquire_train_lock
            try:
                probe = acquire_train_lock(path=self.train_lock_path,
                                           owner={"role": "observatory-probe"},
                                           label="visualizer device check")
                probe.release()
            except TrainLockError:
                busy = True
        mps_available = (hasattr(torch.backends, "mps")
                         and torch.backends.mps.is_available())
        return {"cpu_available": True, "cuda_available": torch.cuda.is_available(),
                "mps_available": mps_available, "training_busy": busy,
                "cuda_held_by_visualizer": held_by_visualizer and self._device == "cuda",
                "mps_held_by_visualizer": held_by_visualizer and self._device == "mps"}

    def _tokenizer_path(self, value):
        if not value:
            return None
        raw = Path(value)
        path = (raw if raw.is_absolute() else self.project_root / raw).resolve()
        if self.project_root not in path.parents or not path.is_file():
            raise ValueError("checkpoint tokenizer must be an existing file under the project root")
        return path

    def load(self, checkpoint_id, device="cpu"):
        is_hf = checkpoint_id in OPEN_WEIGHT_SOURCES
        path = (resolve_open_weight(checkpoint_id, self.project_root) if is_hf
                else resolve_checkpoint(checkpoint_id, self.checkpoint_root))
        if device not in ("cpu", "cuda", "mps"):
            raise ValueError("device must be cpu, cuda, or mps")
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in this environment")
        if device == "mps" and not (hasattr(torch.backends, "mps")
                                     and torch.backends.mps.is_available()):
            raise RuntimeError("MPS is not available in this environment")
        with self._state_lock:
            if self._active_generation:
                raise RuntimeError("wait for the current generation before switching models")
            if self._model is not None and self._checkpoint_id == checkpoint_id and self._device == device:
                return self.status()
            self._release_gpu_lock()
            self._clear_model()
            if device in ("cuda", "mps"):
                from train.runtime_lock import acquire_train_lock
                self._gpu_lock = acquire_train_lock(
                    path=self.train_lock_path,
                    owner={"role": "llm-observatory-inference", "checkpoint": checkpoint_id,
                           "device": device},
                    label=f"{device.upper()} inference")
            try:
                if is_hf:
                    model, config, tokenizer = load_open_weight_model(path, device)
                else:
                    from inference.generate import load_model
                    model, config = load_model(str(path), torch.device(device))
                    tok_path = self._tokenizer_path(config.get("tokenizer"))
                    if tok_path:
                        from train.train import load_tokenizer
                        tokenizer = load_tokenizer(str(tok_path))
                    else:
                        tokenizer = None
            except Exception:
                self._release_gpu_lock()
                raise
            self._model, self._config, self._tokenizer = model, config, tokenizer
            self._checkpoint_id, self._device = checkpoint_id, device
            self._kind = "huggingface" if is_hf else "native"
            return self.status()

    def _release_gpu_lock(self):
        if self._gpu_lock is not None:
            self._gpu_lock.release()
            self._gpu_lock = None

    def _clear_model(self):
        self._model = self._tokenizer = self._config = None
        self._checkpoint_id = self._device = self._kind = None

    def unload(self):
        with self._state_lock:
            if self._active_generation:
                raise RuntimeError("wait for the current generation before unloading")
            self._clear_model()
            self._release_gpu_lock()
            return self.status()

    def start_generation(self, value):
        options = validate_options(value)
        with self._state_lock:
            if self._model is None:
                raise RuntimeError("load a checkpoint before generating")
            if self._active_generation:
                raise RuntimeError("one generation is already running")
            task_id = uuid.uuid4().hex
            task = {"id": task_id, "status": "running", "text": "",
                    "events": [], "error": None, "prompt_tokens": 0,
                    "prompt_truncated": False}
            self._generations[task_id] = task
            self._active_generation = task_id
            worker = threading.Thread(target=self._run_generation,
                                      args=(task_id, options), daemon=True)
            worker.start()
            return {"id": task_id, "status": "running"}

    def generation(self, task_id, after=0):
        if isinstance(after, bool) or not isinstance(after, int) or after < 0:
            raise ValueError("event cursor must be a non-negative integer")
        with self._state_lock:
            task = self._generations.get(task_id)
            if task is None:
                raise KeyError("generation not found")
            return {"id": task_id, "status": task["status"], "text": task["text"],
                    "events": [dict(event) for event in task["events"][after:]],
                    "cursor": len(task["events"]), "error": task["error"],
                    "prompt_tokens": task["prompt_tokens"],
                    "prompt_truncated": task["prompt_truncated"]}

    def inspect(self, value):
        """Capture one bounded prompt pass; optional and separate from fast generation."""
        from viz.trace import DEFAULT_MAX_CONTEXT, build_trace
        if not isinstance(value, dict):
            raise ValueError("trace request must be a JSON object")
        prompt = value.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"prompt must contain 1-{MAX_PROMPT_CHARS} characters")
        max_context = value.get("max_context", DEFAULT_MAX_CONTEXT)
        if isinstance(max_context, bool) or not isinstance(max_context, int) or not 1 <= max_context <= 128:
            raise ValueError("max_context must be between 1 and 128")
        with self._state_lock:
            if self._model is None:
                raise RuntimeError("load a checkpoint before inspecting a trace")
            if self._active_generation:
                raise RuntimeError("wait for the current generation before inspecting a trace")
            model, tokenizer, config, device, kind = self._model, self._tokenizer, self._config, self._device, self._kind
        token_ids = _encode_prompt(prompt, tokenizer, config)
        if not token_ids:
            raise ValueError("prompt produced no tokens")
        if len(token_ids) > max_context:
            raise ValueError(f"prompt has {len(token_ids)} tokens; trace cap is {max_context}")
        selected = value.get("selected_index", len(token_ids) - 1)
        if isinstance(selected, bool) or not isinstance(selected, int) or not 0 <= selected < len(token_ids):
            raise ValueError("selected_index must identify a prompt token")
        import torch
        x = torch.tensor([token_ids], dtype=torch.long, device=device)
        labels = [self._decode_ids([token_id], tokenizer) for token_id in token_ids]
        with torch.inference_mode():
            if kind == "native":
                from viz.hooks import capture_forward
                captures = capture_forward(model, x)
            else:
                result = model(input_ids=x, output_attentions=True,
                               output_hidden_states=True, use_cache=False,
                               return_dict=True)
                captures = {"attentions": result.attentions,
                            "hidden_states": result.hidden_states,
                            "capture_kind": "open-weights-attention-hidden"}
        trace_config = config if isinstance(config, dict) else {}
        trace = build_trace(captures, token_ids, labels,
                            selected_index=selected, max_context=max_context,
                            expected_layers=trace_config.get("num_layers"),
                            expected_heads=trace_config.get("num_heads"))
        trace["checkpoint_id"] = self._checkpoint_id
        trace["tokenizer"] = trace_config.get("tokenizer")
        return trace

    def compare_checkpoints(self, value):
        """Sequential CPU comparison so the browser never keeps two models resident."""
        if not isinstance(value, dict):
            raise ValueError("comparison request must be a JSON object")
        ids = value.get("checkpoint_ids")
        if not isinstance(ids, list) or len(ids) != 2 or not all(isinstance(item, str) for item in ids):
            raise ValueError("checkpoint_ids must contain exactly two catalog IDs")
        if ids[0] == ids[1]:
            raise ValueError("choose two different checkpoints")
        prompt = value.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"prompt must contain 1-{MAX_PROMPT_CHARS} characters")
        for checkpoint_id in ids:
            if checkpoint_id in OPEN_WEIGHT_SOURCES:
                resolve_open_weight(checkpoint_id, self.project_root)
            else:
                resolve_checkpoint(checkpoint_id, self.checkpoint_root)
        with self._state_lock:
            if self._active_generation:
                raise RuntimeError("wait for the current generation before comparing checkpoints")
            original_id, original_device = self._checkpoint_id, self._device
            if original_device in ("cuda", "mps"):
                raise RuntimeError("checkpoint comparison is CPU-only; unload the accelerator model first")
            summaries = []
            try:
                self.unload()
                for checkpoint_id in ids:
                    self.load(checkpoint_id, "cpu")
                    model, tokenizer, config = self._model, self._tokenizer, self._config
                    tokens = _encode_prompt(prompt, tokenizer, config)
                    if not tokens:
                        raise ValueError("prompt produced no tokens")
                    cap = min(128, int(config["context_length"]))
                    if len(tokens) > cap:
                        raise ValueError(f"prompt exceeds comparison context cap ({cap} tokens)")
                    if min(tokens) < 0 or max(tokens) >= int(config["vocab_size"]):
                        raise ValueError(f"tokenizer IDs exceed vocabulary for {checkpoint_id}")
                    import torch
                    x = torch.tensor([tokens], dtype=torch.long, device="cpu")
                    with torch.inference_mode():
                        output = model(x)
                        logits = (output.logits if hasattr(output, "logits") else output)[0, -1].float()
                        probs = torch.softmax(logits, dim=-1)
                        top_probs, top_ids = torch.topk(probs, min(5, probs.numel()))
                        entropy = float((-(probs * probs.clamp_min(1e-12).log()).sum()).item())
                    summaries.append({"checkpoint_id": checkpoint_id,
                                      "token_ids": tokens,
                                      "top": [{"id": int(token_id.item()),
                                               "text": self._decode_ids([int(token_id.item())], tokenizer),
                                               "probability": float(prob.item())}
                                              for prob, token_id in zip(top_probs, top_ids)],
                                      "entropy": entropy})
                    self.unload()
            finally:
                self.unload()
                if original_id is not None:
                    self.load(original_id, original_device)
        same_tokens = summaries[0]["token_ids"] == summaries[1]["token_ids"]
        same_prediction = (same_tokens and summaries[0]["top"][0]["id"] == summaries[1]["top"][0]["id"])
        return {"prompt": prompt, "token_ids_match": same_tokens,
                "same_top_prediction": same_prediction, "models": summaries}

    def _decode_ids(self, ids, tokenizer, *, skip_special_tokens=False):
        if tokenizer is not None:
            try:
                return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
            except TypeError:
                return tokenizer.decode(ids)
        return bytes(int(token) % 256 for token in ids).decode("utf-8", errors="replace")

    def _run_generation(self, task_id, options):
        try:
            import torch
            with self._state_lock:
                model, tokenizer, config, device = self._model, self._tokenizer, self._config, self._device
            ids = _encode_prompt(options["prompt"], tokenizer, config)
            context = int(config["context_length"])
            truncated = len(ids) > context
            ids = ids[-context:]
            if not ids:
                raise ValueError("prompt produced no tokens")
            if min(ids) < 0 or max(ids) >= int(config["vocab_size"]):
                raise ValueError("tokenizer produced an id outside the checkpoint vocabulary")
            x = torch.tensor([ids], dtype=torch.long, device=device)
            generated_ids = []
            generator = torch.Generator(device=device).manual_seed(options["seed"])
            task = self._generations[task_id]
            task["prompt_tokens"] = len(ids)
            task["prompt_truncated"] = truncated
            for step in range(options["max_new_tokens"]):
                with torch.inference_mode():
                    output = model(x[:, -context:])
                    logits_tensor = output.logits if hasattr(output, "logits") else output
                    logits = logits_tensor[0, -1].float()
                    if options["temperature"] == 0:
                        probs = torch.softmax(logits, dim=-1)
                        nxt = torch.argmax(logits).view(1)
                    else:
                        scores = logits / options["temperature"]
                        k = options["top_k"]
                        if k and k < scores.numel():
                            floor = torch.topk(scores, k).values[-1]
                            scores = scores.masked_fill(scores < floor, float("-inf"))
                        probs = torch.softmax(scores, dim=-1)
                        nxt = torch.multinomial(probs, 1, generator=generator)
                    token_id = int(nxt.item())
                    chosen = float(probs[token_id].item())
                    entropy = float((-(probs * probs.clamp_min(1e-12).log()).sum()).item())
                    n = min(5, int(probs.numel()))
                    top_probs, top_ids = torch.topk(probs, n)
                    top = [{"id": int(tid.item()), "text": self._decode_ids([int(tid.item())], tokenizer),
                            "probability": float(prob.item())}
                           for prob, tid in zip(top_probs, top_ids)]
                prior = self._decode_ids(generated_ids, tokenizer, skip_special_tokens=True)
                generated_ids.append(token_id)
                text = self._decode_ids(generated_ids, tokenizer, skip_special_tokens=True)
                piece = text[len(prior):] if text.startswith(prior) else text
                event = {"step": step + 1, "token_id": token_id, "token": piece,
                         "probability": chosen, "entropy": entropy, "top": top}
                x = torch.cat((x, nxt.reshape(1, 1)), dim=1)
                with self._state_lock:
                    task["text"] = text
                    task["events"].append(event)
            with self._state_lock:
                task["status"] = "completed"
        except Exception as exc:
            with self._state_lock:
                task = self._generations.get(task_id)
                if task is not None:
                    task["status"] = "error"
                    task["error"] = str(exc)
        finally:
            with self._state_lock:
                if self._active_generation == task_id:
                    self._active_generation = None
