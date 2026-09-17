"""Experiment 002 — first real training loop.

Usage:
    python -m train.train --steps 500 --batch 32 --preset s11m
    python -m train.train --steps 5000 --batch 64 --preset s11m --data tinystories
    python -m train.train --steps 5000 --batch 64 --preset s12m --data tinystories --tokenizer checkpoints/bpe2k.json --use_rope
    python -m train.train --steps 5000 --batch 64 --preset s12m --corpus data/librivox --tokenizer checkpoints/bpe8k.json --use_rope

Presets (tier + params + shape; train.py prints the true count per run):
    t1m      = toy 4-layer (~0.9M, proves the loop works)
    s11m     = 6L384, byte-level vocab 256 (train THIS first)
    m49m     = 6L384, GPT-2 BPE vocab 50304 (needs BPE tokenizer, later)
    s12m     = 6L384, TinyStories BPE vocab ~2256 (the efficiency win)

Data:
    default  = synthetic byte stream (no downloads, proves loss decreases)
    tinystories = TinyStories train split, byte-encoded (real English words,
                 still byte tokens — better signal before you build BPE)
    + --tokenizer checkpoints/bpe2k.json = BPE-encoded instead of bytes.
      First run pays a one-time encode (~20min, cached to --tok_cache);
      later runs load the cache in seconds.

Checkpoints land in checkpoints/exp002_<preset>_step<N>.pt
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

# Allow `python -m train.train` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import L112M_14L768, L194M_14L1024, M49M_6L384, M50M_10L640, M52M_10L640, M60M_10L640, S11M_6L384, S12M_6L384, S17M_6L384, T1M_4L128
from model.bpe import BPETokenizer
from model.transformer import Transformer

PRESETS = {"t1m": T1M_4L128, "s11m": S11M_6L384, "m49m": M49M_6L384,
           "s12m": S12M_6L384, "s17m": S17M_6L384,
           "m50m": M50M_10L640, "m52m": M52M_10L640, "m60m": M60M_10L640,
           "l194m": L194M_14L1024,
           "l112m": L112M_14L768}



def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synthetic_batch(batch: int, ctx: int, vocab: int, device) -> torch.Tensor:
    # Random bytes with local structure (repeating motifs) so there is
    # something learnable — pure uniform noise trains to flat loss.
    data = torch.randint(0, vocab, (batch, ctx + 1), device=device)
    # Inject a repeat rule: every 16th token copies the token 8 back.
    # A working transformer should beat chance on this.
    data[:, 16::16] = data[:, 8:-7:16]
    return data


def resolve_corpus_files(corpus_path: str) -> list[str]:
    # --corpus accepts a .txt file or a dir of .txt files (book chapters).
    # Dirs sort for determinism; nested dirs included for LibriVox-style trees.
    import glob
    if os.path.isfile(corpus_path):
        return [corpus_path]
    if os.path.isdir(corpus_path):
        files = sorted(glob.glob(os.path.join(corpus_path, "**", "*.txt"),
                                 recursive=True))
        if not files:
            raise FileNotFoundError(f"no .txt files under {corpus_path}")
        return files
    raise FileNotFoundError(f"corpus not found: {corpus_path}")


def load_bytes_corpus(files: list[str]) -> torch.Tensor:
    # Raw utf-8 bytes concatenated — any text file or dir works.
    import numpy as _np
    parts = []
    for path in files:
        with open(path, "rb") as f:
            raw = f.read()
        print(f"  {path}: {len(raw) / 1e6:.1f}M bytes")
        parts.append(_np.frombuffer(raw, dtype="uint8").copy())
    return torch.from_numpy(_np.concatenate(parts))


def load_bpe_corpus(tok: BPETokenizer, files: list[str],
                    cache_pt: str) -> torch.Tensor:
    # Encode once, reuse forever. Cache key includes corpus+vocab names
    # so TinyStories_bpe2k.pt and librivox_bpe8k.pt never collide.
    if os.path.exists(cache_pt):
        print(f"loading BPE cache {cache_pt} ...")
        return torch.load(cache_pt, map_location="cpu", weights_only=True)
    print(f"encoding {len(files)} file(s) -> {cache_pt} (one-time cost) ...")
    ids: list[int] = []
    for path in files:
        ids.extend(encode_file_lines(path, tok))
    print(f"  {len(ids) / 1e6:.1f}M tokens")
    t = torch.tensor(ids, dtype=torch.int32)
    os.makedirs(os.path.dirname(cache_pt) or ".", exist_ok=True)
    torch.save(t, cache_pt)
    return t


def default_bpe_cache(corpus_files: list[str], tok_path: str) -> str:
    # data/<corpus-stem>_<tok-stem>.pt — e.g. data/librivox_bpe8k.pt.
    # Single file: stem of the file. Dir: stem of the dir.
    first = corpus_files[0]
    if len(corpus_files) == 1:
        corpus_stem = os.path.splitext(os.path.basename(first))[0]
    else:
        # common dir name, e.g. data/librivox/*.txt -> librivox
        corpus_stem = os.path.basename(os.path.dirname(os.path.commonprefix(corpus_files)))
        if not corpus_stem:
            corpus_stem = "mixed"
    tok_stem = os.path.splitext(os.path.basename(tok_path))[0]
    return os.path.join("data", f"{corpus_stem}_{tok_stem}.pt")


def split_corpus(corpus: torch.Tensor, val_frac: float,
                 ctx: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    # Closed-book tail split. A val tail shorter than one window can't
    # even fill a batch — val goes off with a warning instead of a
    # cryptic randint crash deep in the loop.
    cut = int(len(corpus) * (1.0 - val_frac))
    train, val = corpus[:cut], corpus[cut:]
    if len(val) < ctx + 1:
        print(f"  val tail too small ({len(val)} < ctx+1={ctx + 1}) — val off")
        val = None
    if len(train) < ctx + 1:
        raise SystemExit(
            f"train corpus too small ({len(train)} < ctx+1={ctx + 1}) — "
            f"need a bigger --corpus for ctx={ctx}")
    return train, val


def load_tinystories(ctx: int, device, cache="data/TinyStories.txt") -> torch.Tensor | None:
    """Download TinyStories once, cache as raw text, return byte-encoded tensor."""
    if not os.path.exists(cache):
        try:
            import urllib.request
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            url = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
                   "resolve/main/TinyStoriesV2-GPT4-train.txt")
            print(f"downloading TinyStories (~2GB) -> {cache} ...")
            urllib.request.urlretrieve(url, cache)
        except Exception as e:
            print(f"download failed ({e}); falling back to synthetic data")
            return None
    print(f"loading {cache} ...")
    with open(cache, "rb") as f:
        raw = f.read()
    print(f"  {len(raw) / 1e6:.1f}M bytes")
    return torch.from_numpy(
        __import__("numpy").frombuffer(raw, dtype="uint8").copy()
    )


def encode_file_lines(path: str, tok: BPETokenizer, limit: int = 0) -> list[int]:
    # File -> flat BPE ids, blank lines skipped. limit=0 means the whole file.
    ids: list[int] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if line:
                ids.extend(tok.encode(line))
    return ids


def load_tinystories_bpe(tok: BPETokenizer, device,
                         cache_txt="data/TinyStories.txt",
                         cache_pt="data/TinyStories_bpe2k.pt") -> torch.Tensor | None:
    # Encode once, reuse forever: 2.2GB through pure-Python BPE takes
    # ~20min, but the int32 cache loads in seconds and is never rewritten.
    if os.path.exists(cache_pt):
        print(f"loading BPE cache {cache_pt} ...")
        return torch.load(cache_pt, map_location="cpu", weights_only=True)
    if not os.path.exists(cache_txt):
        print(f"missing {cache_txt}; falling back to synthetic data")
        return None
    print(f"encoding {cache_txt} -> {cache_pt} (one-time cost, go make tea) ...")
    ids = encode_file_lines(cache_txt, tok)
    print(f"  {len(ids) / 1e6:.1f}M tokens")
    t = torch.tensor(ids, dtype=torch.int32)
    torch.save(t, cache_pt)
    return t


def get_batch(source: torch.Tensor | None, batch: int, ctx: int,
              vocab: int, device, pos: list) -> tuple[torch.Tensor, torch.Tensor]:
    if source is None:
        data = synthetic_batch(batch, ctx, vocab, device)
        return data[:, :-1], data[:, 1:]
    # Random crops from the corpus.
    idx = torch.randint(0, len(source) - ctx - 1, (batch,)).tolist()
    x = torch.stack([source[i:i + ctx] for i in idx]).long().to(device)
    y = torch.stack([source[i + 1:i + ctx + 1] for i in idx]).long().to(device)
    return x, y


def lr_schedule(step: int, warmup: int, total: int, peak: float) -> float:
    # Linear warmup, then cosine decay to 10% of peak.
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.1 * peak + 0.9 * peak * 0.5 * (1 + math.cos(math.pi * p))


def early_stop_update(best: float, val: float, bad: int, patience: int,
                      min_delta: float = 1e-4) -> tuple[float, int, bool]:
    # One val check, one decision: better resets, flat burns patience.
    if patience <= 0:
        return best, bad, False
    if val < best - min_delta:
        return val, 0, False
    bad += 1
    return best, bad, bad >= patience


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="s11m", choices=list(PRESETS))
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--data", default="synthetic", choices=["synthetic", "tinystories"])
    ap.add_argument("--corpus", default=None,
                    help="generic corpus: .txt file or dir of .txt files (e.g. data/librivox). Overrides --data when set.")
    ap.add_argument("--tokenizer", default=None,
                    help="BPE vocab json (e.g. checkpoints/bpe2k.json). Unset = bytes.")
    ap.add_argument("--tok_cache", default=None,
                    help="encoded-id cache; built once, loaded after (default: data/<corpus>_<tok>.pt)")
    ap.add_argument("--val_frac", type=float, default=0.01,
                    help="held-out tail fraction for closed-book val (default 0.01)")
    ap.add_argument("--use_rope", action="store_true",
                    help="Exp 003: rotary positions instead of learned absolute")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--val_every", type=int, default=100,
                    help="eval held-out loss every N steps (0 = off)")
    ap.add_argument("--dropout", type=float, default=None,
                    help="residual/attn dropout (default: preset cfg value)")
    ap.add_argument("--patience", type=int, default=0,
                    help="val checks without improvement before stopping (0 = off)")
    ap.add_argument("--min_delta", type=float, default=1e-4,
                    help="val must beat best by this much to reset patience")
    ap.add_argument("--run_dir", default="runs",
                    help="loss.jsonl + samples land here per run")
    ap.add_argument("--accum", type=int, default=1,
                    help="gradient accumulation steps: effective batch = batch * accum")
    ap.add_argument("--resume", default=None,
                    help="checkpoint .pt to resume from (model + optimizer restored, steps continue to --steps)")
    args = ap.parse_args()

    cfg = PRESETS[args.preset]
    if args.dropout is not None:
        # CLI wins: one variable per run, preset stays the control.
        cfg.dropout = args.dropout
    device = get_device()
    tok = None
    if args.tokenizer:
        # File is truth: vocab size follows the tokenizer, not the preset.
        tok = BPETokenizer.load(args.tokenizer)
        cfg.vocab_size = len(tok.vocab)
        print(f"tokenizer={args.tokenizer} vocab={len(tok.vocab)}")
    print(f"preset={args.preset} params~{cfg.num_params() / 1e6:.1f}M "
          f"ctx={cfg.context_length} device={device}")

    torch.manual_seed(0)
    model = Transformer(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        embedding_dim=cfg.embedding_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        use_rope=args.use_rope,
        dropout=cfg.dropout,
    ).to(device)
    model.train()

    # 4070 Ti SUPER: tf32 + fused AdamW is the free speedup.
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    warmup = min(200, args.steps // 10)

    # Resume: weights + optimizer back, step counter continues to --steps.
    # Shape mismatch (e.g. ctx/vocab change) fails loud, not silent.
    start_step = 0
    if args.resume:
        r = torch.load(args.resume, map_location=device, weights_only=False)
        rc = r["cfg"]
        for k in ("vocab_size", "context_length", "embedding_dim",
                  "num_layers", "num_heads"):
            if rc.get(k) != getattr(cfg, k):
                raise ValueError(
                    f"resume mismatch: ckpt {k}={rc.get(k)} vs cfg {k}={getattr(cfg, k)} "
                    f"(use the same --preset/--tokenizer as the original run)")
        model.load_state_dict(r["model"])
        if "optimizer" in r:
            opt.load_state_dict(r["optimizer"])
        start_step = int(r.get("step", 0))
        print(f"resumed {args.resume} @ step {start_step}")

    corpus = None
    train_corpus, val_corpus = None, None
    corpus_label = args.data
    if args.corpus:
        # Generic path: any file or dir. Same closed-book tail split.
        files = resolve_corpus_files(args.corpus)
        corpus_label = os.path.splitext(os.path.basename(
            args.corpus.rstrip("/")))[0] if os.path.isfile(args.corpus) \
            else os.path.basename(os.path.normpath(args.corpus))
        print(f"corpus={args.corpus} ({len(files)} file(s))")
        if tok is not None:
            cache_pt = args.tok_cache or default_bpe_cache(files, args.tokenizer)
            corpus = load_bpe_corpus(tok, files, cache_pt)
            unit = "tokens"
        else:
            corpus = load_bytes_corpus(files)
            unit = "bytes"
        train_corpus, val_corpus = split_corpus(corpus, args.val_frac,
                                               cfg.context_length)
        print(f"  train {len(train_corpus) / 1e6:.1f}M {unit} / "
              f"val {len(val_corpus) / 1e6:.1f}M {unit}" if val_corpus is not None
              else f"  train {len(train_corpus) / 1e6:.1f}M {unit} / val off")
    elif args.data == "tinystories":
        if tok is not None:
            # BPE path: ids, not bytes. Same 99/1 closed-book split.
            cache_pt = args.tok_cache or "data/TinyStories_bpe2k.pt"
            corpus = load_tinystories_bpe(tok, device, cache_pt=cache_pt)
            unit = "tokens"
        else:
            corpus = load_tinystories(cfg.context_length, device)
            unit = "bytes"
        if corpus is None:
            print("TinyStories unavailable, using synthetic.")
        else:
            # Exp 002b: closed-book exam. Last tail is never trained on —
            # if train loss drops but val stalls, it's memorizing.
            train_corpus, val_corpus = split_corpus(corpus, args.val_frac,
                                                   cfg.context_length)
            print(f"  train {len(train_corpus) / 1e6:.1f}M {unit} / "
                  f"val {len(val_corpus) / 1e6:.1f}M {unit}" if val_corpus is not None
                  else f"  train {len(train_corpus) / 1e6:.1f}M {unit} / val off")

    # Visibility layer: every run gets its own dir with loss.jsonl + samples.
    import datetime
    import json
    tag = f"{args.preset}{'_rope' if args.use_rope else ''}"
    if tok is not None:
        # Run dir says which vocab it trained on: bpe2k_rope, not just rope.
        tag += "_" + os.path.splitext(os.path.basename(args.tokenizer))[0]
    run_name = f"{datetime.datetime.now():%Y%m%d_%H%M}_{tag}_{corpus_label}"
    run_stamp = run_name.split("_")[0] + run_name.split("_")[1]
    run_path = os.path.join(args.run_dir, run_name)
    os.makedirs(os.path.join(run_path, "samples"), exist_ok=True)
    log_f = open(os.path.join(run_path, "loss.jsonl"), "w")
    json.dump({"args": vars(args), "cfg": vars(cfg), "params_m": cfg.num_params() / 1e6},
              log_f)
    log_f.write("\n")
    print(f"  run dir: {run_path}")

    # AMP: mixed-precision forward (fp16/bf16) + fp32 gradients.
    # Free speedup on CUDA — ~1.5-2x throughput, negligible quality impact.
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    accum = args.accum
    if accum > 1:
        print(f"  gradient accumulation: {accum} micro-batches, effective batch = {args.batch * accum}")

    @torch.no_grad()
    def eval_loss(data, batches: int = 10) -> float:
        model.eval()
        total = 0.0
        for _ in range(batches):
            x, y = get_batch(data, args.batch, cfg.context_length,
                             cfg.vocab_size, device, pos)
            with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=use_amp):
                total += F.cross_entropy(
                    model(x).reshape(-1, cfg.vocab_size), y.reshape(-1)).item()
        model.train()
        return total / batches

    @torch.no_grad()
    def write_samples(step: int):
        # Fixed prompts + fixed seed = comparable across runs and steps.
        # This is where you HEAR it learning.
        import torch.nn.functional as F_
        prompts = ["Once there was a princess", "Lily played with"]
        lines = [f"--- step {step} ---"]
        for p in prompts:
            if tok is not None:
                # BPE ids in, BPE decode out. Same fixed prompts, fair compare.
                ids = tok.encode(p)
            else:
                ids = list(p.encode("utf-8"))
            x = torch.tensor([ids], dtype=torch.long, device=device)
            torch.manual_seed(0)
            with torch.no_grad():
                for _ in range(80):
                    nxt_logits = model(x[:, -cfg.context_length:])[0, -1] / 0.8
                    topk = torch.topk(nxt_logits, 40)
                    probs = F_.softmax(
                        torch.full_like(nxt_logits, float("-inf")).scatter(
                            0, topk.indices, topk.values), dim=-1)
                    nxt = torch.multinomial(probs, 1)
                    x = torch.cat([x, nxt.view(1, 1)], dim=1)
            if tok is not None:
                text = tok.decode([int(i) for i in x[0].tolist()])
            else:
                text = bytes(b % 256 for b in x[0].tolist()).decode(
                    "utf-8", errors="replace")
            lines += [f"> {p}", text, ""]
        with open(os.path.join(run_path, "samples", f"step{step}.txt"), "w",
                  encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"  samples -> samples/step{step}.txt")

    pos = [0]
    os.makedirs(args.out, exist_ok=True)
    running = 0.0
    # Early-stop ledger: best val seen, strikes since, stop flag.
    best_val, bad_checks = float("inf"), 0

    for step in range(start_step + 1, args.steps + 1):
        lr = lr_schedule(step, warmup, args.steps, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr

        # Train split only — val split is never trained on.
        # Gradient accumulation: accumulate over N micro-batches, step once.
        # Loss is scaled by 1/accum so the gradient magnitude is correct.
        src = train_corpus if train_corpus is not None else corpus
        opt.zero_grad(set_to_none=True)
        micro_loss = 0.0
        for micro in range(accum):
            x, y = get_batch(src, args.batch, cfg.context_length,
                             cfg.vocab_size, device, pos)
            with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1)) / accum
            scaler.scale(loss).backward()
            micro_loss += loss.item()

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        loss_val = micro_loss  # accumulated loss (already scaled by 1/accum)

        # Val check on the held-out tail.
        val, stop = None, False
        if val_corpus is not None and args.val_every and step % args.val_every == 0:
            val = eval_loss(val_corpus)
            if args.patience > 0:
                # New best: snapshot it, reset strikes. Flat: burn patience.
                improved = val < best_val - args.min_delta
                best_val, bad_checks, stop = early_stop_update(
                    best_val, val, bad_checks, args.patience, args.min_delta)
                if improved:
                    best_ckpt = os.path.join(
                        args.out, f"exp002_{tag}_{run_stamp}_best.pt")
                    saved_best = dict(vars(cfg))
                    saved_best["use_rope"] = args.use_rope
                    saved_best["tokenizer"] = args.tokenizer
                    saved_best["corpus"] = args.corpus or args.data
                    torch.save({"cfg": saved_best, "model": model.state_dict(),
                                "optimizer": opt.state_dict(),
                                "step": step, "val": val}, best_ckpt)
                    print(f"  new best val={val:.4f} -> {best_ckpt}")
                if stop:
                    print(f"  early stop: val flat for {args.patience} checks "
                          f"(best={best_val:.4f} @ step {step})")

        running += (loss_val - running) / min(step, 50)
        log_f.write(json.dumps({"step": step, "train": round(loss_val, 4),
                                "avg50": round(running, 4),
                                "val": round(val, 4) if val else None,
                                "lr": lr}) + "\n")
        if step % 25 == 0 or step == 1:
            vstr = f" val={val:.4f}" if val else ""
            print(f"step {step:5d}/{args.steps} loss={loss_val:.4f} "
                  f"avg50={running:.4f}{vstr} lr={lr:.1e}", flush=True)
        if step % 500 == 0 or step == args.steps:
            ckpt = os.path.join(
                args.out, f"exp002_{tag}_{run_stamp}_step{step}.pt")
            saved_cfg = dict(vars(cfg))
            saved_cfg["use_rope"] = args.use_rope
            # Generate needs this to speak the same vocab. Bytes runs: None.
            saved_cfg["tokenizer"] = args.tokenizer
            saved_cfg["corpus"] = args.corpus or args.data
            torch.save({"cfg": saved_cfg, "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "step": step}, ckpt)
            print(f"  saved {ckpt}")
            write_samples(step)
        if stop:
            # Patience spent: the keeper is the best ckpt, not this one.
            log_f.write(json.dumps({"early_stop": True, "step": step,
                                    "best_val": round(best_val, 4)}) + "\n")
            break

    log_f.close()
    print(f"done. final avg50 loss={running:.4f}  run dir: {run_path}")


if __name__ == "__main__":
    main()
