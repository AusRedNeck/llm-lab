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

from model.config import L112M_14L768, L194M_14L1024, M49M_6L384, M50M_10L640, M50M_10L640_1K, M52M_10L640, M60M_10L640, PYTHIA_6L512, S11M_6L384, S12M_6L384, S17M_6L384, T1M_4L128
from model.bpe import BPETokenizer
from model.transformer import Transformer

PRESETS = {"t1m": T1M_4L128, "s11m": S11M_6L384, "m49m": M49M_6L384,
           "s12m": S12M_6L384, "s17m": S17M_6L384,
           "m50m": M50M_10L640, "m50m_1k": M50M_10L640_1K,
           "m52m": M52M_10L640, "m60m": M60M_10L640,
           "pythia": PYTHIA_6L512,
           "l194m": L194M_14L1024,
           "l112m": L112M_14L768}



def get_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            # is_available() can be True while no usable device is exposed (an empty
            # CUDA_VISIBLE_DEVICES does this): commit to CUDA only after probing it, or
            # every later cuda call raises 'Invalid device id' during startup. A run that
            # dies before step 1 is the worst case for an unattended supervisor.
            torch.cuda.get_device_properties(torch.cuda.current_device())
            return torch.device("cuda")
        except Exception as e:  # noqa: BLE001
            print(f"CUDA reported available but is unusable ({e}); falling back to CPU")
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


def load_token_cache(cache_path: str) -> torch.Tensor:
    # Raw int32 .bin token files are memory-mapped, never read into RAM: a
    # 38GB corpus cannot fit in 34GB, and waiting on torch.load of a huge
    # .pt is what killed run 010 around step 17k. A memmap streams from disk
    # and keeps the process small.
    if cache_path.endswith(".bin"):
        import warnings
        import numpy as _np
        n = os.path.getsize(cache_path) // 4
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # torch nags about read-only arrays
            t = torch.from_numpy(_np.memmap(cache_path, dtype="<i4", mode="r",
                                            shape=(n,)))
        print(f"  memmap {cache_path}: {n:,} tokens ({n * 4 / 1e9:.1f}GB on disk, 0 RAM)")
        return t
    return torch.load(cache_path, map_location="cpu", weights_only=True)


def load_bpe_corpus(tok: BPETokenizer, files: list[str],
                    cache_pt: str) -> torch.Tensor:
    # Encode once, reuse forever. Cache key includes corpus+vocab names
    # so TinyStories_bpe2k.pt and librivox_bpe8k.pt never collide.
    if os.path.exists(cache_pt):
        print(f"loading BPE cache {cache_pt} ...")
        return load_token_cache(cache_pt)
    total_bytes = sum(os.path.getsize(p) for p in files)
    if total_bytes > 1_000_000_000:
        # Encoding this in-process builds a Python list of ints: ~10x the file
        # size in RAM, slower than the disk it lives on. Refuse loudly instead
        # of hanging for a day and then dying.
        raise SystemExit(
            f"no token cache at {cache_pt} and the corpus is {total_bytes / 1e9:.1f}GB.\n"
            f"Build it once with the streaming tokenizer, then pass it back:\n"
            f"    python tokenize_owt_16k.py\n"
            f"    python -m train.train ... --tok_cache data/<name>.bin")
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


def val_bytes_per_token(val_data, tok, sample: int = 200_000) -> float:
    """Source bytes per token — the constant that makes loss comparable.

    Per-token cross-entropy is NOT comparable between tokenizers. A 16k BPE
    splits the same text into fewer, more informative tokens, so its per-token
    loss is mechanically higher at identical compression quality. Exp 010 vs
    011 is the worked example: the raw gap said "16k is 18.6% worse", and the
    byte-normalised gap said 16k is 1.8% BETTER. bits/byte = nats/ln(2) / bpt.
    """
    if tok is None:
        return 1.0                      # byte tokenizer: 1 token == 1 byte
    ids = [int(i) for i in val_data[:sample].tolist()]
    if not ids:
        return 1.0
    text = tok.decode(ids)
    nbytes = len(text.encode("utf-8", errors="ignore"))
    return nbytes / len(ids) if nbytes else 1.0


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
        return load_token_cache(cache_pt)
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
                    help="val must beat best by this much to reset patience "
                         "(now applied to bits/byte, not per-token loss)")
    ap.add_argument("--patience-frac", type=float, default=0.0,
                    help="plateau patience as a FRACTION of --steps (e.g. 0.10 = "
                         "2000 steps at 20k). Preferred over --patience: a check "
                         "count silently depends on --val_every, so 10 checks was "
                         "only 1000 steps and killed exp 011 at 15%% of schedule. "
                         "0 = fall back to --patience")
    ap.add_argument("--min-steps-frac", type=float, default=0.0,
                    help="no plateau-based early stop before this fraction of the "
                         "schedule (e.g. 0.5). Mid-schedule a flat val is the LR "
                         "still being high, not convergence. A real climb still "
                         "aborts -- see --degrade-frac")
    ap.add_argument("--degrade-frac", type=float, default=0.30,
                    help="before min-steps-frac, abort anyway if val exceeds best "
                         "by this fraction (catches genuine overfit/divergence). "
                         "CALIBRATE THIS FROM THE VAL NOISE BAND, not by feel: on "
                         "the m50m/OWT runs, val swings a median 5.2%% and up to "
                         "10.4%% between consecutive checks, with excursions up to "
                         "5.0%% above the running best purely from noise, while real "
                         "memorisation (exp 010) was +143%%. A 0.05 threshold sits "
                         "INSIDE the noise band and aborts a healthy run -- keep it "
                         "well clear of the noise (default 0.30)")
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

    # === METRIC 1: Real parameter counts ===
    total_params = sum(p.numel() for p in model.parameters())
    embed_params = sum(p.numel() for n, p in model.named_parameters()
                       if 'embedding' in n or 'emb' in n or 'lm_head' in n)
    non_embed_params = total_params - embed_params
    print(f"  params: {total_params:,} total ({total_params/1e6:.1f}M), "
          f"{non_embed_params:,} non-embedding ({non_embed_params/1e6:.1f}M)")

    # 4070 Ti SUPER: tf32 + fused AdamW is the free speedup.
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    warmup = min(200, args.steps // 10)

    # Resume: weights + optimizer back, step counter continues to --steps.
    # Shape mismatch (e.g. ctx/vocab change) fails loud, not silent.
    start_step = 0
    # Continuity across restarts (added 2026-09-18 after exp011c was killed by an app
    # restart). Weights+optimizer+step were already restored, but three things were NOT,
    # and each one gets worse the longer the run is:
    #   * best_bpb/bad_checks -- the early-stop ledger. Reset to inf on every resume, so a
    #     long run that restarts repeatedly silently loses its divergence guard.
    #   * the run stamp in checkpoint filenames -- a resumed run minted a NEW stamp, so one
    #     logical run scattered across several ckpt families and "newest ckpt" became
    #     ambiguous (this is what a supervisor needs to find).
    #   * the run dir -- the loss curve for one logical run got split across run dirs.
    import re
    resume_best_bpb = None
    resume_bad_checks = 0
    resume_stamp = None
    resume_run_name = None
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
        # Inherit the ledger (new ckpts carry best_bpb; older _best.pt carries val_bpb).
        if r.get("best_bpb") is not None:
            resume_best_bpb = float(r["best_bpb"])
        elif r.get("val_bpb"):
            resume_best_bpb = float(r["val_bpb"])
        resume_bad_checks = int(r.get("bad_checks") or 0)
        # Inherit the run identity from the filename stamp and/or the saved run_name.
        _m = re.search(r"_(\d{12})_", os.path.basename(args.resume))
        resume_stamp = _m.group(1) if _m else None
        resume_run_name = r.get("run_name")
        _ledger = (f"best_bpb={resume_best_bpb:.4f} bad_checks={resume_bad_checks}"
                   if resume_best_bpb else
                   "best_bpb NOT in ckpt (guard baseline will restart)")
        print(f"resumed {args.resume} @ step {start_step} | {_ledger}"
              f" | stamp={resume_stamp} run={resume_run_name}")

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
    # A resumed run keeps the ORIGINAL identity: same ckpt stamp and same run dir, so one
    # logical run stays one family of checkpoints (what a supervisor must find) and one
    # continuous loss curve (what analysis reads). Appending rather than truncating is what
    # makes an interrupted run's curve readable end-to-end; the header is only written once.
    reuse_run = bool(resume_run_name) and os.path.isdir(os.path.join(args.run_dir, resume_run_name))
    if reuse_run:
        run_name = resume_run_name
    run_stamp = resume_stamp or (run_name.split("_")[0] + run_name.split("_")[1])
    run_path = os.path.join(args.run_dir, run_name)
    os.makedirs(os.path.join(run_path, "samples"), exist_ok=True)
    log_path = os.path.join(run_path, "loss.jsonl")
    fresh = (not reuse_run) or not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    log_f = open(log_path, "a" if reuse_run else "w")
    if fresh:
        json.dump({"args": vars(args), "cfg": vars(cfg), "params_m": cfg.num_params() / 1e6},
                  log_f)
        log_f.write("\n")
    print(f"  run dir: {run_path}{' (resumed - appending)' if reuse_run else ''}")

    # AMP: mixed-precision forward (fp16/bf16) + fp32 gradients.
    # Free speedup on CUDA — ~1.5-2x throughput, negligible quality impact.
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # bf16 probe is guarded: it touches the CUDA driver and can raise on an odd context,
    # which would kill the run before step 1.
    if use_amp:
        try:
            _bf16 = torch.cuda.is_bf16_supported()
        except Exception as e:  # noqa: BLE001
            print(f"  bf16 probe failed ({e}); using fp16 autocast")
            _bf16 = False
    else:
        _bf16 = False
    autocast_dtype = torch.bfloat16 if _bf16 else torch.float16
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

    # === METRIC 2: Three-way eval (loader check) ===
    # Buffer the last N training batches so we can re-score them at val time.
    # If "served" loss is far below "random train" loss, we're recycling data.
    SERVED_BUFFER_SIZE = 200  # keep last 200 micro-batches
    served_buffer = []  # list of (x, y) tensors on CPU

    @torch.no_grad()
    def eval_three_way(train_data, val_data, served_buf, batches=20):
        """Score: served batches, random train, and val. Returns (served, random_train, val)."""
        model.eval()
        # 1) Served: replay the last N batches the loader actually served
        if served_buf:
            s_total = 0.0
            n = min(batches, len(served_buf))
            indices = torch.randperm(len(served_buf))[:n]
            for i in indices:
                x, y = served_buf[i]
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=use_amp):
                    s_total += F.cross_entropy(
                        model(x).reshape(-1, cfg.vocab_size), y.reshape(-1)).item()
            served_loss = s_total / n
        else:
            served_loss = float('nan')

        # 2) Random train: fresh random samples from the train split
        rt_total = 0.0
        for _ in range(batches):
            x, y = get_batch(train_data, args.batch, cfg.context_length,
                             cfg.vocab_size, device, pos)
            with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=use_amp):
                rt_total += F.cross_entropy(
                    model(x).reshape(-1, cfg.vocab_size), y.reshape(-1)).item()
        random_train_loss = rt_total / batches

        # 3) Val: held-out tail
        v_total = 0.0
        for _ in range(batches):
            x, y = get_batch(val_data, args.batch, cfg.context_length,
                             cfg.vocab_size, device, pos)
            with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=use_amp):
                v_total += F.cross_entropy(
                    model(x).reshape(-1, cfg.vocab_size), y.reshape(-1)).item()
        val_loss = v_total / batches

        model.train()
        return served_loss, random_train_loss, val_loss

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

    # Byte-normalised val: the only loss number comparable across vocabularies.
    bpt = val_bytes_per_token(val_corpus, tok) if val_corpus is not None else 1.0
    bpb_factor = math.log2(math.e) / bpt
    # Patience in STEPS (via --patience-frac), not checks: a check count quietly
    # depends on --val_every and killed exp 011 at 15% of its schedule.
    if args.patience_frac > 0:
        patience_checks = max(1, round(args.patience_frac * args.steps / max(1, args.val_every)))
    else:
        patience_checks = args.patience
    if val_corpus is not None:
        print(f"  val normalisation: {bpt:.4f} bytes/token -> "
              f"bpb = val * {bpb_factor:.4f}")
        print(f"  early stop: {patience_checks} checks "
              f"({patience_checks * args.val_every} steps) of flat bpb; "
              f"no plateau stop before step {int(args.min_steps_frac * args.steps)}; "
              f"abort early if bpb > best x {1 + args.degrade_frac:.2f}")

    # Early-stop ledger: best BITS/BYTE seen, strikes since, stop flag.
    # A resumed run INHERITS this ledger: it used to reset to inf on every resume, so a
    # long run that restarted repeatedly lost its divergence guard each time.
    best_bpb, bad_checks = float("inf"), 0
    if resume_best_bpb:
        best_bpb, bad_checks = resume_best_bpb, resume_bad_checks
        print(f"  early-stop ledger inherited: best_bpb={best_bpb:.4f} "
              f"strikes={bad_checks}")

    # === METRIC 3: Throughput tracking ===
    import time
    throughput_start = time.monotonic()
    throughput_tokens = 0  # accumulated over first 100 steps
    THROUGHPUT_WINDOW = 100
    throughput_reported = False

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
            # METRIC 2: buffer this micro-batch for three-way eval
            if len(served_buffer) < SERVED_BUFFER_SIZE:
                served_buffer.append((x.cpu(), y.cpu()))

        # METRIC 3: throughput tracking
        throughput_tokens += args.batch * accum * cfg.context_length

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        loss_val = micro_loss  # accumulated loss (already scaled by 1/accum)

        # Val check on the held-out tail. All decisions use bits/byte.
        val, val_bpb, stop = None, None, False
        served_loss, random_train_loss = None, None
        if val_corpus is not None and args.val_every and step % args.val_every == 0:
            # METRIC 2: three-way eval instead of just val
            served_loss, random_train_loss, val = eval_three_way(
                train_corpus if train_corpus is not None else corpus,
                val_corpus, served_buffer, batches=20)
            val_bpb = val * bpb_factor
            served_bpb = served_loss * bpb_factor if served_loss == served_loss else None  # nan check
            random_train_bpb = random_train_loss * bpb_factor
            print(f"  three-way: served={served_loss:.4f} "
                  f"random_train={random_train_loss:.4f} val={val:.4f} "
                  f"(bpb: {served_bpb:.4f} / {random_train_bpb:.4f} / {val_bpb:.4f})")
            if patience_checks > 0:
                if val_bpb < best_bpb - args.min_delta:
                    # New best: snapshot it, reset strikes.
                    best_bpb, bad_checks = val_bpb, 0
                    best_ckpt = os.path.join(
                        args.out, f"exp002_{tag}_{run_stamp}_best.pt")
                    saved_best = dict(vars(cfg))
                    saved_best["use_rope"] = args.use_rope
                    saved_best["tokenizer"] = args.tokenizer
                    saved_best["corpus"] = args.corpus or args.data
                    saved_best["val_bpb"] = val_bpb
                    saved_best["bytes_per_token"] = bpt
                    torch.save({"cfg": saved_best, "model": model.state_dict(),
                                "optimizer": opt.state_dict(),
                                "step": step, "val": val, "val_bpb": val_bpb,
                                "best_bpb": val_bpb, "bad_checks": bad_checks,
                                "run_name": os.path.basename(run_path)},
                               best_ckpt)
                    print(f"  new best bpb={val_bpb:.4f} (val={val:.4f}) -> {best_ckpt}")
                else:
                    bad_checks += 1
                    if step < args.min_steps_frac * args.steps:
                        # Mid-schedule a flat bpb is the LR still being high, not
                        # convergence -- only a genuine climb is worth stopping for.
                        if val_bpb > best_bpb * (1.0 + args.degrade_frac):
                            stop = True
                            print(f"  abort: bpb {val_bpb:.4f} > best {best_bpb:.4f} "
                                  f"+{args.degrade_frac:.0%} at step {step} (mid-schedule)")
                    elif bad_checks >= patience_checks:
                        stop = True
                        print(f"  early stop: bpb flat for {bad_checks} checks "
                              f"({bad_checks * args.val_every} steps) — "
                              f"best={best_bpb:.4f} @ step {step}")

        running += (loss_val - running) / min(step, 50)
        log_entry = {"step": step, "train": round(loss_val, 4),
                     "avg50": round(running, 4),
                     "val": round(val, 4) if val else None,
                     "val_bpb": round(val_bpb, 5) if val_bpb else None,
                     "lr": lr}
        # METRIC 2: log three-way eval
        if served_loss is not None:
            log_entry["served"] = round(served_loss, 4)
            log_entry["served_bpb"] = round(served_loss * bpb_factor, 5)
            log_entry["random_train"] = round(random_train_loss, 4)
            log_entry["random_train_bpb"] = round(random_train_loss * bpb_factor, 5)
        log_f.write(json.dumps(log_entry) + "\n")
        log_f.flush()   # a watcher should see steps appear, not wait for 4KB

        # METRIC 3: throughput report after first THROUGHPUT_WINDOW steps
        if not throughput_reported and step >= THROUGHPUT_WINDOW:
            elapsed = time.monotonic() - throughput_start
            tok_per_sec = throughput_tokens / elapsed
            remaining_budget = 7 * 3600  # 7 hours in seconds
            estimated_tokens = tok_per_sec * remaining_budget
            print(f"\n  === THROUGHPUT (first {THROUGHPUT_WINDOW} steps) ===")
            print(f"  {tok_per_sec:,.0f} tokens/sec")
            print(f"  7-hour budget: ~{estimated_tokens/1e9:.2f}B tokens")
            print(f"  Steps to 1B tokens: ~{1e9 / tok_per_sec:,.0f}")
            print(f"  ===\n")
            throughput_reported = True
        if step % 25 == 0 or step == 1:
            vstr = f" val={val:.4f} bpb={val_bpb:.4f}" if val else ""
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
                        "step": step,
                        # Continuity fields: a supervisor reads run_name/stamp to find the
                        # newest checkpoint of THIS logical run, and the trainer restores
                        # the ledger on resume.
                        "best_bpb": None if best_bpb == float("inf") else best_bpb,
                        "bad_checks": bad_checks,
                        "run_name": os.path.basename(run_path)}, ckpt)
            print(f"  saved {ckpt}")
            write_samples(step)
        if stop:
            # Patience spent: the keeper is the best ckpt, not this one.
            log_f.write(json.dumps({"early_stop": True, "step": step,
                                    "best_val": round(best_bpb / bpb_factor, 4),
                                    "best_bpb": round(best_bpb, 5)}) + "\n")
            break

    log_f.close()
    _best = "n/a" if best_bpb == float("inf") else f"{best_bpb:.4f}"
    print(f"done. final avg50 loss={running:.4f}  best bpb={_best}  run dir: {run_path}")


if __name__ == "__main__":
    main()
