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

# NOTE: torch and model.transformer are deliberately NOT imported here.
# train.train runs on a machine-wide singleton lock (train/runtime_lock.py) so two
# trainers can never share one GPU. The lock must be acquired BEFORE torch/CUDA
# initialises, so those imports live at the top of _train(), after main() holds
# the lock. model.config and model.bpe are torch-free and stay top-level so that
# PRESETS and the pure-python helpers remain importable without torch.

# Allow `python -m train.train` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import L112M_14L768, L194M_14L1024, M49M_6L384, M50M_10L640, M50M_10L640_1K, M52M_10L640, M60M_10L640, M66M_10L704, PYTHIA_6L256, PYTHIA_6L384, PYTHIA_6L512, PYTHIA_12L768, PYTHIA160_12L768, S11M_6L384, S12M_6L384, S17M_6L384, T1M_4L128
from model.bpe import BPETokenizer

PRESETS = {"t1m": T1M_4L128, "s11m": S11M_6L384, "m49m": M49M_6L384,
           "s12m": S12M_6L384, "s17m": S17M_6L384,
           "m50m": M50M_10L640, "m50m_1k": M50M_10L640_1K, "m66m": M66M_10L704,
           "m52m": M52M_10L640, "m60m": M60M_10L640,
           "pythia": PYTHIA_6L512,
           # Overtraining ladder -- same data/tokenizer/val as `pythia`.
           # Ordered small -> large so a partial sweep is still a curve.
           "pythia256": PYTHIA_6L256,
           "pythia384": PYTHIA_6L384,
           "pythia768": PYTHIA_12L768,
           "pythia160": PYTHIA160_12L768,
           "l194m": L194M_14L1024,
           "l112m": L112M_14L768}


def load_tokenizer(path: str):
    """Load either tokenizer flavour, always with .encode(text) -> list[int].

    HF `tokenizer.json` (keys "model" + "pre_tokenizer") -> HfBpeShim over the
    real HF encoder. Legacy BPETokenizer json ({vocab, merges}) -> BPETokenizer.

    This has to be the same encoder that built the vocab: a re-derived legacy
    copy of the Pythia vocab measures ~1.1% fewer tokens than the real thing
    and first diverges around curly quotes/accented characters.
    """
    import json
    with open(path, encoding="utf-8") as f:
        head = json.load(f)
    if "model" in head and "pre_tokenizer" in head:
        scripts = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        from hf_bpe_shim import HfBpeShim
        return HfBpeShim.from_file(path)
    return BPETokenizer.load(path)


def tokenizer_vocab_size(tok, path: str) -> int:
    """Embedding rows, which is NOT the tokenizer's entry count.

    Pythia is the trap: model.vocab holds 50,254 entries, the encoder also emits
    23 added whitespace-run tokens (ids 50,254..50,276), so get_vocab() is
    50,277 and a real token stream contains id 50,276. pythia-70m's config.json
    then pads the embedding to 50,304 for efficiency. Sizing our model to
    50,254 or 50,277 makes a token the encoder actually produces an
    out-of-range cross-entropy target.

    Legacy vocabs have no padding concept, so use the entry count there.
    """
    import json
    with open(path, encoding="utf-8") as f:
        head = json.load(f)
    if "model" in head and "pre_tokenizer" in head:
        cfg_path = os.path.join(os.path.dirname(path), "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                v = json.load(f).get("vocab_size")
            if isinstance(v, int) and v >= len(tok._hf.get_vocab()):
                print(f"  vocab: {len(tok._hf.get_vocab())} encoder ids, "
                      f"embedding padded to {v} (config.json)")
                return v
        return len(tok._hf.get_vocab())
    return len(tok.vocab)



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


def eval_cycle_length(evals_bpb: list, max_period: int = 8) -> int:
    """Recover the eval sampler's period by autocorrelation of the val series.

    Measured on the 70M Pile run: the eval is NOT fresh noise. eval_three_way() draws
    val windows with get_batch(), which calls torch.randint on the GLOBAL RNG, and the
    number of draws between consecutive evals is constant (1000 steps x batch). So the
    RNG advances by a fixed amount per eval and the sampler lands on the SAME handful of
    window sets in rotation, forever. Median |diff| between v[i] and v[i+5] was 0.0004 bpb
    versus 0.035 for non-multiples of 5 -- a deterministic 5-cycle, not jitter.

    That matters because a min over a periodic series is a biased statistic: it always
    lands on whichever phase is easiest, so it sits a fixed distance BELOW the true
    score and never moves even when the model has stopped learning. Both the keeper and
    the saturation guard must work on the unwrapped envelope instead.

    Returns the period in evals (1 = no detectable cycle, i.e. genuinely noisy).
    """
    n = len(evals_bpb)
    if n < 2 * max_period + 2:
        return 1
    lo, hi = n // 2, n  # ignore the descending warmup; structure is a late-run property
    tail = evals_bpb[lo:hi]
    best_p, best_err = 1, None
    for p in range(2, max_period + 1):
        diffs = [abs(tail[i + p] - tail[i]) for i in range(len(tail) - p)]
        if not diffs:
            continue
        err = sorted(diffs)[len(diffs) // 2]  # median, robust to the trend itself
        if best_err is None or err < best_err:
            best_p, best_err = p, err
    # Require the cycle to be clearly better than "no structure" or it is just a fit
    # to the residual trend. A median within 40% of the p=1-ish noise floor is noise.
    if best_err is None:
        return 1
    floor = sorted([abs(tail[i + 1] - tail[i]) for i in range(len(tail) - 1)])
    floor_err = floor[len(floor) // 2] if floor else 0.0
    if floor_err <= 0 or best_err > 0.6 * floor_err:
        return 1
    return best_p


def envelope(evals_bpb: list, period: int = 0) -> tuple[list, int]:
    """Unwrap the eval cycle into a per-round MEAN, the unbiased score estimate.

    Returns (envelope_values, period_used). The mean over a whole cycle cancels the
    per-window-set difficulty differences, which is exactly what the min cannot do: a
    min over a periodic series tracks the easiest phase and is biased low by the cycle
    amplitude (measured 0.0399 bpb on the 70M run).

    A trailing partial round is DROPPED. A partial round mixes phases and reintroduces
    the very bias this removes; better to lose up to period-1 points of resolution than
    to report a number that is not comparable to earlier rounds.
    """
    p = period or eval_cycle_length(evals_bpb)
    if p <= 1 or len(evals_bpb) < 2 * p:
        return list(evals_bpb), 1
    rounds = []
    for i in range(len(evals_bpb) // p):
        rounds.append(sum(evals_bpb[i * p:(i + 1) * p]) / p)
    return rounds, p


def envelope_min(evals_bpb: list, window: int) -> list:
    """Lower envelope of a val series: the min of each block of `window` evals.

    DEPRECATED as a decision statistic (2026-10-01). This tracks the LOW phase of the
    eval cycle, which is the same biased-low statistic that made best_bpb read 0.0399 bpb
    below the true score. Kept only because the saturation guard's block arithmetic still
    references it and existing replays import it. New code should use envelope(), which
    takes the per-round MEAN and is unbiased.

    Prefer: envelope(evals_bpb) -> (per-round means, period)
    """
    w = max(1, int(window))
    full = len(evals_bpb) // w
    return [min(evals_bpb[i * w:(i + 1) * w]) for i in range(full)]


def flat_envelope_stop(evals_bpb: list, window: int, flat_delta: float,
                       step: int, steps: int, flat_frac: float,
                       val_every: int, min_step: int = 0) -> bool:
    """True when the UNBIASED envelope has stalled: no per-round-mean improvement.

    Guards the failure mode neither --degrade-frac nor --patience-frac covers: a curve
    that stops improving WITHOUT rising. On the 70M full-epoch run the envelope pinned
    at ~1.4432 bpb from step ~6200 onward while the train-val gap widened 0.1841 ->
    0.2786 nats -- overfitting onset, invisible to a rise-only threshold.

    CALIBRATED ON THE ENVELOPE, NOT ON best_bpb (2026-10-01). The first version of this
    guard was tuned against the min-tracked series and inherited that statistic's
    0.0399 bpb low bias, so its thresholds were meaningless: it read a saturated run as
    still improving. It now unwraps the eval cycle and thresholds the per-round MEAN.

    `window` is retained for signature compatibility but is no longer used to build the
    statistic -- the period is detected from the data. A stall budget of N ROUNDS
    (not blocks of N evals) is what actually bounds the decision.

    The stall must persist for `flat_frac` x steps so ordinary slow patches and a cooling
    cosine tail cannot trip it, and it never fires before `min_step`.
    """
    if flat_frac <= 0 or steps <= 0:
        return False
    if step < min_step:
        return False
    env, period = envelope(evals_bpb)
    if period <= 1 or len(env) < 2:
        # No detectable cycle: fall back to the raw series only if it is long enough to
        # judge. A short, cycle-free series is not evidence of a stall.
        if len(evals_bpb) < 8:
            return False
        env = list(evals_bpb)
        period = 1
    # How many ROUNDS of envelope fit inside the stall budget?
    round_steps = max(1, period * max(1, val_every))
    rounds_back = max(1, int(round(flat_frac * steps / round_steps)))
    if rounds_back + 1 > len(env):
        return False  # not enough history yet to judge a stall this long
    newest = env[-1]
    prior = env[-(rounds_back + 1):-1]
    # Still improving if ANY round in the budget beat the newest by more than the delta.
    if any(p > newest + flat_delta for p in prior):
        return False
    return True


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
                    help="Exp 003: rotary positions instead of learned absolute. "
                         "Pythia presets already set this in config.")
    ap.add_argument("--rotary-pct", type=float, default=None,
                    help="Fraction of head dims for RoPE (Pythia=0.25). "
                         "Default: the preset's rotary_pct (1.0 for legacy "
                         "presets, 0.25 for the pythia presets).")
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
    ap.add_argument("--flat-frac", type=float, default=0.0,
                    help="SATURATION guard: abort if the lower envelope of val has not "
                         "improved by more than --flat-delta within this fraction of "
                         "--steps. 0 = off. A curve that flattens without RISING is "
                         "overfitting onset, and neither --degrade-frac (needs a rise) "
                         "nor --patience-frac (fires on raw-eval jitter) can see it. "
                         "Envelope = min val_bpb per block of --flat-window evals; "
                         "raw evals are not usable directly because this harness emits a "
                         "repeating sawtooth (~4-eval period) that would reset any "
                         "threshold. Calibrate --flat-window to that period, not to 1.")
    ap.add_argument("--flat-window", type=int, default=4,
                    help="evals per envelope block for --flat-frac (default 4, the "
                         "measured sawtooth period on the 70M Pile run)")
    ap.add_argument("--flat-delta", type=float, default=0.005,
                    help="minimum envelope improvement (bpb) that counts as progress "
                         "for --flat-frac. Must exceed the eval-to-eval noise band or "
                         "jitter resets the guard forever (default 0.005 bpb)")
    ap.add_argument("--run_dir", default="runs",
                    help="loss.jsonl + samples land here per run")
    ap.add_argument("--accum", type=int, default=1,
                    help="gradient accumulation steps: effective batch = batch * accum")
    ap.add_argument("--warmup", type=int, default=None,
                    help="warmup steps (default: min(200, steps//10))")
    ap.add_argument("--resume", default=None,
                    help="checkpoint .pt to resume from (model + optimizer restored, steps continue to --steps)")
    args = ap.parse_args()

    from train.runtime_lock import acquire_train_lock
    lock = acquire_train_lock(owner={"preset": args.preset, "lr": args.lr})
    try:
        return _train(args)
    finally:
        lock.release()


def _train(args):
    # Heavy imports happen HERE, after main() holds the singleton lock -- importing
    # torch (or model.transformer, which imports torch) initialises CUDA, and a
    # second trainer must be refused before it touches the GPU. Declared global so
    # the module-level helpers (get_device, get_batch, ...) resolve them.
    global torch, F, Transformer
    import torch
    import torch.nn.functional as F
    from model.transformer import Transformer
    cfg = PRESETS[args.preset]
    if args.dropout is not None:
        # CLI wins: one variable per run, preset stays the control.
        cfg.dropout = args.dropout
    if args.rotary_pct is not None:
        # CLI wins; unset means "the preset's own value" (Pythia = 0.25).
        cfg.rotary_pct = args.rotary_pct
    use_rope = args.use_rope or cfg.use_rope
    device = get_device()
    tok = None
    if args.tokenizer:
        # File is truth: vocab size follows the tokenizer, not the preset.
        tok = load_tokenizer(args.tokenizer)
        cfg.vocab_size = tokenizer_vocab_size(tok, args.tokenizer)
        print(f"tokenizer={args.tokenizer} vocab={cfg.vocab_size}")
    print(f"preset={args.preset} params~{cfg.num_params() / 1e6:.1f}M "
          f"ctx={cfg.context_length} device={device} "
          f"rope={use_rope} rotary_pct={cfg.rotary_pct} "
          f"parallel_residual={cfg.parallel_residual}")

    torch.manual_seed(0)
    model = Transformer(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        embedding_dim=cfg.embedding_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        use_rope=use_rope,
        rotary_pct=cfg.rotary_pct,
        dropout=cfg.dropout,
        parallel_residual=cfg.parallel_residual,
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
    warmup = args.warmup if args.warmup is not None else min(200, args.steps // 10)

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
    resume_eval_bpb = None
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
        # The saturation guard's envelope needs the val history, not just the best. A
        # checkpoint that carries only best_bpb would let every restart silently reset
        # stall detection, which is exactly the bug class this guard exists to prevent.
        resume_eval_bpb = r.get("eval_bpb")
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
    unit = None
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
        # Corpus coverage: denominator for tokens_seen / train_tokens = epochs.
        # Random sampling reuses data, so epochs can exceed 1 — that IS the signal
        # (Chinchilla: ~20 tok/param; our 111M looked starved at single digits).
        _unit = unit or ("tokens" if tok is not None else "bytes")
        try:
            _corpus_n = int(len(corpus)) if corpus is not None else None
        except TypeError:
            _corpus_n = None
        try:
            _train_n = int(len(train_corpus)) if train_corpus is not None else _corpus_n
        except TypeError:
            _train_n = _corpus_n
        try:
            _val_n = int(len(val_corpus)) if val_corpus is not None else None
        except TypeError:
            _val_n = None
        json.dump({"args": vars(args), "cfg": vars(cfg), "params_m": cfg.num_params() / 1e6,
                   "corpus_tokens": _corpus_n, "train_tokens": _train_n,
                   "val_tokens": _val_n, "unit": _unit},
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

    # === METRIC 5: fixed train probe (memorization pair for val) ===
    # Ten deterministic windows from the TRAIN split, chosen by stride from the file
    # start, never re-randomized. Scored at every eval. tval - val is the memorization
    # gap: a healthy run keeps it small and stable; a widening one is fitting the
    # seen-data instead of learning the distribution (the ~2400-2700 wall we hit in
    # exp014/exp015/p160-pre-fix was invisible because we only watched train & val raw).
    probe_pairs = []
    try:
        src_probe = train_corpus if train_corpus is not None else corpus
        if src_probe is not None:
            n_probe = 10
            stride = max(1, (len(src_probe) - cfg.context_length - 1) // n_probe)
            for i in range(n_probe):
                off = i * stride
                xs = src_probe[off:off + cfg.context_length]
                ys = src_probe[off + 1:off + 1 + cfg.context_length]
                probe_pairs.append((xs.long().to(device), ys.long().to(device)))
    except Exception:
        probe_pairs = []

    @torch.no_grad()
    def eval_probe() -> "float|None":
        if not probe_pairs:
            return None
        model.eval()
        t = 0.0
        for x, y in probe_pairs:
            with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=use_amp):
                t += F.cross_entropy(
                    model(x.unsqueeze(0)).reshape(-1, cfg.vocab_size),
                    y.reshape(-1)).item()
        model.train()
        return t / len(probe_pairs)

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
        # Report each guard by its OWN trigger. The old single line reported a plateau
        # rule that was switched off while implying a degrade tripwire was active -- and
        # the degrade rule was inside the same gate, so it was off too. Both are armed
        # independently now (fix 2026-10-01); print what is actually running.
        armed = [f"degrade abort if bpb > best x {1 + args.degrade_frac:.2f}"
                 if args.degrade_frac > 0 else "degrade OFF",
                 f"plateau stop after {patience_checks} flat checks "
                 f"({patience_checks * args.val_every} steps)"
                 if patience_checks > 0 else "plateau OFF",
                 f"saturation stop if {args.flat_window}-eval envelope flat "
                 f"{args.flat_frac:.0%} of schedule (> {args.flat_delta} bpb)"
                 if args.flat_frac > 0 else "saturation OFF"]
        print(f"  guards: {' | '.join(armed)}; "
              f"no stop before step {int(args.min_steps_frac * args.steps)}")

    # Early-stop ledger: best BITS/BYTE seen, strikes since, stop flag.
    # A resumed run INHERITS this ledger: it used to reset to inf on every resume, so a
    # long run that restarted repeatedly lost its divergence guard each time.
    best_bpb, bad_checks = float("inf"), 0
    # val_bpb per eval, for the saturation guard's envelope. Rebuilt from the curve on
    # resume so a restarted run cannot reset its own stall detection.
    eval_bpb_history: list = []
    recycle_hits = 0  # consecutive recycling tripwire hits (needs 2 past step 100)
    if resume_best_bpb:
        best_bpb, bad_checks = resume_best_bpb, resume_bad_checks
        eval_bpb_history = list(resume_eval_bpb or [])
        print(f"  early-stop ledger inherited: best_bpb={best_bpb:.4f} "
              f"strikes={bad_checks} evals={len(eval_bpb_history)}")

    # === METRIC 3: Throughput tracking ===
    import time
    throughput_start = None  # start after warmup
    throughput_tokens = 0
    THROUGHPUT_WARMUP = 30  # skip first N steps (torch.compile + CUDA warmup)
    THROUGHPUT_WINDOW = 100  # measure this many steps after warmup
    throughput_reported = False
    peak_mem_mb = 0
    peak_temp = 0

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
        if step == THROUGHPUT_WARMUP:
            throughput_start = time.monotonic()
            throughput_tokens = 0  # reset after warmup

        scaler.unscale_(opt)
        # METRIC 4 (leading indicators): pre-clip grad norm is the clip return value -- free.
        gnorm = None
        try:
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        except Exception:
            pass
        if gnorm is None:  # metrics bug must never kill a run; fall back to plain clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # lmax/lstd/ent: last micro-batch's output-distribution shape. Rising logit scale
        # + collapsing entropy is the pre-collapse signature we were blind to at step 3900
        # (p160-long-pre-fix). Subsampled for cost; wrapped so it can never kill a run.
        lmax = lstd = ent = None
        try:
            with torch.no_grad():
                lg = logits.detach().reshape(-1, cfg.vocab_size)
                if lg.shape[0] > 4096:  # deterministic slice, not randperm: zero sync cost
                    lg = lg[:4096]
                lmax = float(lg.max(-1).values.mean())
                lstd = float(lg.std(-1).mean())
                lp = torch.log_softmax(lg, dim=-1)
                ent = float(-(lp.exp() * lp).sum(-1).mean())
        except Exception:
            pass
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
            # TRIPWIRE: data recycling detection. Gated three ways:
            # warmup (early batches are noisy, served buffer not full),
            # bpb margin (fair across vocabs), consecutive hits (one noisy
            # check never kills a run -- the 160M smoke proved that).
            if step >= 100:
                if served_bpb is not None and served_bpb < random_train_bpb - 0.15:
                    recycle_hits += 1
                    if recycle_hits >= 2:
                        print(f"\n  !!! DATA RECYCLING DETECTED !!!")
                        print(f"  served_bpb={served_bpb:.4f} < random_train_bpb={random_train_bpb:.4f} - 0.15")
                        print(f"  (2 consecutive checks, past step 100)")
                        print(f"  Stopping. The loader is recycling data.")
                        stop = True
                else:
                    recycle_hits = 0
            # GUARD BLOCK. The keeper (_best.pt) is deliberately OUTSIDE every stop-rule
            # gate: it is a safety artifact, not a reward for arming a stop rule. It used
            # to live inside `if patience_checks > 0`, so any run launched with
            # --patience-frac 0 (the full-epoch 70M run, and the chain specs that inherit
            # it) silently produced NO keeper at all: no *_best.pt, best_bpb=None in the
            # step checkpoint, and `best bpb=n/a` in the trainer's own exit line. A
            # 15,136-step run finished with its best score recoverable only by reading the
            # log. Fix 2026-10-01: keeper always runs; each stop rule arms on its own flag.
            #
            # The keeper triggers on the UNBIASED per-round envelope, not a single eval.
            # One eval is a single phase of a fixed 5-eval cycle (see eval_cycle_length),
            # so tracking the raw minimum saves whichever checkpoint happened to align
            # with the easiest window set and reports ~0.0399 bpb below true quality. On
            # the 70M run that turned a miss at the 1.42 target into an apparent pass.
            eval_bpb_history.append(val_bpb)
            round_env, round_period = envelope(eval_bpb_history)
            # Use the per-round mean WHEN a cycle is detectable and a full round is
            # available; otherwise fall back to the raw eval.
            #
            # The fallback is not a compromise, it is the short-run case: with too few
            # evals there is no cycle to unwrap (eval_cycle_length needs 2*max_period+2
            # samples), and returning inf here made the keeper never fire at all --
            # `best bpb=n/a` on a smoke run, i.e. the exact P0 bug this guard block was
            # written to fix, reintroduced by my own over-correction. A short run has no
            # cycle bias to correct, so raw evals are unbiased enough there.
            if round_period > 1 and len(round_env) >= 1 \
                    and len(eval_bpb_history) >= round_period:
                cand_bpb, basis = round_env[-1], "envelope_mean"
            else:
                cand_bpb, basis = val_bpb, "raw_eval"
            if cand_bpb < best_bpb - args.min_delta:
                # New best: snapshot it, reset strikes.
                best_bpb, bad_checks = cand_bpb, 0
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
                            "best_bpb": best_bpb,
                            "best_bpb_basis": basis,
                            "eval_bpb": list(eval_bpb_history),
                            "bad_checks": bad_checks,
                            "run_name": os.path.basename(run_path)},
                           best_ckpt)
                print(f"  new best bpb={best_bpb:.4f} (basis {basis}, raw "
                      f"{val_bpb:.4f} @ step {step}) -> {best_ckpt}")
            else:
                bad_checks += 1

            # Guard 1: DEGRADE. Arms on --degrade-frac alone. It never needed patience
            # (it is a RISE test, not a plateau test); sharing the gate meant a run with
            # --patience-frac 0 also lost its only overfit/divergence tripwire.
            if args.degrade_frac > 0:
                if step < args.min_steps_frac * args.steps:
                    # Mid-schedule a flat bpb is the LR still being high, not
                    # convergence -- only a genuine climb is worth stopping for.
                    if val_bpb > best_bpb * (1.0 + args.degrade_frac):
                        stop = True
                        print(f"  abort: bpb {val_bpb:.4f} > best {best_bpb:.4f} "
                              f"+{args.degrade_frac:.0%} at step {step} (mid-schedule)")
                elif val_bpb > best_bpb * (1.0 + args.degrade_frac):
                    stop = True
                    print(f"  abort: bpb {val_bpb:.4f} > best {best_bpb:.4f} "
                          f"+{args.degrade_frac:.0%} at step {step}")
            # Guard 2: PLATEAU. Arms on --patience-frac alone (unchanged semantics).
            if patience_checks > 0 and bad_checks >= patience_checks:
                stop = True
                print(f"  early stop: bpb flat for {bad_checks} checks "
                      f"({bad_checks * args.val_every} steps) — "
                      f"best={best_bpb:.4f} @ step {step}")
            # Guard 3: SATURATION / flat envelope. Arms on --flat-frac alone. Covers the
            # failure the other two structurally cannot: a curve that stops improving
            # WITHOUT rising. See flat_envelope_stop().
            if args.flat_frac > 0 and flat_envelope_stop(
                    eval_bpb_history, args.flat_window, args.flat_delta,
                    step, args.steps, args.flat_frac, args.val_every,
                    min_step=args.min_steps_frac * args.steps):
                stop = True
                env, _p = envelope(eval_bpb_history)
                print(f"  flat stop: per-round envelope stalled at bpb={env[-1]:.4f} "
                      f"for {args.flat_frac:.0%} of the schedule ({_p}-eval cycle x "
                      f"{args.flat_delta} bpb); best={best_bpb:.4f} @ step {step}")

        # METRIC 5: fixed-train probe, only at val cadence (it's a forward pass)
        tval = None
        if probe_pairs and args.val_every and step % args.val_every == 0:
            try:
                tval = eval_probe()
            except Exception:
                tval = None

        running += (loss_val - running) / min(step, 50)
        log_entry = {"step": step, "train": round(loss_val, 4),
                     "avg50": round(running, 4),
                     "val": round(val, 4) if val else None,
                     "val_bpb": round(val_bpb, 5) if val_bpb else None,
                     "lr": lr}
        # METRIC 4: leading indicators on every step (~free); tval on eval cadence.
        if gnorm is not None:
            log_entry["gnorm"] = round(gnorm, 3)
        if lmax is not None:
            log_entry["lmax"] = round(lmax, 3)
            log_entry["lstd"] = round(lstd, 3)
            log_entry["ent"] = round(ent, 4)
        if tval is not None:
            log_entry["tval"] = round(tval, 4)
        # METRIC 2: log three-way eval
        if served_loss is not None:
            log_entry["served"] = round(served_loss, 4)
            log_entry["served_bpb"] = round(served_loss * bpb_factor, 5)
            log_entry["random_train"] = round(random_train_loss, 4)
            log_entry["random_train_bpb"] = round(random_train_loss * bpb_factor, 5)
        log_f.write(json.dumps(log_entry) + "\n")
        log_f.flush()   # a watcher should see steps appear, not wait for 4KB

        # METRIC 3: throughput report after measuring THROUGHPUT_WINDOW steps
        if throughput_start and not throughput_reported:
            if step >= THROUGHPUT_WARMUP + THROUGHPUT_WINDOW:
                elapsed = time.monotonic() - throughput_start
                tok_per_sec = throughput_tokens / elapsed
                remaining_budget = 7 * 3600
                estimated_tokens = tok_per_sec * remaining_budget
                # GPU stats
                if device.type == "cuda":
                    mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                    temp = torch.cuda.temperature() if hasattr(torch.cuda, 'temperature') else 0
                    print(f"\n  === THROUGHPUT (steps {THROUGHPUT_WARMUP+1}-{THROUGHPUT_WARMUP+THROUGHPUT_WINDOW}) ===")
                    print(f"  {tok_per_sec:,.0f} tokens/sec")
                    print(f"  7-hour budget: ~{estimated_tokens/1e9:.2f}B tokens")
                    print(f"  Steps to 1B: ~{1e9 / tok_per_sec:,.0f}")
                    print(f"  Peak GPU mem: {mem_mb:,.0f}MB / 16376MB")
                    print(f"  ===\n")
                else:
                    mem_mb = 0
                    # MPS: unified memory, no separate gauge. Report rate only.
                    print(f"\n  === THROUGHPUT === {tok_per_sec:,.0f} tok/s ===\n")
                throughput_reported = True
                # Persist for the viz: ETA + OOM headroom cards read this row.
                log_f.write(json.dumps({"throughput": True, "step": step,
                                        "tok_per_sec": round(tok_per_sec, 1),
                                        "peak_mem_mb": round(mem_mb, 1)}) + "\n")
                log_f.flush()
        if step % 25 == 0 or step == 1:
            vstr = f" val={val:.4f} bpb={val_bpb:.4f}" if val else ""
            print(f"step {step:5d}/{args.steps} loss={loss_val:.4f} "
                  f"avg50={running:.4f}{vstr} lr={lr:.1e}", flush=True)
        if step % 500 == 0 or step == args.steps:
            ckpt = os.path.join(
                args.out, f"exp002_{tag}_{run_stamp}_step{step}.pt")
            saved_cfg = dict(vars(cfg))
            saved_cfg["use_rope"] = use_rope
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
                        # Full val history so a resumed run's saturation guard starts
                        # with its envelope intact instead of re-learning "no progress
                        # yet" from scratch and therefore never firing.
                        "eval_bpb": list(eval_bpb_history),
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
