LLM Lab

Purpose
-------
Learn by building a small language model from scratch,
then progressively introduce modern techniques and a tool harness.

Principles
----------
- Understand before abstracting.
- Keep model and harness separate.
- Prefer measurable experiments.
- Reproduce experiments across machines.
- Record what changed and what happened.

Setup
-----
After git pull, run:

    bash setup.sh

This creates the venv, syncs deps, and installs CUDA torch on Windows.
Mac just needs `uv sync` (setup.sh handles both).

To verify CUDA: `uv run python -c "import torch; print(torch.cuda.is_available())"`

Machines
--------
- PC (Ronin, 4070 Ti Super)  — primary dev/training (CUDA)
- Mac (MacBook Pro, MPS)      — training/inference
- Mini (Mac Mini, Win10)      — inference/testing

Experiments
-----------

### 001 — Tiny Transformer
Status: DONE
Baseline byte-level transformer. Minimal architecture to validate the harness.

### 002 — BPE Tokenization (BPE2K)
Status: DONE
Switched from byte-level to BPE with vocab size 2048 (actual vocab: 2256 with special tokens).
Config: embed 384, 6 layers, 6 heads, context 256, ~10M params.
Data: TinyStories.

### 003 — RoPE (Rotary Position Embeddings)
Status: DONE
Replaced learned positional embeddings with rotary position embeddings on top of the BPE2K config.
Config: same as 002 — vocab 2256, ctx 256, embed 384, 6L, 6H, ~10M params.
Batch 64, lr 3e-4, TinyStories.

### 004 — BPE Efficiency Win
Status: DONE
BPE vs byte-level on same architecture (bpe2k + RoPE, 5k steps):

| Model       | Val NLL/token | Bytes/token | Nats/byte |
|-------------|---------------|-------------|-----------|
| bytes rope  | 0.7314        | 1.00        | 0.7314    |
| bpe2k rope  | 2.1150        | 4.11        | 0.5148    |

~30% less surprise per byte of story. BPE era confirmed.

### 005 — Overfitting Knee (20k steps)
Status: DONE
Same bpe2k+rope recipe, 4x the steps. Result: val peaked at 2.086 @ step 4500,
then climbed monotonically to 2.89 by step 20k. Train kept learning; generalization
peaked early. Discovered early stopping empirically.

Cross-machine reproducibility confirmed (Sep 5, 2026):

| Machine | Hardware    | Time (20k steps) | Train Loss | Val Loss (best) | Val Loss (20k) |
|---------|-------------|-------------------|------------|-----------------|----------------|
| PC      | 4070 Ti     | 31 min            | 0.8717     | 2.0858          | 2.8924         |
| Mac     | MPS         | ~2h 15m           | 0.8934     | ~2.09           | ~2.89          |

Val loss matches to 4 decimal places across CUDA and MPS. Same overfit curve.
4.3x speedup on PC vs Mac.

### 006 — Dropout + Early Stopping
Status: DONE
Wired `ModelConfig.dropout` (was dead config) + `--patience` / `--min_delta` early stopping.
Two runs:

| Run         | Dropout | Steps | Best Val | Step | Train (final) | Notes |
|-------------|---------|-------|----------|------|---------------|-------|
| 20260905_1910 | 0.1   | 5k    | 2.1626   | 4900 | 1.9885        | Short horizon — dropout slightly worse (expected) |
| 20260913_1007 | 0.1   | 4.9k  | 2.154    | 4900 | 1.8809        | Clean early stop, no overfit |

Verdict: dropout 0.1 trades a tiny bit of peak val (2.15 vs 2.09 no-dropout) for
flat val curve — no climb back to 2.89. That's exactly what dropout is supposed to do.
Early stopping fires correctly: patience=5 on val checks without improvement.

### 007 — Book Era First Crunch (LibriSpeech)
Status: DONE (Sep 14, 2026 — Ronin 4070 Ti)
Config: libri10m preset — vocab 8256 (bpe8k clean), ctx 512, embed 384, 6L6H,
dropout 0.1, ~10M params. 20k-step budget, early stopped.
Result: early stop @ step 6800, best val 4.84, train 9.18→3.72.
Verdict: pipeline works end-to-end on real literature (193M tokens). Val 4.84 vs
TinyStories 2.15 is expected — books are a much harder corpus. Train/val gap
(3.72/4.84) says capacity wall, not broken code. Next: go bigger (008).

### 008 — TinyStories 50M Control
Status: DONE (Sep 14, 2026 — Ronin 4070 Ti)
Config: bpe50m preset — vocab 2256 (bpe2k), ctx 256, embed 640, 10L10H,
dropout 0.1, ~52M params. 20k-step budget, full run.
Result: step 20000, train 0.003, val 5.89.
Verdict: massive overfit — model memorized TinyStories completely but didn't
generalize (val 5.89). 50M is too much model for TinyStories. Dropout kept it
from collapsing but can't prevent memorization on a small, simple corpus.
This run confirms 50M architecture is functional on CUDA. The real test is
Gutenberg (009) where data volume matches the capacity.

### 009 — Gutenberg 50M (LibriSpeech)
Status: QUEUED
Config: libri50m preset — vocab 8256 (bpe8k), ctx 512, embed 640, 10L10H,
dropout 0.1, ~52M params. 193M tokens from Gutenberg books.
Hypothesis: 50M should generalize better on 193M tokens than TinyStories —
more data matches more capacity. Val should land below 4.84 (007's 10M result).

### 010 — M50M Full OpenWebText (4k vocab)
Status: DONE (Sep 17, 2026 — Ronin 4070 Ti)
Config: m50m preset — vocab 4512 (bpe_owt4k), ctx 512, embed 640, 10L10H,
dropout 0.1, ~55M params. Full OpenWebText (~4.8B tokens).
Budget: 20k steps, batch 32, LR 3e-4. No early stopping (patience=0).
Result: best val 4.0986 @ step 2600. Process died at step ~17,050 (likely OOM or
crash). Final state: train 0.10, val 9.98 — completely overfit.
Verdict: 4k vocab is too small for 55M params. Model runs out of embedding
resolution early and memorizes training data. Best val (4.09) is the same as
every other vocab size — the embedding layer is the bottleneck, not data or
architecture. Next: try 16k vocab (experiment 011).
KEEPER: checkpoints/exp002_m50m_rope_bpe_owt4k_202609172151_step2500.pt
LESSON: **Always use --patience with early stopping.** Without it, the model
wastes hours memorizing after the val inflection point. Set patience=10
(val must improve by min_delta=0.001 within 10 checks = 1000 steps).

### 011 — M50M Full OpenWebText (16k vocab)
Status: QUEUED
Config: same as 010 but with bpe_owt16k tokenizer (~16k vocab).
Hypothesis: larger vocab gives the embedding layer more resolution, which
should improve generalization for 55M params. Target val < 4.09.
NOTE: smoke 200 first, then --patience 10 --min_delta 0.001 for full run.

### 011 — M50M Full OpenWebText (16k vocab)
Status: TOKENIZING (Sep 18, 2026 — Ronin)
Config: same as 010 but with bpe_owt16k tokenizer (16,256 vocab). ~70M params
(10M of that is the bigger embedding table).

Pipeline (rewritten Sep 18 — the first three attempts all died):
  run_16k_smoke.bat   rehearsal: 3 shards x 20M chars -> verify -> 200 train steps (~2.5 min)
  run_16k_full.bat    the real thing: 80 shards, serialized, resumable (~7h)
  tokenize_owt_16k.py orchestrator (manifest, resume, preflight disk check)
  token_io.py         streaming encoder + memmap reader (no %TEMP%, no RAM blowup)
  verify_tokens.py    7 acceptance checks — run before trusting any long encode
Train with the .bin:  --tok_cache data/openwebtext_combined_bpe_owt16k.bin

What broke before, and why it can't now:
  * tempfile.mkdtemp() put per-chunk .npy files on C: -> writes only under data/ now
  * torch.cat() of all 80 shard tensors needed ~80GB RAM on a 34GB box -> the
    output is appended shard by shard; peak RAM is one 500K-char chunk (~4MB)
  * torch.load() of a 47GB .pt is how run 010 died at step ~17k -> .bin token
    files are memmapped, so the corpus streams from disk at 0 RAM
  * loss.jsonl was unbuffered-free, so a watcher saw an empty log -> flushed per step

Smoke evidence (Sep 18): encode 40s (370k tok/s), verify 7/7, resume skip all 3
shards in 0.7s, 200 train steps in 81s (0.4s/step -> 20k steps ~2.2h on the 4070 Ti).

## Current state (2026-09-18) — read this first, it's the handoff

**16k OWT corpus:** `data/openwebtext_combined_bpe_owt16k.bin` (raw int32, ~38GB, ~9.46B
tokens expected). Encoded in parallel (8 worker processes, `run_16k_full_par.bat`),
~1.6M tok/s aggregate.

One worker died mid-run to an unreproducible fault in the BPE encoder
(`TypeError: slice indices must be integers` in model/bpe.py). Its block (shards
33-42) is therefore missing and the run **deliberately refuses to merge**. The
encoder is now hardened (`sentinel -1` + `token_io.encode_resilient` bisect with
evidence capture in `<dst>.encode_errors.jsonl`) and the parent reports worker
deaths immediately. Commit `9a41716`.

Following up unattended: `followup_wait.ps1` holds until the encode exits, then runs
a second pass that reads the merge plan off the parts on disk, encodes only the gap,
merges in corpus order and verifies. **Outcome is one line in `logs/followup.log`**
(VERIFIED OK / VERIFY FAILED / ENCODE INCOMPLETE).

Check progress:  `tail -2 logs/16k_tokenize_par.log`   (status lines every 60s)
Rerun by hand:   `run_16k_full_par.bat`   (resumable — finished shards are skipped)
Verify:          `python verify_tokens.py --bin data/openwebtext_combined_bpe_owt16k.bin --vocab data/bpe_owt16k.json --src data/openwebtext/shards/train-00000-of-00080.txt --manifest`
Tests:           `python -m pytest tests/test_tokenizer_safety.py tests/test_bpe_resilience.py -q`  (16 tests)

**Next: exp 011 training** — m50m, 20k steps, `--patience 10 --min_delta 0.001`,
training off the memmapped `.bin` (never a 38GB `.pt` — that's how run 010 died at
step ~17k). Measured 0.4s/step → ~2.2h. Hypothesis: 16k vocab gives the embedding
more resolution than 4k did (exp 010 best val 4.0986 @ step 2600, then overfit).

**Curated corpora on disk** (~149GB, fetched 2026-09-18, don't re-download):
`data/cosmopedia/` (336 parquet, 86 GiB), `data/finewiki/` (15 parquet, 36 GiB, en
only), `data/open_web_math/` (114 parquet, 26 GiB). Fetch script: `fetch_corpus.py`
(validates patterns against the real file list first; resumable). Next big target if
we want volume: FineWeb-Edu sample-100BT (267GB, ~100B tokens) — but note the cost:
4 bytes/token means a 373GB file and ~65h of CPU encode single-core (~8h with 8 workers).
