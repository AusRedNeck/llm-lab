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
