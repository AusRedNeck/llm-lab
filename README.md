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

Cross-machine reproducibility confirmed (Sep 5, 2026):

| Run       | Machine | Hardware    | Time (20k steps) | Train Loss | Avg50  | Val Loss |
|-----------|---------|-------------|-------------------|------------|--------|----------|
| BPE2K+RoPE | PC    | 4070 Ti     | 31 min            | 0.8717     | 0.9148 | 2.8924   |
| BPE2K+RoPE | Mac   | MPS         | ~2h 15m           | 0.8934     | 0.9166 | 2.8925   |

Val loss matches to 4 decimal places across CUDA and MPS. Same overfit curve.
4.3x speedup on PC vs Mac (31 min vs 2h 15m).

5k-step runs (both machines): early-stopped at step 4900, best val 2.154.
These are intermediate checkpoints, not final.

Keeper: PC checkpoint stashed to Google Drive (llm-lab keepers folder).

### 004 — Next
TBD. Candidates:
- BPE vocabulary size sweep (how much does vocab impact loss?)
- Dropout/regularization (address the train-val gap)
- Larger dataset or data mix experiments
- Align with "Build a Large Language Model From Scratch" book chapters
