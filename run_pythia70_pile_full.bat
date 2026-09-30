@echo off
REM ===========================================================================
REM Pythia-70M on the FULL Pile — the open-weights gap experiment.
REM
REM Goal: train our 70.6M on the same diet Pythia-70m used (the Pile) and
REM compare val bits/byte against EleutherAI/pythia-70m. Pythia saw 300B
REM tokens; this run sees 2.0B (0.67%). The gap is the finding, not a failure.
REM
REM WHY THESE VALUES (each is a correction of the aborted 2026-09-27 run):
REM
REM   lr 5e-4, not 1.5e-3. The OWT run held 1.5e-3 and gnorm climbed
REM   0.255 -> 0.711 while val bpb rose 1.6134 -> 1.8682. It reached a floor
REM   around step 2300 (mean bpb 4000-6000 = 1.6758, BETTER than the mean
REM   before it) and then the too-hot LR ate that floor for 9200 steps. 5e-4
REM   is the rung that survived the 160M probe.
REM
REM   No plateau guard. At 121k steps a "flat bpb" is the LR still being
REM   high, not convergence. min-steps-frac 0.5 keeps the divergence guard
REM   live for the second half; degrade-frac 0.30 is deliberately loose
REM   because the healthy shape here is monotone decline, not a spike.
REM
REM   eff batch 32 (16x2). Measured 64,332 tok/s at 9.4GB/16GB VRAM.
REM   eff 16 ran 53,994 tok/s; eff 64 did not fit. Bigger batch = fewer
REM   optimizer steps for the same tokens, so the schedule is shorter but
REM   each step is better conditioned.
REM
REM   Steps 121,091 = exactly one epoch over the 1.98B train tokens.
REM   cosine decays to 10% of peak, so the low-LR endgame is REACHED —
REM   the thing the last run never got to do.
REM
REM DATA: 2,003,992,003 tokens from data/pile_train_full.txt via the
REM byte-faithful parallel tokenizer (scripts/tokenize_pile_parallel.py) and
REM the REAL HF Pythia tokenizer. Do not re-tokenize with text-mode reads:
REM CRLF stripping cost 0.77% of the corpus (see tests/test_corpus_newlines.py).
REM
REM Runtime ~8.6h. Checkpoints every 1000 steps, val every 1000.
REM ===========================================================================
cd /d D:\Projects\llm-lab

D:\Projects\llm-lab\.venv\Scripts\python.exe -u -m train.train ^
  --preset pythia ^
  --tokenizer data/incoming/pythia70m_hf/tokenizer.json ^
  --tok_cache data/pile_train_full_bpe_pythia70m.bin ^
  --corpus data/pile_train_full.txt ^
  --lr 5e-4 ^
  --steps 121091 ^
  --batch 16 ^
  --accum 2 ^
  --val_every 1000 ^
  --val_frac 0.01 ^
  --out checkpoints ^
  --patience-frac 0 ^
  --min-steps-frac 0.5 ^
  --degrade-frac 0.30 ^
  >> logs\pythia70_pile_full.log 2>&1
