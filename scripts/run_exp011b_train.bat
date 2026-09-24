@echo off
REM Exp 011b: resume exp 011 (m50m, 16k OWT vocab) with the MEASUREMENT fixed.
REM
REM WHY THIS RUN EXISTS
REM   Exp 011 was read as a failure ("16k vocab is worse than 4k") because val loss
REM   was compared PER TOKEN across two different vocabularies. Per-token CE is not
REM   comparable that way: a 16k BPE emits fewer, more informative tokens, so its
REM   per-token loss is mechanically higher at identical compression quality.
REM   Byte-normalised, exp 011 is actually AHEAD -- 1.7271 vs 1.7596 bits/byte
REM   (16k vs 4k, measured on the same held-out tail text). The raw 18.6% "worse"
REM   per-token gap was arithmetic artifact.
REM
REM   The run was also stopped early for the wrong reason: patience was
REM   `10 checks x val_every 100` = 1000 steps = 5% of a 20k-step cosine schedule,
REM   so it early-stopped at step 3300 while the LR was still 2.8e-4. A flat val
REM   at that point is the schedule, not convergence. Exp 010 (best @ step 2600,
REM   run died ~step 17k) is premature for the same reason, so its "the embedding
REM   layer is the bottleneck" conclusion is unproven too.
REM
REM WHAT CHANGED IN train/train.py
REM   - val is reported and DECIDED ON in bits/byte (--val_every rows in loss.jsonl
REM     now carry val_bpb alongside val)
REM   - patience is a FRACTION of the schedule (--patience-frac 0.10 = 2000 steps),
REM     no longer a check count that silently depends on --val_every
REM   - no plateau stop before --min-steps-frac of the schedule (0.5 = step 10000)
REM   - a genuine climb (--degrade-frac 0.05, i.e. >5% over best) still aborts, so
REM     the exp 010-style memorisation blowup is still caught
REM
REM INTERPRETER -- READ THIS
REM   Bare `python` resolves to the Hermes venv: torch 2.5.1+cu121, CUDA available.
REM   Do NOT launch with .venv\Scripts\python.exe -- llm-lab's own .venv carries
REM   torch 2.14.0+cpu and will silently train on CPU (the run still "works",
REM   it is just ~50x slower). Confirm the FIRST line of output says device=cuda.
REM
REM RESUMES from exp 011's best checkpoint (step 3300), so only ~16700 steps remain
REM   (~1.9h at the measured 0.4s/step) rather than a fresh 20000.
cd /d D:\Projects\llm-lab
if not exist logs mkdir logs
echo === exp 011b start %date% %time% === > logs\exp011b_train.log
python -u -m train.train --steps 20000 --batch 32 --preset m50m --corpus data/openwebtext_combined.txt --tokenizer data/bpe_owt16k.json --tok_cache data/openwebtext_combined_bpe_owt16k.bin --use_rope --val_every 100 --patience-frac 0.10 --min-steps-frac 0.5 --degrade-frac 0.05 --resume checkpoints/exp002_m50m_rope_bpe_owt16k_202609181516_best.pt >> logs\exp011b_train.log 2>&1
echo === exp 011b exit %errorlevel% at %date% %time% === >> logs\exp011b_train.log
