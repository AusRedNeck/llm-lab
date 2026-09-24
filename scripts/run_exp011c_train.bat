@echo off
REM Exp 011c: exp 011b relaunched with the abort threshold CALIBRATED.
REM
REM WHAT HAPPENED TO 011b
REM   Run 011b proved the fix works (best bpb 1.7013 vs exp 010/4k 1.7596) and was
REM   then killed by my own mid-schedule guard at step 3700/20000:
REM       abort: bpb 1.7870 > best 1.7013 +5% at step 3700
REM   That guard was miscalibrated. Measuring the val series (runs/*/loss.jsonl):
REM       exp 011  : median |delta| 5.09% between checks, max +3.08% over best
REM       exp 011b : median |delta| 5.25%, max |delta| 10.38%, max +5.03% over best
REM       exp 010  : real memorisation was +143% over best
REM   A 5% threshold sits INSIDE the noise band, so it fired on the single largest
REM   ordinary wobble in 14 samples. Noise and divergence are separated by ~30x,
REM   so the threshold belongs well clear of the noise: 0.30.
REM
REM WHAT ELSE CHANGED
REM   --min-steps-frac  0.5 -> 0.6  (no plateau stop before step 12000)
REM   --patience-frac   0.10 -> 0.15 (3000 steps, not 2000, since 5% wiggles make a
REM                                  20-check flat stretch cheap to hit by chance)
REM   --degrade-frac    0.05 -> 0.30 (default in train.py also corrected)
REM   Net effect: the earliest a plateau can stop this run is step 15000, where the
REM   cosine LR has decayed to ~7e-5 from a 3e-4 peak -- i.e. late enough that a
REM   flat curve is evidence, not just the schedule.
REM
REM RESUMES from run 011b's best checkpoint (step 2500, val 4.7666 / bpb 1.7013),
REM   so the aborted 1200 steps are simply redone.
REM
REM INTERPRETER: bare `python` = Hermes venv (torch 2.5.1+cu121, CUDA). NEVER
REM   .venv\Scripts\python.exe -- that is torch 2.14.0+cpu and trains on CPU.
REM   Confirm the 2nd line of the log says device=cuda.
REM
REM LAUNCH (from git-bash): MSYS_NO_PATHCONV=1 cmd.exe /c "run_exp011c_train.bat"
REM   Plain `cmd //c foo.bat` does NOT work from git-bash -- it opens cmd, prints
REM   the banner and exits 0 without running anything (a masked failure).
cd /d D:\Projects\llm-lab
if not exist logs mkdir logs
echo === exp 011c start %date% %time% === > logs\exp011c_train.log
python -u -m train.train --steps 20000 --batch 32 --preset m50m --corpus data/openwebtext_combined.txt --tokenizer data/bpe_owt16k.json --tok_cache data/openwebtext_combined_bpe_owt16k.bin --use_rope --val_every 100 --patience-frac 0.15 --min-steps-frac 0.6 --degrade-frac 0.30 --resume checkpoints/exp002_m50m_rope_bpe_owt16k_202609181737_best.pt >> logs\exp011c_train.log 2>&1
echo === exp 011c exit %errorlevel% at %date% %time% === >> logs\exp011c_train.log
