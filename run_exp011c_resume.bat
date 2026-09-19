@echo off
REM ============================================================================
REM Exp 011c RESUME  (written 2026-09-18 19:3x)
REM
REM WHY THIS EXISTS
REM   The original exp 011c run was HARD-KILLED at ~step 5170 / 20000 when the
REM   Hermes desktop app restarted at 19:05. It had been launched from a Hermes
REM   terminal call, so it was a child of the app's process tree and died with it.
REM   The log has no "exp 011c exit" marker -> killed, not a clean exit or a crash.
REM
REM WHAT IT RESUMES FROM
REM   This run's OWN periodic checkpoint at step 5000 (weights + optimizer + step).
REM   NOT the *_best.pt (that one is step 3500, val_bpb 1.6436) - resuming from
REM   step 5000 keeps 1500 extra steps of work. train.py re-inits best_bpb to inf
REM   on resume, which is harmless with --degrade-frac 0.30.
REM
REM INTERPRETER IS PINNED ON PURPOSE
REM   D:\Projects\llm-lab\.venv has torch 2.14.0+CPU (cuda False).
REM   ONLY C:\Users\shane\AppData\Local\hermes\hermes-agent\venv has torch
REM   2.5.1+cu121 with cuda True. Letting PATH decide silently falls back to CPU.
REM   PYTHONPATH is cleared so no foreign site-packages can shadow torch.
REM
REM DETACHMENT: launched by run_exp011c_resume_hidden.vbs via the
REM   Hermes_Exp011cResume scheduled task, so an app restart cannot kill it again.
REM ============================================================================

set PYTHONPATH=
cd /d D:\Projects\llm-lab
if not exist logs mkdir logs

echo === exp 011c RESUME start %date% %time% === >> logs\exp011c_resume.log

"C:\Users\shane\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe" -u -m train.train --steps 20000 --batch 32 --preset m50m --corpus data/openwebtext_combined.txt --tokenizer data/bpe_owt16k.json --tok_cache data/openwebtext_combined_bpe_owt16k.bin --use_rope --val_every 100 --patience-frac 0.15 --min-steps-frac 0.6 --degrade-frac 0.30 --resume checkpoints/exp002_m50m_rope_bpe_owt16k_202609181806_step5000.pt >> logs\exp011c_resume.log 2>&1

echo === exp 011c RESUME exit %errorlevel% at %date% %time% === >> logs\exp011c_resume.log
