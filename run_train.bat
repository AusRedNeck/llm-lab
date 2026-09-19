@echo off
REM ============================================================================
REM Training launch (DETACHED). Fired by the Hermes_TrainRun scheduled task, which
REM the watchdog (train_watchdog.py) invokes ONLY when the job is dead + incomplete.
REM
REM The task is the parent, not Hermes - that is the whole point: a run launched
REM from a Hermes terminal dies when the app restarts (exp011c was killed that way
REM at 19:05 on 2026-09-18, mid-run). Task Scheduler survives the app.
REM
REM Optional arg is passed through to the launcher; "--dry-run" is used to test the
REM dispatch wiring without starting a second trainer.
REM
REM INTERPRETER PINNED ON PURPOSE: D:\Projects\llm-lab\.venv is torch 2.14+CPU.
REM Only the Hermes venv python has CUDA. Never let PATH decide this.
REM ============================================================================

set PYTHONPATH=
cd /d D:\Projects\llm-lab
if not exist logs mkdir logs

echo === run_train.bat start %date% %time% args=[%*] === >> logs\train_launch.out
"C:\Users\shane\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe" train_launch.py %* >> logs\train_launch.out 2>&1
echo === run_train.bat exit %errorlevel% at %date% %time% === >> logs\train_launch.out
