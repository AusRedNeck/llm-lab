@echo off
REM Rehearsal for the 16k pipeline: 3 shards x 20M chars -> verify -> 200 train steps.
REM About 2.5 minutes. Run this before the full tokenizer, every time.
cd /d D:\Projects\llm-lab
if not exist logs mkdir logs
echo === 16k SMOKE start %date% %time% === > logs\16k_smoke.log

python -u tokenize_owt_16k.py --smoke >> logs\16k_smoke.log 2>&1
if errorlevel 1 goto :fail

python -u verify_tokens.py --bin data/tok16k_smoke.bin --vocab data/bpe_owt16k.json --src data/openwebtext/shards/train-00000-of-00080.txt --manifest >> logs\16k_smoke.log 2>&1
if errorlevel 1 goto :fail

python -u -m train.train --steps 200 --batch 32 --preset m50m --corpus data/openwebtext_combined.txt --tokenizer data/bpe_owt16k.json --tok_cache data/tok16k_smoke.bin --use_rope --val_every 50 --patience 10 >> logs\16k_smoke.log 2>&1
if errorlevel 1 goto :fail

echo === SMOKE PASSED %date% %time% === >> logs\16k_smoke.log
goto :eof

:fail
echo === SMOKE FAILED at %date% %time% === >> logs\16k_smoke.log
exit /b 1
