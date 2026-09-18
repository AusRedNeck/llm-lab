@echo off
REM Full 16k tokenization, PARALLEL (8 worker processes), then self-verify.
REM
REM Safe to kill and rerun at any point:
REM   * finished shards are recorded in the manifest and skipped next time
REM   * a half-written shard is truncated and redone (token streams can't have holes)
REM   * an existing serial output is adopted as part00, never re-encoded
REM   * orphans from a crashed run are reaped at startup, and workers die with
REM     the parent — no stray processes eating RAM
cd /d D:\Projects\llm-lab
if not exist logs mkdir logs
echo === 16k PARALLEL tokenize start %date% %time% === > logs\16k_tokenize_par.log

python -u tokenize_owt_16k.py --workers 8 --status-every 60 >> logs\16k_tokenize_par.log 2>&1
echo === encode exit %errorlevel% at %date% %time% === >> logs\16k_tokenize_par.log

python -u verify_tokens.py --bin data/openwebtext_combined_bpe_owt16k.bin --vocab data/bpe_owt16k.json --src data/openwebtext/shards/train-00000-of-00080.txt --manifest >> logs\16k_tokenize_par.log 2>&1
echo === verify exit %errorlevel% === >> logs\16k_tokenize_par.log
echo === ALL DONE %date% %time% === >> logs\16k_tokenize_par.log
