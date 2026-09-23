@echo off
rem LR ladder completion rungs 2026-09-22 (post-app-update resume).
rem 50k rungs match the earlier probe exactly: 1000 steps, batch 16 x accum 4 = eff 64.
rem 16k rung matches its sweep: 2500 steps, batch 32 x accum 2 = eff 64.
set PY=C:\Users\shane\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe
cd /d D:\Projects\llm-lab

echo === 50k RETRY: lr=1.5e-3 (rerun of the ^C'd arm) %date% %time% === >> logs\lr_sweep_50k.log
%PY% -m train.train --steps 1000 --batch 16 --accum 4 --lr 1.5e-3 --preset pythia --corpus data/openwebtext_combined.txt --tokenizer checkpoints/bpe_owt50k_v2.json --tok_cache data/openwebtext_combined_bpe_owt50kv2.bin --use_rope --val_every 100 >> logs\lr_sweep_50k.log 2>&1
echo === 50k RETRY DONE: lr=1.5e-3 exit=%errorlevel% %date% %time% === >> logs\lr_sweep_50k.log

echo === 50k RUNG: lr=2e-3 %date% %time% === >> logs\lr_sweep_50k.log
%PY% -m train.train --steps 1000 --batch 16 --accum 4 --lr 2e-3 --preset pythia --corpus data/openwebtext_combined.txt --tokenizer checkpoints/bpe_owt50k_v2.json --tok_cache data/openwebtext_combined_bpe_owt50kv2.bin --use_rope --val_every 100 >> logs\lr_sweep_50k.log 2>&1
echo === 50k RUNG DONE: lr=2e-3 exit=%errorlevel% %date% %time% === >> logs\lr_sweep_50k.log

echo === 50k RUNG: lr=3e-3 %date% %time% === >> logs\lr_sweep_50k.log
%PY% -m train.train --steps 1000 --batch 16 --accum 4 --lr 3e-3 --preset pythia --corpus data/openwebtext_combined.txt --tokenizer checkpoints/bpe_owt50k_v2.json --tok_cache data/openwebtext_combined_bpe_owt50kv2.bin --use_rope --val_every 100 >> logs\lr_sweep_50k.log 2>&1
echo === 50k RUNG DONE: lr=3e-3 exit=%errorlevel% %date% %time% === >> logs\lr_sweep_50k.log

echo === 16k EXTENSION RUNG: lr=4e-3 %date% %time% === >> logs\lr_sweep.log
%PY% -m train.train --steps 2500 --batch 32 --accum 2 --lr 4e-3 --preset pythia --corpus data/openwebtext_combined.txt --tokenizer data/bpe_owt16k.json --tok_cache data/openwebtext_combined_bpe_owt16k.bin --use_rope --val_every 100 >> logs\lr_sweep.log 2>&1
echo === 16k RUNG DONE: lr=4e-3 exit=%errorlevel% %date% %time% === >> logs\lr_sweep.log

echo === ALL RUNGS COMPLETE %date% %time% === >> logs\lr_rungs_20260922.done
