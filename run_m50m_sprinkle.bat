@echo off
cd /d D:\Projects\llm-lab

echo === Sprinkler tokenize + M50M train === > overnight_m50m_full.log
echo Started at %date% %time% >> overnight_m50m_full.log

REM Phase 1: Parallel tokenization (8 workers)
echo Phase 1: Sprinkler tokenization... >> overnight_m50m_full.log
python -u sprinkle_tokenize.py data/openwebtext_combined.txt data/bpe_owt4k.json data/openwebtext_combined_bpe_owt4k.pt --workers 8 >> overnight_m50m_full.log 2>&1

REM Phase 2: Train M50M on full OWT
echo Phase 2: Training M50M at %date% %time% >> overnight_m50m_full.log
python -u -m train.train --steps 20000 --batch 32 --preset m50m --corpus data/openwebtext_combined.txt --tokenizer data/bpe_owt4k.json --use_rope --val_every 100 --tok_cache data/openwebtext_combined_bpe_owt4k.pt >> overnight_m50m_full.log 2>&1

echo === ALL DONE at %date% %time% === >> overnight_m50m_full.log
