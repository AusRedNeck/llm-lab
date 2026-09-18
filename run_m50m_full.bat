@echo off
cd /d D:\Projects\llm-lab
echo === Phase 1: Tokenize full OWT (4K) at %date% %time% === > overnight_m50m_full.log
python -u tokenize_stream.py data/openwebtext_combined.txt data/bpe_owt4k.json data/openwebtext_combined_bpe_owt4k.pt >> overnight_m50m_full.log 2>&1
echo === Phase 1 done at %date% %time% === >> overnight_m50m_full.log
echo === Phase 2: Train M50M at %date% %time% === >> overnight_m50m_full.log
python -u -m train.train --steps 20000 --batch 32 --preset m50m --corpus data/openwebtext_combined.txt --tokenizer data/bpe_owt4k.json --use_rope --val_every 100 >> overnight_m50m_full.log 2>&1
echo === ALL DONE at %date% %time% === >> overnight_m50m_full.log
