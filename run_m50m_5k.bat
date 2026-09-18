@echo off
cd /d D:\Projects\llm-lab
echo Starting M50M 5k on full OWT at %date% %time% > overnight_m50m_5k.log
python -u -m train.train --steps 5000 --batch 32 --preset m50m --corpus data/openwebtext_combined.txt --tokenizer data/bpe_owt4k.json --use_rope --val_every 100 --tok_cache data/openwebtext_combined_bpe_owt4k.pt >> overnight_m50m_5k.log 2>&1
echo Finished at %date% %time% >> overnight_m50m_5k.log
