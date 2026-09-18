@echo off
cd /d D:\Projects\llm-lab
echo === M50M 20k Full OpenWebText === > overnight_m50m_20k.log
echo Start: %date% %time% >> overnight_m50m_20k.log
python -u -m train.train --steps 20000 --batch 32 --preset m50m --corpus data/openwebtext_combined.txt --tokenizer data/bpe_owt4k.json --use_rope --val_every 100 >> overnight_m50m_20k.log 2>&1
echo === DONE %date% %time% === >> overnight_m50m_20k.log
