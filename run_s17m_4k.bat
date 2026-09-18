@echo off
cd /d D:\Projects\llm-lab
echo Starting S17M on 4K vocab at %date% %time% > overnight_s17m_4k.log
python -u -m train.train --steps 20000 --batch 64 --preset s17m --corpus data/openwebtext_sample_1gb.txt --tokenizer data/bpe_owt4k.json --use_rope --val_every 100 >> overnight_s17m_4k.log 2>&1
echo Finished at %date% %time% >> overnight_s17m_4k.log
