@echo off
cd /d D:\Projects\llm-lab
echo === Training 16k BPE tokenizer on full OWT === > overnight_tokenizer_16k.log
echo Start: %date% %time% >> overnight_tokenizer_16k.log
python -u scripts\train_tokenizer_openwebtext.py --input data/openwebtext_combined.txt --merges 16000 --out data/bpe_owt16k.json >> overnight_tokenizer_16k.log 2>&1
echo === DONE %date% %time% === >> overnight_tokenizer_16k.log
