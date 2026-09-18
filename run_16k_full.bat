@echo off
REM Full 16k tokenization of all 80 OpenWebText shards, then self-verify.
REM Serialized, resumable, writes only under llm-lab\data (never C:).
cd /d D:\Projects\llm-lab
if not exist logs mkdir logs
echo === 16k tokenize start %date% %time% === > logs\16k_tokenize.log

python -u tokenize_owt_16k.py >> logs\16k_tokenize.log 2>&1
echo === encode finished %date% %time% === >> logs\16k_tokenize.log

python -u verify_tokens.py --bin data/openwebtext_combined_bpe_owt16k.bin --vocab data/bpe_owt16k.json --src data/openwebtext/shards/train-00000-of-00080.txt --manifest >> logs\16k_tokenize.log 2>&1
echo === verify exit code %errorlevel% === >> logs\16k_tokenize.log
echo === ALL DONE %date% %time% === >> logs\16k_tokenize.log
