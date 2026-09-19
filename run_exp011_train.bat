@echo off
REM Exp 011: M50M on the full OWT corpus with the 16k vocab.
REM Hypothesis: 16k vocab gives the embedding more resolution than 4k did
REM (exp 010 peaked at val 4.0986 @ step 2600, then memorised to val 9.98).
REM Trains off the memmapped .bin — never a 38GB .pt (that is how run 010 died ~step 17k).
REM Patience is set: without it the model burns hours memorising after the knee.
cd /d D:\Projects\llm-lab
if not exist logs mkdir logs
echo === exp 011 start %date% %time% === > logs\exp011_train.log
python -u -m train.train --steps 20000 --batch 32 --preset m50m --corpus data/openwebtext_combined.txt --tokenizer data/bpe_owt16k.json --tok_cache data/openwebtext_combined_bpe_owt16k.bin --use_rope --val_every 100 --patience 10 --min_delta 0.001 >> logs\exp011_train.log 2>&1
echo === exp 011 exit %errorlevel% at %date% %time% === >> logs\exp011_train.log
