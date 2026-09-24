@echo off
REM Overnight LLM training — dense111m on OpenWebText
REM Resume from step 2500, runs to 40000
REM This runs independently of Hermes — tab switches won't kill it.
cd /d D:\Projects\llm-lab
echo %date% %time% > overnight_resume.log
python -m train.train --preset dense111m --corpus data/openwebtext_sample_1gb.txt --tokenizer checkpoints/bpe8k.json --use_rope --steps 40000 --batch 8 --accum 4 --dropout 0.1 --val_every 500 --resume checkpoints/exp002_dense111m_rope_bpe8k_202609152207_step2500.pt >> overnight_resume.log 2>&1
echo %date% %time% DONE >> overnight_resume.log
