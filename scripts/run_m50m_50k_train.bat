@echo off
REM Exp 014: the m50m trunk at the 50k vocab, 20k-step ceiling, early stop armed.
REM
REM WHAT THIS RUN IS
REM   Preset m50m (10L x d640 x 10H, ctx 512, RoPE) pointed at the 50k GPT-NeoX-style
REM   vocab instead of the 16k one. Corpus is the v2 re-encode of full OpenWebText
REM   (data/openwebtext_combined_bpe_owt50kv2.bin), produced with the FIXED tokenizer
REM   (full 256-byte alphabet -- see below).
REM
REM THE MODEL IS 113.9M PARAMS, NOT 55M
REM   The preset name is stale. Embedding and lm_head are untied and both carry a
REM   bias, so vocab scales three tensors: delta = (v2-v1)*d*2 + (v2-v1).
REM   Measured from the state dict: 55.34M @ 4,512 / 70.39M @ 16,256 / 113.94M @ 50,256.
REM   Real counts come from the model, never from the name.
REM
REM WHY --accum 2 AND WHY batch 32 IS NOT AN OPTION
REM   Measured on this box (4070 Ti SUPER, 16GB), same model, same corpus slice:
REM     batch 32 accum 1   ~2.7k tok/s   VRAM 15,383 / 16,376 MiB  <-- thrashing
REM     batch 16 accum 2   42.2k tok/s   VRAM 12,831 / 16,376 MiB  <-- this one
REM     batch 8  accum 4   41.8k tok/s   VRAM  7,987 / 16,376 MiB
REM   batch 32 fits but sits at 94% of the card and runs ~15x slower per token than
REM   the same effective batch split across accumulation. Accumulation does not add
REM   VRAM -- same micro-batch, same activations -- so use it. If VRAM ever becomes
REM   tight (a val pass or sample burst on top), drop to batch 8 accum 4: same
REM   throughput, 8GB used instead of 12.8GB.
REM   Effective batch = 16 x 2 = 32, matching every prior m50m run, so bpb stays
REM   comparable across exp011c / exp012 / exp014.
REM
REM EARLY STOP IS THE POINT OF THE 20k CEILING
REM   20,000 steps is a CEILING, not a target. At effective batch 32 that is 328M
REM   tokens ~ 3.8% of the corpus, and this line has measured its best bpb at
REM   ~90M tokens seen (0.9%) every time it has been run. The run is expected to end
REM   itself near the knee, well before 20k.
REM   patience-frac 0.15 = 3,000 steps of flat bpb (check count is derived from
REM   --val_every, so it cannot silently change length); min-steps-frac 0.6 holds the
REM   plateau stop off until step 12,000, where the cosine LR is far enough down that
REM   a flat curve is evidence rather than the schedule; degrade-frac 0.15 is 3x the
REM   largest noise excursion measured on this line and is the guard that actually
REM   catches overfitting (a plateau rule cannot -- val climbing while train falls
REM   is not flat).
REM
REM INTERPRETER: bare `python` = Hermes venv (torch 2.5.1+cu121, CUDA). NEVER
REM   .venv\Scripts\python.exe -- that is torch 2.14.0+cpu and trains on CPU.
REM   Confirm the 2nd line of the log says device=cuda.
REM
REM LAUNCH (from git-bash): MSYS_NO_PATHCONV=1 cmd.exe /c "run_m50m_50k_train.bat"
REM   Plain `cmd //c foo.bat` does NOT work from git-bash -- it opens cmd, prints
REM   the banner and exits 0 without running anything (a masked failure).
cd /d D:\Projects\llm-lab
if not exist logs mkdir logs
echo === exp 014 start %date% %time% === > logs\exp014_m50m_50k.log
python -u -m train.train --steps 20000 --batch 16 --accum 2 --preset m50m --corpus data/openwebtext_combined.txt --tokenizer checkpoints/bpe_owt50k_v2.json --tok_cache data/openwebtext_combined_bpe_owt50kv2.bin --use_rope --val_every 100 --patience-frac 0.15 --min-steps-frac 0.6 --degrade-frac 0.15 --dropout 0.1 >> logs\exp014_m50m_50k.log 2>&1
echo === exp 014 exit %errorlevel% at %date% %time% === >> logs\exp014_m50m_50k.log
