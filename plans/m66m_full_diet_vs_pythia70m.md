# M66M vs Pythia-70M — full-diet fight plan (Ronin)

Status: READY. Waiting on Ronin mesh/app recovery. Everything below runs
as-is the moment 8644 is back. STATE sections 20–22 for the backstory.

## The fight

Our `m66m` (66.2M total, ~60M transformer, 4k vocab) vs HF
`EleutherAI/pythia-70m-deduped` (70M total, ~19M transformer, 50k vocab).
Same bpb metric, both turfs. Targets to beat:

| turf | ours s17m (now) | theirs 70M (target) |
|---|---|---|
| OWT home | 1.90 | 1.24 |
| Pile away | 2.89 | 1.49 |

## Diet math (the part we kept getting wrong — see STATE 22)

66.2M × 20 tok/param = **1.32B tokens**. 9 OWT shards ≈ 4.5GB ≈ 1.4B
4k-tokens ≈ **one epoch, no repeats**. At batch 64 × ctx 512 (32768
tok/step): **40500 steps**. Schedule to the diet, stop on signal —
never the 5k timer again.

## Phase 0 — smoke gate (Shane's rule, 200 steps max)

```
python -u -m train.train --steps 200 --batch 8 --preset m66m ^
  --corpus D:\Projects\llm-lab\data\owt_4p5gb_combined.txt ^
  --tokenizer D:\Projects\llm-lab\data\bpe_owt4k.json --use_rope --val_every 25
```

Gate: `AMP: cuda=True, bf16_supported=True` in the log head (fp16 NaNs
killed a d768 run before — memory), `torch.cuda.is_available()` True,
`__pycache__`-free `params~66.2M` line. Anything else: stop, report, no launch.

## Phase 1 — tokenize 9 shards (CPU, ~1hr, resume-safe)

```bat
cd /d D:\Projects\llm-lab
copy /b data\shard00.txt+data\shard01.txt+data\shard02.txt+data\shard03.txt+data\shard04.txt+data\shard05.txt+data\shard06.txt+data\shard07.txt+data\shard08.txt data\owt_4p5gb_combined.txt
python -u scripts\tokenize_stream.py data\owt_4p5gb_combined.txt data\bpe_owt4k.json data\owt_4p5gb_bpe_owt4k.bin
```

Shard names on Ronin are `E:\llm-lab-data\...` — adjust paths to wherever
the 38GB corpus actually lives (last verified: full OWT on Ronin disk,
2026-09-17). Any 9 consecutive shards ≈ 4.5GB work; note which in the run log.
`.bin` is the artifact (memmap, 0 RAM); `--resume` continues if killed.

## Phase 2 — train (overnight+, 4070 Ti)

```
python -u -m train.train --steps 40500 --batch 64 --preset m66m ^
  --corpus data\owt_4p5gb_combined.txt --tokenizer data\bpe_owt4k.json ^
  --tok_cache data\owt_4p5gb_bpe_owt4k.bin ^
  --use_rope --dropout 0.1 --val_every 100 ^
  --patience-frac 0.10 --min-steps-frac 0.5
```

Policy: 40500-step cosine schedule (knee follows the clock — STATE 22),
plateau guard armed but silent before 50%, degrade guard (default 0.30)
catches real divergence. LR peak default 3e-4 (M50M precedent); if val
knees with LR still high and batch-noise is suspected, the t1m protocol
(STATE 22) says scale LR, not steps.

## Phase 3 — score both turfs (Mac, MPS, minutes)

```
python -m eval.pile_eval --ckpt checkpoints/<m66m_best>.pt
python -m eval.hf_eval --model EleutherAI/pythia-70m-deduped --val_text data/incoming/openwebtext_sample_1gb.txt --max_samples 200
```

pile_eval covers Pile turf for ours; hf_eval covers OWT turf for theirs
(Pile numbers for theirs already logged: 1.49). Fill the table above,
append STATE 23, commit.

## If Ronin stays dark

Fallback: same plan on Mac at batch 16 (MPS RAM), ~4x slower. Not
recommended — 40500 steps becomes a multi-day burn. Better: shrink to a
33M preset (half diet, 660M tokens) as the Mac-sized version of this fight.
