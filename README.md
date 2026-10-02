# LLM Lab

A language model built from scratch — no HuggingFace trainer, no `transformers`,
our own BPE, our own Transformer, our own loop — plus the experiment harness that
keeps receipts on every claim it makes.

The point isn't the model. The point is that every number here traces to a run
directory, and every experiment is an *arm* with a declared lever, a control, and
a verdict. Several of our worst mistakes turned out to be measurement bugs.

**Status: we beat `EleutherAI/pythia-70m` at matched tokens.**

| | bpb @ ctx 512 | tokens seen |
|---|---|---|
| **Ours** (hand-rolled trainer, 70.7M) | **1.3947** | 2.004B |
| `pythia-70m` @ step 1000 | 1.4715 | 2.097B |
| **Gap** | **−0.0768 (5.2% ahead)** | 1.046× our budget |

Scored on identical token ids, one harness, same seeds. An earlier figure of
"+18.7% behind" is **superseded** — it compared our 2.0B tokens against Pythia's
*final* checkpoint at 299.9B, a 150× token mismatch reported as a quality gap.
Don't quote it. Reasoning lives in `reports/matched_tokens_70m.json` and commit
`a3a3b8b`.

At 28.3 tokens/param we sit deliberately past Chinchilla's 20 — which is why
the samples contain memorized fragments. Tradeoff made knowingly.

## Quickstart

```bash
git clone <repo> && cd llm-lab
bash scripts/setup.sh                     # venv + deps + CUDA torch on Windows
uv run python -c "import torch; print(torch.cuda.is_available())"
```

Mac only needs `uv sync`. To train off a token cache, pass `--tok_cache`:

```bash
uv run python train/train.py --tok_cache data/<corpus>.bin --preset pythia ...
```

> **Never launch training as a child of a running app.** An update will kill it.
> Use the detached path — `viz/arm.py` then the watchdog — or Task Scheduler.

## State

The block below is generated from the runs themselves, not typed by hand, so it
cannot silently drift from disk:

```bash
python scripts/readme_state.py          # rewrite the block
python scripts/readme_state.py --check  # exit 1 if stale (CI / cron)
```

<!-- AUTO:STATE:BEGIN -->
*Generated 2026-10-02 00:12 from `runs/*/loss.jsonl` + `experiments.json`, git `a3a3b8b`. Do not hand-edit — run `python scripts/readme_state.py`.*

### Best on Pile

| Rank | Run | bpb | @ step | Tokens seen | Params |
|------|-----|-----|--------|-------------|--------|
| 1 | `20260930_2055_pythia_tokenizer_pile_train_full` ⭐ | **1.4033** | 13400 | 1.76B | 70.7M |
| 2 | `20260930_0907_pythia_tokenizer_pile_train_full` | **1.5226** | 4900 | 642M | 70.7M |
| 3 | `20260929_2133_pythia_tokenizer_pile_train_full` | **1.9683** | 3000 | 49M | 70.7M |
| 4 | `20260929_2159_pythia_tokenizer_pile_train_full` | **2.0132** | 4000 | 66M | 70.7M |

_Slices are separate. `20260930_2055_pythia_tokenizer_pile_train_full` scored against `data/pile_train_full.txt` — do not rank it against a run from another slice._

### Best on OpenWebText

| Rank | Run | bpb | @ step | Tokens seen | Params |
|------|-----|-----|--------|-------------|--------|
| 1 | `20260922_1918_pythia_rope_bpe_owt50k_v2_openwebtext_combined` ⭐ | **1.4699** | 12000 | 393M | 70.7M |
| 2 | `20260922_1241_pythia_rope_bpe_owt16k_openwebtext_combined` | **1.5192** | 2500 | 82M | 35.8M |
| 3 | `20260919_1655_pythia_rope_bpe_owt16k_openwebtext_combined` | **1.5271** | 2500 | 82M | 35.8M |
| 4 | `20260919_1638_pythia_rope_bpe_owt16k_openwebtext_combined` | **1.5433** | 2500 | 82M | 35.8M |
| 5 | `20260923_0628_pythia160_rope_bpe_owt50k_v2_openwebtext_combined` | **1.5572** | 2400 | 79M | 162.6M |
| 6 | `20260919_1625_pythia_rope_bpe_owt16k_openwebtext_combined` | **1.5728** | 2500 | 82M | 35.8M |

_Slices are separate. `20260922_1918_pythia_rope_bpe_owt50k_v2_openwebtext_combined` scored against `data/openwebtext_combined.txt` — do not rank it against a run from another slice._

### Best on owt_4p5gb_combined

| Rank | Run | bpb | @ step | Tokens seen | Params |
|------|-----|-----|--------|-------------|--------|
| 1 | `20260926_1326_m66m_rope_bpe_owt4k_owt_4p5gb_combined` ⭐ | **1.6221** | 2300 | 75M | 66.2M |

_Slices are separate. `20260926_1326_m66m_rope_bpe_owt4k_owt_4p5gb_combined` scored against `data/owt_4p5gb_combined.txt` — do not rank it against a run from another slice._

### In flight

None. No curve has moved in the last 60 minutes. GPU is idle.

### Registry — 27 arms

closed 10 · fail 8 · pass 7 · inconclusive 1 · running 1

Open arms (not `closed`):

| Arm | Family | Verdict | Goal |
|-----|---------|---------|------|
| `ctx1024-exp013` | context-length | **inconclusive** | Does 1024 ctx beat 512 at equal tokens on the m50m trunk? |
| `exp014-m50m-50k` | capacity-50k | **fail** | Vocab axis: 113.9M trunk on the 50k stream. |
| `exp015-pythia-50k` | capacity-50k | **pass** | Capacity axis rung 1: SMALLER trunk on 50k — is the m50m failure shape-specific? |
| `p160-fix-long` | capacity-50k | **pass** | Single-lever (architecture parity) re-run of the collapsed 160M arm. Questions: survive step 4000? beat 70M... |
| `p160-fix-validation` | capacity-50k | **pass** | 200-step smoke: does the fixed stack train cleanly? |
| `p160-long-pre-fix` | capacity-50k | **fail** | Capacity axis rung 3: does 2.3x params beat the 70M? |
| `p160-lr-15e-4` | pythia160-lr-probe | **fail** | 160M LR probe rung. |
| `p160-lr-1e-3` | pythia160-lr-probe | **fail** | 160M LR probe rung. |
| `p160-lr-5e-4` | pythia160-lr-probe | **pass** | 160M LR probe (1000 steps, eff64). |
| `p50k-lr-15e-4` | pythia50k-70m-lr-probe | **pass** | 50k 70M LR probe; winner of the ladder. |
| `p50k-lr-3e-3` | pythia50k-70m-lr-probe | **fail** | 50k 70M LR probe rung (bracket test). |
| `pile-eff128k-5k` | pile-batch-knee | **pass** | Knee probe at eff batch 131,072 (micro 16 x accum 16), LR held at 5e-4. Single-lever vs pile-full-diet-70m:... |
| `pile-full-diet-70m` | pile-full-diet | **fail** | Pythia-70M shape on the FULL 2.0B-token Pile diet, one epoch, to measure the open-weights gap vs EleutherAI... |
| `pile-full-diet-70m-lr25` | pile-full-diet | **fail** | Same full-diet Pile arm at lr 2.5e-4 (half of 5e-4). The 5e-4 arm reached best bpb 1.9683 @ step 3000 then ... |
| `pile-pythia-full-epoch` | pile-chinchilla-epoch | **running** | FULL EPOCH over the Pile at Pythia-70M parity: one pass over the 2,003,992,003-token cache = 15,136 steps a... |
| `pythia16k-lr-3e-3` | pythia16k-lr-ladder | **pass** | 16k LR ladder rung; winner. |
| `pythia16k-lr-4e-3` | pythia16k-lr-ladder | **fail** | 16k LR ladder rung (beyond bracket). |

> ⚠️ **Verdict drift.** Marked `running` in `experiments.json`, but nothing is training: `pile-pythia-full-epoch`. Update the registry (`viz/arm.py` / `experiments.json`) — the run finished, this row didn't.

_Full arm history with levers and controls: `experiments.json`, rendered at `viz/training.html`._
<!-- AUTO:STATE:END -->

## Layout

```
llm-lab/
├── train/train.py          # trainer — resume, early stopping, leading metrics
├── model/
│   ├── transformer.py      # pre-norm, GELU, RoPE, parallel residuals
│   ├── block.py            # parallel residual, rotary_pct
│   ├── attention.py        # MHA + rotary position embeddings
│   ├── config.py           # ModelConfig presets (m50m, pythia, pythia160, …)
│   └── bpe.py              # BPE tokenizer
├── viz/
│   ├── dashboard.py        # zero-deps HTML decision board (--watch)
│   ├── registry.py         # run registry checker (anti-circularity layer)
│   ├── arm.py              # arm a new experiment (writes spec + registry row)
│   ├── probe.py            # offline checkpoint X-ray (layer norms, attn entropy)
│   └── hooks.py            # forward-capture connector (used by probe.py)
├── scripts/readme_state.py # generates the State block above
├── experiments.json        # experiment registry — one row per ARM
├── train_job.json          # active job spec (arm.py writes this)
├── train_watchdog.py       # cron supervisor
├── train_launch.py         # detached launcher (called by watchdog)
├── prune_checkpoints.py    # keeper-aware checkpoint retention
├── audit_run_data.py       # data integrity checker
├── reports/                # score reports (matched tokens, audits)
└── data/                   # corpora + tokenized binaries
```

## Running things

```bash
python viz/arm.py --id <name> -- <train args>   # arm an experiment
python train_watchdog.py                        # supervisor, cron every 10 min
python viz/dashboard.py [--watch 30]            # rebuild the board
python viz/probe.py checkpoints/<ckpt>.pt       # offline checkpoint X-ray
python viz/registry.py --check                  # registry vs disk integrity
python prune_checkpoints.py                     # keeper-aware retention
python audit_run_data.py                        # data integrity audit
```

**Supervision chain:** watchdog → schtasks `Hermes_TrainRun` → hidden vbs →
`run_train.bat` → `train_launch.py` → trainer. Task Scheduler is the parent, so
app restarts can't kill a run. Resume always takes the newest **step**
checkpoint, never `_best.pt`.

## Principles

- Understand before abstracting.
- Keep model and harness separate.
- Prefer measurable experiments.
- Reproduce across machines.
- Record what changed and what happened.
- **Compare like-for-like, or don't quote the percentage.** Every wrong verdict
  we produced came from a mismatched denominator — token count, val slice, vocab.

## Machines

| Machine | Role | Backend |
|---|---|---|
| PC (Ronin, 4070 Ti Super) | primary dev/training | CUDA |
| Mac (MacBook Pro) | training/inference | MPS |
| Mini (Mac Mini) | inference/testing | CPU |

## History

The full experiment log — all 16 numbered experiments, LR ladders, viz pass
history, audit findings, keeper backups, corpus inventory, and the shell traps
that cost us runs — lives in **[ARCHIVE.md](ARCHIVE.md)**.