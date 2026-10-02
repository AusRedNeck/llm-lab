# Authors

## Built and directed by

**Shane Lowry** — [@AusRedNeck](https://github.com/AusRedNeck)

Author, designer, and the person who decided what was worth measuring. Every
experiment arm in `experiments.json` was chosen, launched, and judged by hand;
the code is the record of those decisions, not the other way around.

## How this was built

Hand-rolled from scratch on purpose. No `transformers`, no HuggingFace trainer,
no downloaded model weights — the BPE tokenizer, the Transformer, the training
loop, and the evaluation harness are all in this repo and all readable in one
sitting.

AI tooling assisted with drafting and review. It did not choose the
architecture, the levers, or the verdicts, and it is not the author of record.
Some of the best bugs in this repo were found by having that tooling read the
code carefully and disagree out loud: the `--resume` overwrite in
`scripts/token_io.py` and the self-satisfying staleness gate in
`scripts/readme_state.py` were both caught that way, not by a test suite that
had been passing green the whole time.

The commit history is authored under one identity because it is one person's
project across three machines. The one exception is preserved in the backup tag
`pre-author-rewrite-20261002`, which still holds the original attribution for
`daffc35`.

## Machines this ran on

| Machine | Role | Backend |
|---|---|---|
| PC (Ronin, 4070 Ti Super) | primary dev/training | CUDA |
| Mac (MacBook Pro) | training/inference | MPS |
| Mini (Mac Mini) | inference/testing | CPU |

Loss values matched to 4 decimals across CUDA and MPS for identical configs,
which is the check that made cross-machine reproduction worth trusting.