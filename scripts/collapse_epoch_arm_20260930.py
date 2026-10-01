"""Collapse the 2026-09-30 Pile arms into the registry's ONE-ROW-PER-ARM shape.

Background: the 09:07 eff-128k probe finished (best val bpb 1.53965 @ step 5000). The
full-epoch arm was stopped by Shane at step 856 to free the GPU, then restarted fresh at
20:55 rather than resuming aborted weights into a restarted cosine.

SHAPE (the part that is easy to get wrong):
experiments.json is ONE ROW PER ARM, and `run_dirs` holds that arm's launch attempts,
newest last. So the 11:43 abort and the 20:55 restart are TWO run_dirs on the SAME row
`pile-pythia-full-epoch`. An earlier pass of this file split them into two rows
(`pile-pythia-full-epoch-1143` + an empty `pile-pythia-full-epoch`), which invents an
experiment that does not exist, strands the abort's evidence on a row that can never
carry a verdict (registry.py only ever diffs the NEWEST run dir against the control's
newest), and breaks the control chain. This collapses it back.

An operator abort is a RESULT, not debris: the abort keeps its curve, its step, and its
reason, on the arm it belongs to.

Run:  .venv/Scripts/python.exe scripts/register_closed_arms_20260930.py
Then: .venv/Scripts/python.exe viz/registry.py --check   (must PASS)
"""

import json
import pathlib
import sys

REG = pathlib.Path(__file__).resolve().parent.parent / "experiments.json"

ABORTED_DIR = "20260930_1143_pythia_tokenizer_pile_train_full"
ACTIVE_DIR = "20260930_2055_pythia_tokenizer_pile_train_full"
SPLIT_ID = "pile-pythia-full-epoch-1143"


def main():
    data = json.loads(REG.read_text(encoding="utf-8"))
    raw = data["arms"]
    # registry.py iterates arms as a LIST. A dict here dies with a confusing
    # "string indices must be integers" and takes the whole gate down silently.
    if isinstance(raw, dict):
        order = list(raw)
        arms = dict(raw)
    else:
        order = [a["id"] for a in raw]
        arms = {a["id"]: a for a in raw}
    before_n = len(order)

    # --- the full-epoch arm: one row, two launch attempts, live verdict ---
    # The only real difference vs the 5k probe is resolved cfg.dropout 0.1 -> 0.0, and
    # that lives in GIT (a966f1e, model/config.py PYTHIA_6L512), not in the trainer args:
    # every header carries dropout: null and both presets resolve to identical
    # batch/accum/lr/tokenizer. registry.py diffs train ARGS only, so the honest
    # declaration is a `code` lever; declaring dropout as an arg lever fails the gate as
    # claimed-but-unchanged, which is the exact failure the gate exists to catch.
    epoch = {
        "id": "pile-pythia-full-epoch",
        "family": "pile-chinchilla-epoch",
        "run_dirs": [ABORTED_DIR, ACTIVE_DIR],
        "control_id": "pile-eff128k-5k",
        "lever": {
            "code": "a966f1e pythia preset dropout 0.1->0.0 (EleutherAI parity; resolved cfg 0.1->0.0)"
        },
        "verdict": "running",
        "goal": (
            "FULL EPOCH over the Pile at Pythia-70M parity: one pass over the "
            "2,003,992,003-token cache = 15,136 steps at eff batch 131,072 (micro 16 x "
            "accum 16), ctx 512, dropout 0.0, lr 5e-4 cosine. TARGET IS CHINCHILLA, not "
            "pythia-70m's 1.2727: 20 tok/param = step 10,794 predicts ~1.36 bpb; pass mark "
            "~1.42 bpb at step 10,794. pythia-70m saw 4,258 tok/param (212x Chinchilla, 300B "
            "tokens), so its number is an ASYMPTOTE, not a reachable target."
        ),
        "note": (
            "TWO LAUNCH ATTEMPTS ON ONE ARM. (1) 20260930_1143 ABORTED BY OPERATOR at step "
            "856 (~19 min GPU) so Shane could use the card; curve kept at "
            f"runs/{ABORTED_DIR}/loss.jsonl (val bpb 2.0138 @600, 1.9785 @800, still "
            "descending). Its numbers are NOT comparable to the 5k probe - the probe's cosine "
            "was fully cooled by step 5,000 while this attempt was still at ~5.0e-4, so the "
            "hotter LR is schedule position, not a regression. (2) 20260930_2055 RESTARTED "
            "FRESH at step 0 rather than resuming the aborted weights, because a mid-cosine "
            "resume would restart the LR schedule against a stale early-stop ledger. Do NOT "
            "compare this arm's step-5000 to the probe's 1.5396: same tokens seen, but the "
            "cosine here is stretched over 15,136 so LR is still ~3.9e-4 at step 5,000. "
            "Compare final-to-final. GRADED READING: a small constant offset (+0.16 to +0.22 "
            "nats vs the Chinchilla fit, shrinking) is expected from a population regression; "
            "a GROWING gap is the failure mode and is what killed the eff-16k arm. Disk: "
            "checkpoints already 163G, D: has 1.15TB free, this run adds ~21G."
        ),
    }
    changed = []
    if arms.get("pile-pythia-full-epoch") != epoch:
        arms["pile-pythia-full-epoch"] = epoch
        changed.append("pile-pythia-full-epoch -> running, 2 run_dirs (aborted 1143 + live 2055)")

    if SPLIT_ID in arms:
        del arms[SPLIT_ID]
        order = [i for i in order if i != SPLIT_ID]
        changed.append(f"removed split row {SPLIT_ID} (one arm, not two - abort belongs on the epoch row)")

    merged = [arms[i] for i in order]
    merged += [row for row in arms.values() if row["id"] not in order]
    data["arms"] = merged
    REG.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    for c in changed:
        print("changed:", c)
    if not changed:
        print("no change (already registered)")
    print(f"arms on disk: {len(merged)} (was {before_n})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
