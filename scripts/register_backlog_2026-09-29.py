#!/usr/bin/env python3
"""Backfill experiments.json for the runs that were never registered (2026-09-29).

WHY THIS EXISTS
    `viz/registry.py --check` failed with 21 unregistered run dirs. The gate is
    the anti-circular layer: it refuses to let a comparison be quoted when the
    arms it rests on are unrecorded. But blanket-ignoring all 21 would mute the
    gate rather than satisfy it, and silently discard real evidence.

    The 21 dirs split into two genuinely different classes, so they get two
    different treatments:

      1. REAL EVIDENCE -> registered as arms with their control + lever.
         These carry curves that later reasoning depends on. In particular
         20260927_0752 is the OWT run the full-diet arm's lr choice is a
         correction OF, so without it "lr 5e-4, not 1.5e-3" is an unsourced
         claim.

      2. CALIBRATION SMOKES -> ignore_dirs, with the reason recorded.
         Short throughput/shape probes (batch sweeps, a 3-step ladder rung
         check). They answer "does it fit / how fast", not "what does the
         model learn", so they have no verdict to carry.

Run: .venv/Scripts/python.exe scripts/register_backlog_2026-09-29.py [--dry-run]
"""
import json
import os
import sys

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(LAB, "experiments.json")

# --- 1. real evidence: registered arms -------------------------------------
# control_id points at the arm that run was a rung/lift OF, where one exists.
NEW_ARMS = [
    {
        "id": "m66m-4p5gb-40k",
        "family": "m66m-owt4k-long",
        "run_dirs": ["20260926_1326_m66m_rope_bpe_owt4k_owt_4p5gb_combined"],
        "control_id": None,
        "lever": {"corpus": "openwebtext_combined->owt_4p5gb", "steps": "40500",
                  "lr": "3e-4", "eff_batch": "64", "vocab": "bpe_owt4k"},
        "goal": "66.2M on the 4.5GB OWT slice, 40.5k steps, long-schedule run.",
        "verdict": "closed",
        "note": "early_stop fired; 66.19M params. Pre-50k-vocab era (4k vocab).",
    },
    {
        "id": "pythia-owt50kv2-lr-15e-4-long",
        "family": "pythia-owt50kv2-lr",
        "run_dirs": ["20260927_0752_pythia_rope_bpe_owt50k_v2_openwebtext_combined"],
        "control_id": "p50k-lr-15e-4",
        "lever": {"corpus": "owt_subset->openwebtext_combined", "steps": "1000->20000",
                  "lr": "1.5e-3", "eff_batch": "64", "vocab": "owt50k_v2"},
        "goal": "The 20k OWT run the full-diet Pile arm is a correction of: hold "
                "lr 1.5e-3 long enough to watch the floor get eaten.",
        "verdict": "closed",
        "note": "early_stop; 70.65M. gnorm 0.255->0.711 while val bpb rose "
                "1.6134->1.8682; floor ~step 2300 then eroded ~9200 steps. "
                "This is the evidence behind the Pile arm's lr 5e-4 choice.",
    },
    {
        "id": "pile-native-tok-200step",
        "family": "pile-corpus-probe",
        "run_dirs": ["20260928_0937_pythia_bpe_pythia_native_pile_train"],
        "control_id": None,
        "lever": {"corpus": "owt50k_v2->pile(native tok)", "steps": "200",
                  "lr": "3e-4", "eff_batch": "8", "vocab": "bpe_pythia_native"},
        "goal": "First Pile-corpus probe: does the hand-rolled BPE read Pile at all.",
        "verdict": "closed",
        "note": "200 steps, 70.65M, 18MB tokenized sample. Proved the pipeline; "
                "superseded by the byte-faithful HF-tokenizer cache (f5713c3).",
    },
]

# --- 2. calibration smokes: ignored, with reason ---------------------------
IGNORE_DIRS = [
    # m66m 4k-vocab short probes (200/29/15 steps) - shape + speed, no verdict.
    "20260926_1157_m66m_rope_bpe_owt4k_openwebtext_combined",
    "20260926_1308_m66m_rope_bpe_owt4k_owt_4p5gb_combined",
    "20260926_1320_m66m_rope_bpe_owt4k_owt_4p5gb_combined",
    # pythia 130 steps, no early_stop and no exit marker - launch casualty.
    "20260926_2200_pythia_rope_bpe_owt50k_v2_openwebtext_combined",
    # 2026-09-29 batch-size sweep @ lr 5e-4 on the full Pile cache.
    # Measured 63,332 tok/s at eff32 (16x2), which is the number the
    # full-diet arm's window estimate rests on.
    "20260929_1538_pythia_tokenizer_pile_train_full",
    "20260929_1539_pythia_tokenizer_pile_train_full",
    "20260929_1540_pythia_tokenizer_pile_train_full",
    "20260929_1543_pythia_tokenizer_pile_train_full",
    "20260929_1547_pythia_tokenizer_pile_train_full",
    "20260929_1554_pythia_tokenizer_pile_train_full",
    "20260929_1557_pythia_tokenizer_pile_train_full",
    "20260929_1604_pythia_tokenizer_pile_train_full",
    "20260929_1606_pythia_tokenizer_pile_train_full",
    "20260929_1613_pythia_tokenizer_pile_train_full",
    "20260929_1614_pythia_tokenizer_pile_train_full",
    # 3-step ladder rungs: confirms pythia256/384/768 build + load at
    # 30.7M/49.5M/162.6M under the new parallel_residual stack.
    "20260929_1726_pythia256_tokenizer_pile_train_full",
    "20260929_1726_pythia384_tokenizer_pile_train_full",
    "20260929_1726_pythia768_tokenizer_pile_train_full",
]

IGNORE_REASON = (
    "2026-09-29: unregistered dirs were classified, not blanket-ignored. "
    "Real evidence runs (m66m-4p5gb-40k, pythia-owt50kv2-lr-15e-4-long, "
    "pile-native-tok-200step) were registered as arms; the rest are "
    "calibration smokes (batch sweep for tok/s, 3-step ladder build checks, "
    "short 4k-vocab probes) plus one launch casualty. Calibration runs answer "
    "'does it fit and how fast', not 'what did the model learn', so they carry "
    "no verdict and are ignored by name."
)


def main():
    dry = "--dry-run" in sys.argv
    meta = json.load(open(EXP, encoding="utf-8"))
    ids = {a["id"] for a in meta["arms"]}
    owned = {d for a in meta["arms"] for d in a["run_dirs"]}

    added, skipped = [], []
    for arm in NEW_ARMS:
        if arm["id"] in ids:
            skipped.append(arm["id"])
            continue
        for d in arm["run_dirs"]:
            if not os.path.isdir(os.path.join(LAB, "runs", d)):
                sys.exit(f"run dir missing: {d}")
            if d in owned:
                sys.exit(f"dir already owned by another arm: {d}")
        added.append(arm)

    known_ignore = set(meta.get("ignore_dirs", []))
    new_ignore = [d for d in IGNORE_DIRS
                  if d not in known_ignore and d not in owned]

    if dry:
        print(json.dumps({"would_add_arms": [a["id"] for a in added],
                          "already_registered": skipped,
                          "would_ignore": new_ignore}, indent=2))
        return

    meta["arms"].extend(added)
    meta.setdefault("ignore_dirs", [])
    for d in new_ignore:
        meta["ignore_dirs"].append(d)
    meta["ignore_dirs_note"] = IGNORE_REASON
    json.dump(meta, open(EXP, "w", encoding="utf-8"), indent=2)

    print(f"registered {len(added)} arm(s): {', '.join(a['id'] for a in added)}")
    if skipped:
        print(f"  already present: {', '.join(skipped)}")
    print(f"ignored {len(new_ignore)} calibration dir(s)")
    print(f"arms={len(meta['arms'])} ignore_dirs={len(meta['ignore_dirs'])}")


if __name__ == "__main__":
    main()
