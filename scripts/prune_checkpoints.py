#!/usr/bin/env python3
"""Prune intermediate checkpoints under a keep-policy, with the keepers hard-protected.

Policy (Shane approved 2026-09-18): per run (family, stamp) KEEP
  * the `_best.pt`              - the run's keeper
  * the newest step ckpt (last 1) - the resume point
  * every 5000th step ckpt        - coarser fallbacks along the curve
  * the pythia-matched anchor step - the step whose tokens equal a Pythia
    checkpoint we score against (step 16,000 of the 131,072-tok/step parity
    run == pythia-160m@step1000, to the token). It is NOT on the 5000 grid,
    so without this the matched-token head-to-head loses its checkpoint.
  * legacy step-less finals       - they predate the _best convention
and DELETE the rest.

Protections that override the policy (a bug here is unrecoverable, so these are explicit):
  * every file named in llm-lab-private/checkpoints/keepers.sha256.json is NEVER deleted
  * the ACTIVE job's newest checkpoint (train_job.json -> newest step of its stamp) is never
    deleted, because the supervisor resumes from exactly that file
  * nothing outside the checkpoints dir is ever touched

Safety model: the policy is only safe because the keepers exist in TWO verified locations
(keeper_backup.py: private store + second SSD). Run keeper_backup.py first if in doubt —
this script refuses to run when a keeper has no verified copy.

Usage: python prune_checkpoints.py [--dry-run] [--apply]
"""
import glob
import hashlib
import json
import os
import re
import sys
from datetime import datetime

CKPT = r"D:/Projects/llm-lab/checkpoints"
MANIFEST = r"D:/Projects/llm-lab-private/checkpoints/keepers.sha256.json"
JOB = r"D:/Projects/llm-lab/train_job.json"
LOG = r"D:/Projects/llm-lab/logs/prune_log_{}.txt".format(datetime.now().strftime("%Y%m%d_%H%M"))
MILESTONE = 5000
KEEP_LAST = 1
RUNS = r"D:/Projects/llm-lab/runs"
# Pythia checkpoints we score against, in TOKENS: branch step1000 and step2000.
# Our parity run trains at 131,072 tok/step, so pythia step1000 is EXACTLY our
# step 16,000 -- and 16000 is not a multiple of MILESTONE, so the milestone grid
# would happily delete the one checkpoint the matched-token head-to-head is
# measured from. Anchors are derived per run from that run's OWN effective batch,
# so a run whose steps never land on one is unaffected (20k runs at eff 32,768
# anchor at step 64,000, which they do not have).
PYTHIA_ANCHOR_TOKENS = (2_097_152_000, 4_194_304_000)


def eff_tokens_per_step(stamp, runs_dir=RUNS):
    """batch x accum x ctx for the run with this stamp, from its loss.jsonl header.

    None when unreadable: never guess an effective batch, because a wrong one
    "protects" a step that is not the anchor at all -- the exact failure this
    exists to prevent.
    """
    if not stamp or len(stamp) != 12:
        return None
    # run dirs are "<stamp>_<preset>_<tokenizer>_<corpus...>", so the stamp is a
    # PREFIX of the directory name, not the whole name.
    pattern = os.path.join(runs_dir, f"{stamp[:8]}_{stamp[8:]}*", "loss.jsonl")
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1:
        return None
    try:
        with open(hits[0], encoding="utf-8") as f:
            head = json.loads(f.readline())
        a, cfg = head["args"], head["cfg"]
        return int(a["batch"]) * int(a.get("accum", 1)) * int(cfg["context_length"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def anchor_steps(stamp, runs_dir=RUNS):
    """Steps of this run that are token-exact matches for a Pythia checkpoint."""
    eff = eff_tokens_per_step(stamp, runs_dir)
    if not eff:
        return set()
    return {t // eff for t in PYTHIA_ANCHOR_TOKENS if t % eff == 0}


def parse(name):
    m = re.match(r"(exp\d+_.+?)_(\d{12})_(step(\d+)|best)\.pt$", name)
    if m:
        return m.group(1), m.group(2), (int(m.group(4)) if m.group(4) else None), m.group(3)
    return None, None, None, None


def sha256(path, bs=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bs)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    apply = "--apply" in sys.argv
    files = sorted(f for f in os.listdir(CKPT) if f.endswith(".pt"))
    total = sum(os.path.getsize(os.path.join(CKPT, f)) for f in files)

    # --- protections -------------------------------------------------------
    keepers, keeper_copies_ok = set(), 0
    if os.path.exists(MANIFEST):
        man = json.load(open(MANIFEST, encoding="utf-8"))
        for name, meta in man["files"].items():
            keepers.add(name)
            dests = meta.get("destinations", {})
            if any(v in ("copied", "already-ok") for v in dests.values()):
                keeper_copies_ok += 1
    job = json.load(open(JOB, encoding="utf-8")) if os.path.exists(JOB) else {}
    live_stamp, live_newest = job.get("stamp"), None
    for f in files:
        fam, stamp, step, kind = parse(f)
        if stamp == live_stamp and step is not None:
            live_newest = max(live_newest or 0, step)

    print(f"checkpoints: {len(files)} files, {total/1e9:.1f} GB")
    print(f"keepers protected: {len(keepers)} ({keeper_copies_ok} with a verified copy)")
    print(f"live job: {job.get('name')} stamp={live_stamp} newest step={live_newest}")
    if keeper_copies_ok < len(keepers):
        print("REFUSING: some keepers have no verified copy (run keeper_backup.py first)")
        return 1

    # --- policy -----------------------------------------------------------
    runs = {}
    for f in files:
        fam, stamp, step, kind = parse(f)
        runs.setdefault((fam, stamp), []).append((step, kind, f))
    keep, drop = set(), []
    anchors_found = {}
    for (fam, stamp), items in runs.items():
        steps = sorted([i for i in items if i[0] is not None], key=lambda x: x[0])
        keep |= {i[2] for i in items if i[1] == "best" or i[0] is None}
        keep |= {i[2] for i in steps[-KEEP_LAST:]} if KEEP_LAST else set()
        keep |= {i[2] for i in steps if i[0] % MILESTONE == 0}
        # token-exact match for a Pythia checkpoint (e.g. step 16,000 of the
        # 131,072-tok/step parity run == pythia-160m@step1000, to the token).
        anchors = anchor_steps(stamp)
        hit = sorted({i[0] for i in steps if i[0] in anchors})
        if hit:
            anchors_found[stamp] = hit
            keep |= {i[2] for i in steps if i[0] in anchors}
        if fam and stamp == live_stamp and steps and steps[-1][0] == live_newest:
            keep.add(steps[-1][2])                     # explicit: the supervisor's resume point
        for i in items:
            if i[2] not in keep:
                drop.append(i[2])
    keep |= (keepers & set(files))                      # keepers always win

    drop = sorted(set(drop) - keepers)
    drop_bytes = sum(os.path.getsize(os.path.join(CKPT, f)) for f in drop)
    keep_bytes = sum(os.path.getsize(os.path.join(CKPT, f)) for f in set(files) - set(drop))
    print(f"\npolicy: keep best + last {KEEP_LAST} + every {MILESTONE} + "
          f"pythia-matched anchors + finals + keepers")
    if anchors_found:
        for stamp, hits in sorted(anchors_found.items()):
            print(f"  anchor steps kept for {stamp}: {hits}")
    else:
        print("  no run currently lands on a pythia anchor step (protection arms "
              "when one exists)")
    print(f"  KEEP   {len(set(files) - set(drop)):>3} files / {keep_bytes/1e9:>6.1f} GB")
    print(f"  DELETE {len(drop):>3} files / {drop_bytes/1e9:>6.1f} GB")
    by_fam = {}
    for f in drop:
        fam, stamp, step, kind = parse(f)
        by_fam[fam or "(legacy)"] = by_fam.get(fam or "(legacy)", 0) + os.path.getsize(os.path.join(CKPT, f))
    print("  largest reductions:")
    for fam, b in sorted(by_fam.items(), key=lambda x: -x[1])[:6]:
        print(f"     {fam:<42} {b/1e9:>6.1f} GB")

    if not apply:
        print("\nDRY RUN - nothing deleted (pass --apply)")
        print("  sample of the deletion list:")
        for f in drop[:8]:
            print(f"     {os.path.getsize(os.path.join(CKPT, f))/1e6:>6.0f} MB  {f}")
        return 0

    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "w", encoding="utf-8") as log:
        log.write(f"prune {datetime.now().isoformat(timespec='seconds')} "
                  f"policy=best+last{KEEP_LAST}+every{MILESTONE}\n")
        freed = 0
        for f in drop:
            p = os.path.join(CKPT, f)
            sz = os.path.getsize(p)
            os.remove(p)
            freed += sz
            log.write(f"deleted {sz:>12} {f}\n")
        log.write(f"total freed {freed}\n")
    print(f"\ndeleted {len(drop)} files, freed {drop_bytes/1e9:.1f} GB")
    print(f"log -> {LOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
