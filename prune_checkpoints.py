#!/usr/bin/env python3
"""Prune intermediate checkpoints under a keep-policy, with the keepers hard-protected.

Policy (Shane approved 2026-09-18): per run (family, stamp) KEEP
  * the `_best.pt`              - the run's keeper
  * the newest step ckpt (last 1) - the resume point
  * every 5000th step ckpt        - coarser fallbacks along the curve
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
    for (fam, stamp), items in runs.items():
        steps = sorted([i for i in items if i[0] is not None], key=lambda x: x[0])
        keep |= {i[2] for i in items if i[1] == "best" or i[0] is None}
        keep |= {i[2] for i in steps[-KEEP_LAST:]} if KEEP_LAST else set()
        keep |= {i[2] for i in steps if i[0] % MILESTONE == 0}
        if fam and stamp == live_stamp and steps and steps[-1][0] == live_newest:
            keep.add(steps[-1][2])                     # explicit: the supervisor's resume point
        for i in items:
            if i[2] not in keep:
                drop.append(i[2])
    keep |= (keepers & set(files))                      # keepers always win

    drop = sorted(set(drop) - keepers)
    drop_bytes = sum(os.path.getsize(os.path.join(CKPT, f)) for f in drop)
    keep_bytes = sum(os.path.getsize(os.path.join(CKPT, f)) for f in set(files) - set(drop))
    print(f"\npolicy: keep best + last {KEEP_LAST} + every {MILESTONE} + finals + keepers")
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
