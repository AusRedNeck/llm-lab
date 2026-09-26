#!/usr/bin/env python3
"""Arm a new experiment: writes train_job.json AND the registry row in one action.

The registry only prevents going in circles if hand-launching is harder than
registering. So: don't hand-edit train_job.json; arm through here.

  python viz/arm.py --id p160-lr2e-4-fix --family capacity-50k \
      --goal "cooler LR on the fixed 160M stack" --control p160-fix-long \
      --lever lr=5e-4->2e-4 \
      -- --steps 20000 --batch 8 --accum 8 --lr 2e-4 --preset pythia160 ...

Everything after -- becomes the trainer arg list (goes into train_args verbatim).
Refuses to overwrite an unfinished arm. After arming:
  python train_watchdog.py            # launches detached + supervises
"""
import argparse
import json
import os
import re
import sys

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(LAB, "experiments.json")
SPEC = os.path.join(LAB, "train_job.json")
PY = "C:/Users/shane/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe"


def arg_val(args, flag):
    try:
        return args[args.index(flag) + 1]
    except ValueError:
        return None


def match_from(args):
    m = ["train.train", f"--preset {arg_val(args, '--preset')}"]
    lr = arg_val(args, "--lr")
    if lr:
        m.append(f"--lr {lr}")
    rp = arg_val(args, "--rotary-pct")
    if rp:
        m.append(f"--rotary-pct {rp}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--goal", required=True)
    ap.add_argument("--control", default=None)
    ap.add_argument("--lever", action="append", default=[],
                    help="key=claim (repeatable), e.g. lr=1.5e-3->5e-4")
    ap.add_argument("--python", default=PY)
    ap.add_argument("--log", default=None)
    ap.add_argument("--max-restarts", type=int, default=6)
    ap.add_argument("--next-job", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print spec+row, write nothing, ignore the active-job guard")
    ap.add_argument("trainer_args", nargs=argparse.REMAINDER)
    ns = ap.parse_args()

    args = [a for a in ns.trainer_args if a != "--"]
    if "--preset" not in args or "--steps" not in args:
        sys.exit("arm.py: trainer args must include --preset and --steps")
    if not all(a.startswith("--") for a in args[0::2]) or len(args) % 2:
        # tolerant: flags like --use_rope take no value; just forbid file edits
        pass

    lever = {}
    for kv in ns.lever:
        if "=" not in kv:
            sys.exit(f"arm.py: --lever needs key=claim, got {kv!r}")
        k, v = kv.split("=", 1)
        lever[k] = v

    meta = json.load(open(EXP, encoding="utf-8"))
    ids = {a["id"] for a in meta["arms"]}
    if ns.id in ids:
        sys.exit(f"arm.py: arm id {ns.id!r} already registered (pick a new id)")
    if ns.control and ns.control not in ids:
        sys.exit(f"arm.py: control {ns.control!r} not in registry")
    # refuse to clobber an unfinished job
    if os.path.exists(SPEC):
        cur = json.load(open(SPEC, encoding="utf-8"))
        if not cur.get("completed") and not ns.dry_run:
            sys.exit(f"arm.py: active job {cur.get('name')!r} is not completed. "
                     "Let it finish or mark it done before arming a new one.")

    steps = int(arg_val(args, "--steps"))
    log = ns.log or f"logs/{ns.id}.log"
    spec = {
        "name": ns.id,
        "desc": ns.goal,
        "ckpt_dir": "checkpoints",
        "stamp": "",
        "run_name": "",
        "target_steps": steps,
        "python": ns.python,
        "train_args": args,
        "match": match_from(args),
        "log": log,
        "max_restarts": ns.max_restarts,
        "min_progress_steps": 25,
        "completed": False,
        "next_job": ns.next_job,
        "notes": f"armed by viz/arm.py {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}",
    }
    row = {"id": ns.id, "family": ns.family, "run_dirs": [],
           "control_id": ns.control, "lever": lever or {"note": "declare levers"},
           "goal": ns.goal, "verdict": "running", "note": ""}
    if ns.dry_run:
        print(json.dumps({"spec": spec, "row": row}, indent=2))
        return
    meta["arms"].append(row)
    json.dump(meta, open(EXP, "w", encoding="utf-8"), indent=2)
    json.dump(spec, open(SPEC, "w", encoding="utf-8"), indent=2)
    print(f"armed {ns.id} (family {ns.family})")
    print(f"  spec -> train_job.json   (match: {spec['match']})")
    print(f"  row  -> experiments.json (control: {ns.control})")
    print("next: python train_watchdog.py    # detached launch + supervision")


def args_dryrun():
    return "--dry-run" in sys.argv


if __name__ == "__main__":
    main()
