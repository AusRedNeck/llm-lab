"""Prepare an explicitly approved continuation of one paused chunked run."""

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _arg_value(args, flag):
    try:
        return args[args.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def _set_arg(args, flag, value):
    value = str(value)
    try:
        index = args.index(flag)
    except ValueError:
        args.extend([flag, value])
    else:
        if index + 1 >= len(args):
            raise ValueError(f"missing value after {flag}")
        args[index + 1] = value


def prepare_next_chunk(spec, state, next_stop, checkpoint_path, checkpoint):
    if state.get("status") != "paused_at_chunk":
        raise ValueError("run must be paused at a chunk boundary")
    if spec.get("completed"):
        raise ValueError("completed run cannot be continued")
    if not spec.get("run_name") or not spec.get("stamp"):
        raise ValueError("run identity must be pinned before continuing")
    if state.get("job") != spec.get("name"):
        raise ValueError("supervisor state belongs to another job")
    current = int(spec["target_steps"])
    paused_step = state.get("paused_step")
    if paused_step is not None and int(paused_step) != current:
        raise ValueError("supervisor paused step does not match current chunk target")
    horizon = int(spec["schedule_steps"])
    size = int(spec.get("chunk_steps") or 0)
    expected = min(current + size, horizon)
    if not size or int(next_stop) != expected:
        raise ValueError(f"next chunk must end at step {expected}")
    if not checkpoint_path or checkpoint.get("step") != current:
        raise ValueError("checkpoint step does not match paused boundary")
    if not checkpoint.get("rng_state"):
        raise ValueError("checkpoint lacks RNG state needed for chunk continuity")
    if checkpoint.get("schedule_steps") != horizon:
        raise ValueError("checkpoint schedule horizon mismatch")
    if checkpoint.get("run_name") != spec.get("run_name"):
        raise ValueError("checkpoint run identity mismatch")
    args = list(spec["train_args"])
    if int(_arg_value(args, "--steps") or 0) != horizon:
        raise ValueError("train args do not preserve the schedule horizon")
    if int(_arg_value(args, "--stop-at-step") or 0) != current:
        raise ValueError("train args do not match paused chunk")
    _set_arg(args, "--stop-at-step", next_stop)
    _set_arg(args, "--resume", checkpoint_path)
    new = dict(spec)
    new.update({"train_args": args, "target_steps": int(next_stop),
                "chunk_index": int(spec.get("chunk_index", 1)) + 1,
                "completed": False, "next_job": None})
    new["log"] = f"logs/{spec['name']}_chunk{new['chunk_index']:02d}.log"
    next_state = dict(state)
    next_state.update({"status": "armed", "target_steps": int(next_stop),
                       "schedule_steps": horizon, "chunk_index": new["chunk_index"],
                       "chunk_start_step": current, "relaunch_count": 0,
                       "strikes": 0, "watchdog_last_step": None})
    for key in ("pid", "last_relaunch", "last_resume_step", "resume_requested",
                "paused_step", "pause_warning", "watchdog_last_at"):
        next_state.pop(key, None)
    return new, next_state


def _running_trainers():
    if LAB not in sys.path:
        sys.path.insert(0, LAB)
    scripts = os.path.join(LAB, "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import train_watchdog
    return train_watchdog.running_trainers()


def _atomic_json(path, value):
    temporary = f"{path}.{os.getpid()}.tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write(chr(10))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="train_job.json")
    parser.add_argument("--stop-at-step", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args(argv)
    spec_path = (options.spec if os.path.isabs(options.spec) else
                 os.path.join(LAB, options.spec))
    state_path = os.path.splitext(spec_path)[0] + ".state.json"
    with open(spec_path, encoding="utf-8") as stream:
        spec = json.load(stream)
    with open(state_path, encoding="utf-8") as stream:
        state = json.load(stream)
    trainers, can_tell = _running_trainers()
    if not can_tell:
        raise RuntimeError("global trainer preflight is blind; refusing to continue")
    if trainers:
        raise RuntimeError(f"trainer already active: {trainers}")
    checkpoint_dir = spec.get("ckpt_dir", "checkpoints")
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(LAB, checkpoint_dir)
    pattern = f"*_{spec['stamp']}_step{spec['target_steps']}.pt"
    candidates = glob.glob(os.path.join(checkpoint_dir, pattern))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one boundary checkpoint, found {len(candidates)}")
    import torch
    checkpoint_path = candidates[0]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        resume_path = Path(os.path.relpath(checkpoint_path, LAB)).as_posix()
    except ValueError:
        resume_path = Path(checkpoint_path).as_posix()
    new_spec, new_state = prepare_next_chunk(
        spec, state, options.stop_at_step, resume_path, checkpoint)
    if options.dry_run:
        print(json.dumps({"spec": new_spec, "state": new_state,
                          "resume_checkpoint": resume_path}, indent=2))
        return 0
    _atomic_json(spec_path, new_spec)
    _atomic_json(state_path, new_state)
    command = [new_spec["python"], os.path.join("scripts", "train_watchdog.py")]
    if os.path.abspath(spec_path) != os.path.join(LAB, "train_job.json"):
        command += ["--spec", spec_path]
    return subprocess.run(command, cwd=LAB, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
