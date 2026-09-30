#!/usr/bin/env python3
"""Batch-size ladder smokes for the Pile knee investigation.

THE HYPOTHESIS UNDER TEST
    Both LR arms knee at ~0.7-0.9 tok/param regardless of LR, and the three-way
    eval shows `served` RISING after the knee in both runs. Rising served means
    the model stops even fitting the batches it just trained on - that is
    optimization instability from steps that are too large, NOT overfitting in
    the usual train-beats-val sense. Both val-randTr and randTr-served gaps are
    FLAT, so it is not a data or recycling problem either.

    Remedy for too-large steps is a larger batch. Pythia-70m trained at a
    2,097,152-token batch (128x ours) at lr 1.0e-3 (2x hotter) for 143k steps
    and never kneed at 3k. So: does the knee move with batch size?

    BUT batch and LR are coupled - scaling rules say 4x batch wants ~2x LR.
    So the ladder varies BOTH to separate the effects:
      A  eff 16k  lr 5e-4   control, we already have this
      B  eff 256k lr 1e-3   batch 16x AND lr 2x  (Pythia's actual LR)
      C  eff 256k lr 5e-4   batch 16x, lr held   <- isolates batch alone

    Verdict logic:
      C knee moves  -> batch is the lever, no LR change needed
      only B moves  -> it is the batch x LR interaction, not batch alone
      neither moves -> knee is capacity-bound; the answer is a bigger model

WHAT THESE SMOKES ARE FOR
    NOT a val-quality check. A 200-step smoke cannot see a knee at 3-4k steps.
    They measure THROUGHPUT and PEAK VRAM per rung, because that decides which
    rung can actually run for 8 hours. 256k tokens/batch is micro 32 x accum 8
    and logits are batch x ctx x vocab = 32 x 512 x 50304, four times today's
    micro-batch 8 which already peaked near 12.4GB of 16.4GB. B and C may not
    fit at all; that is the thing worth learning, cheaply.

Usage:
    python scripts/batch_ladder_smokes.py [--rungs A,B,C] [--steps 200]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(LAB, ".venv", "Scripts", "python.exe")

# (id, micro_batch, accum, lr, hypothesis)
# A = control, B = measured, C = the batch-alone arm.
# C uses micro 16 (which fits: rung A peaked 12,581 MB) with accum 16 to reach
# the SAME eff batch as B (256 seqs = 131,072 tokens) at a fraction of the
# allocator pressure. B proved micro 32 fills 98% of the card and thrashes.
RUNGS = [
    ("A", 16, 2, "5e-4", "control: current production batch. knee 0.69 tok/param"),
    ("B", 32, 8, "1e-3", "batch 8x + lr 2x (Pythia's actual LR). OOM-adjacent: 16,001MB peak"),
    ("C", 16, 16, "5e-4", "batch 8x at micro 16 (fits), LR held -> isolates batch alone"),
]

COMMON = [
    "--preset", "pythia",
    "--tokenizer", "data/incoming/pythia70m_hf/tokenizer.json",
    "--tok_cache", "data/pile_train_full_bpe_pythia70m.bin",
    "--corpus", "data/pile_train_full.txt",
    "--val_every", "0",
    "--out", "checkpoints",
    "--patience-frac", "0",
    "--min-steps-frac", "0.5",
    "--degrade-frac", "0.15",
]


def gpu_mb():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30)
    try:
        return int(r.stdout.strip().split("\n")[0])
    except Exception:
        return None


def run_rung(rid, micro, accum, lr, steps, timeout=900):
    tag = f"ladder{rid}_b{micro}x{accum}_lr{lr}"
    log = os.path.join(LAB, "logs", f"{tag}.log")
    args = [PY, "-u", "-m", "train.train",
            "--batch", str(micro), "--accum", str(accum), "--lr", lr,
            "--steps", str(steps)] + COMMON
    # The curve is the LIVENESS signal. The log is block-buffered when stdout
    # is a file, so a run can be at step 150 while the log still shows step 100;
    # reading the log alone made a healthy rung look frozen and nearly caused a
    # healthy trainer to be killed. Poll the newest run dir's loss.jsonl instead.
    def curve_step():
        rd = os.path.join(LAB, "runs")
        cands = [os.path.join(rd, d) for d in os.listdir(rd)
                 if os.path.isdir(os.path.join(rd, d))]
        cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        for p in cands[:3]:
            lj = os.path.join(p, "loss.jsonl")
            if not os.path.exists(lj):
                continue
            try:
                if time.time() - os.path.getmtime(lj) > 900:
                    continue
                last = ""
                with open(lj, "rb") as f:
                    for line in f:
                        if line.strip():
                            last = line
                if last:
                    import json as _j
                    return _j.loads(last.decode("utf-8", "replace")).get("step", 0)
            except Exception:
                continue
        return 0

    base = gpu_mb() or 0
    t0 = time.time()
    last_seen, last_change = 0, time.time()
    peak = base
    with open(log, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(args, cwd=LAB, stdout=f, stderr=subprocess.STDOUT)
        deadline = t0 + timeout
        while proc.poll() is None and time.time() < deadline:
            m = gpu_mb()
            if m and m > peak:
                peak = m
            st = curve_step()
            if st > last_seen:
                last_seen = st
                last_change = time.time()
            elif time.time() - last_change > 300 and last_seen > 0:
                # 5 min with a flat curve = genuinely wedged, stop waiting.
                f.write(f"\n[ladder] FROZEN at step {last_seen} for 5min - killing\n")
                f.flush()
                proc.kill()
                break
            time.sleep(2.0)
        rc = proc.returncode
    wall = time.time() - t0
    return {"rung": rid, "micro": micro, "accum": accum, "lr": lr,
            "eff_batch": micro * accum, "log": log, "rc": rc,
            "wall_s": round(wall, 1), "peak_vram_mb": peak,
            "baseline_vram_mb": base, "curve_step_reached": last_seen}


def parse_log(path):
    """Pull tok/s and the final loss out of a smoke log."""
    out = {"tok_s": None, "final_avg50": None, "oom": False,
           "ctx": None, "params": None, "steps_done": 0}
    if not os.path.exists(path):
        return out
    txt = open(path, encoding="utf-8", errors="replace").read()
    out["oom"] = ("CUDA out of memory" in txt) or ("out of memory" in txt.lower())
    m = re.search(r"preset=\S+ params~[\d.]+M ctx=(\d+)", txt)
    if m:
        out["ctx"] = int(m.group(1))
    m = re.search(r"params: ([\d,]+) total \(([\d.]+)M\), ([\d,]+) non-embedding", txt)
    if m:
        out["params"] = f"{m.group(2)}M total / {m.group(3)} non-emb"
    m = re.search(r"([\d,]+) tokens/sec", txt)
    if m:
        out["tok_s"] = int(m.group(1).replace(",", ""))
    steps = re.findall(r"step\s+(\d+)/", txt)
    if steps:
        out["steps_done"] = int(steps[-1])
    m = re.findall(r"avg50=([\d.]+)", txt)
    if m:
        out["final_avg50"] = float(m[-1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", default="A,B,C")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    want = [r.strip().upper() for r in a.rungs.split(",")]

    if gpu_mb() and gpu_mb() > 4000:
        sys.exit("GPU already busy (>4GB) - refusing to add a second job. "
                 "The singleton lock would refuse anyway; check for an orphan.")

    results = []
    for rid, micro, accum, lr, why in RUNGS:
        if rid not in want:
            continue
        print(f"\n=== rung {rid}: micro {micro} x accum {accum} "
              f"(eff {micro*accum} seqs = {micro*accum*512:,} tok/batch), lr {lr}")
        print(f"    {why}")
        r = run_rung(rid, micro, accum, lr, a.steps, a.timeout)
        r.update(parse_log(r["log"]))
        eff = r["eff_batch"]
        r["eff_tokens_per_batch"] = eff * 512
        r["tok_per_param_at_1k_steps"] = round(a.steps * eff * 512 / 70_739_072, 2)
        results.append(r)
        status = "OOM" if r["oom"] else ("ok" if r["rc"] == 0 else f"rc={r['rc']}")
        print(f"    -> {status}  steps {r['steps_done']}  {r['wall_s']}s  "
              f"peak VRAM {r['peak_vram_mb']} MB  tok/s {r['tok_s']}")
        if r["oom"]:
            print("    OOM: this rung does not fit. Do not retry at this micro-batch.")

    out = os.path.join(LAB, "reports", "batch_ladder_smokes.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(results, open(out, "w", encoding="utf-8"), indent=2)

    print(f"\n=== SUMMARY (wrote {out}) ===")
    print(f"{'rung':>4} {'eff seqs':>9} {'tok/batch':>11} {'lr':>7} "
          f"{'steps':>6} {'wall_s':>8} {'peakMB':>7} {'tok/s':>8}  status")
    for r in results:
        st = "OOM" if r["oom"] else ("ok" if r["rc"] == 0 else f"rc={r['rc']}")
        print(f"{r['rung']:>4} {r['eff_batch']:>9} {r['eff_tokens_per_batch']:>11,} "
              f"{r['lr']:>7} {r['steps_done']:>6} {r['wall_s']:>8} "
              f"{r['peak_vram_mb']:>7} {str(r['tok_s'] or '-'):>8}  {st}")
    print("\nPick the real run from THIS table, not from hope: the largest eff")
    print("batch that fits under ~14GB is the one that can move the knee.")


if __name__ == "__main__":
    main()
