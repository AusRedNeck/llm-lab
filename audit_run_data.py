#!/usr/bin/env python3
"""Audit the llm-lab run data for integrity and consistency.

Answers "is the data we draw conclusions from actually clean?" -- independent of whether the
science is right. Checks, in order of what has actually bitten us:

  loss.jsonl    parses, not truncated, monotonic steps, no duplicate steps, stable schema,
                val checks present/consistent, early_stop records well-formed
  identity      run stamp in the header == the run dir it lives in (the resume-continuity
                contract: one logical run = one dir = one curve)
  checkpoints   non-empty, torch-loadable, filename step == internal step, no orphans
  corpora       header-declared corpus vs what is actually on disk
  keepers       every sha256 in the index still matches the file
  hygiene       stray .part/.tmp, empty dirs, duplicate stamps

Writes reports/audit_run_data.json and prints a summary. Exit 0 = clean, 1 = findings.
Usage: python audit_run_data.py [--no-hash] [--quiet]
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import zipfile

LAB = os.path.dirname(os.path.abspath(__file__))
findings: list[dict] = []
info: list[str] = []


def note(level, kind, where, detail):
    findings.append({"level": level, "kind": kind, "where": where, "detail": detail})


def audit_curve(run_dir):
    """Integrity of one run's loss.jsonl."""
    lj = os.path.join(run_dir, "loss.jsonl")
    name = os.path.basename(run_dir)
    if not os.path.exists(lj):
        note("ERROR", "missing_curve", name, "no loss.jsonl")
        return None
    raw = open(lj, "rb").read()
    if not raw.strip():
        note("WARN", "abandoned_run", name, "loss.jsonl is 0 bytes - the run never started")
        return None
    if not raw.endswith(b"\n"):
        note("WARN", "truncated_tail", name, "no trailing newline (possible truncation)")

    rows, bad = [], 0
    for i, line in enumerate(raw.decode("utf-8", "replace").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            bad += 1
            if bad <= 2:
                note("ERROR", "unparsable_line", name, f"line {i} is not valid JSON")
    if bad > 2:
        note("ERROR", "unparsable_line", name, f"{bad} total unparsable lines")

    header = rows[0] if rows and "args" in rows[0] else None
    early = [r for r in rows if r.get("early_stop")]
    data = [r for r in rows if "step" in r and not r.get("early_stop")]
    if not data:
        note("WARN", "abandoned_run", name,
             "no step rows (header only) - a stub run that never trained")
        return None

    steps = [r["step"] for r in data]
    # monotonic + duplicates
    dups = {s for s in steps if steps.count(s) > 1}
    if dups:
        note("ERROR", "duplicate_steps", name, f"repeated step numbers: {sorted(dups)[:6]}")
    if steps != sorted(steps):
        note("WARN", "unordered_steps", name, "steps are not in ascending order")

    # schema drift: a mid-run change of keys silently breaks downstream tooling
    keysets = {tuple(sorted(r.keys())) for r in data}
    if len(keysets) > 1:
        note("WARN", "schema_drift", name, f"{len(keysets)} distinct row schemas in one curve")

    vals = [r for r in data if r.get("val") is not None]
    for e in early:
        if e.get("step") != data[-1]["step"]:
            note("WARN", "early_stop_step_mismatch", name,
                 f"early_stop at step {e.get('step')} but last data row is {data[-1]['step']}")

    # duplicate val checks at the same step (double-counted measurements)
    vsteps = [r["step"] for r in vals]
    vdups = {s for s in vsteps if vsteps.count(s) > 1}
    if vdups:
        note("WARN", "duplicate_val_checks", name, f"val measured twice at steps {sorted(vdups)[:6]}")

    # identity: header run_name must match the directory (resume-continuity contract)
    if header:
        hdr_run = str(header["args"].get("run_name") or "")
        if hdr_run and os.path.basename(hdr_run.rstrip("/\\")) != name:
            note("WARN", "identity_mismatch", name,
                 f"header run_name={os.path.basename(hdr_run)} != dir name")

    # value sanity
    for r in vals:
        if not (0 < r["val"] < 20):
            note("WARN", "implausible_val", name, f"step {r['step']}: val={r['val']}")
    for r in early:
        if r.get("best_bpb") is None and r.get("best_val") is None:
            note("WARN", "empty_early_stop", name, "early_stop record carries no best value")

    return {
        "run": name,
        "rows": len(data),
        "val_checks": len(vals),
        "first_step": steps[0],
        "last_step": steps[-1],
        "best_val": min((r["val"] for r in vals), default=None),
        "best_val_step": min(vals, key=lambda r: r["val"])["step"] if vals else None,
        "early_stop": bool(early),
        "corpus": os.path.basename(str(header["args"].get("corpus", ""))) if header else None,
        "tokenizer": os.path.basename(str(header["args"].get("tokenizer", ""))) if header else None,
        "params_m": header.get("params_m") if header else None,
        "vocab": (header.get("cfg") or {}).get("vocab_size") if header else None,
    }


def audit_checkpoints(deep=True):
    ckpts = sorted(glob.glob(os.path.join(LAB, "checkpoints", "*.pt")))
    stamps = {}
    for p in ckpts:
        base = os.path.basename(p)
        size = os.path.getsize(p)
        if size == 0:
            note("ERROR", "empty_checkpoint", base, "0 bytes")
            continue
        with open(p, "rb") as f:
            magic = f.read(4)
        if magic[:2] != b"PK":
            note("WARN", "non_zip_checkpoint", base, f"magic={magic!r} (not a torch zip)")
        # stamp = everything before _step/_best/_final
        stem = base[:-3]
        for suffix in ("_best", "_final"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
        if "_step" in stem:
            stem = stem.split("_step")[0]
        stamps.setdefault(stem, []).append(base)

    if deep:
        import torch
        for stem in sorted(stamps):
            # verify filename step == internal step on the NEWEST file of each family
            newest = sorted(stamps[stem])[-1]
            p = os.path.join(LAB, "checkpoints", newest)
            try:
                ck = torch.load(p, map_location="cpu", weights_only=False)
                inner = ck.get("step") if isinstance(ck, dict) else None
                fn = None
                for part in os.path.basename(newest)[:-3].split("_"):
                    if part.startswith("step") and part[4:].isdigit():
                        fn = int(part[4:])
                if fn is not None and inner is not None and fn != inner:
                    note("ERROR", "step_mismatch", newest, f"filename step {fn} != internal {inner}")
            except Exception as e:
                note("ERROR", "unloadable_checkpoint", newest, f"{type(e).__name__}: {e}")
    return stamps


def audit_keepers(verify_hash=True):
    idx = os.path.join(LAB, "..", "llm-lab-private", "checkpoints", "keepers.sha256.json")
    kdir = os.path.join(LAB, "..", "llm-lab-private", "checkpoints", "keepers")
    idx = os.path.normpath(idx)
    kdir = os.path.normpath(kdir)
    if not os.path.exists(idx):
        note("WARN", "no_keeper_index", "keepers", "keepers.sha256.json not found")
        return 0, 0
    man = json.load(open(idx, encoding="utf-8"))
    if isinstance(man, list):
        entries = man
    elif isinstance(man.get("files"), list):
        entries = man["files"]
    elif isinstance(man.get("files"), dict):
        entries = [dict(v if isinstance(v, dict) else {"sha256": v},
                        name=(v.get("name") if isinstance(v, dict) else None) or k)
                   for k, v in man["files"].items()]
    else:
        entries = []
    ok = bad = 0
    for e in entries if isinstance(entries, list) else []:
        nm = os.path.basename(str(e.get("name") or e.get("file") or e.get("path") or ""))
        want = str(e.get("sha256") or e.get("hash") or "").lower()
        p = os.path.join(kdir, nm)
        if not os.path.exists(p):
            note("ERROR", "keeper_missing", nm, "in index but not on disk")
            bad += 1
            continue
        if not verify_hash:
            ok += 1
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(8 << 20), b""):
                h.update(chunk)
        got = h.hexdigest().lower()
        mismatch = (not got.startswith(want)) if len(want) < 64 else (got != want)
        if want and mismatch:
            note("ERROR", "keeper_hash_mismatch", nm, "sha256 differs from the index")
            bad += 1
        else:
            ok += 1
    return ok, bad


def main():
    quiet = "--quiet" in sys.argv
    no_hash = "--no-hash" in sys.argv

    run_dirs = sorted(d for d in glob.glob(os.path.join(LAB, "runs", "*"))
                      if os.path.isdir(d) and not os.path.basename(d).startswith("_"))
    curves = []
    for d in run_dirs:
        r = audit_curve(d)
        if r:
            curves.append(r)

    # hygiene: stray temporaries, empty dirs
    stray = glob.glob(os.path.join(LAB, "runs", "**", "*.part"), recursive=True) + \
            glob.glob(os.path.join(LAB, "runs", "**", "*.tmp"), recursive=True) + \
            glob.glob(os.path.join(LAB, "checkpoints", "*.part"))
    for s in stray:
        note("WARN", "stray_temp_file", os.path.basename(s), "leftover partial file")

    empty = [os.path.basename(d) for d in run_dirs
             if not os.listdir(d) or (len(os.listdir(d)) == 1 and "samples" in os.listdir(d)
                                      and not os.listdir(os.path.join(d, "samples")))]
    for e in empty:
        note("WARN", "empty_run_dir", e, "no curve and no samples")

    # cross-check: checkpoints vs run stamps
    stamps = audit_checkpoints(deep=True)
    run_stamps = set()
    for d in run_dirs:
        base = os.path.basename(d)
        run_stamps.add(base)
    digits = lambda s: "".join(c for c in s if c.isdigit())
    norm_runs = [digits(os.path.basename(d)) for d in run_dirs]
    orphan_fams = []
    for stem in stamps:
        # exp002_<preset>_<tokenizer>_<stamp>
        parts = stem.split("_")
        stamp = next((p for p in reversed(parts) if len(p) == 12 and p.isdigit()), None)
        if stamp and not any(stamp in rs for rs in norm_runs):
            orphan_fams.append(stem)
    for o in orphan_fams:
        stamp = next((q for q in reversed(o.split("_")) if len(q) == 12 and q.isdigit()), "")
        # Drift means the SAME run stamped twice a few minutes apart (family vs run dir), not
        # merely anything on the same day: 20:57 vs 10:07 is a different run, not drift.
        def mins(s):
            try:
                from datetime import datetime
                return datetime.strptime(s[:12], "%Y%m%d%H%M").timestamp() / 60
            except Exception:
                return None
        near = []
        if stamp:
            for r in norm_runs:
                d = mins(r)
                if d is not None and abs(d - mins(stamp)) <= 90:
                    near.append(r)
        if near:
            note("WARN", "checkpoint_stamp_drift", o,
                 f"run dir exists but stamped differently ({near[0][:12]}) - identity drift")
        else:
            note("WARN", "orphan_checkpoints", o, "weights with no run dir anywhere (curve lost?)")

    k_ok, k_bad = audit_keepers(verify_hash=not no_hash)

    errs = [f for f in findings if f["level"] == "ERROR"]
    warns = [f for f in findings if f["level"] == "WARN"]

    report = {
        "run_dirs": len(run_dirs),
        "curves_audited": len(curves),
        "checkpoint_files": sum(len(v) for v in stamps.values()),
        "checkpoint_families": len(stamps),
        "keepers_hashed_ok": k_ok,
        "keepers_bad": k_bad,
        "errors": errs,
        "warnings": warns,
        "curves": curves,
    }
    os.makedirs(os.path.join(LAB, "reports"), exist_ok=True)
    out = os.path.join(LAB, "reports", "audit_run_data.json")
    json.dump(report, open(out, "w", encoding="utf-8"), indent=2)

    if not quiet:
        print(f"run dirs {len(run_dirs)}   curves {len(curves)}   ckpts "
              f"{report['checkpoint_files']} in {len(stamps)} families   "
              f"keepers {k_ok} verified ({k_bad} bad)")
        print(f"\nERRORS: {len(errs)}   WARNINGS: {len(warns)}")
        for f in errs[:25]:
            print(f"  [ERROR] {f['kind']:<22} {f['where'][:44]:<44} {f['detail']}")
        for f in warns[:25]:
            print(f"  [warn ] {f['kind']:<22} {f['where'][:44]:<44} {f['detail']}")
        print(f"\nreport -> {out}")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
