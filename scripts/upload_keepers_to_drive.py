#!/usr/bin/env python3
"""Upload the llm-lab KEEPERS to the Google Drive "llm-lab keepers" folder.

Third copy, and the only OFF-MACHINE one: \\MAC\\EasyStore (E:) is read-only, so Google Drive
is what survives this machine being lost. Idempotent: a keeper whose name is already in the
folder is skipped, so a re-run after an interruption resumes rather than duplicating.

Records Drive file IDs to keepers.drive.json so KEEPERS.md can point at them.
Usage: python upload_keepers_to_drive.py [--dry-run]
"""
import glob
import json
import os
import subprocess
import sys
import time

GAPI = r"C:/Users/shane/AppData/Local/hermes/skills/productivity/google-workspace/scripts/google_api.py"
GAPI_PY = r"C:/Users/shane/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe"
FOLDER = "1uckC6WYMpum-2n7nOJ33pxMQyIUuVljC"      # "llm-lab keepers"
KEEPERS_DIR = r"D:/Projects/llm-lab-private/checkpoints/keepers"
OUT = r"D:/Projects/llm-lab-private/checkpoints/keepers.drive.json"


def run(args, timeout=3600):
    r = subprocess.run([GAPI_PY, GAPI] + args, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def in_drive(name):
    q = f"name = '{name}' and '{FOLDER}' in parents and trashed = false"
    rc, out = run(["drive", "search", q, "--raw-query", "--max", "3"])
    return rc == 0 and name in out


def main():
    dry = "--dry-run" in sys.argv
    files = sorted(glob.glob(os.path.join(KEEPERS_DIR, "*.pt")))
    # LARGEST FIRST. Biggest here means most valuable: the 845 MB m50m bests (including the
    # record run) would otherwise upload dead last on alphabetical order, so a failure late in
    # the run would leave the crown-jewel checkpoints with no off-machine copy.
    files.sort(key=os.path.getsize, reverse=True)
    total = sum(os.path.getsize(f) for f in files)
    print(f"keepers: {len(files)} files, {total/1e9:.2f} GB -> Drive folder {FOLDER}", flush=True)
    results = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    for i, path in enumerate(files, 1):
        name = os.path.basename(path)
        mb = os.path.getsize(path) / 1e6
        if results.get(name, {}).get("id"):
            print(f"  [{i:>2}/{len(files)}] {name:<58} already uploaded (recorded)", flush=True)
            continue
        if in_drive(name):
            print(f"  [{i:>2}/{len(files)}] {name:<58} already in Drive (skipped)", flush=True)
            continue
        if dry:
            print(f"  [{i:>2}/{len(files)}] {name:<58} {mb:>6.0f} MB would upload", flush=True)
            continue
        t0 = time.time()
        rc, out = run(["drive", "upload", path, "--parent", FOLDER])
        dt = time.time() - t0
        fid = None
        try:
            j = json.loads(out[out.index("{"):out.rindex("}") + 1])
            fid = j.get("id")
        except Exception:
            pass
        if rc == 0 and fid:
            results[name] = {"id": fid, "bytes": os.path.getsize(path)}
            json.dump(results, open(OUT, "w", encoding="utf-8"), indent=2)
            print(f"  [{i:>2}/{len(files)}] {name:<58} {mb:>6.0f} MB OK in {dt:>5.0f}s "
                  f"({mb/max(dt,0.1):.1f} MB/s)", flush=True)
        else:
            print(f"  [{i:>2}/{len(files)}] {name:<58} FAILED rc={rc}: {out.strip()[-200:]}", flush=True)
    print(f"\nuploaded {len(results)}/{len(files)} keepers; index -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
