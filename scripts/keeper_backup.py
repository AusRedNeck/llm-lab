#!/usr/bin/env python3
"""Copy llm-lab KEEPER checkpoints out of the working dir, and prove the copies match.

WHY: keepers currently exist ONLY under D:/Projects/llm-lab/checkpoints/ -- the same disk the
training writes to. Anything that eats that directory eats the experiments. The private repo
cannot hold them (its .gitignore ignores *.pt by an earlier deliberate decision, and 253 of
337 ckpts exceed GitHub's 100 MB hard limit -- the m50m bests are 845 MB), so the bytes go to
real locations and the private repo carries a MANIFEST instead.

DESTINATIONS (each is pre-checked for writability; an unavailable one is skipped up front
rather than costing a per-file timeout):
  1. D:/Projects/llm-lab-private/checkpoints/keepers/  -- the private store.
     NOTE: same physical disk as the source, so this protects against accidental deletion of
     the working dir, NOT against losing that disk.
  2. C:/llm-lab-keepers/  -- Disk 0 (CT2000P3PSSD8); the working dir is Disk 1 (CT4000P3PSSD8),
     so this survives the training disk failing. Verified distinct: 2026-09-18.
  3. E:/llm-lab-keepers/  -- \\MAC\EasyStore, the Mac's external disk. READ-ONLY as of
     2026-09-18 ("Permission denied" on mkdir): it is listed so the pre-check reports it, but
     it cannot be a target until the share grants write access.

WHAT COUNTS AS A KEEPER
  * every *_best.pt   (the best-val checkpoint of its run)
  * the finals named in llm-lab-private/checkpoints/README.md (runs predating _best.pt)

Idempotent: a destination file already hashing equal to the source is skipped.
Usage: python keeper_backup.py [--dry-run]
"""
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime

CKPT = r"D:/Projects/llm-lab/checkpoints"
DESTINATIONS = [
    ("private store (D:)", r"D:/Projects/llm-lab-private/checkpoints/keepers"),
    ("second SSD (C:)", r"C:/llm-lab-keepers"),
    ("Mac share (E:)", r"E:/llm-lab-keepers"),
]
MANIFEST = r"D:/Projects/llm-lab-private/checkpoints/KEEPERS.md"
MANIFEST_JSON = r"D:/Projects/llm-lab-private/checkpoints/keepers.sha256.json"
DOCUMENTED = ["exp002_bytes10m_step5000.pt", "exp002_bytes10m_rope_step5000.pt"]


def sha256(path, bs=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bs)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def writable(directory):
    """mkdir + probe file. Returns (ok, reason)."""
    try:
        os.makedirs(directory, exist_ok=True)
        probe = os.path.join(directory, ".write_probe")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
        return True, "writable"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def keepers():
    names = sorted(f for f in os.listdir(CKPT) if f.endswith("_best.pt"))
    names += [d for d in DOCUMENTED if os.path.exists(os.path.join(CKPT, d))]
    return names


def copy_verified(src, dst_dir, dry):
    name = os.path.basename(src)
    dst = os.path.join(dst_dir, name)
    h_src = sha256(src)
    size = os.path.getsize(src)
    if os.path.exists(dst) and os.path.getsize(dst) == size and sha256(dst) == h_src:
        return "already-ok", h_src, size
    if dry:
        return "would-copy", h_src, size
    tmp = dst + ".part"
    shutil.copy2(src, tmp)
    if sha256(tmp) != h_src:
        os.remove(tmp)
        return "MISMATCH", h_src, size
    os.replace(tmp, dst)
    return "copied", h_src, size


def main():
    dry = "--dry-run" in sys.argv
    names = keepers()
    total = sum(os.path.getsize(os.path.join(CKPT, n)) for n in names)
    print(f"keepers: {len(names)} files, {total / 1e9:.2f} GB from {CKPT}\n")

    dests = []
    for label, path in DESTINATIONS:
        ok, why = writable(path)
        print(f"  {label:<22} {path:<52} {why}")
        if ok:
            dests.append((label, path))
    if not dests:
        print("\nno writable destination - nothing to do")
        return 1
    print()
    if dry:
        for n in names:
            print(f"   {os.path.getsize(os.path.join(CKPT, n)) / 1e6:>6.0f} MB  {n}")
        print("\nDRY RUN - nothing copied")
        return 0

    manifest = {}
    for i, n in enumerate(names, 1):
        src = os.path.join(CKPT, n)
        row = {}
        h = None
        for label, path in dests:
            try:
                st, h, size = copy_verified(src, path, dry=False)
            except Exception as e:  # noqa: BLE001
                st = f"FAILED: {type(e).__name__}"
            row[label] = st
        manifest[n] = {"sha256": h, "bytes": os.path.getsize(src), "destinations": row}
        print(f"  [{i:>2}/{len(names)}] {n:<58} {os.path.getsize(src)/1e6:>6.0f} MB  "
              + "  ".join(f"{k.split(' (')[0]}={v}" for k, v in row.items()))

    with open(MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump({"generated": datetime.now().isoformat(timespec="seconds"), "source": CKPT,
                   "destinations": [p for _, p in dests], "files": manifest}, f, indent=2)

    lines = [
        "# Keeper checkpoints - copies + hashes",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `keeper_backup.py`.",
        "",
        "Weights are NOT committed to this repo (`.gitignore` excludes `*.pt` deliberately: they",
        "balloon the repo, and 253 of 337 working ckpts exceed GitHub's 100 MB hard limit - the",
        "m50m bests are 845 MB). **This file is the index**: what each keeper is, its sha256, and",
        "where verified copies live.",
        "",
        f"- **{len(names)} keepers, {total/1e9:.2f} GB**",
    ]
    for label, path in dests:
        n_ok = sum(1 for v in manifest.values()
                   if v["destinations"].get(label) in ("copied", "already-ok"))
        lines.append(f"- {label}: `{path}` ({n_ok}/{len(names)} verified)")
    lines += [
        "",
        "Off-machine copies are currently BLOCKED (2026-09-18): `\\\\MAC\\EasyStore` (E:) is",
        "read-only - `Permission denied` on mkdir - and the Google Drive token is expired",
        "(`invalid_grant: Token has been expired or revoked`). Until one of those is fixed, a",
        "fire/theft of this machine loses the keepers even though two disks carry them.",
        "",
        "| keeper | MB | sha256 (first 16) | " + " | ".join(l.split(" (")[0] for l, _ in dests) + " |",
        "|---|---|---|" + "---|" * len(dests),
    ]
    for n in names:
        v = manifest[n]
        cells = " | ".join(str(v["destinations"].get(l, "-")) for l, _ in dests)
        lines.append(f"| `{n}` | {v['bytes']/1e6:.0f} | `{v['sha256'][:16]}` | {cells} |")
    lines += [
        "",
        "## A keeper is",
        "every `*_best.pt` (the best-val checkpoint of its run) plus the finals named in",
        "`checkpoints/README.md` for runs that predate that convention.",
        "",
        "## Restore",
        "Copy the file back into `D:/Projects/llm-lab/checkpoints/` and verify the sha256 above.",
        "",
        "## Safe to prune from the working dir?",
        "Only ckpts that are NOT in this table. Step-intermediates are rebuildable by resuming",
        "from any earlier checkpoint of the same family; keepers are not rebuildable back to the",
        "same weights, which is why they live in more than one place.",
    ]
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nmanifest -> {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
