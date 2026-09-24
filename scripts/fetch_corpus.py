"""Fetch curated corpora from the Hub — patterns validated before anything moves.

Why not the `hf` CLI: its --include handling through cmd/bash quoting was a mess
(multi-pattern calls silently became positional filenames; `data/*` matched zero
files while `*.parquet` matched 336). driving snapshot_download() directly means
the patterns are Python strings, checked with fnmatch against the real repo file
list, and printed as a count + GB before a single byte is downloaded.

Resumable: files already complete in the local dir are skipped, so re-running
after a kill picks up where it stopped.

Usage:
    python fetch_corpus.py --list          # just show what each pattern matches
    python fetch_corpus.py                 # fetch everything below
    python fetch_corpus.py --only cosmopedia
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import time
import urllib.request

from huggingface_hub import HfApi, snapshot_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# What we're landing, and the flat patterns we validated against each tree.
CORPORA = [
    {
        "name": "cosmopedia",
        "repo": "HuggingFaceTB/cosmopedia",
        "local_dir": os.path.join(DATA, "cosmopedia"),
        "patterns": ["*.parquet"],          # every config: web samples v1/v2, stories, math, ...
        "why": "textbook-style synthetic prose — the quality play for small models",
    },
    {
        "name": "finewiki_en",
        "repo": "HuggingFaceFW/finewiki",
        "local_dir": os.path.join(DATA, "finewiki"),
        "patterns": ["*enwiki*"],           # English Wikipedia only, not the other 400 files
        "why": "cleaned Wikipedia (en) — dense factual prose",
    },
    {
        "name": "open_web_math",
        "repo": "open-web-math/open-web-math",
        "local_dir": os.path.join(DATA, "open_web_math"),
        "patterns": ["*.parquet"],
        "why": "math text — reasoning scaffold, small and cheap",
    },
]
GB = 1024 ** 3


def repo_file_sizes(repo: str) -> dict[str, int]:
    """Repo path -> size in bytes, straight from the tree API."""
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true"
    with urllib.request.urlopen(url, timeout=60) as r:
        tree = json.load(r)
    return {t["path"]: (t.get("size") or 0) for t in tree if t["type"] == "file"}


def matched(repo: str, patterns: list[str]) -> tuple[int, int]:
    sizes = repo_file_sizes(repo)
    hits = [(p, s) for p, s in sizes.items()
            if any(fnmatch.fnmatch(p, pat) for pat in patterns)]
    return len(hits), sum(s for _, s in hits)


def fetch(spec: dict, dry: bool) -> None:
    n, total = matched(spec["repo"], spec["patterns"])
    print(f"\n=== {spec['name']}  ({spec['repo']})")
    print(f"    patterns {spec['patterns']} -> {n} files, {total / GB:.2f} GiB")
    print(f"    why: {spec['why']}")
    if dry or n == 0:
        if n == 0:
            print("    !! pattern matched nothing — refusing to run")
        return
    os.makedirs(spec["local_dir"], exist_ok=True)
    t0 = time.time()
    path = snapshot_download(
        repo_id=spec["repo"], repo_type="dataset",
        local_dir=spec["local_dir"], allow_patterns=spec["patterns"],
        max_workers=8,
    )
    got = previous = 0
    for root, _, files in os.walk(spec["local_dir"]):
        if os.path.sep + ".cache" in root:
            continue
        got += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    print(f"    done in {(time.time() - t0) / 60:.1f} min -> {path}")
    print(f"    on disk: {got / GB:.2f} GiB", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="validate patterns only")
    ap.add_argument("--only", default="", help="run a single corpus by name")
    args = ap.parse_args()

    specs = [s for s in CORPORA if not args.only or s["name"] == args.only]
    if not specs:
        return 1
    for spec in specs:
        fetch(spec, dry=args.list)
    if not args.list:
        print("\nall requested corpora fetched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
