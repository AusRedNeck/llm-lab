#!/usr/bin/env python3
"""scripts/readme_state.py -- write the README's state block from real data.

The README used to carry a hand-maintained "Current State" section. Hand-
maintained sections rot: they get appended to instead of replaced, they go
chronologically inverted, and eventually they contradict the runs on disk.
This regenerates that block from the two things that actually know the truth:

  runs/*/loss.jsonl      -- via viz/dashboard.py load_runs (best bpb, tokens)
  experiments.json       -- the arm registry (verdicts, levers, goals)

Writes only between the AUTO fences, so prose around it stays hand-written.

Two rules this script will not break, because both were learned the hard way:

  1. bpb is comparable ONLY within one val slice. A Pile run and an
     OpenWebText run are scored on different text; ranking them against each
     other is the same mismatched-denominator mistake as the superseded
     +18.7% "pythia gap" (see commit a3a3b8b). So the scoreboard is grouped
     by val slice and groups are never merged.

  2. Steps are not comparable across batches. The x-unit is tokens seen.

Usage:
    python scripts/readme_state.py            # rewrite the block in place
    python scripts/readme_state.py --check    # exit 1 if stale (for CI/cron)
"""
import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
EXPERIMENTS = ROOT / "experiments.json"
FENCE_BEGIN = "<!-- AUTO:STATE:BEGIN -->"
FENCE_END = "<!-- AUTO:STATE:END -->"

# A run is "in flight" if its curve moved recently. Same window the dashboard
# uses, so the two never disagree about what is live.
LIVE_WINDOW_S = 3600

# Only runs that got far enough for a best-bpb to mean something. A 200-step
# smoke scoring 1.88 is not a result, it is a launch receipt.
MIN_STEP = 1000


def short_corpus(path):
    """Turn a corpus path into a human label that still says WHICH slice."""
    if not path:
        return "unknown"
    name = Path(path).name.replace(".txt", "")
    for prefix, label in (("pile", "Pile"), ("openwebtext", "OpenWebText"),
                          ("cosmopedia", "Cosmopedia"), ("finewiki", "FineWiki"),
                          ("tiny", "TinyStories"), ("libri", "LibriSpeech"),
                          ("racing", "Racing")):
        if name.startswith(prefix):
            return label
    # OWT slices share a prefix but are different text. Whatever we don't
    # recognise stays verbatim -- a real path beats a guess.
    return name


def git_head():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=ROOT, capture_output=True, text=True,
                             timeout=10).stdout.strip()
        return out or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def load_registry():
    if not EXPERIMENTS.exists():
        return []
    data = json.loads(EXPERIMENTS.read_text(encoding="utf-8"))
    return [a for a in data.get("arms", []) if isinstance(a, dict)]


def fmt_tokens(n):
    if not n:
        return "-"
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.0f}M"
    return f"{n:,.0f}"


def build_state():
    """Render the markdown block. Pure reads of disk -- no writes."""
    sys.path.insert(0, str(ROOT))
    from viz.dashboard import load_runs

    runs = load_runs(ROOT / "runs")
    arms = load_registry()
    now = datetime.now()

    scored = [r for r in runs
              if r.get("best_bpb") is not None
              and (r.get("best_step") or 0) >= MIN_STEP]

    live = [r for r in runs if r.get("live")]

    #-- Scoreboard, grouped by val slice. Never merge slices.
    by_slice = {}
    for r in scored:
        by_slice.setdefault(short_corpus(r.get("corpus")), []).append(r)

    lines = []
    lines.append(f"*Generated {now.strftime('%Y-%m-%d %H:%M')} from "
                 f"`runs/*/loss.jsonl` + `experiments.json`, git `{git_head()}`. "
                 f"Do not hand-edit — run `python scripts/readme_state.py`.*")
    lines.append("")

    #-- Headline: the best result per slice, only if it cleared MIN_STEP.
    if not scored:
        lines.append("No run has scored a usable best-bpb yet "
                     f"(need >={MIN_STEP} steps).")
        lines.append("")
    for slice_name, group in sorted(by_slice.items(),
                                   key=lambda kv: min(r["best_bpb"]
                                                      for r in kv[1])):
        best = min(group, key=lambda r: r["best_bpb"])
        rank_rows = sorted(group, key=lambda r: r["best_bpb"])[:6]
        lines.append(f"### Best on {slice_name}")
        lines.append("")
        lines.append("| Rank | Run | bpb | @ step | Tokens seen | Params |")
        lines.append("|------|-----|-----|--------|-------------|--------|")
        for i, r in enumerate(rank_rows, 1):
            name = r["name"]
            tok = (r.get("best_step") or 0) * (r.get("eff") or 0) * (r.get("ctx") or 0)
            star = " ⭐" if i == 1 else ""
            lines.append(
                f"| {i} | `{name}`{star} | **{r['best_bpb']:.4f}** | "
                f"{r['best_step']} | {fmt_tokens(tok)} | {r['params_m']:.1f}M |")
        lines.append("")
        lines.append(f"_Slices are separate. `{best['name']}` scored against "
                     f"`{best.get('corpus') or 'unknown'}` — do not rank it "
                     f"against a run from another slice._")
        lines.append("")

    #-- In flight. Empty is a legitimate, useful answer.
    lines.append("### In flight")
    lines.append("")
    if not live:
        lines.append(f"None. No curve has moved in the last "
                     f"{LIVE_WINDOW_S // 60} minutes. GPU is idle.")
    else:
        lines.append("| Run | last step | best bpb | tokens |")
        lines.append("|-----|-----------|----------|--------|")
        for r in sorted(live, key=lambda r: r["name"]):
            best = f"{r['best_bpb']:.4f}" if r.get("best_bpb") else "-"
            tok = (r.get("best_step") or 0) * (r.get("eff") or 0) * (r.get("ctx") or 0)
            lines.append(f"| `{r['name']}` | {r.get('last_step')} | {best} | "
                         f"{fmt_tokens(tok)} |")
    lines.append("")

    #-- Registry roll-up straight from experiments.json verdicts.
    tally = {}
    for a in arms:
        v = a.get("verdict") or "unset"
        tally[v] = tally.get(v, 0) + 1
    tally_txt = " · ".join(f"{v} {n}" for v, n in
                           sorted(tally.items(), key=lambda kv: -kv[1]))
    lines.append(f"### Registry — {len(arms)} arms")
    lines.append("")
    lines.append(f"{tally_txt}")
    lines.append("")
    lines.append("Open arms (not `closed`):")
    lines.append("")
    lines.append("| Arm | Family | Verdict | Goal |")
    lines.append("|-----|---------|---------|------|")
    open_arms = [a for a in arms if a.get("verdict") != "closed"]
    for a in sorted(open_arms, key=lambda a: a.get("id", "")):
        goal = (a.get("goal") or "").replace("|", "/")
        if len(goal) > 110:
            goal = goal[:107] + "..."
        lines.append(f"| `{a.get('id')}` | {a.get('family')} | "
                     f"**{a.get('verdict')}** | {goal} |")
    if not open_arms:
        lines.append("| _none_ | | | every arm closed |")
    lines.append("")

    #-- Arms whose verdict disagrees with the disk. A verdict of "running" on
    # a run whose curve stopped days ago is the exact rot this script exists
    # to surface, so we shout about it instead of printing it as fact.
    stale = []
    for a in arms:
        if a.get("verdict") != "running":
            continue
        dirs = a.get("run_dirs") or []
        done, moving = False, False
        for d in dirs:
            f = ROOT / "runs" / d / "loss.jsonl"
            if not f.exists():
                continue
            age = now.timestamp() - f.stat().st_mtime
            moving = moving or age < LIVE_WINDOW_S
            done = done or (age >= LIVE_WINDOW_S and f.stat().st_size > 0)
        if done and not moving:
            stale.append(a.get("id"))
    if stale:
        lines.append("> ⚠️ **Verdict drift.** Marked `running` in "
                     f"`experiments.json`, but nothing is training: "
                     f"{', '.join('`' + str(s) + '`' for s in stale)}. "
                     f"Update the registry (`viz/arm.py` / `experiments.json`) "
                     f"— the run finished, this row didn't.")
        lines.append("")

    lines.append(f"_Full arm history with levers and controls: "
                 f"`experiments.json`, rendered at `viz/training.html`._")
    return "\n".join(lines)


PROVENANCE_RE = re.compile(
    r"^_?\*?Generated .*?(?:\*|_)?$", re.MULTILINE | re.DOTALL)


def strip_provenance(block):
    """Drop the "Generated <timestamp> ... git <sha>" line.

    That line records WHEN and FROM WHICH COMMIT the block was written, so it
    can never match the block it is embedded in: committing a regenerated README
    changes HEAD, which changes the sha the block reports, which makes the next
    --check fail. The check would be red forever on a correct tree. Everything
    else in the block is derived from runs/ + experiments.json and IS comparable.
    """
    return PROVENANCE_RE.sub("", block).strip()


def splice(text, block):
    """Replace what's between the fences. Everything outside stays untouched."""
    if FENCE_BEGIN not in text or FENCE_END not in text:
        raise SystemExit(f"{README.name} is missing the AUTO fences. "
                         f"Add:\n  {FENCE_BEGIN}\n  {FENCE_END}")
    pre, rest = text.split(FENCE_BEGIN, 1)
    _, post = rest.split(FENCE_END, 1)
    return f"{pre}{FENCE_BEGIN}\n{block}\n{FENCE_END}{post}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the README block is out of date")
    args = ap.parse_args()

    block = build_state()
    current = README.read_text(encoding="utf-8")
    updated = splice(current, block)

    # Only the substance has to match. See strip_provenance(): the timestamp and
    # sha are a record of the write, not of the content, and comparing them makes
    # --check permanently red on a tree whose state block is actually correct.
    same_substance = (strip_provenance(current) == strip_provenance(updated))
    if same_substance and updated != current:
        # Provenance line is older than HEAD. Refresh it if we were asked to
        # write, but do not call the block stale.
        if not args.check:
            README.write_text(updated, encoding="utf-8")
            print(f"{README.name}: state block current; provenance line refreshed.")
            return 0
        print(f"{README.name}: state block already current "
              f"(provenance line lags HEAD by design).")
        return 0

    if updated == current:
        print(f"{README.name}: state block already current.")
        return 0

    if args.check:
        print(f"{README.name}: state block is STALE. "
              f"Run: python scripts/readme_state.py")
        return 1

    README.write_text(updated, encoding="utf-8")
    print(f"{README.name}: state block rewritten ({len(block)} chars).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())