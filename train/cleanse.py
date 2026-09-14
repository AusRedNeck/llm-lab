# Cleanse Gutenberg books for training: boilerplate out, story stays.
# Rule: cut everything through the *** START OF line, cut everything
# from the *** END OF line on. No markers -> quarantine, never guess.
# Old-style headers and Australia HTML go to quarantine for a human pass.
#
# Run: uv run python -m train.cleanse --src data/librispeech --out data/librispeech_clean
"""Strip Project Gutenberg boilerplate from book files."""
from __future__ import annotations

import argparse
import glob
import os
import re

# Variants seen: "*** START OF ...", "***START OF THE PROJECT...".
START_RE = re.compile(r"\*\*\*\s*START OF")
# Full marker only — bare "END OF VOL. I." is body text, not boilerplate.
END_RE = re.compile(r"\*\*\*\s*END OF")
# Old-style headers ("Project Gutenberg Etext ...", no *** markers).
OLDSTYLE_RE = re.compile(r"(?i)^(?:the\s+)?project gutenberg\s?(etext|etext of|['’]s)")
# Gutenberg Australia HTML editions.
AUSTRALIA_RE = re.compile(r"gutenberg\.net\.au")


def cleanse_text(text: str) -> tuple[str, str]:
    # Returns (cleaned_body, status): ok | no_markers | oldstyle | australia.
    # Status is the quarantine reason when the body is returned unchanged.
    if AUSTRALIA_RE.search(text[:2000]):
        return text, "australia"
    lines = text.split("\n")
    start_idx = next((i for i, ln in enumerate(lines) if START_RE.search(ln)), None)
    if start_idx is None:
        if any(OLDSTYLE_RE.search(ln) for ln in lines[:5]):
            return text, "oldstyle"
        return text, "no_markers"
    body = "\n".join(lines[start_idx + 1:])
    body_lines = body.split("\n")
    end_idx = next((i for i, ln in enumerate(body_lines) if END_RE.search(ln)), None)
    if end_idx is not None:
        body = "\n".join(body_lines[:end_idx])
    return body.strip() + "\n", "ok"


def cleanse_tree(src: str, out: str) -> dict:
    # Mirror src tree under out; quarantined files are SKIPPED (listed, kept).
    files = sorted(glob.glob(os.path.join(src, "**", "*.txt"), recursive=True))
    stats: dict = {"total": len(files), "ok": 0, "quarantine": {}}
    for fp in files:
        with open(fp, encoding="utf-8", errors="replace") as f:
            text = f.read()
        body, status = cleanse_text(text)
        if status != "ok":
            stats["quarantine"].setdefault(status, []).append(fp)
            continue
        rel = os.path.relpath(fp, src)
        dest = os.path.join(out, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(body)
        stats["ok"] += 1
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/librispeech")
    ap.add_argument("--out", default="data/librispeech_clean")
    ap.add_argument("--quarantine_list", default="data/quarantine.txt")
    args = ap.parse_args()

    stats = cleanse_tree(args.src, args.out)
    q = stats["quarantine"]
    print(f"total={stats['total']} ok={stats['ok']} "
          f"quarantined={sum(len(v) for v in q.values())}")
    with open(args.quarantine_list, "w", encoding="utf-8") as f:
        for reason in sorted(q):
            for fp in q[reason]:
                f.write(f"{reason}\t{fp}\n")
                print(f"  Q[{reason}] {fp}")
    print(f"clean -> {args.out}  list -> {args.quarantine_list}")


if __name__ == "__main__":
    main()
