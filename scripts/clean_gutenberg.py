#!/usr/bin/env python3
"""clean_gutenberg.py — Strip Gutenberg headers, footers, and boilerplate.

Reads all .txt.utf-8 files in data/LibriSpeech/books/utf-8/,
writes cleaned versions to data/LibriSpeech/books/clean/,
and reports before/after token counts.
"""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UTF8_DIR = ROOT / "data" / "LibriSpeech" / "books" / "utf-8"
CLEAN_DIR = ROOT / "data" / "LibriSpeech" / "books" / "clean"

# Header markers: everything before the first match is stripped
HEADER_PATTERNS = [
    r"\*\*\*START OF (?:THIS|THE) PROJECT GUTENBERG",
    r"\*\*\*START OF THE PROJECT GUTENBERG",
    r"END OF THE PROJECT GUTENBERG LICENSE",
    r"E-text prepared by",
    r"Produced by",
    r"\[PG Etext",
    r"THE PROJECT GUTENBERG ETEXT",
]

# Footer markers: everything from the first match onward is stripped
FOOTER_PATTERNS = [
    r"\*\*\*END OF (?:THIS|THE) PROJECT GUTENBERG",
    r"\*\*\*END OF THE PROJECT GUTENBERG",
    r"End of the Project Gutenberg Etext",
    r"End of Project Gutenberg",
    r"End of Project Gutenberg-tm",
    r"THE END",
    r"End of the Project Gutenberg",
    r"\[End of.*Gutenberg",
]

# Boilerplate lines to remove (exact or substring match)
BOILERPLATE_SUBSTRINGS = [
    "Project Gutenberg",
    "Gutenberg License",
    "Gutenberg-tm",
    "Gutenberg Legal Advisor",
    "Gutenberg Association",
    "Gutenberg Literary Archive",
    "Small Print! and all other references to Project Gutenberg",
    "Information about Project Gutenberg",
    "Information on contacting Project Gutenberg",
    "The Goal of Project Gutenberg",
    "The official release date",
    "You can use this eBook for",
    "Please take a look at the important information",
    "These Etexts Prepared By Hundreds of Volunteers",
    "Welcome To The World of Free Plain Vanilla Electronic Texts",
    "Michael S. Hart through the Project Gutenberg",
    "Copyright laws are changing all over the world",
]


def clean_book(text: str) -> str:
    """Strip Gutenberg boilerplate from a book text."""
    # Phase 1: Strip header — find first content marker
    lines = text.split("\n")
    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Skip empty lines and common header patterns
        if any(re.search(p, stripped, re.IGNORECASE) for p in HEADER_PATTERNS):
            # Found a header marker — skip everything up to and including this block
            # Skip until we hit a blank line followed by non-boilerplate
            j = i + 1
            while j < len(lines):
                if lines[j].strip() == "":
                    # Blank line — check if next line looks like content
                    if j + 1 < len(lines):
                        next_line = lines[j + 1].strip()
                        if next_line and not any(
                            bp.lower() in next_line.lower()
                            for bp in BOILERPLATE_SUBSTRINGS
                        ):
                            start_idx = j + 1
                            break
                    j += 1
                elif any(
                    bp.lower() in lines[j].strip().lower()
                    for bp in BOILERPLATE_SUBSTRINGS
                ):
                    j += 1
                else:
                    break
            if start_idx > 0:
                break
        # If line looks like actual content (not boilerplate), start here
        if (
            stripped
            and not any(bp.lower() in stripped.lower() for bp in BOILERPLATE_SUBSTRINGS)
            and not any(re.search(p, stripped, re.IGNORECASE) for p in HEADER_PATTERNS)
        ):
            start_idx = i
            break

    # Phase 2: Strip footer
    end_idx = len(lines)
    for i in range(len(lines) - 1, start_idx, -1):
        stripped = lines[i].strip()
        if any(re.search(p, stripped, re.IGNORECASE) for p in FOOTER_PATTERNS):
            end_idx = i
            break
        if any(bp.lower() in stripped.lower() for bp in BOILERPLATE_SUBSTRINGS):
            # Walk back until we hit content
            j = i - 1
            while j > start_idx:
                if lines[j].strip() and not any(
                    bp.lower() in lines[j].strip().lower()
                    for bp in BOILERPLATE_SUBSTRINGS
                ):
                    end_idx = j + 1
                    break
                j -= 1
            break

    # Phase 3: Clean remaining boilerplate lines
    cleaned = []
    for line in lines[start_idx:end_idx]:
        stripped = line.strip()
        # Skip boilerplate lines
        if any(bp.lower() in stripped.lower() for bp in BOILERPLATE_SUBSTRINGS):
            continue
        # Skip lines that are mostly asterisks or dashes (decorative headers)
        if re.match(r"^[\*\-\=]{5,}$", stripped):
            continue
        cleaned.append(line)

    # Phase 4: Collapse multiple blank lines
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned))
    return result.strip()


def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    total_before = 0
    total_after = 0
    file_count = 0

    for book_dir in sorted(UTF8_DIR.iterdir()):
        if not book_dir.is_dir():
            continue
        for txt_file in book_dir.glob("*.utf-8"):
            text = txt_file.read_text(encoding="utf-8", errors="replace")
            before_chars = len(text)
            total_before += before_chars

            cleaned = clean_book(text)
            after_chars = len(cleaned)
            total_after += after_chars

            # Write cleaned version
            out_dir = CLEAN_DIR / book_dir.name
            out_dir.mkdir(exist_ok=True)
            out_file = out_dir / txt_file.name
            out_file.write_text(cleaned, encoding="utf-8")

            file_count += 1
            if before_chars > after_chars:
                pct = (1 - after_chars / before_chars) * 100
                print(
                    f"  {book_dir.name}/{txt_file.name}: "
                    f"{before_chars:,} → {after_chars:,} chars (-{pct:.0f}%)"
                )

    print(f"\n=== Summary ===")
    print(f"Files cleaned: {file_count}")
    print(f"Before: {total_before:,} chars")
    print(f"After:  {total_after:,} chars")
    print(f"Removed: {total_before - total_after:,} chars ({(1 - total_after / total_before) * 100:.1f}%)")


if __name__ == "__main__":
    main()
