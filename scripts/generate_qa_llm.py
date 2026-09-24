#!/usr/bin/env python3
"""generate_qa_llm.py — Use LongCat 2.0 to generate racing Q&A training pairs.

Sends corpus chunks to LongCat via HFM and collects natural Q&A pairs.
"""
import json
import os
import sys
import time
import requests

import winreg
def get_api_key():
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Environment')
        val, _ = winreg.QueryValueEx(key, 'COMMANDCODE_API_KEY')
        return val
    except Exception:
        return os.environ.get('COMMANDCODE_API_KEY', '')

HFM_URL = "https://api.commandcode.ai/provider/v1/chat/completions"
MODEL = "meituan/LongCat-2.0:free"

SYSTEM_PROMPT = """You are a sim racing coach data generator. Given a text excerpt about racing technique, driving, setup, or motorsport knowledge, generate 5-8 high-quality Question/Answer training pairs.

Rules:
- Questions should sound like real questions a sim racer would ask their coach
- Answers should be specific, technical, and directly supported by the source text
- Include numbers, specific values, and concrete advice when available
- Vary question styles: "how do I...", "what happens when...", "why does...", "when should I..."
- Each answer should be 2-4 sentences, concise but complete
- Output ONLY valid JSON array: [{"q": "...", "a": "..."}, ...]
- Do NOT include any text outside the JSON array"""


def generate_qa_from_chunk(text, chunk_idx, max_retries=3):
    """Send a chunk to LongCat and get Q&A pairs back."""
    prompt = f"""Generate 5-8 Q&A training pairs from this racing knowledge:

{text[:4000]}"""

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2000,
        "temperature": 0.7
    }

    api_key = get_api_key()
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    for attempt in range(max_retries):
        try:
            resp = requests.post(HFM_URL, json=payload, headers=headers, timeout=120)
            if resp.status_code != 200:
                print(f"  Chunk {chunk_idx}: HTTP {resp.status_code}, retrying...")
                time.sleep(2)
                continue

            data = resp.json()
            msg = data["choices"][0]["message"]
            content = msg.get("content", "").strip()
            if not content:
                content = msg.get("reasoning_content", "").strip()

            # Extract JSON from response (may have markdown fencing)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            # Find the JSON array
            start = content.find("[")
            end = content.rfind("]") + 1
            if start >= 0 and end > start:
                content = content[start:end]

            qa_pairs = json.loads(content)
            return qa_pairs

        except json.JSONDecodeError as e:
            print(f"  Chunk {chunk_idx}: JSON parse error (attempt {attempt+1})")
            if attempt < max_retries - 1:
                time.sleep(1)
        except Exception as e:
            print(f"  Chunk {chunk_idx}: Error {e} (attempt {attempt+1})")
            if attempt < max_retries - 1:
                time.sleep(2)

    return []


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="Start chunk index")
    parser.add_argument("--end", type=int, default=999, help="End chunk index")
    parser.add_argument("--output", default="data/racing_finetune/racing_qa_llm.jsonl")
    args = parser.parse_args()

    chunk_dir = "data/racing_finetune/llm_chunks"
    chunks = sorted([f for f in os.listdir(chunk_dir) if f.endswith(".txt")])
    chunks = chunks[args.start:min(args.end, len(chunks))]

    print(f"Processing {len(chunks)} chunks (index {args.start}-{args.start + len(chunks) - 1})")
    print(f"Model: {MODEL}")
    print(f"Output: {args.output}")

    all_qa = []
    output_mode = "a" if args.start > 0 else "w"

    for i, chunk_file in enumerate(chunks):
        chunk_idx = args.start + i
        with open(os.path.join(chunk_dir, chunk_file), "r") as f:
            text = f.read()

        if len(text.strip()) < 200:
            print(f"  Chunk {chunk_idx}: SKIP (too short)")
            continue

        print(f"  Chunk {chunk_idx}/{args.start + len(chunks) - 1} ({len(text):,} chars)...", end=" ", flush=True)

        qa_pairs = generate_qa_from_chunk(text, chunk_idx)

        if qa_pairs:
            all_qa.extend(qa_pairs)
            print(f"-> {len(qa_pairs)} pairs")
        else:
            print("-> 0 pairs (failed)")

        # Write incrementally
        if qa_pairs:
            with open(args.output, output_mode, encoding="utf-8") as f:
                for qa in qa_pairs:
                    record = {
                        "instruction": qa.get("q", qa.get("question", "")),
                        "response": qa.get("a", qa.get("answer", "")),
                        "source": f"llm_chunk_{chunk_idx}",
                        "category": "llm_generated"
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_mode = "a"

        # Rate limit
        time.sleep(0.5)

    print(f"\nTotal: {len(all_qa)} Q&A pairs generated")
    print(f"Written to: {args.output}")


if __name__ == "__main__":
    main()
