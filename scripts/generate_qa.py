#!/usr/bin/env python3
"""generate_qa.py — Generate Q&A training pairs from racing corpus.

Reads the consolidated racing corpus and produces instruction-following
training data in Alpaca-style format:

    ### Instruction: [question]
    ### Response: [answer]

Two modes:
1. Template-based: extract facts, generate structured Q&A pairs
2. LLM-assisted: use local model to generate natural Q&A pairs

Usage:
    python generate_qa.py                    # template mode
    python generate_qa.py --llm              # LLM-assisted mode
    python generate_qa.py --output qa.jsonl  # custom output
"""
import json
import os
import re
import sys
import random
from pathlib import Path


def load_corpus(path):
    """Load the consolidated racing corpus."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_into_sections(text):
    """Split corpus into topical sections."""
    sections = []
    # Split on ## headers or === separators
    parts = re.split(r'\n(?:## (?:SOURCE: )?[A-Z]|={3,})', text)
    for part in parts:
        part = part.strip()
        if len(part) > 200:
            sections.append(part)
    return sections


def split_into_paragraphs(text, min_len=100):
    """Split text into paragraphs."""
    paras = [p.strip() for p in text.split('\n\n') if len(p.strip()) > min_len]
    return paras


# ============================================================
# TEMPLATE-BASED Q&A GENERATION
# ============================================================

# Racing-specific question templates keyed to topic patterns
QUESTION_TEMPLATES = {
    "tire_pressure": [
        "What tire pressure should I run in {car}?",
        "How do I adjust tire pressures for {track}?",
        "What's the optimal cold tire pressure for iRacing?",
        "How does tire pressure affect grip?",
        "When should I increase front tire pressure?",
        "How does tire pressure change over a stint?",
    ],
    "braking": [
        "How do I trail brake effectively?",
        "What brake bias should I use?",
        "How do I reduce brake lockups?",
        "When should I brake earlier vs harder?",
        "How does brake bias affect car rotation?",
        "What's the difference between threshold braking and trail braking?",
    ],
    "setup": [
        "How do I set up a {car} for {track}?",
        "What suspension changes help with oversteer?",
        "How do I fix understeer in my setup?",
        "What anti-roll bar settings should I use?",
        "How does ride height affect aero balance?",
        "When should I stiffen the rear springs?",
    ],
    "driving_technique": [
        "How do I carry more speed through corners?",
        "What's the racing line for {corner}?",
        "How do I improve my corner exit speed?",
        "When should I start turning in?",
        "How do I manage weight transfer under braking?",
        "What does smooth steering input mean?",
    ],
    "race_craft": [
        "How do I defend a position without blocking?",
        "When is the best time to overtake?",
        "How do I set up a pass on the straight?",
        "How do I race side by side through a corner?",
        "What's the safest way to dive bomb?",
        "How do I manage轮胎 wear in a long stint?",
    ],
    "car_specific": [
        "How does the BMW M2 handle compared to the GT3?",
        "What's unique about driving the Dallara P217?",
        "How do I adjust my driving for the Mercedes AMG GT3?",
        "What brake bias works for the NASCAR vehicles?",
        "How do I set up the Aston Martin Vantage for endurance?",
    ],
    "telemetry": [
        "What telemetry data tells me where I'm losing time?",
        "How do I analyze my brake trace?",
        "What does my throttle trace reveal about my driving?",
        "How do I compare my lap to a faster driver?",
        "What's the most important telemetry channel?",
        "How do I use delta time to improve?",
    ],
    "mental_game": [
        "How do I stay focused during a long race?",
        "How do I recover from a mistake mid-race?",
        "How do I handle the pressure of leading?",
        "What should I think about during formation lap?",
        "How do I deal with getting wrecked?",
    ],
    "weather": [
        "How do I adjust for rain conditions?",
        "What tire pressures work in the wet?",
        "When should I switch to wet tires?",
        "How do I find grip on a drying track?",
        "How does rain affect brake bias?",
    ],
    "track_specific": [
        "How do I get a good lap at {track}?",
        "What's the trickiest corner at {track}?",
        "How do I set up for a street circuit?",
        "What's the best passing zone at {track}?",
        "How do I manage elevation changes?",
    ],
}


def extract_facts_from_section(section):
    """Extract key facts from a text section for Q&A generation."""
    facts = []
    sentences = re.split(r'[.!?]+', section)
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 30 or len(sent) > 300:
            continue
        # Look for sentences with specific info
        keywords = ['should', 'will', 'because', 'when', 'how', 'always',
                     'never', 'increase', 'decrease', 'optimal', 'best',
                     'higher', 'lower', 'more', 'less', 'affects',
                     'causes', 'results', 'improves', 'reduces']
        if any(kw in sent.lower() for kw in keywords):
            facts.append(sent)
    return facts


def generate_qa_from_facts(facts, section_text):
    """Generate Q&A pairs from extracted facts."""
    qa_pairs = []
    for fact in facts:
        # Generate a question that the fact answers
        fact_lower = fact.lower()

        if any(w in fact_lower for w in ['tire', 'pressure', 'psi']):
            q = "What should I know about tire pressures?"
            category = "tire_pressure"
        elif any(w in fact_lower for w in ['brake', 'braking', 'bias']):
            q = "What's important about braking technique?"
            category = "braking"
        elif any(w in fact_lower for w in ['setup', 'suspension', 'spring', 'damper']):
            q = "How should I think about car setup?"
            category = "setup"
        elif any(w in fact_lower for w in ['corner', 'turn', 'apex', 'racing line']):
            q = "How do I approach corners effectively?"
            category = "driving_technique"
        elif any(w in fact_lower for w in ['overtake', 'pass', 'defend', 'draft']):
            q = "What's the key to race craft?"
            category = "race_craft"
        elif any(w in fact_lower for w in ['throttle', 'acceleration', 'exit']):
            q = "How should I manage the throttle?"
            category = "driving_technique"
        elif any(w in fact_lower for w in ['weight transfer', 'balance', 'grip']):
            q = "How does weight transfer affect the car?"
            category = "driving_technique"
        else:
            q = "What should I know about racing technique?"
            category = "general"

        qa_pairs.append({
            "instruction": q,
            "response": fact.strip(),
            "category": category,
            "source": "corpus_extraction"
        })
    return qa_pairs


# ============================================================
# CURATED Q&A PAIRS (from known racing knowledge)
# ============================================================

CURATED_QA = [
    {
        "instruction": "What is trail braking?",
        "response": "Trail braking is a technique where you keep some brake pressure applied as you turn into a corner, gradually releasing the brake as you approach the apex. This keeps weight on the front tires, giving you more grip for the initial turn-in. Think of it as 'trailing' the brake into the corner rather than releasing it completely before turning.",
        "category": "braking"
    },
    {
        "instruction": "How do I fix understeer?",
        "response": "Understeer is when the car pushes wide and won't turn. To fix it: 1) Increase front tire pressure slightly, 2) Soften the front anti-roll bar or stiffen the rear, 3) Lower front ride height or raise rear, 4) Increase front downforce if available, 5) Check your brake bias — too far forward can cause entry understeer. But first, check your driving: are you turning in too early or carrying too much speed?",
        "category": "setup"
    },
    {
        "instruction": "How do I fix oversteer?",
        "response": "Oversteer is when the rear steps out. To fix it: 1) Stiffen the front anti-roll bar or soften the rear, 2) Increase rear tire pressure slightly, 3) Raise front ride height or lower rear, 4) Increase rear downforce, 5) Check brake bias — too far rear can cause snap oversteer on entry. In the moment, catch it with smooth countersteer and a slight lift — don't slam the brakes.",
        "category": "setup"
    },
    {
        "instruction": "What is the racing line?",
        "response": "The racing line is the fastest path through a corner. It typically goes: outside on approach, clip the apex (inside), and exit wide. The key is to make the corner as straight as possible — this lets you carry more speed. The geometric apex isn't always the fastest; a late apex lets you get on the throttle earlier for better exit speed.",
        "category": "driving_technique"
    },
    {
        "instruction": "What brake bias should I use?",
        "response": "Brake bias determines how braking force is split between front and rear. A typical starting point is 55-58% front. More front bias = more stable under braking but less rotation. More rear bias = more rotation but risk of rear lockup. Start conservative (more front) and move it rearward as you get comfortable. In the wet, move bias slightly forward to avoid rear lockup.",
        "category": "braking"
    },
    {
        "instruction": "How do I improve my lap times?",
        "response": "The biggest time gains come from: 1) Consistency — a clean lap is faster than a fast lap with mistakes. 2) Braking — brake later and harder, then trail brake into the corner. 3) Exit speed — prioritize getting on the throttle early. The exit of one corner is the entry to the next straight. 4) Vision — look further ahead. 5) Smoothness — every sudden input upsets the car's balance.",
        "category": "driving_technique"
    },
    {
        "instruction": "What is threshold braking?",
        "response": "Threshold braking is braking at the maximum grip of the tires without locking up. You apply brake pressure up to the point just before the wheels lock, then modulate. The goal is to use 100% of available grip. Modern ABS in real cars does this automatically, but in sims without ABS you need to feel for the lockup point and stay right at the edge.",
        "category": "braking"
    },
    {
        "instruction": "How does weight transfer work?",
        "response": "Weight transfer is how the car's weight shifts during acceleration, braking, and cornering. Braking shifts weight forward (more front grip, less rear). Acceleration shifts weight backward (more rear grip, less front). Cornering shifts weight to the outside. Understanding this is key: you're managing grip by controlling where the weight goes. Smooth inputs = smooth weight transfer = more total grip.",
        "category": "driving_technique"
    },
    {
        "instruction": "How do I race in the rain?",
        "response": "Rain racing tips: 1) Smooth inputs become even more critical — every jerk upsets the car more on low grip. 2) Brake earlier and more gently. 3) Stay off the painted lines and curbs — they're like ice when wet. 4) Follow the rubbered-in line when it's wet, but when it's drying, the offline line has more grip (the rubber is slippery when wet). 5) Tire pressures should be slightly lower. 6) Be patient — lap times will be much slower.",
        "category": "weather"
    },
    {
        "instruction": "How do I manage tire wear?",
        "response": "Tire wear management: 1) Smooth driving preserves tires — sliding = wear. 2) Avoid locking brakes and wheelspin. 3) In endurance races, you may need to drive at 95% to make tires last. 4) Higher tire pressures run cooler but may lose grip. 5) Watch your temps — if they spike, back off for a lap to let them cool. 6) The last 10 laps on worn tires is where races are won or lost.",
        "category": "tire_pressure"
    },
    {
        "instruction": "How do I set up a car for qualifying vs race?",
        "response": "Qualifying setup: softer front springs, higher rear ride height for more rotation, lower tire pressures for maximum grip, aggressive camber for cornering grip. Race setup: stiffer springs for consistency, slightly higher pressures to manage heat over long runs, more conservative camber to spread tire wear. The key difference: qualifying is one lap, race is about consistency over many laps.",
        "category": "setup"
    },
    {
        "instruction": "What is the difference between oversteer and understeer?",
        "response": "Understeer: the front tires lose grip first and the car pushes wide — you turn the wheel but the car goes straight. Oversteer: the rear tires lose grip first and the back of the car slides out — the car turns more than you asked. Understeer is generally safer (the car goes where it's pointed) but oversteer can be faster if you can control it with throttle and steering.",
        "category": "driving_technique"
    },
    {
        "instruction": "How do I improve my consistency?",
        "response": "Consistency comes from: 1) Reference points — pick braking, turn-in, apex, and exit markers for every corner. 2) Same approach every lap — don't try to be fast, try to be repeatable. 3) Vision — look at your reference points, not the car ahead. 4) Smooth inputs — consistent inputs produce consistent laps. 5) Mental discipline — drive the track, not the delta. The fastest drivers are consistent first, fast second.",
        "category": "mental_game"
    },
    {
        "instruction": "How do I set up for a street circuit?",
        "response": "Street circuit setup: stiffer suspension to handle bumps and curbs, higher ride height to avoid bottoming out, more compliance in the dampers (slower rebound), slightly higher tire pressures to handle the rough surface. Driving-wise: braking zones are shorter, walls punish mistakes, and grip changes lap to lap as rubber goes down. Be aggressive on curbs but respect the walls.",
        "category": "track_specific"
    },
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/racing_finetune/racing_corpus_consolidated.txt")
    parser.add_argument("--output", default="data/racing_finetune/racing_qa.jsonl")
    parser.add_argument("--llm", action="store_true", help="Use LLM to generate additional Q&A")
    args = parser.parse_args()

    corpus_path = args.corpus
    output_path = args.output

    print(f"Loading corpus: {corpus_path}")
    corpus = load_corpus(corpus_path)
    sections = split_into_sections(corpus)
    print(f"  {len(sections)} sections, {len(corpus):,} chars")

    all_qa = []

    # 1. Template-based extraction
    print("\nExtracting facts from corpus...")
    for i, section in enumerate(sections):
        facts = extract_facts_from_section(section)
        qa_pairs = generate_qa_from_facts(facts, section)
        all_qa.extend(qa_pairs)
        if facts:
            print(f"  Section {i}: {len(facts)} facts -> {len(qa_pairs)} Q&A pairs")

    # 2. Add curated pairs
    all_qa.extend(CURATED_QA)
    print(f"\nTemplate Q&A: {len(all_qa) - len(CURATED_QA)} pairs")
    print(f"Curated Q&A: {len(CURATED_QA)} pairs")
    print(f"Total: {len(all_qa)} pairs")

    # 3. LLM-assisted generation (if requested)
    if args.llm:
        print("\nLLM-assisted Q&A generation...")
        try:
            from hermes_tools import execute_code
            llm_qa = generate_llm_qa(corpus, sections)
            all_qa.extend(llm_qa)
            print(f"LLM Q&A: {len(llm_qa)} pairs")
        except Exception as e:
            print(f"LLM generation failed: {e}")

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for qa in all_qa:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    print(f"\nWritten: {output_path}")
    print(f"Total Q&A pairs: {len(all_qa)}")

    # Also write Alpaca format for compatibility
    alpaca_path = output_path.replace(".jsonl", "_alpaca.json")
    alpaca_data = []
    for qa in all_qa:
        alpaca_data.append({
            "instruction": qa["instruction"],
            "input": "",
            "output": qa["response"]
        })
    with open(alpaca_path, "w", encoding="utf-8") as f:
        json.dump(alpaca_data, f, indent=2, ensure_ascii=False)
    print(f"Alpaca format: {alpaca_path}")


if __name__ == "__main__":
    main()
