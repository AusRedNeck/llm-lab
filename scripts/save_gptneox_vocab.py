#!/usr/bin/env python3
"""Save GPT-NeoX (Pythia-70M) vocab to our BPETokenizer format.

Source: EleutherAI/pythia-70m-deduped (HF cache, no retrain).
Out: data/bpe_gptneox.json (vocab + merges + eos flag).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model.bpe import BPETokenizer, EOS
from model.train_bpe_hf import _decode_bytelevel_token

SRC = "EleutherAI/pythia-70m-deduped"
OUT = "data/bpe_gptneox.json"
SPECIALS = ("<|endoftext|>", "<|end_of_text|>")


def main() -> None:
    t0 = time.time()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SRC)  # cached after first pull
    js = json.loads(tok._tokenizer.to_str())
    hf_vocab = js["model"]["vocab"]
    hf_merges = js["model"].get("merges", [])
    added = {a["content"]: a["id"] for a in js.get("added_tokens", [])}
    print(f"  HF vocab={len(hf_vocab)} merges={len(hf_merges)} added={len(added)}",
          flush=True)

    # ids -> raw bytes (ByteLevel unmapped, same trick as train_bpe_hf).
    our_vocab: dict[int, bytes] = {}
    for s, i in hf_vocab.items():
        our_vocab[i] = EOS.encode("utf-8") if s in SPECIALS \
            else _decode_bytelevel_token(s)
    for s, i in added.items():  # multi-space helpers live past the base vocab
        if i not in our_vocab:
            our_vocab[i] = EOS.encode("utf-8") if s in SPECIALS \
                else _decode_bytelevel_token(s)

    our_merges: dict[tuple[bytes, bytes], int] = {}
    for rank, (a, b) in enumerate(hf_merges):
        our_merges[(_decode_bytelevel_token(a),
                    _decode_bytelevel_token(b))] = rank

    btok = BPETokenizer(our_vocab, our_merges, eos=True)
    print(f"  ours: vocab={len(btok.vocab)} merges={len(btok.merges)} "
          f"eos_id={btok.eos_id}", flush=True)

    # Probe: ours (pure Python) vs HF (Rust) on the same line.
    probe = "Once upon a time there was a little princess."
    ours = btok.encode(probe)
    hf_ids = tok.encode(probe, add_special_tokens=False)
    print(f"  probe: ours={len(ours)} toks HF={len(hf_ids)} toks "
          f"match={ours == hf_ids}", flush=True)
    assert btok.decode(ours) == probe, "roundtrip broke"
    print("  roundtrip: OK", flush=True)

    btok.save(OUT)
    print(f"  saved {OUT} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
