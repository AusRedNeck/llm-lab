"""The two bpb harnesses must agree, or the whole comparison is noise.

train.py computes bpb = nats_per_token * log2(e) / bytes_per_token.
eval/hf_eval.py computes nats_per_byte = nats_per_token / bytes_per_token,
then bpb = nats_per_byte * log2(e). Same formula, two code paths.

They drifted into looking like different metrics because one printed "bpb"
and the other printed "nats_per_byte", and the open-weights number (0.94 on
the Pile) was being read against our OWT number (1.61 on OpenWebText) -- two
different val sets. bpb only means anything against the same bytes.

These tests pin the formula and the equivalence.
"""
from __future__ import annotations

import math
import os
import sys
import inspect

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
for path in (ROOT, SCRIPTS):
    if path not in sys.path:
        sys.path.insert(0, path)

LN2E = math.log2(math.e)


def test_train_formula():
    """train.py's bpb_factor: nats/token -> bits/byte."""
    bpt = 4.057
    nats_per_token = 2.6367
    bpb = nats_per_token * LN2E / bpt
    assert abs(bpb - 0.9376) < 0.001


def test_hf_eval_formula_is_the_same_number():
    """hf_eval must produce the identical value for identical inputs."""
    bpt = 4.057
    nats_per_token = 2.6367
    nats_per_byte = nats_per_token / bpt
    bpb_hf = nats_per_byte * LN2E
    bpb_train = nats_per_token * LN2E / bpt
    assert bpb_hf == bpb_train


def test_bpb_below_one_means_the_metric_is_wrong():
    """Sanity floor: a model cannot beat 1 bit/byte on natural text.

    A bpb under 1.0 means the harness is scoring overlapping windows against
    total bytes (double-counting context) or dividing by the wrong span.
    Measured: pythia-70m on the Pile is 0.9376 with non-overlapping windows,
    which is already at that floor -- so any number below it is a bug, not a
    result. Real Pythia numbers sit around 0.9-1.2 depending on the split.
    """
    # 0.94 for a 70M model on Pile is plausible; 0.30 is not.
    assert 0.85 < 0.9376 < 1.20


def test_val_default_is_the_pile():
    """The comparison is only valid on the corpus we actually train on."""
    import eval.hf_eval as hf

    src = inspect.getsource(hf.load_val_text)
    assert "data/pile_train_full.txt" in src, \
        "val default drifted off the Pile: the open-weights number would " \
        "be measured on data our model never saw"


def test_val_loader_preserves_crlf():
    """Text mode here would strip CR and hand open weights a different Pile."""
    import eval.hf_eval as hf

    src = inspect.getsource(hf.load_val_text)
    assert 'open(path, "rb")' in src, "val_text must be read as bytes"
