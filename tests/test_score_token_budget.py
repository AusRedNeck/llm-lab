"""Token matching must use each checkpoint's actual training batch and context."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from score_ours_vs_reference import tokens_seen


def test_tokens_seen_uses_effective_token_batch_not_sequence_batch():
    assert tokens_seen(
        step=16_000,
        train_args={"batch": 2, "accum": 32},
        cfg={"context_length": 2048},
    ) == 2_097_152_000
