# Rope vs no-rope (or any checkpoints) head-to-head on held-out val.
# Usage:
#   python -m eval.bakeoff                                    # default pair
#   python -m eval.bakeoff --ckpts ckpt1.pt ckpt2.pt ckpt3.pt
#   python -m eval.bakeoff --val_batches 30
#
# Now powered by eval_report — same metrics, same fair nats-per-byte.

import argparse
import sys

from eval.benchmark import eval_report, print_report

DEFAULT_CKPTS = {
    "rope-1801": "checkpoints/exp002_bytes10m_rope_202609032333_step5000.pt",
    "norope-2313": "checkpoints/exp002_bytes10m_202609032313_step5000.pt",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_batches", type=int, default=20)
    ap.add_argument("--ckpts", nargs="*", default=None,
                    help="Checkpoint paths (overrides default pair)")
    args = ap.parse_args()

    if args.ckpts:
        paths = args.ckpts
    else:
        paths = list(DEFAULT_CKPTS.values())

    results = eval_report(paths, val_batches=args.val_batches)
    print_report(results)


if __name__ == "__main__":
    main()
