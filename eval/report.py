"""CLI for the eval harness.

Usage:
    python -m eval.report checkpoints/exp004_best.pt checkpoints/exp006_best.pt
    python -m eval.report checkpoints/*step5000*.pt --batches 30
"""
import argparse
import glob
import sys

from eval.benchmark import eval_report, print_report


def main():
    ap = argparse.ArgumentParser(description="Eval harness: fair cross-vocab metrics")
    ap.add_argument("checkpoints", nargs="*", help="Checkpoint .pt files (glob OK)")
    ap.add_argument("--batches", type=int, default=20,
                    help="Val batches per checkpoint (default: 20)")
    args = ap.parse_args()

    # Expand globs
    paths = []
    for p in args.checkpoints:
        expanded = glob.glob(p)
        if expanded:
            paths.extend(expanded)
        else:
            paths.append(p)  # let eval_checkpoint report the error

    if not paths:
        print("No checkpoints specified.")
        print("Usage: python -m eval.report checkpoints/*.pt")
        sys.exit(1)

    print(f"Evaluating {len(paths)} checkpoint(s) with {args.batches} val batches...")
    results = eval_report(paths, val_batches=args.batches)
    print_report(results)


if __name__ == "__main__":
    main()
