#!/usr/bin/env python3
"""Calibrate the detection threshold on clean SD images and save output/threshold.json.

CPU cost is about n_neg full generations, each searched at 10 degree steps.
"""

from __future__ import annotations

import argparse
import json
import os

from ringid.codec import calibrate_threshold
from ringid.key import load_key
from ringid.pipeline import load_pipeline

ROOT = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-neg", type=int, default=16)
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--prompt", default="a photo of a beautiful woman")
    p.add_argument("--seed-base", type=int, default=9000)
    p.add_argument("--key", default=None)
    args = p.parse_args()

    key = load_key(args.key)
    pipe = load_pipeline()
    threshold = calibrate_threshold(
        key,
        pipe,
        n_neg=args.n_neg,
        prompt=args.prompt,
        steps=args.steps,
        seed_base=args.seed_base,
    )
    out = os.path.join(ROOT, "output", "threshold.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold, "n_neg": args.n_neg, "steps": args.steps}, f, indent=2)
    print(f"threshold={threshold:.4f}")
    print(out)


if __name__ == "__main__":
    main()
