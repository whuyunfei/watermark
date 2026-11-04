#!/usr/bin/env python3
"""Generate one watermarked SD image.

  python generate.py --prompt "a photo of a cat" --seed 8 --steps 15
"""

from __future__ import annotations

import argparse
import os

from ringid.key import load_key
from ringid.pipeline import generate_watermarked, load_pipeline

ROOT = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="a photo of a beautiful woman")
    p.add_argument("--seed", type=int, default=8)
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--key", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    key = load_key(args.key)
    pipe = load_pipeline()
    image = generate_watermarked(pipe, key, args.prompt, seed=args.seed, steps=args.steps)
    out = args.out or os.path.join(ROOT, "output", f"wm_seed{args.seed}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    image.save(out)
    print(out)


if __name__ == "__main__":
    main()
