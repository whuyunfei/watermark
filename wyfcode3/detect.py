#!/usr/bin/env python3
"""Detect a RingID v4 watermark. Needs the key, the image, and the SD VAE.

  python detect.py --image output/wm_seed8.png
  python detect.py --image output/wm_seed8.png --threshold 0.557
"""

from __future__ import annotations

import argparse

import json
import os

from PIL import Image

from ringid.codec import calibrate_threshold, verify_image
from ringid.key import load_key
from ringid.pipeline import load_pipeline

ROOT = os.path.dirname(os.path.abspath(__file__))


def resolve_threshold(key: str, given: float | None) -> float:
    if given is not None:
        return given
    path = os.path.join(ROOT, "output", "threshold.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return float(json.load(f)["threshold"])
    print("No output/threshold.json. Using a fast latent-noise threshold.")
    print("For clean-SD calibration run: python calibrate.py")
    return calibrate_threshold(key, pipe=None)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--key", default=None)
    p.add_argument("--threshold", type=float, default=None)
    args = p.parse_args()

    key = load_key(args.key)
    pipe = load_pipeline()
    image = Image.open(args.image).convert("RGB")
    result = verify_image(key, image, pipe, threshold=resolve_threshold(key, args.threshold))
    print(
        f"verified={result.verified} peak={result.peak:.4f} "
        f"threshold={result.threshold:.4f} estimated_deg={result.estimated_deg:.1f} "
        f"zero_deg_peak={result.zero_deg_peak:.4f}"
    )


if __name__ == "__main__":
    main()
