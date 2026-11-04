#!/usr/bin/env python3
"""First SD test: download weights if needed, then clean vs watermark vs one rotation.

Saves a side-by-side PNG for visual quality. Does not print PSNR or SSIM.

  python eval_quick.py
  python eval_quick.py --threshold 0.557 --angle 37.0
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image

from ringid.attacks import rotate
from detect import resolve_threshold
from ringid.codec import verify_image
from ringid.key import load_key
from ringid.pipeline import generate_clean, generate_watermarked, load_pipeline

ROOT = os.path.dirname(os.path.abspath(__file__))


def _strip(images: list[Image.Image]) -> Image.Image:
    w, h = images[0].size
    canvas = Image.new("RGB", (w * len(images), h))
    for i, img in enumerate(images):
        canvas.paste(img.convert("RGB"), (i * w, 0))
    return canvas


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="a photo of a beautiful woman")
    p.add_argument("--seed", type=int, default=8)
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--angle", type=float, default=None, help="Rotation degrees. Default: one random angle.")
    p.add_argument("--angle-seed", type=int, default=100)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--key", default=None)
    args = p.parse_args()

    key = load_key(args.key)
    pipe = load_pipeline()
    angle = args.angle
    if angle is None:
        rng = np.random.default_rng(args.angle_seed)
        angle = round(float(rng.uniform(-179.0, 179.0)), 1)

    print("Generating clean SD image ...", flush=True)
    clean = generate_clean(pipe, args.prompt, seed=args.seed, steps=args.steps)
    print("Generating watermarked SD image ...", flush=True)
    wm = generate_watermarked(pipe, key, args.prompt, seed=args.seed, steps=args.steps)
    spun = rotate(wm, angle)

    out_dir = os.path.join(ROOT, "output", "eval_quick")
    os.makedirs(out_dir, exist_ok=True)
    clean.save(os.path.join(out_dir, f"clean_seed{args.seed}.png"))
    wm.save(os.path.join(out_dir, f"wm_seed{args.seed}.png"))
    spun.save(os.path.join(out_dir, f"wm_seed{args.seed}_rot_{angle:+.1f}.png"))
    strip_path = os.path.join(out_dir, f"quality_clean_wm_rot_seed{args.seed}.png")
    _strip([clean, wm, spun]).save(strip_path)

    print("Detecting ...", flush=True)
    threshold = resolve_threshold(key, args.threshold)
    clean_res = verify_image(key, clean, pipe, threshold=threshold)
    wm_res = verify_image(key, wm, pipe, threshold=threshold)
    rot_res = verify_image(key, spun, pipe, threshold=threshold)
    print(f"angle={angle:+.1f}")
    print(f"clean  verified={clean_res.verified} peak={clean_res.peak:.4f}")
    print(f"wm     verified={wm_res.verified} peak={wm_res.peak:.4f}")
    print(f"rotated verified={rot_res.verified} peak={rot_res.peak:.4f} est={rot_res.estimated_deg:.1f}")
    print(f"threshold={clean_res.threshold:.4f}")
    print(f"quality strip: {strip_path}")


if __name__ == "__main__":
    main()
