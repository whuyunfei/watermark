"""Detection rate, false-positive rate, rotation rate, and saved images.

  python eval_metrics.py
  python eval_metrics.py --n-pos 8 --n-neg 8 --steps 15 --whiten
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image

from anyangle.codec import THRESHOLD, detect_image, embed_torch, load_key_pem, pseudorandom_score
from anyangle.sd import PROMPT, generate, load_pipeline, project_root, random_latent


def rotate(img: Image.Image, deg: float) -> Image.Image:
    if abs(deg) < 1e-9:
        return img.copy()
    turned = img.rotate(deg, expand=True, resample=Image.BICUBIC, fillcolor=(128, 128, 128))
    return turned.resize(img.size, Image.LANCZOS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pos", type=int, default=4)
    ap.add_argument("--n-neg", type=int, default=4)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--seed-base", type=int, default=100)
    ap.add_argument("--whiten", action="store_true")
    args = ap.parse_args()

    key = load_key_pem()
    pr = pseudorandom_score(key)
    pipe = load_pipeline()
    rng = np.random.default_rng(20261012)
    pos_seeds = [args.seed_base + 11 * i for i in range(args.n_pos)]
    out = os.path.join(project_root(), "output", "eval")
    os.makedirs(out, exist_ok=True)

    clean_hit = rot_hit = rot_n = 0
    for i, seed in enumerate(pos_seeds):
        marked = embed_torch(random_latent(pipe, seed), key)
        wm = generate(pipe, marked, seed=seed, steps=args.steps)
        peak = detect_image(pipe, wm, key, whiten=args.whiten)
        clean_hit += int(peak >= THRESHOLD)
        angles = [float(rng.uniform(0, 360)), float(rng.uniform(0, 360))]
        for deg in angles:
            rot_n += 1
            rot_hit += int(detect_image(pipe, rotate(wm, deg), key, whiten=args.whiten) >= THRESHOLD)
        if i == 0:
            wm.save(os.path.join(out, f"seed{seed}_field1.png"))
        print(f"pos seed={seed} peak={peak:.3f}", flush=True)

    fp = 0
    for i in range(args.n_neg):
        seed = 910 + i
        img = generate(pipe, random_latent(pipe, seed), seed=seed, steps=args.steps)
        peak = detect_image(pipe, img, key, whiten=args.whiten)
        fp += int(peak >= THRESHOLD)
        if i == 0:
            img.save(os.path.join(out, f"seed{seed}_neg.png"))
        print(f"neg seed={seed} peak={peak:.3f}", flush=True)

    summary = {
        "field": 1.0,
        "whiten": bool(args.whiten),
        "threshold": THRESHOLD,
        "pr": round(pr, 4),
        "clean_detect": f"{clean_hit}/{len(pos_seeds)}",
        "rotation_detect": f"{rot_hit}/{rot_n}",
        "false_positive": f"{fp}/{args.n_neg}",
        "images": out,
    }
    path = os.path.join(out, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
