"""Generate one clean image and one FIELD=1.0 watermarked image.

First run downloads Stable Diffusion weights into ./model_cache.

  python demo.py
  python demo.py --seed 28 --steps 15 --whiten
"""

from __future__ import annotations

import argparse
import os

from anyangle.codec import FIELD, THRESHOLD, detect_image, embed_torch, load_key_pem, pseudorandom_score
from anyangle.sd import PROMPT, generate, load_pipeline, project_root, random_latent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=28)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--whiten", action="store_true", help="whiten the detection spectrum; pixels are unchanged")
    args = ap.parse_args()

    key = load_key_pem()
    print(f"FIELD={FIELD}  PR={pseudorandom_score(key):.4f}", flush=True)
    print("Loading Stable Diffusion (downloads on first use)...", flush=True)
    pipe = load_pipeline()
    base = random_latent(pipe, args.seed)
    marked = embed_torch(base, key)
    clean = generate(pipe, base, seed=args.seed, steps=args.steps, prompt=PROMPT)
    wm = generate(pipe, marked, seed=args.seed, steps=args.steps, prompt=PROMPT)

    out = os.path.join(project_root(), "output")
    os.makedirs(out, exist_ok=True)
    clean_path = os.path.join(out, f"seed{args.seed}_clean.png")
    wm_path = os.path.join(out, f"seed{args.seed}_field1.png")
    clean.save(clean_path)
    wm.save(wm_path)
    peak_clean = detect_image(pipe, clean, key, whiten=args.whiten)
    peak_wm = detect_image(pipe, wm, key, whiten=args.whiten)
    print(f"clean peak={peak_clean:.3f}  watermark peak={peak_wm:.3f}  threshold={THRESHOLD}", flush=True)
    print(f"saved\n  {clean_path}\n  {wm_path}", flush=True)


if __name__ == "__main__":
    main()
