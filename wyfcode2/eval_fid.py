"""FID of FIELD=0.60 images against real photographs.

Reference set: Oxford-IIIT Pet test split, downloaded by torchvision on first use.
Also reports FID between clean SD images and watermarked images.

  python eval_fid.py
  python eval_fid.py --n-seeds 12 --max-ref 200 --steps 15
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import scipy.linalg
import torch
import torchvision

from anyangle.codec import embed_torch, load_key_pem
from anyangle.sd import PROMPT, generate, load_pipeline, project_root, random_latent

# scipy>=1.14 dropped the disp argument that clean-fid still passes.
_orig_sqrtm = scipy.linalg.sqrtm


def _sqrtm_compat(a, disp=False):
    del disp
    out = _orig_sqrtm(a)
    return out if isinstance(out, tuple) else (out, None)


scipy.linalg.sqrtm = _sqrtm_compat

from cleanfid import fid  # noqa: E402


def export_pets(folder: str, max_images: int) -> int:
    os.makedirs(folder, exist_ok=True)
    existing = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))]
    if len(existing) >= max_images:
        return len(existing)
    root = os.path.join(project_root(), "data")
    ds = torchvision.datasets.OxfordIIITPet(root=root, split="test", download=True)
    n = min(int(max_images), len(ds))
    for i in range(n):
        path = os.path.join(folder, f"pet_{i:05d}.jpg")
        if os.path.isfile(path):
            continue
        img, _ = ds[i]
        img.convert("RGB").save(path, quality=95)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--max-ref", type=int, default=200)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--seed-base", type=int, default=100)
    args = ap.parse_args()

    key = load_key_pem()
    pipe = load_pipeline()
    out = os.path.join(project_root(), "output", "fid")
    ref = os.path.join(out, "ref_oxford_pet")
    clean_dir = os.path.join(out, "gen_clean")
    wm_dir = os.path.join(out, "gen_field1")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(wm_dir, exist_ok=True)

    n_ref = export_pets(ref, args.max_ref)
    print(f"reference images: {n_ref}", flush=True)
    for i in range(args.n_seeds):
        seed = args.seed_base + 11 * i
        clean_path = os.path.join(clean_dir, f"seed{seed}.png")
        wm_path = os.path.join(wm_dir, f"seed{seed}.png")
        if not os.path.isfile(clean_path):
            generate(pipe, random_latent(pipe, seed), seed=seed, steps=args.steps).save(clean_path)
        if not os.path.isfile(wm_path):
            marked = embed_torch(random_latent(pipe, seed), key)
            generate(pipe, marked, seed=seed, steps=args.steps).save(wm_path)
        print(f"seed={seed}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fid_clean = float(fid.compute_fid(ref, clean_dir, mode="clean", num_workers=0, device=dev))
    fid_wm = float(fid.compute_fid(ref, wm_dir, mode="clean", num_workers=0, device=dev))
    fid_pair = float(fid.compute_fid(clean_dir, wm_dir, mode="clean", num_workers=0, device=dev))
    summary = {
        "metric": "FID (lower is closer)",
        "field": 1.0,
        "reference": "Oxford-IIIT Pet test",
        "n_ref": n_ref,
        "n_generated": args.n_seeds,
        "fid_clean_to_real": round(fid_clean, 4),
        "fid_field1_to_real": round(fid_wm, 4),
        "fid_clean_to_field1": round(fid_pair, 4),
        "note": "Pet photos and portrait prompts are different domains. Use fid_clean_to_field1 for the watermark gap.",
    }
    path = os.path.join(out, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
