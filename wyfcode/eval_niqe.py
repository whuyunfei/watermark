"""NIQE of clean SD images versus FIELD=1.0 watermarked images.

Lower is more natural. This does not change the watermark.

  python eval_niqe.py
  python eval_niqe.py --n-seeds 4 --steps 15
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from anyangle.codec import embed_torch, load_key_pem
from anyangle.sd import PROMPT, device, generate, load_pipeline, project_root, random_latent


def niqe_score(image: Image.Image) -> float:
    import pyiqa

    metric = pyiqa.create_metric("niqe", device=str(device()))
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device())
    with torch.inference_mode():
        score = metric(tensor)
    return float(score.item() if score.numel() == 1 else score.mean().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--seed-base", type=int, default=8)
    args = ap.parse_args()

    key = load_key_pem()
    pipe = load_pipeline()
    out = os.path.join(project_root(), "output", "niqe")
    os.makedirs(out, exist_ok=True)
    rows = []
    for i in range(args.n_seeds):
        seed = args.seed_base + 11 * i
        clean = generate(pipe, random_latent(pipe, seed), seed=seed, steps=args.steps, prompt=PROMPT)
        wm = generate(pipe, embed_torch(random_latent(pipe, seed), key), seed=seed, steps=args.steps, prompt=PROMPT)
        clean_s = niqe_score(clean)
        wm_s = niqe_score(wm)
        rows.append({"seed": seed, "clean": round(clean_s, 4), "field1": round(wm_s, 4)})
        if i == 0:
            clean.save(os.path.join(out, f"seed{seed}_clean.png"))
            wm.save(os.path.join(out, f"seed{seed}_field1.png"))
        print(f"seed={seed} clean={clean_s:.3f} field1={wm_s:.3f}", flush=True)

    summary = {
        "metric": "NIQE (lower is more natural)",
        "field": 1.0,
        "clean_mean": round(float(np.mean([r["clean"] for r in rows])), 4),
        "field1_mean": round(float(np.mean([r["field1"] for r in rows])), 4),
        "per_seed": rows,
    }
    path = os.path.join(out, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
