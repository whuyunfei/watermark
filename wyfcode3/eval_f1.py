#!/usr/bin/env python3
"""F1 and rotation robustness for RingID v4.

Protocol (SD full pipeline, same prompt and step count as generate.py):

- 20 watermarked images (positive) and 20 clean SD images (negative).
- Threshold is frozen before those images are scored. It is the gaussian-latent
  threshold from ``calibrate_threshold(pipe=None)``, which is what ``detect.py``
  uses when ``output/threshold.json`` does not exist yet. The value is then
  written to ``output/threshold.json``.
- A sample is positive when ``peak >= threshold`` after the 10-degree search
  implemented by ``verify_image``.
- F1 uses only the unrotated positive and negative sets.
- Rotation robustness is a separate pass rate on the same 20 watermarked images
  after ``rotate`` (expand, bicubic, resize back to 512) at one uniform angle
  in [0.5, 359.5] degrees per seed.

Resumable: finished samples are appended to ``output/eval_f1/records.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from PIL import Image

from ringid.attacks import rotate
from ringid.codec import calibrate_threshold, verify_image
from ringid.key import load_key
from ringid.pipeline import generate_clean, generate_watermarked, load_pipeline

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
OUT_DIR = os.path.join(ROOT, "output", "eval_f1")
RECORD_PATH = os.path.join(OUT_DIR, "records.jsonl")
THRESHOLD_PATH = os.path.join(ROOT, "output", "threshold.json")
SUMMARY_PATH = os.path.join(REPO, "results", "summary_wyfcode3.json")

PROMPT = "a photo of a beautiful woman"
STEPS = 15
POS_BASE = 51000
NEG_BASE = 52000


def _seeds(base: int, n: int) -> list[int]:
    return [base + i for i in range(n)]


def _angle(seed: int) -> float:
    rng = np.random.default_rng(seed + 17_000_003)
    return round(float(rng.uniform(0.5, 359.5)), 2)


def _load_records() -> list[dict]:
    if not os.path.isfile(RECORD_PATH):
        return []
    rows = []
    with open(RECORD_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _append(row: dict) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(RECORD_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _metrics(pos: list[float], neg: list[float], thr: float) -> dict:
    tp = sum(s >= thr for s in pos)
    fn = len(pos) - tp
    fp = sum(s >= thr for s in neg)
    tn = len(neg) - fp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "tpr": round(recall, 6),
        "fpr": round(fp / len(neg), 6) if neg else 0.0,
    }


def _ang_err(estimated: float, true: float) -> float:
    delta = (estimated - true + 180.0) % 360.0 - 180.0
    return abs(delta)


def _write_summary(rows: list[dict], threshold: float, n: int, elapsed: float | None) -> None:
    pos = [r for r in rows if r["split"] == "pos"]
    neg = [r for r in rows if r["split"] == "neg"]
    rot = [r for r in rows if r["split"] == "rot"]
    pos_scores = [float(r["peak"]) for r in pos]
    neg_scores = [float(r["peak"]) for r in neg]
    rot_scores = [float(r["peak"]) for r in rot]
    metrics = _metrics(pos_scores, neg_scores, threshold)
    rot_pass = sum(float(r["peak"]) >= threshold for r in rot)
    rot_err = [_ang_err(float(r["estimated_deg"]), float(r["angle"])) for r in rot]
    summary = {
        "scheme": "RingID latent v4 (wyfcode3): FFT ring on denoised latent channel 3, VAE detect, 10-degree search",
        "prompt": PROMPT,
        "steps": STEPS,
        "n_pos": n,
        "n_neg": n,
        "n_rot": n,
        "threshold": threshold,
        "threshold_source": "calibrate_threshold(pipe=None) gaussian latents, n_neg=16, quantile 0.995 + 0.08; frozen before scoring the 40 images",
        "attack": "rotate(expand=True, bicubic, fill 128) then resize to 512; one Unif[0.5, 359.5] degree per positive seed",
        "pos": pos,
        "neg": neg,
        "rot": rot,
        "clean_pass": f"{metrics['tp']}/{n}",
        "neg_correct": f"{metrics['tn']}/{n}",
        "rot_pass": f"{rot_pass}/{n}",
        "rot_pass_rate": round(rot_pass / n, 6) if n else 0.0,
        "rot_angle_err_median_deg": round(float(np.median(rot_err)), 4) if rot_err else None,
        "rot_angle_err_mean_deg": round(float(np.mean(rot_err)), 4) if rot_err else None,
        "pos_peak_min": round(min(pos_scores), 4) if pos_scores else None,
        "pos_peak_mean": round(float(np.mean(pos_scores)), 4) if pos_scores else None,
        "neg_peak_max": round(max(neg_scores), 4) if neg_scores else None,
        "neg_peak_mean": round(float(np.mean(neg_scores)), 4) if neg_scores else None,
        "rot_peak_min": round(min(rot_scores), 4) if rot_scores else None,
        "rot_peak_mean": round(float(np.mean(rot_scores)), 4) if rot_scores else None,
        **metrics,
        "elapsed_sec": None if elapsed is None else round(elapsed, 1),
    }
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(
        f"F1={metrics['f1']:.4f} precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} tp={metrics['tp']} fp={metrics['fp']} "
        f"tn={metrics['tn']} fn={metrics['fn']}",
        flush=True,
    )
    print(
        f"rotation pass {rot_pass}/{n} ({summary['rot_pass_rate']:.4f}) "
        f"median angle error {summary['rot_angle_err_median_deg']}",
        flush=True,
    )
    print(f"summary: {SUMMARY_PATH}", flush=True)


def _ensure_threshold(key: str) -> float:
    if os.path.isfile(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, encoding="utf-8") as f:
            thr = float(json.load(f)["threshold"])
        print(f"threshold={thr:.4f} (existing {THRESHOLD_PATH})", flush=True)
        return thr
    print("Calibrating gaussian-latent threshold ...", flush=True)
    thr = float(calibrate_threshold(key, pipe=None, n_neg=16))
    os.makedirs(os.path.dirname(THRESHOLD_PATH), exist_ok=True)
    with open(THRESHOLD_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold": thr,
                "n_neg": 16,
                "source": "gaussian_latent",
                "rule": "quantile 0.995 of rotation-searched gaussian peaks + 0.08",
            },
            f,
            indent=2,
        )
        f.write("\n")
    print(f"threshold={thr:.4f} -> {THRESHOLD_PATH}", flush=True)
    return thr


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--key", default=None)
    args = p.parse_args()
    n = int(args.n)
    if n < 1:
        raise SystemExit("--n must be >= 1")

    os.makedirs(OUT_DIR, exist_ok=True)
    key = load_key(args.key)
    threshold = _ensure_threshold(key)
    done = {(r["split"], int(r["seed"])) for r in _load_records()}
    pos_seeds = _seeds(POS_BASE, n)
    neg_seeds = _seeds(NEG_BASE, n)
    need = 3 * n - len(done & ({("pos", s) for s in pos_seeds} | {("neg", s) for s in neg_seeds} | {("rot", s) for s in pos_seeds}))
    print(f"samples remaining: {need} (pos/neg/rot, n={n})", flush=True)

    t_run = time.time()
    pipe = None
    if need:
        pipe = load_pipeline()
        pipe.set_progress_bar_config(disable=True)

    def _detect(split: str, seed: int, image: Image.Image, extra: dict | None = None) -> None:
        t0 = time.time()
        result = verify_image(key, image, pipe, threshold=threshold)
        row = {
            "split": split,
            "seed": int(seed),
            "peak": round(float(result.peak), 4),
            "zero_deg_peak": round(float(result.zero_deg_peak), 4),
            "estimated_deg": round(float(result.estimated_deg), 1),
            "verified": bool(result.verified),
            "threshold": threshold,
            "detect_sec": round(time.time() - t0, 2),
        }
        if extra:
            row.update(extra)
        _append(row)
        print(
            f"{split} seed={seed} verified={row['verified']} peak={row['peak']:.4f} "
            f"zero={row['zero_deg_peak']:.4f} est={row['estimated_deg']:.1f} "
            f"({row['detect_sec']:.1f}s)",
            flush=True,
        )

    for seed in pos_seeds:
        path = os.path.join(OUT_DIR, f"pos_seed{seed}.png")
        if ("pos", seed) in done and os.path.isfile(path):
            continue
        if not os.path.isfile(path):
            t0 = time.time()
            print(f"generate pos seed={seed}", flush=True)
            img = generate_watermarked(pipe, key, PROMPT, seed=seed, steps=STEPS)
            img.save(path)
            print(f"saved {path} ({time.time()-t0:.1f}s)", flush=True)
        else:
            img = Image.open(path).convert("RGB")
        if ("pos", seed) not in done:
            _detect("pos", seed, img)
        del img

    for seed in neg_seeds:
        path = os.path.join(OUT_DIR, f"neg_seed{seed}.png")
        if ("neg", seed) in done and os.path.isfile(path):
            continue
        if not os.path.isfile(path):
            t0 = time.time()
            print(f"generate neg seed={seed}", flush=True)
            img = generate_clean(pipe, PROMPT, seed=seed, steps=STEPS)
            img.save(path)
            print(f"saved {path} ({time.time()-t0:.1f}s)", flush=True)
        else:
            img = Image.open(path).convert("RGB")
        if ("neg", seed) not in done:
            _detect("neg", seed, img)
        del img

    for seed in pos_seeds:
        angle = _angle(seed)
        src = os.path.join(OUT_DIR, f"pos_seed{seed}.png")
        path = os.path.join(OUT_DIR, f"rot_seed{seed}.png")
        if ("rot", seed) in done and os.path.isfile(path):
            continue
        base = Image.open(src).convert("RGB")
        spun = rotate(base, angle)
        spun.save(path)
        if ("rot", seed) not in done:
            _detect("rot", seed, spun, {"angle": angle})
        del base, spun

    rows = _load_records()
    rows = [r for r in rows if int(r["seed"]) in set(pos_seeds) | set(neg_seeds)]
    # Keep the latest row per (split, seed).
    latest: dict[tuple[str, int], dict] = {}
    for r in rows:
        latest[(r["split"], int(r["seed"]))] = r
    rows = list(latest.values())
    have = (
        sum(r["split"] == "pos" for r in rows) == n
        and sum(r["split"] == "neg" for r in rows) == n
        and sum(r["split"] == "rot" for r in rows) == n
    )
    if not have:
        raise SystemExit(f"incomplete records: {len(rows)} rows, expected {3*n}")
    _write_summary(rows, threshold, n, time.time() - t_run)


if __name__ == "__main__":
    main()
