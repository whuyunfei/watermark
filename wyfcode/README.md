# Any-angle watermark, FIELD = 1.0

Stable Diffusion watermark that only changes the initial noise. Detection uses the key, the RGB image, and the VAE. It does not search for the rotation angle.

Embedding strength is fixed at **1.0** on latent channel 3, annulus radius 10–18. The other channels stay Gaussian.

## Setup

```bash
pip install -r requirements.txt
python demo.py --seed 28 --steps 15
```

The first run downloads `Manojb/stable-diffusion-2-1-base` into `model_cache/` when that cache is empty. Later runs load the local files and do not contact the hub. Override the cache or the model with `ANYANGLE_CACHE` and `ANYANGLE_MODEL`.

Outputs:

- `output/seed28_clean.png` — same seed, no watermark
- `output/seed28_field1.png` — FIELD=1.0 watermark

## Metrics

```bash
python eval_metrics.py --n-pos 4 --n-neg 4 --steps 15
python eval_niqe.py --n-seeds 4 --steps 15
python eval_fid.py --n-seeds 8 --max-ref 200 --steps 15
```

| Script | What it reports |
|--------|-----------------|
| `eval_metrics.py` | Clean detection rate, rotation detection rate (expand + resize, two random angles), false-positive rate. Saves images under `output/eval/`. |
| `eval_niqe.py` | NIQE of clean SD vs FIELD=1.0. Lower is more natural. Saves one pair under `output/niqe/`. |
| `eval_fid.py` | FID against Oxford-IIIT Pet photos (downloaded on first use) and FID between clean and watermarked SD images. |

`--whiten` on `eval_metrics.py` whitens the 12-bin detection spectrum only. It does not change the image. On the development set this removed the one false positive and left the rotation rate unchanged.

## What is in the box

| File | Role |
|------|------|
| `anyangle/codec.py` | embed and detect |
| `anyangle/sd.py` | download SD, generate, VAE encode |
| `anyangle/key.pem` | demo key (symmetric; the PEM text is the seed) |
| `demo.py` | one clean / watermark pair |
| `eval_metrics.py` | detection rate, false-positive rate, rotation |
| `eval_niqe.py` | NIQE image quality |
| `eval_fid.py` | FID against real photos |
