"""Crop test for the rotation-invariant initial-latent watermark.

A center square is cut from the generated image, keeping ``keep`` of each side.
In ``resize`` mode the existing detector then scales that square back to 512,
which is what happens when a cropped file is verified. Removed area is
``1 - keep**2``. Detection is one inversion and the same Chernoff threshold as
the rotation test; there is no crop-offset search.
"""
import argparse
import json
import os
import pickle
import random
import time

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image

from inversion import exact_inversion, generate, stable_diffusion_pipe
from src.invariant_watermark import InvariantWatermark, chernoff_threshold

parser = argparse.ArgumentParser('Args')
parser.add_argument('--test_num', type=int, default=20)
parser.add_argument('--ids', type=int, nargs='*', default=None)
parser.add_argument('--keep', type=float, nargs='+', required=True,
                    help='Fraction of each side retained by the center crop. 1 is uncropped.')
parser.add_argument('--mode', choices=('resize', 'pad'), default='resize')
parser.add_argument('--ring_width', type=float, default=2.0)
parser.add_argument('--arc', type=float, default=2.0)
parser.add_argument('--max_harmonic', type=int, default=16)
parser.add_argument('--guidance', type=float, default=3.0)
parser.add_argument('--fpr', type=float, default=0.00001)
parser.add_argument('--model_id', type=str, default='sd2-community/stable-diffusion-2-1-base')
parser.add_argument('--dataset_id', type=str, default='Gustavosta/Stable-Diffusion-Prompts')
parser.add_argument('--out', type=str, default='results/invariant_crop_20')
args = parser.parse_args()
print(args, flush=True)

os.makedirs(f'{args.out}/images', exist_ok=True)
os.makedirs('keys', exist_ok=True)
key_path = 'keys/invariant_key.pkl'
if not os.path.exists(key_path):
    with open(key_path, 'wb') as f:
        pickle.dump(InvariantWatermark.new().key, f)
with open(key_path, 'rb') as f:
    key = pickle.load(f)
layout = dict(ring_width=args.ring_width, arc=args.arc, max_harmonic=args.max_harmonic)
wm = InvariantWatermark(key, **layout)
threshold = chernoff_threshold(len(wm.harmonics), args.fpr)
print(f'{wm.num_coefficients} template coefficients over {len(wm.harmonics)} harmonics, '
      f'threshold {threshold:.2f}', flush=True)

hf_cache_dir = os.path.abspath(os.environ.get('HF_MODEL_CACHE', 'hf_models'))
pipe = stable_diffusion_pipe(solver_order=1, model_id=args.model_id, cache_dir=hf_cache_dir)
pipe.set_progress_bar_config(disable=True)

random.seed(42)
prompts = random.sample([sample['Prompt'] for sample in load_dataset(args.dataset_id)['test']], args.test_num)
ids = list(range(args.test_num) if args.ids is None else args.ids)


def release_cached_memory():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def render(prompt, latents):
    image, _, _ = generate(prompt=prompt, init_latents=latents.float().to(pipe.device),
                           guidance_scale=args.guidance, num_inference_steps=50, solver_order=1, pipe=pipe)
    release_cached_memory()
    return image


def invert(image):
    z = exact_inversion(image, prompt='', test_num_inference_steps=50, inv_order=0, pipe=pipe,
                        decoder_inv=False).detach().float().cpu()
    release_cached_memory()
    return z


def watermarked(i):
    path = f'{args.out}/images/{i}_wm.png'
    if not os.path.exists(path):
        np.random.seed(i)
        g = torch.from_numpy(np.random.randn(1, 4, 64, 64))
        render(prompts[i], wm.embed(g)).save(path)
    return Image.open(path).convert('RGB')


def attack(image, keep):
    w, h = image.size
    side = max(1, int(round(min(w, h) * keep)))
    left = (w - side) // 2
    top = (h - side) // 2
    cropped = image.crop((left, top, left + side, top + side))
    if args.mode == 'resize':
        return cropped
    canvas = Image.new('RGB', image.size, (0, 0, 0))
    canvas.paste(cropped, (left, top))
    return canvas


rows_path = f'{args.out}/detections.jsonl'
done = {}
if os.path.exists(rows_path):
    with open(rows_path) as f:
        for line in f:
            row = json.loads(line)
            done[(row['image'], row['mode'], row['keep'])] = row

rows = []
for keep in args.keep:
    for i in ids:
        row = done.get((i, args.mode, keep))
        if row is None:
            t0 = time.time()
            z = invert(attack(watermarked(i), keep))
            removed_area = 1.0 - keep ** 2
            row = {'image': i, 'mode': args.mode, 'keep': keep, 'removed_area': removed_area,
                   **wm.detect(z, args.fpr), 'seconds': time.time() - t0}
            with open(rows_path, 'a') as f:
                f.write(json.dumps(row) + '\n')
            done[(i, args.mode, keep)] = row
        rows.append(row)
        print(f"[detect] img {i:02d} {args.mode} keep {keep:.3f} "
              f"removed_area {1 - keep ** 2:6.1%}: detected={row['detected']!s:5s} "
              f"D={row['score']:7.1f}/{row['threshold']:.1f} ({row['seconds']:.0f}s)", flush=True)

by_keep = {}
for row in rows:
    by_keep.setdefault(row['keep'], []).append(row)
summary_levels = []
for keep, group in sorted(by_keep.items(), reverse=True):
    summary_levels.append({
        'keep_side': keep,
        'remaining_area': keep ** 2,
        'removed_area': 1 - keep ** 2,
        'detected': f"{sum(r['detected'] for r in group)}/{len(group)}",
        'min_score': min(r['score'] for r in group),
        'mean_score': sum(r['score'] for r in group) / len(group),
        'threshold': group[0]['threshold'],
    })
    print(f"[level] keep {keep:.3f} removed {1 - keep ** 2:.1%}: "
          f"{summary_levels[-1]['detected']} min D={summary_levels[-1]['min_score']:.1f}", flush=True)

result = {'args': vars(args), 'num_coefficients': wm.num_coefficients, 'levels': summary_levels}
with open(f'{args.out}/summary.json', 'w') as f:
    json.dump(result, f, indent=2)
print(json.dumps(summary_levels, indent=2), flush=True)
