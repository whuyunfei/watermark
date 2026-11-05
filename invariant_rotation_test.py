"""Rotation test for the rotation-invariant initial-latent watermark: one inversion, no angle search."""
import argparse
import json
import math
import os
import pickle
import random
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import InterpolationMode

from inversion import exact_inversion, generate, stable_diffusion_pipe
from src.invariant_watermark import InvariantWatermark, chernoff_threshold

parser = argparse.ArgumentParser('Args')
parser.add_argument('--test_num', type=int, default=8)
parser.add_argument('--ring_width', type=float, default=2.0)
parser.add_argument('--arc', type=float, default=2.0)
parser.add_argument('--max_harmonic', type=int, default=16)
parser.add_argument('--guidance', type=float, default=3.0)
parser.add_argument('--fixed_angles', type=float, nargs='*', default=[0])
parser.add_argument('--random_angles', type=int, default=5)
parser.add_argument('--bilinear_random', type=int, default=1)
parser.add_argument('--nowm_random', type=int, default=2)
parser.add_argument('--fpr', type=float, default=0.00001)
parser.add_argument('--model_id', type=str, default='sd2-community/stable-diffusion-2-1-base')
parser.add_argument('--dataset_id', type=str, default='Gustavosta/Stable-Diffusion-Prompts')
parser.add_argument('--out', type=str, default='results/invariant_rotation')
args = parser.parse_args()
print(args)

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
wrong_key = InvariantWatermark.new(**layout)
print(f'{wm.num_coefficients} template coefficients over {len(wm.harmonics)} harmonics, '
      f'threshold {chernoff_threshold(len(wm.harmonics), args.fpr):.2f}')

hf_cache_dir = os.path.abspath(os.environ.get('HF_MODEL_CACHE', 'hf_models'))
pipe = stable_diffusion_pipe(solver_order=1, model_id=args.model_id, cache_dir=hf_cache_dir)
pipe.set_progress_bar_config(disable=True)

random.seed(42)
prompts = random.sample([sample['Prompt'] for sample in load_dataset(args.dataset_id)['test']], args.test_num)


def release_cached_memory():
    # The MPS allocator keeps freed buffers wired; on 16 GB machines that pushes the system into swap.
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


def pair(i):
    paths = (f'{args.out}/images/{i}_nowm.png', f'{args.out}/images/{i}_wm.png')
    if not all(map(os.path.exists, paths)):
        np.random.seed(i)
        g = torch.from_numpy(np.random.randn(1, 4, 64, 64))
        render(prompts[i], g).save(paths[0])
        render(prompts[i], wm.embed(g)).save(paths[1])
    return [Image.open(p).convert('RGB') for p in paths]


rows_path = f'{args.out}/detections.jsonl'
done = {}
if os.path.exists(rows_path):
    with open(rows_path) as f:
        for line in f:
            row = json.loads(line)
            done[(row['image'], row['label'], row['attack_angle'], row['interpolation'])] = row

rows = []
for i in range(args.test_num):
    nowm, marked = pair(i)
    rng = np.random.default_rng(1000 + i)
    random_angles = [round(float(a), 1) for a in rng.uniform(-180, 180, args.random_angles + args.bilinear_random + args.nowm_random)]
    plan = [('wm', a, InterpolationMode.NEAREST) for a in args.fixed_angles + random_angles[:args.random_angles]]
    plan += [('wm', a, InterpolationMode.BILINEAR) for a in random_angles[args.random_angles:args.random_angles + args.bilinear_random]]
    plan += [('nowm', a, InterpolationMode.NEAREST) for a in args.fixed_angles + random_angles[args.random_angles + args.bilinear_random:]]
    for label, angle, mode in plan:
        row = done.get((i, label, angle, mode.value))
        if row is None:
            t0 = time.time()
            z = invert(TF.rotate(marked if label == 'wm' else nowm, angle, interpolation=mode))
            row = {'image': i, 'label': label, 'attack_angle': angle, 'interpolation': mode.value,
                   **wm.detect(z, args.fpr), 'wrong_key_score': wrong_key.detect(z, args.fpr)['score'],
                   'seconds': time.time() - t0}
            with open(rows_path, 'a') as f:
                f.write(json.dumps(row) + '\n')
        rows.append(row)
        print(f"[detect] img {i} {label:4s} rot {angle:>6g} {mode.value:8s}: detected={row['detected']!s:5s} "
              f"D={row['score']:7.1f}/{row['threshold']:.1f} wrong_key_D={row['wrong_key_score']:5.1f} "
              f"({row['seconds']:.0f}s, 1 inversion)", flush=True)

wm_rows = [r for r in rows if r['label'] == 'wm']
nowm_rows = [r for r in rows if r['label'] == 'nowm']
result = {
    'args': vars(args), 'num_coefficients': wm.num_coefficients, 'detection': rows,
    'wm_detected': f"{sum(r['detected'] for r in wm_rows)}/{len(wm_rows)}",
    'wm_min_score': min(r['score'] for r in wm_rows) if wm_rows else None,
    'nowm_detected': f"{sum(r['detected'] for r in nowm_rows)}/{len(nowm_rows)}",
    'nowm_max_score': max(r['score'] for r in nowm_rows) if nowm_rows else None,
    'wrong_key_max_score': max(r['wrong_key_score'] for r in rows) if rows else None,
}
with open(f'{args.out}/summary.json', 'w') as f:
    json.dump(result, f, indent=2)
print({k: v for k, v in result.items() if k not in ('args', 'detection')})

thumb = 256
grid = Image.new('RGB', (thumb * 3, thumb * args.test_num), 'white')
for i in range(args.test_num):
    nowm, marked = pair(i)
    angle = next(r['attack_angle'] for r in wm_rows if r['image'] == i)
    for c, tile in enumerate((nowm, marked, TF.rotate(marked, angle))):
        grid.paste(tile.resize((thumb, thumb)), (c * thumb, i * thumb))
grid.save(f'{args.out}/compare_grid.png')
print(f'Saved {args.out}/summary.json and {args.out}/compare_grid.png')
