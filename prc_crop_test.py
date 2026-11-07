"""Crop test for the PRC watermark. PRC codewords are LDPC parity checks on the initial latent.

A center square keeps ``keep`` of each side. ``resize`` lets the detector scale that
square back to 512. ``pad`` pastes it back on black so the center stays at the
original scale. Removed area is ``1 - keep**2``. Detection is one inversion and
the PRC posterior test; decoding is reported separately.
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

import src.pseudogaussians as prc_gaussians
from inversion import exact_inversion, generate, stable_diffusion_pipe
from src.prc import Decode, Detect, Encode, KeyGen

parser = argparse.ArgumentParser('Args')
parser.add_argument('--test_num', type=int, default=20)
parser.add_argument('--ids', type=int, nargs='*', default=None)
parser.add_argument('--keep', type=float, nargs='+', required=True)
parser.add_argument('--mode', choices=('resize', 'pad'), default='resize')
parser.add_argument('--guidance', type=float, default=3.0)
parser.add_argument('--inf_steps', type=int, default=50)
parser.add_argument('--fpr', type=float, default=0.00001)
parser.add_argument('--prc_t', type=int, default=3)
parser.add_argument('--var', type=float, default=1.5)
parser.add_argument('--decoder_inv', type=int, default=0)
parser.add_argument('--key', type=str, default='keys/prc_num_1_steps_50_fpr_1e-05_nowm_0.pkl')
parser.add_argument('--model_id', type=str, default='sd2-community/stable-diffusion-2-1-base')
parser.add_argument('--dataset_id', type=str, default='Gustavosta/Stable-Diffusion-Prompts')
parser.add_argument('--out', type=str, default='results/prc_crop_20')
args = parser.parse_args()
print(args, flush=True)

os.makedirs(f'{args.out}/images', exist_ok=True)
os.makedirs('keys', exist_ok=True)
n = 4 * 64 * 64
if not os.path.exists(args.key):
    with open(args.key, 'wb') as f:
        pickle.dump(KeyGen(n, false_positive_rate=args.fpr, t=args.prc_t), f)
with open(args.key, 'rb') as f:
    encoding_key, decoding_key = pickle.load(f)

hf_cache_dir = os.path.abspath(os.environ.get('HF_MODEL_CACHE', 'hf_models'))
pipe = stable_diffusion_pipe(solver_order=1, model_id=args.model_id, cache_dir=hf_cache_dir)
pipe.set_progress_bar_config(disable=True)

random.seed(42)
prompts = random.sample([sample['Prompt'] for sample in load_dataset(args.dataset_id)['test']], args.test_num)
ids = list(range(args.test_num) if args.ids is None else args.ids)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def render(prompt, latents):
    image, _, _ = generate(prompt=prompt, init_latents=latents.float().to(pipe.device),
                           guidance_scale=args.guidance, num_inference_steps=args.inf_steps,
                           solver_order=1, pipe=pipe)
    return image


def invert(image):
    return exact_inversion(image, prompt='', guidance_scale=args.guidance,
                           test_num_inference_steps=args.inf_steps, inv_order=0, pipe=pipe,
                           decoder_inv=bool(args.decoder_inv)).detach().float().cpu()


def watermarked(i):
    path = f'{args.out}/images/{i}_wm.png'
    if not os.path.exists(path):
        seed_everything(i)
        codeword = Encode(encoding_key)
        latents = prc_gaussians.sample(codeword).reshape(1, 4, 64, 64)
        render(prompts[i], latents).save(path)
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


def score(image):
    z = invert(image)
    posteriors = prc_gaussians.recover_posteriors(z.flatten().to(torch.float64), variances=float(args.var))
    detected = bool(Detect(decoding_key, posteriors))
    decoded = Decode(decoding_key, posteriors) is not None
    return detected, decoded


rows_path = f'{args.out}/detections.jsonl'
done = {}
if os.path.exists(rows_path):
    with open(rows_path) as f:
        for line in f:
            row = json.loads(line)
            if row.get('decoder_inv') == bool(args.decoder_inv):
                done[(row['image'], row['mode'], row['keep'])] = row

rows = []
for keep in args.keep:
    for i in ids:
        row = done.get((i, args.mode, keep))
        if row is None:
            t0 = time.time()
            detected, decoded = score(attack(watermarked(i), keep))
            row = {
                'image': i, 'mode': args.mode, 'keep': keep, 'removed_area': 1.0 - keep ** 2,
                'decoder_inv': bool(args.decoder_inv), 'detected': detected, 'decoded': decoded,
                'seconds': time.time() - t0,
            }
            with open(rows_path, 'a') as f:
                f.write(json.dumps(row) + '\n')
            done[(i, args.mode, keep)] = row
        rows.append(row)
        print(f"[detect] img {i:02d} {args.mode} keep {keep:.3f} removed {1 - keep ** 2:6.1%}: "
              f"detect={row['detected']!s:5s} decode={row['decoded']!s:5s} ({row['seconds']:.0f}s)", flush=True)

by_keep = {}
for row in rows:
    by_keep.setdefault(row['keep'], []).append(row)
summary_levels = []
for keep, group in sorted(by_keep.items(), reverse=True):
    level = {
        'keep_side': keep,
        'remaining_area': keep ** 2,
        'removed_area': 1 - keep ** 2,
        'detected': f"{sum(r['detected'] for r in group)}/{len(group)}",
        'decoded': f"{sum(r['decoded'] for r in group)}/{len(group)}",
    }
    summary_levels.append(level)
    print(f"[level] keep {keep:.3f} removed {1 - keep ** 2:.1%}: "
          f"detect {level['detected']} decode {level['decoded']}", flush=True)

with open(f'{args.out}/summary_{args.mode}.json', 'w') as f:
    json.dump({'args': vars(args), 'levels': summary_levels}, f, indent=2)
print(json.dumps(summary_levels, indent=2), flush=True)
