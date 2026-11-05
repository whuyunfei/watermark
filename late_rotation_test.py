"""Rotation and fidelity test for the late-step polar watermark."""
import argparse
import json
import os
import pickle
import random
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF
from datasets import load_dataset
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision.transforms import InterpolationMode

from inversion import stable_diffusion_pipe
from src.late_watermark import LateStepWatermark
from src.polar_watermark import PolarWatermark

parser = argparse.ArgumentParser('Args')
parser.add_argument('--test_num', type=int, default=8)
parser.add_argument('--step', type=int, default=45)
parser.add_argument('--strength', type=float, default=1.0)
parser.add_argument('--guidance', type=float, default=3.0)
parser.add_argument('--wm_angles', type=float, nargs='+', default=[0, 15, 22, 45, 75, -67, 90, -158, 180])
parser.add_argument('--bilinear_angles', type=float, nargs='*', default=[45])
parser.add_argument('--nowm_angles', type=float, nargs='+', default=[0, 22, 90, 180])
parser.add_argument('--fpr', type=float, default=0.00001)
parser.add_argument('--model_id', type=str, default='sd2-community/stable-diffusion-2-1-base')
parser.add_argument('--dataset_id', type=str, default='Gustavosta/Stable-Diffusion-Prompts')
parser.add_argument('--out', type=str, default='results/late_rotation')
args = parser.parse_args()
print(args)

os.makedirs(f'{args.out}/images', exist_ok=True)
os.makedirs('keys', exist_ok=True)
key_path = 'keys/late_key.pkl'
if not os.path.exists(key_path):
    with open(key_path, 'wb') as f:
        pickle.dump(PolarWatermark.new().key, f)
with open(key_path, 'rb') as f:
    key = pickle.load(f)

hf_cache_dir = os.path.abspath(os.environ.get('HF_MODEL_CACHE', 'hf_models'))
pipe = stable_diffusion_pipe(solver_order=1, model_id=args.model_id, cache_dir=hf_cache_dir)
pipe.set_progress_bar_config(disable=True)
pipe.text_encoder.to('cpu')
wm = LateStepWatermark(pipe, key, step=args.step, strength=args.strength)
wrong_key = PolarWatermark.new()

random.seed(42)
prompts = random.sample([sample['Prompt'] for sample in load_dataset(args.dataset_id)['test']], args.test_num)

config = {'model_id': args.model_id, 'step': args.step, 'strength': args.strength, 'guidance': args.guidance}
meta_path = f'{args.out}/meta.json'
meta = {'config': config, 'images': {}}
if os.path.exists(meta_path):
    with open(meta_path) as f:
        saved = json.load(f)
    if saved['config'] == config:
        meta = saved


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def release_cached_memory():
    # The MPS allocator keeps freed buffers wired; on 16 GB machines that pushes the system into swap.
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def attack_plan(angles, bilinear_angles):
    return [(a, InterpolationMode.NEAREST) for a in angles] + [(a, InterpolationMode.BILINEAR) for a in bilinear_angles]


def generated_pair(i):
    paths = (f'{args.out}/images/{i}_nowm.png', f'{args.out}/images/{i}_wm.png')
    entry = meta['images'].get(str(i))
    if entry is None or entry['prompt'] != prompts[i] or not all(map(os.path.exists, paths)):
        seed_everything(i)
        g = torch.from_numpy(np.random.randn(1, 4, 64, 64))
        nonce = wm.polar.random_nonce()
        wm.generate(prompts[i], g, nonce=None, guidance_scale=args.guidance).save(paths[0])
        wm.generate(prompts[i], g, nonce=nonce, guidance_scale=args.guidance).save(paths[1])
        release_cached_memory()
        entry = meta['images'][str(i)] = {'prompt': prompts[i], 'nonce': nonce}
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
    return Image.open(paths[0]).convert('RGB'), Image.open(paths[1]).convert('RGB'), entry['nonce']


rows_path = f'{args.out}/detections.jsonl'
finished = {}
if os.path.exists(rows_path):
    with open(rows_path) as f:
        for line in f:
            row = json.loads(line)
            finished[(row['image'], row['generation_nonce'], row['label'], row['attack_angle'], row['interpolation'])] = row

quality, rows = [], []
for i in range(args.test_num):
    t0 = time.time()
    nowm, marked, nonce = generated_pair(i)
    a, b = np.asarray(nowm), np.asarray(marked)
    quality.append({
        'image': i,
        'prompt': prompts[i],
        'psnr': float(peak_signal_noise_ratio(a, b, data_range=255)),
        'ssim': float(structural_similarity(a, b, channel_axis=2, data_range=255)),
    })
    print(f"[quality] img {i}: PSNR={quality[-1]['psnr']:.2f} dB SSIM={quality[-1]['ssim']:.4f} "
          f"({time.time() - t0:.0f}s) prompt={prompts[i][:60]!r}", flush=True)

    plans = (('wm', marked, attack_plan(args.wm_angles, args.bilinear_angles)),
             ('nowm', nowm, attack_plan(args.nowm_angles, [])))
    for label, image, plan in plans:
        for angle, mode in plan:
            row = finished.get((i, nonce, label, angle, mode.value))
            if row is None:
                t0 = time.time()
                attacked = TF.rotate(image, angle, interpolation=mode)
                cache = {}
                result = wm.detect(attacked, fpr=args.fpr, cache=cache)
                release_cached_memory()
                seconds = time.time() - t0
                row = {
                    'image': i, 'generation_nonce': nonce, 'label': label, 'attack_angle': angle,
                    'interpolation': mode.value, **result,
                    'nonce_embedded': nonce if label == 'wm' else None,
                    'nonce_correct': label == 'wm' and result['nonce'] == nonce,
                    'wrong_key_coarse_score': wm.detect(attacked, args.fpr, wrong_key, cache, refine=False)['score'],
                    'seconds': seconds,
                }
                with open(rows_path, 'a') as f:
                    f.write(json.dumps(row) + '\n')
            rows.append(row)
            print(f"[detect] img {i} {label:4s} rot {angle:>5g} {mode.value:8s}: detected={row['detected']!s:5s} "
                  f"T={row['score']:5.2f}/{row['threshold']:.2f} (coarse {row['coarse_score']:5.2f}) "
                  f"angle_est={row['angle']:+6.1f} nonce_ok={row['nonce_correct']} "
                  f"wrong_key_T={row['wrong_key_coarse_score']:.2f} ({row['seconds']:.0f}s, "
                  f"{row['inversions']} inversions)", flush=True)

with open(f'{args.out}/summary.json', 'w') as f:
    json.dump({'args': vars(args), 'num_bits': wm.polar.num_bits, 'quality': quality, 'detection': rows}, f, indent=2)

thumb = 256
grid = Image.new('RGB', (thumb * 4, thumb * args.test_num), 'white')
for i in range(args.test_num):
    nowm = Image.open(f'{args.out}/images/{i}_nowm.png')
    marked = Image.open(f'{args.out}/images/{i}_wm.png')
    diff = np.clip(np.abs(np.asarray(nowm, dtype=np.int16) - np.asarray(marked, dtype=np.int16)) * 8, 0, 255)
    tiles = (nowm, marked, Image.fromarray(diff.astype(np.uint8)), TF.rotate(marked, 45))
    for c, tile in enumerate(tiles):
        grid.paste(tile.resize((thumb, thumb)), (c * thumb, i * thumb))
grid.save(f'{args.out}/compare_grid.png')
print(f'Saved {args.out}/summary.json and {args.out}/compare_grid.png')
