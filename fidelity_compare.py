"""Per-image similarity between watermarked and same-seed unwatermarked generations."""
import glob
import json
import os

import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

lpips_net = lpips.LPIPS(net='alex', verbose=False)


def to_tensor(image):
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)[None]


def compare(pairs):
    scores = []
    for nowm_path, wm_path in pairs:
        a, b = Image.open(nowm_path).convert('RGB'), Image.open(wm_path).convert('RGB')
        x, y = np.asarray(a), np.asarray(b)
        with torch.no_grad():
            perceptual = float(lpips_net(to_tensor(a), to_tensor(b)))
        scores.append({
            'nowm': nowm_path, 'wm': wm_path,
            'psnr': float(peak_signal_noise_ratio(x, y, data_range=255)),
            'ssim': float(structural_similarity(x, y, channel_axis=2, data_range=255)),
            'lpips': perceptual,
        })
    return scores


def pairs_in(folder, wm_suffix):
    found = []
    for wm_path in sorted(glob.glob(f'{folder}/*_{wm_suffix}.png')):
        nowm_path = wm_path.replace(f'_{wm_suffix}.png', '_nowm.png')
        if os.path.exists(nowm_path):
            found.append((nowm_path, wm_path))
    return found


schemes = {
    'PRC (original paper)': [('results/polar_rotation/images/0_nowm.png', 'results/polar_rotation/images/0_prc.png')],
    'polar sectors in the initial noise': pairs_in('results/polar_rotation/images', 'wm'),
    'polar sectors at late step (new)': pairs_in('results/late_rotation/images', 'wm'),
}
report = {}
for name, pairs in schemes.items():
    scores = compare(pairs)
    report[name] = scores
    stat = {k: (np.mean([s[k] for s in scores]), np.min([s[k] for s in scores]), np.max([s[k] for s in scores]))
            for k in ('psnr', 'ssim', 'lpips')}
    print(f"{name:36s} n={len(scores)} | " + " | ".join(
        f"{k.upper()} mean {m:.3f} (min {lo:.3f}, max {hi:.3f})" for k, (m, lo, hi) in stat.items()), flush=True)

with open('results/fidelity_compare.json', 'w') as f:
    json.dump(report, f, indent=2)
