"""Rotation test: polar-sector watermark vs. the PRC (LDPC) detector."""
import argparse
import json
import math
import os
import pickle
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from datasets import load_dataset
from PIL import Image
from scipy import stats

import src.pseudogaussians as prc_gaussians
from inversion import exact_inversion, generate, get_device, stable_diffusion_pipe
from src.polar_watermark import PolarWatermark
from src.prc import Decode, Detect, Encode, KeyGen

parser = argparse.ArgumentParser('Args')
parser.add_argument('--test_num', type=int, default=4)
parser.add_argument('--angles', type=float, nargs='+', default=[0, 15, 45, 75])
parser.add_argument('--model_id', type=str, default='sd2-community/stable-diffusion-2-1-base')
parser.add_argument('--dataset_id', type=str, default='Gustavosta/Stable-Diffusion-Prompts')
parser.add_argument('--inf_steps', type=int, default=50)
parser.add_argument('--guidance', type=float, default=3.0)
parser.add_argument('--fpr', type=float, default=0.00001)
parser.add_argument('--decoder_inv', type=int, default=0)
parser.add_argument('--prc_baseline', type=int, default=1)
parser.add_argument('--prc_angles', type=float, nargs='+', default=None)
parser.add_argument('--prc_key', type=str, default='keys/prc_num_1_steps_50_fpr_1e-05_nowm_0.pkl')
parser.add_argument('--out', type=str, default='results/polar_rotation')
args = parser.parse_args()
if args.prc_angles is None:
    args.prc_angles = args.angles
print(args)

hf_cache_dir = os.path.abspath(os.environ.get('HF_MODEL_CACHE', 'hf_models'))
device = get_device()
os.makedirs(f'{args.out}/images', exist_ok=True)
os.makedirs(f'{args.out}/attacked', exist_ok=True)
os.makedirs('keys', exist_ok=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def distribution_check(wm, n=400, seed=123):
    rng = np.random.default_rng(seed)
    plain = torch.from_numpy(rng.standard_normal((n, *wm.shape)))
    variants = {
        'no watermark': plain,
        'polar, fresh nonce per image': torch.cat([wm.embed(plain[k:k + 1], wm.random_nonce()) for k in range(n)]),
        'polar, one fixed nonce': torch.cat([wm.embed(plain[k:k + 1], 0) for k in range(n)]),
    }
    report = {}
    for name, z in variants.items():
        pixels = z.reshape(n, -1).numpy()
        sectors = wm.features(z).numpy()
        chi2 = float(n * (sectors.mean(0) ** 2).sum())
        report[name] = {
            'pixel_mean': float(pixels.mean()),
            'pixel_std': float(pixels.std()),
            'pixel_ks_pvalue': float(stats.kstest(pixels[:, ::13].ravel(), 'norm').pvalue),
            'sector_ks_pvalue': float(stats.kstest(sectors.ravel(), 'norm').pvalue),
            'max_abs_sector_mean_across_images': float(np.abs(sectors.mean(0)).max()),
            'shared_pattern_chi2_pvalue': float(stats.chi2.sf(chi2, wm.num_bits)),
        }
    return report


def prc_statistic(decoding_key, posteriors):
    _, parity_check_matrix, one_time_pad, _, noise_rate, _, _, _, t = decoding_key
    post = (1 - 2 * noise_rate) * (1 - 2 * np.array(one_time_pad, dtype=float)) * posteriors.numpy(force=True)
    r = parity_check_matrix.shape[0]
    pi = np.prod(post[parity_check_matrix.indices.reshape(r, t)], axis=1)
    log_plus, log_minus = np.log((1 + pi) / 2), np.log((1 - pi) / 2)
    log_prod = log_plus + log_minus
    const = 0.5 * np.sum(log_plus ** 2 + log_minus ** 2 - 0.5 * log_prod ** 2)
    return float((log_plus.sum() - 0.5 * log_prod.sum()) / np.sqrt(2 * const))


def posteriors(latents):
    return prc_gaussians.recover_posteriors(latents.reshape(-1).to(torch.float64), variances=1.5).flatten()


if args.prc_baseline and not os.path.exists(args.prc_key):
    print(f'PRC baseline key {args.prc_key} is missing; generating it', flush=True)
    with open(args.prc_key, 'wb') as f:
        pickle.dump(KeyGen(4 * 64 * 64, false_positive_rate=args.fpr, t=3), f)

key_path = 'keys/polar_key.pkl'
if not os.path.exists(key_path):
    with open(key_path, 'wb') as f:
        pickle.dump(PolarWatermark.new().key, f)
with open(key_path, 'rb') as f:
    wm = PolarWatermark(pickle.load(f))
wrong_key = PolarWatermark.new()
print(f'polar watermark: {wm.num_bits} sector signs, {2 ** wm.nonce_bits} nonces', flush=True)

summary = {'args': vars(args), 'num_bits': wm.num_bits, 'distribution': distribution_check(wm)}
print(json.dumps(summary['distribution'], indent=2), flush=True)

random.seed(42)
all_prompts = [sample['Prompt'] for sample in load_dataset(args.dataset_id)['test']]
prompts = random.sample(all_prompts, args.test_num)

pipe = stable_diffusion_pipe(solver_order=1, model_id=args.model_id, cache_dir=hf_cache_dir)
pipe.set_progress_bar_config(disable=True)


def render(prompt, latents):
    image, _, _ = generate(prompt=prompt, init_latents=latents.to(torch.float32).to(device),
                           guidance_scale=args.guidance, num_inference_steps=args.inf_steps,
                           solver_order=1, pipe=pipe)
    return image


def invert(image, decoder_inv):
    return exact_inversion(image, prompt='', test_num_inference_steps=args.inf_steps, inv_order=0,
                           pipe=pipe, decoder_inv=decoder_inv).detach().float().cpu()


def fidelity(true_latents, inverted, angle):
    restored = wm.unrotate(inverted, [angle])
    true = true_latents.reshape(1, *wm.shape).float()
    disk = wm.disk.expand(wm.shape)
    pixel_agree = (torch.sign(restored[0][disk]) == torch.sign(true[0][disk])).double().mean()
    sector_corr = torch.corrcoef(torch.stack([wm.features(restored)[0], wm.features(true)[0]]))[0, 1]
    return float(pixel_agree), float(sector_corr)


polar_rows = []
for i in range(args.test_num):
    seed_everything(i)
    g = torch.from_numpy(np.random.randn(1, 4, 64, 64))
    nonce = wm.random_nonce()
    for label, latents in (('nowm', g), ('wm', wm.embed(g, nonce))):
        image = render(prompts[i], latents)
        image.save(f'{args.out}/images/{i}_{label}.png')
        for angle in args.angles:
            attacked = TF.rotate(image, angle)
            attacked.save(f'{args.out}/attacked/{i}_{label}_rot{angle:g}.png')
            inverted = invert(attacked, bool(args.decoder_inv))
            result = wm.detect(inverted, fpr=args.fpr)
            pixel_agree, sector_corr = fidelity(latents, inverted, angle)
            row = {'image': i, 'label': label, 'attack_angle': angle, **result,
                   'nonce_embedded': nonce if label == 'wm' else None,
                   'nonce_correct': label == 'wm' and result['nonce'] == nonce,
                   'wrong_key_score': wrong_key.detect(inverted, fpr=args.fpr)['score'],
                   'pixel_sign_agreement': pixel_agree, 'sector_correlation': sector_corr}
            polar_rows.append(row)
            print(f"[polar] img {i} {label:4s} rot {angle:>4g}: detected={row['detected']!s:5s} "
                  f"T={row['score']:.2f}/{row['threshold']:.2f} angle_est={row['angle']:+.0f} "
                  f"nonce_ok={row['nonce_correct']} wrong_key_T={row['wrong_key_score']:.2f} "
                  f"pixel_agree={pixel_agree:.3f} sector_corr={sector_corr:.2f}", flush=True)
summary['polar'] = polar_rows

if args.prc_baseline:
    with open(args.prc_key, 'rb') as f:
        encoding_key, decoding_key = pickle.load(f)
    prc_fpr = decoding_key[3]
    search = np.arange(-180.0, 180.0, 1.0)
    seed_everything(0)
    z_prc = prc_gaussians.sample(Encode(encoding_key)).reshape(1, 4, 64, 64)
    image = render(prompts[0], z_prc)
    image.save(f'{args.out}/images/0_prc.png')
    prc_rows = []
    for angle in args.prc_angles:
        attacked = TF.rotate(image, angle)
        attacked.save(f'{args.out}/attacked/0_prc_rot{angle:g}.png')
        inverted = invert(attacked, True)
        direct = posteriors(inverted)
        restored = wm.unrotate(inverted, search)
        scores = [prc_statistic(decoding_key, posteriors(restored[k])) for k in range(len(search))]
        best = int(np.argmax(scores))
        pixel_agree, _ = fidelity(z_prc, inverted, angle)
        row = {
            'attack_angle': angle,
            'direct_detect': bool(Detect(decoding_key, direct)),
            'direct_decode': Decode(decoding_key, direct) is not None,
            'direct_statistic': prc_statistic(decoding_key, direct),
            'direct_threshold': math.sqrt(math.log(1 / prc_fpr)),
            'search_statistic': scores[best],
            'search_threshold': math.sqrt(math.log(len(search) / prc_fpr)),
            'search_angle': float(search[best]),
            'search_decode': Decode(decoding_key, posteriors(restored[best])) is not None,
            'pixel_sign_agreement': pixel_agree,
        }
        row['search_detect'] = row['search_statistic'] >= row['search_threshold']
        prc_rows.append(row)
        print(f"[prc]   rot {angle:>4g}: direct detect={row['direct_detect']} decode={row['direct_decode']} "
              f"S={row['direct_statistic']:.2f}/{row['direct_threshold']:.2f} | with rotation search "
              f"detect={row['search_detect']} decode={row['search_decode']} S={row['search_statistic']:.2f}/"
              f"{row['search_threshold']:.2f} angle_est={row['search_angle']:+.0f} | pixel_agree={pixel_agree:.3f}",
              flush=True)
    summary['prc'] = prc_rows

with open(f'{args.out}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

thumb = 256
columns = [('nowm', 0), ('wm', 0), ('wm', max(args.angles))]
grid = Image.new('RGB', (thumb * len(columns), thumb * args.test_num), 'white')
for i in range(args.test_num):
    for c, (label, angle) in enumerate(columns):
        path = f'{args.out}/images/{i}_{label}.png' if angle == 0 else f'{args.out}/attacked/{i}_{label}_rot{angle:g}.png'
        grid.paste(Image.open(path).resize((thumb, thumb)), (c * thumb, i * thumb))
grid.save(f'{args.out}/compare_grid.png')
print(f'Saved {args.out}/summary.json and {args.out}/compare_grid.png')
