"""
PRC, Gaussian Shading, and Tree-Ring detection.
"""

import argparse
import os
import pickle

import torch
from PIL import Image
from tqdm import tqdm

import src.pseudogaussians as prc_gaussians
from inversion import exact_inversion, stable_diffusion_pipe
from src.baseline.gs_watermark import Gaussian_Shading_chacha
from src.baseline.treering_watermark import tr_detect
from src.prc import Decode, Detect

parser = argparse.ArgumentParser('Args')
parser.add_argument('--test_num', type=int, default=10)
parser.add_argument('--method', type=str, default='prc')  # gs, tr, prc
parser.add_argument('--model_id', type=str, default='sd2-community/stable-diffusion-2-1-base')
parser.add_argument('--dataset_id', type=str, default='Gustavosta/Stable-Diffusion-Prompts')
parser.add_argument('--inf_steps', type=int, default=50)
parser.add_argument('--nowm', type=int, default=0)
parser.add_argument('--fpr', type=float, default=0.00001)
parser.add_argument('--prc_t', type=int, default=3)

parser.add_argument('--test_path', type=str, default='original_images')
args = parser.parse_args()
print(args)

hf_cache_dir = os.path.abspath(os.environ.get('HF_MODEL_CACHE', 'hf_models'))
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f'Using device={device}, hf_cache_dir={hf_cache_dir}')
method = args.method
test_num = args.test_num
model_id = args.model_id
dataset_id = args.dataset_id
fpr = args.fpr
exp_id = f'{method}_num_{test_num}_steps_{args.inf_steps}_fpr_{fpr}_nowm_{args.nowm}'

with open(f'keys/{exp_id}.pkl', 'rb') as f:
    saved_key = pickle.load(f)

gs_watermark = None
tr_key = None
decoding_key = None
if method == 'prc':
    encoding_key, decoding_key = saved_key
elif method == 'gs':
    watermark_m, key, nonce, watermark = saved_key
    gs_watermark = Gaussian_Shading_chacha(ch_factor=1, hw_factor=8, fpr=fpr, user_number=10000)
elif method == 'tr':
    tr_key = saved_key
else:
    raise NotImplementedError(f'Unknown method {method}')

pipe = stable_diffusion_pipe(solver_order=1, model_id=model_id, cache_dir=hf_cache_dir)
pipe.set_progress_bar_config(disable=True)

if dataset_id == 'coco':
    image_dir = f'results/{exp_id}_coco/{args.test_path}'
else:
    image_dir = f'results/{exp_id}/{args.test_path}'

cur_inv_order = 0
var = 1.5
combined_results = []
for i in tqdm(range(test_num)):
    img = Image.open(f'{image_dir}/{i}.png')
    if method == 'prc':
        reversed_latents = exact_inversion(img,
                                           prompt='',
                                           test_num_inference_steps=args.inf_steps,
                                           inv_order=cur_inv_order,
                                           pipe=pipe
                                           )
        reversed_prc = prc_gaussians.recover_posteriors(
            reversed_latents.detach().float().flatten().cpu().to(torch.float64),
            variances=float(var),
        ).flatten().cpu()
        detection_result = Detect(decoding_key, reversed_prc)
        decoding_result = (Decode(decoding_key, reversed_prc) is not None)
        combined_result = detection_result or decoding_result
        print(f'{i:03d}: Detection: {detection_result}; Decoding: {decoding_result}; Combined: {combined_result}')
    elif method == 'gs':
        reversed_latents = exact_inversion(img,
                                           prompt='',
                                           test_num_inference_steps=args.inf_steps,
                                           inv_order=cur_inv_order,
                                           pipe=pipe
                                           )
        accuracy = gs_watermark.eval_watermark_from_file(reversed_latents, key, nonce, watermark)
        combined_result = accuracy >= gs_watermark.tau_onebit
        print(f'{i:03d}: Bit accuracy: {accuracy:.4f}; threshold: {gs_watermark.tau_onebit:.4f}; Detected: {combined_result}')
    else:
        distance, combined_result = tr_detect(img, pipe, 'keys/', tr_key, num_inference_steps=args.inf_steps)
        print(f'{i:03d}: Distance: {distance:.4f}; Detected: {combined_result}')
    combined_results.append(combined_result)

with open('decoded.txt', 'w') as f:
    for result in combined_results:
        f.write(f'{result}\n')

print('Decoded results saved to decoded.txt')
