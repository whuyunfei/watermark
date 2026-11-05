import os
import argparse
import torch
import pickle
import json
from tqdm import tqdm
import random
import numpy as np
from datasets import load_dataset
from src.prc import KeyGen, Encode, str_to_bin, bin_to_str
import src.pseudogaussians as prc_gaussians
from src.baseline.gs_watermark import Gaussian_Shading_chacha
from src.baseline.treering_watermark import tr_detect, tr_get_noise
from inversion import stable_diffusion_pipe, generate

parser = argparse.ArgumentParser('Args')
parser.add_argument('--test_num', type=int, default=10)
parser.add_argument('--method', type=str, default='prc') # gs, tr, prc
parser.add_argument('--model_id', type=str, default='sd2-community/stable-diffusion-2-1-base')
parser.add_argument('--dataset_id', type=str, default='Gustavosta/Stable-Diffusion-Prompts') # coco 
parser.add_argument('--inf_steps', type=int, default=50)
parser.add_argument('--nowm', type=int, default=0)
parser.add_argument('--fpr', type=float, default=0.00001)
parser.add_argument('--prc_t', type=int, default=3)
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
n = 4 * 64 * 64  # the length of a PRC codeword
method = args.method
test_num = args.test_num
model_id = args.model_id
dataset_id = args.dataset_id
nowm = args.nowm
fpr = args.fpr
prc_t = args.prc_t
exp_id = f'{method}_num_{test_num}_steps_{args.inf_steps}_fpr_{fpr}_nowm_{nowm}'
os.makedirs('keys', exist_ok=True)

if method == 'prc':
    if not os.path.exists(f'keys/{exp_id}.pkl'):  # Generate watermark key for the first time and save it to a file
        (encoding_key_ori, decoding_key_ori) = KeyGen(n, false_positive_rate=fpr, t=prc_t)  # Sample PRC keys
        with open(f'keys/{exp_id}.pkl', 'wb') as f:  # Save the keys to a file
            pickle.dump((encoding_key_ori, decoding_key_ori), f)
        with open(f'keys/{exp_id}.pkl', 'rb') as f:  # Load the keys from a file
            encoding_key, decoding_key = pickle.load(f)
        assert encoding_key[0].all() == encoding_key_ori[0].all()
    else:  # Or we can just load the keys from a file
        with open(f'keys/{exp_id}.pkl', 'rb') as f:
            encoding_key, decoding_key = pickle.load(f)
        print(f'Loaded PRC keys from file keys/{exp_id}.pkl')
elif method == 'gs':
    gs_watermark = Gaussian_Shading_chacha(ch_factor=1, hw_factor=8, fpr=fpr, user_number=10000)
    if not os.path.exists(f'keys/{exp_id}.pkl'):
        watermark_m_ori, key_ori, nonce_ori, watermark_ori = gs_watermark.create_watermark_and_return_w()
        with open(f'keys/{exp_id}.pkl', 'wb') as f:
            pickle.dump((watermark_m_ori, key_ori, nonce_ori, watermark_ori), f)
        with open(f'keys/{exp_id}.pkl', 'rb') as f:
            watermark_m, key, nonce, watermark = pickle.load(f)
        assert watermark_m.all() == watermark_m_ori.all()
    else:  # Or we can just load the keys from a file
        with open(f'keys/{exp_id}.pkl', 'rb') as f:
            watermark_m, key, nonce, watermark = pickle.load(f)
            print(f'Loaded GS keys from file keys/{exp_id}.pkl')
elif method == 'tr':
    if not os.path.exists(f'keys/{exp_id}.pkl'):
        _, _, _, tr_key = tr_get_noise((1, 4, 64, 64), keys_path='keys/')
        with open(f'keys/{exp_id}.pkl', 'wb') as f:
            pickle.dump(tr_key, f)
    else:
        with open(f'keys/{exp_id}.pkl', 'rb') as f:
            tr_key = pickle.load(f)
        print(f'Loaded TR keys from file keys/{exp_id}.pkl')
else:
    raise NotImplementedError

if dataset_id == 'coco':
    save_folder = f'./results/{exp_id}_coco/original_images'
else:
    save_folder = f'./results/{exp_id}/original_images'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
print(f'Saving original images to {save_folder}')

random.seed(42)
if dataset_id == 'coco':
    with open('coco/captions_val2017.json') as f:
        all_prompts = [ann['caption'] for ann in json.load(f)['annotations']]
else:
    all_prompts = [sample['Prompt'] for sample in load_dataset(dataset_id)['test']]

prompts = random.sample(all_prompts, test_num)

pipe = stable_diffusion_pipe(solver_order=1, model_id=model_id, cache_dir=hf_cache_dir)
pipe.set_progress_bar_config(disable=True)

def seed_everything(seed, workers=False):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"
    return seed

# for i in tqdm(range(2)):
for i in tqdm(range(test_num)):
    seed_everything(i)
    current_prompt = prompts[i]
    if nowm:
        init_latents_np = np.random.randn(1, 4, 64, 64)
        init_latents = torch.from_numpy(init_latents_np).to(torch.float64)
    else:
        if method == 'prc':
            prc_codeword = Encode(encoding_key)
            init_latents = prc_gaussians.sample(prc_codeword).reshape(1, 4, 64, 64)
        elif method == 'gs':
            init_latents = gs_watermark.truncSampling(watermark_m)
        elif method == 'tr':
            shape = (1, 4, 64, 64)
            init_latents, _, _, _ = tr_get_noise(shape, from_file=tr_key, keys_path='keys/')
        else:
            raise NotImplementedError
    # PRC samples in float64 and Gaussian Shading samples in float16 on CUDA.
    # The UNet is float32. CUDA casts after scaling by init_noise_sigma; CPU and MPS do not.
    if device != 'cuda' and init_latents.dtype != torch.float32:
        init_latents = init_latents.to(torch.float32)
    init_latents = init_latents.to(device)
    orig_image, _, _ = generate(prompt=current_prompt,
                                init_latents=init_latents,
                                num_inference_steps=args.inf_steps,
                                solver_order=1,
                                pipe=pipe
                                )
    orig_image.save(f'{save_folder}/{i}.png')

print(f'Done generating {method} images')