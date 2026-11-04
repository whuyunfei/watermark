"""Stable Diffusion 2.1 pipeline: random z0, denoise, RingID inject, VAE decode."""

from __future__ import annotations

import os

import torch
from PIL import Image

from ringid.codec import inject_latents

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ID = "Manojb/stable-diffusion-2-1-base"
CACHE_DIR = os.path.join(ROOT, "models")


def cache_ready() -> bool:
    if not os.path.isdir(CACHE_DIR):
        return False
    for dirpath, _, filenames in os.walk(CACHE_DIR):
        if "model_index.json" in filenames:
            return True
    return False


def download_weights() -> str:
    """Download SD 2.1 weights into models/ on first use. Safe to call again."""
    from huggingface_hub import snapshot_download

    os.makedirs(CACHE_DIR, exist_ok=True)
    path = snapshot_download(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        resume_download=True,
    )
    return path


def ensure_weights() -> str:
    if cache_ready():
        return CACHE_DIR
    print(f"Weights not found. Downloading {MODEL_ID} into {CACHE_DIR}")
    return download_weights()


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pipeline():
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    ensure_weights()
    scheduler = DDIMScheduler(
        beta_end=0.012,
        beta_schedule="scaled_linear",
        beta_start=0.00085,
        clip_sample=False,
        set_alpha_to_one=True,
        steps_offset=1,
        num_train_timesteps=1000,
        prediction_type="epsilon",
        timestep_spacing="trailing",
        rescale_betas_zero_snr=False,
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        scheduler=scheduler,
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    pipe = pipe.to(_device())
    pipe.set_progress_bar_config(disable=False)
    return pipe


def random_latent(height: int, width: int, generator: torch.Generator, device) -> torch.Tensor:
    return torch.randn(
        1,
        4,
        height // 8,
        width // 8,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    ).to(device)


def _decode(pipe, latents: torch.Tensor) -> Image.Image:
    import numpy as np

    if latents.dim() == 3:
        latents = latents.unsqueeze(0)
    scale = float(pipe.vae.config.scaling_factor)
    with torch.no_grad():
        image = pipe.vae.decode(latents / scale, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        arr = (image.detach().cpu().permute(0, 2, 3, 1).numpy()[0] * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def generate_clean(pipe, prompt: str, *, seed: int, steps: int, height: int = 512, width: int = 512) -> Image.Image:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    out = pipe(
        prompt,
        height=height,
        width=width,
        num_inference_steps=int(steps),
        generator=gen,
        return_dict=False,
    )
    return out[0]


def generate_watermarked(
    pipe,
    key: str,
    prompt: str,
    *,
    seed: int,
    steps: int,
    height: int = 512,
    width: int = 512,
) -> Image.Image:
    """Clean Gaussian z0, full DDIM denoise, then v4 ring inject, then VAE decode."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    latents = random_latent(height, width, gen, pipe.device)
    out = pipe(
        prompt,
        height=height,
        width=width,
        num_inference_steps=int(steps),
        latents=latents,
        generator=gen,
        output_type="latent",
        return_dict=False,
    )
    denoised = out[0]
    if isinstance(denoised, torch.Tensor) and denoised.dim() == 3:
        denoised = denoised.unsqueeze(0)
    marked = inject_latents(denoised, key)
    return _decode(pipe, marked)
