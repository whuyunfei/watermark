"""Stable Diffusion loader. First call downloads weights into the cache directory."""

from __future__ import annotations

import os

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image

MODEL_ID = os.environ.get("ANYANGLE_MODEL", "Manojb/stable-diffusion-2-1-base")
PROMPT = "a photo of a beautiful woman"


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cache_dir() -> str:
    path = os.environ.get("ANYANGLE_CACHE") or os.path.join(project_root(), "model_cache")
    os.makedirs(path, exist_ok=True)
    return path


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ddim_scheduler() -> DDIMScheduler:
    return DDIMScheduler(
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


def _local_snapshot() -> str | None:
    """A finished download stores the unet weights under the hub snapshot folder."""
    root = os.path.join(cache_dir(), f"models--{MODEL_ID.replace('/', '--')}", "snapshots")
    if not os.path.isdir(root):
        return None
    for name in os.listdir(root):
        snap = os.path.join(root, name)
        unet = os.path.join(snap, "unet", "diffusion_pytorch_model.bin")
        vae = os.path.join(snap, "vae", "diffusion_pytorch_model.bin")
        text = os.path.join(snap, "text_encoder", "pytorch_model.bin")
        if os.path.isfile(unet) and os.path.isfile(vae) and os.path.isfile(text):
            return snap
    return None


def load_pipeline():
    """Download the SD checkpoint on first use, then reuse the local snapshot."""
    kwargs = dict(
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
        use_safetensors=False,
    )
    local = _local_snapshot()
    if local is not None:
        pipe = StableDiffusionPipeline.from_pretrained(local, local_files_only=True, **kwargs)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID, cache_dir=cache_dir(), local_files_only=False, **kwargs
        )
    pipe.scheduler = ddim_scheduler()
    pipe = pipe.to(device())
    return pipe


def random_latent(pipe, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(
        1, 4, 64, 64, generator=gen, device=pipe.device, dtype=torch.float32
    )


def generate(pipe, latents: torch.Tensor, *, seed: int, steps: int, prompt: str = PROMPT) -> Image.Image:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    out = pipe(
        prompt,
        height=512,
        width=512,
        num_inference_steps=int(steps),
        latents=latents,
        generator=gen,
        return_dict=False,
    )
    images = out[0]
    if isinstance(images, (list, tuple)):
        return images[0]
    return images


def encode_image(pipe, image: Image.Image) -> np.ndarray:
    img = image.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(pipe.device) * 2.0 - 1.0
    with torch.no_grad():
        latent = pipe.vae.encode(tensor).latent_dist.sample()
    return (0.18215 * latent).detach().cpu().numpy()
