"""RingID v4 codec: key rings on one latent channel, VAE detect, 10 degree search.

Embed happens on the denoised latent immediately before VAE decode.
Detect is VAE encode plus FFT ring correlation. No DDIM inversion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from ringid.attacks import undo_expand_rotation
from ringid.key import rng_from_key

CHANNEL = 3
INNER_R = 1
OUTER_R = 6
RING_SCALE = 1.35
N_LEVELS = 8
BLEND = 0.85
RESIDUAL_CLIP = 2.4
CENTER_SUPPRESS = 0.70
CENTER_SIGMA = 9.0
SEARCH_STEP_DEG = 10.0


@dataclass(frozen=True)
class DetectResult:
    verified: bool
    peak: float
    threshold: float
    estimated_deg: float
    channel: int
    zero_deg_peak: float


def ring_radii(inner_r: int = INNER_R, outer_r: int = OUTER_R) -> np.ndarray:
    return np.arange(int(inner_r), int(outer_r) + 1, dtype=np.int32)


def build_ring_template(key: str) -> np.ndarray:
    radii = ring_radii()
    rng = rng_from_key(key, "ringid_fft_rings")
    levels = np.linspace(-RING_SCALE, RING_SCALE, N_LEVELS)
    idx = rng.integers(0, N_LEVELS, size=radii.size)
    tpl = np.zeros(OUTER_R + 1, dtype=np.float64)
    tpl[radii] = levels[idx]
    return tpl


def _radius_ids(h: int, w: int) -> np.ndarray:
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    return np.rint(np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)).astype(np.int32)


def _fft2c(arr: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(arr, norm="ortho"), axes=(-2, -1))


def _ifft2c(spec: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(np.fft.ifftshift(spec, axes=(-2, -1)), norm="ortho")


def _enforce_hermitian(spec: np.ndarray) -> np.ndarray:
    h, w = spec.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    y2 = (2 * cy - yy) % h
    x2 = (2 * cx - xx) % w
    return 0.5 * (spec + np.conjugate(spec[y2, x2]))


def _chessboard(h: int, w: int) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return np.where((yy + xx) % 2 == 0, 1.0, -1.0)


def _ring_mask(h: int, w: int) -> np.ndarray:
    rid = _radius_ids(h, w)
    return (rid >= INNER_R) & (rid <= OUTER_R)


def _template_spectrum(h: int, w: int, template: np.ndarray) -> np.ndarray:
    rid = _radius_ids(h, w)
    spec = np.zeros((h, w), dtype=np.float64)
    for r in ring_radii():
        spec[rid == int(r)] = float(template[int(r)])
    return spec


def _center_suppress(resid: np.ndarray) -> np.ndarray:
    h, w = resid.shape[-2:]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.ogrid[:h, :w]
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    win = 1.0 - CENTER_SUPPRESS * np.exp(-r2 / (2.0 * CENTER_SIGMA ** 2))
    return resid * win


def imprint_channel(spatial: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Keep host FFT phase. Blend ring amplitudes. Suppress the center lobe."""
    host = np.asarray(spatial, dtype=np.float64)
    spec = _fft2c(host)
    h, w = host.shape[-2:]
    mask = _ring_mask(h, w)
    tpl2d = _template_spectrum(h, w, template) * _chessboard(h, w)
    old = spec.real[mask]
    target = tpl2d[mask] / (float(np.std(tpl2d[mask])) + 1e-8) * (float(np.std(old)) + 1e-8)
    spec.real[mask] = (1.0 - BLEND) * old + BLEND * target
    spec = _enforce_hermitian(spec)
    resid = np.real(_ifft2c(spec)) - host
    resid = _center_suppress(resid)
    rms = float(np.sqrt(np.mean(resid ** 2))) + 1e-8
    resid = np.clip(resid, -RESIDUAL_CLIP * rms, RESIDUAL_CLIP * rms)
    return host + resid


def inject_latents(latents: torch.Tensor, key: str) -> torch.Tensor:
    if latents.dim() == 3:
        latents = latents.unsqueeze(0)
    tpl = build_ring_template(key)
    arr = latents.detach().cpu().numpy().astype(np.float64)
    arr[0, CHANNEL] = imprint_channel(arr[0, CHANNEL], tpl)
    out = torch.from_numpy(arr.astype(np.float32)).to(device=latents.device)
    return out.type_as(latents)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def score_channel(spatial: np.ndarray, template: np.ndarray) -> float:
    spec = np.real(_fft2c(np.asarray(spatial, dtype=np.float64)))
    h, w = spec.shape[-2:]
    mask = _ring_mask(h, w)
    chess = _chessboard(h, w)
    tpl2d = _template_spectrum(h, w, template)
    peak_chess = _pearson(spec[mask] * chess[mask], tpl2d[mask])
    peak_raw = _pearson(spec[mask], tpl2d[mask])
    return max(peak_chess, peak_raw)


def search_angles(step_deg: float = SEARCH_STEP_DEG) -> list[float]:
    step = max(0.5, float(step_deg))
    count = max(1, int(math.ceil(360.0 / step)))
    return [-180.0 + i * step for i in range(count)]


def encode_image(pipe, image: Image.Image) -> np.ndarray:
    from torchvision import transforms

    device = pipe.device
    with torch.no_grad():
        tensor = transforms.ToTensor()(image.convert("RGB")).unsqueeze(0).to(device) * 2 - 1
        latent = pipe.vae.encode(tensor).latent_dist.sample()
        scaled = float(pipe.vae.config.scaling_factor) * latent
    arr = scaled.detach().cpu().numpy().astype(np.float64)
    if arr.ndim == 3:
        arr = arr[None, ...]
    return arr


def _peak(pipe, image: Image.Image, template: np.ndarray) -> tuple[float, int]:
    lat = encode_image(pipe, image)
    peak = score_channel(lat[0, CHANNEL], template)
    return float(peak), CHANNEL


def best_peak(pipe, image: Image.Image, template: np.ndarray, threshold: float | None) -> tuple[float, int, float, float]:
    zero_peak, zero_ch = _peak(pipe, image, template)
    best_peak_v = float(zero_peak)
    best_ch = int(zero_ch)
    best_angle = 0.0
    if threshold is not None and best_peak_v >= threshold:
        return best_peak_v, best_ch, best_angle, float(zero_peak)
    for angle in search_angles():
        if abs(angle) < 1e-9:
            continue
        canonical = undo_expand_rotation(image, angle)
        peak, ch = _peak(pipe, canonical, template)
        if peak > best_peak_v:
            best_peak_v = float(peak)
            best_ch = int(ch)
            best_angle = float(angle)
    return best_peak_v, best_ch, best_angle, float(zero_peak)


def calibrate_threshold(key: str, pipe=None, *, n_neg: int = 16, prompt: str = "a photo of a beautiful woman", steps: int = 15, seed_base: int = 9000) -> float:
    """Threshold from unmatched images. With a pipeline, negatives are clean SD images."""
    template = build_ring_template(key)
    peaks: list[float] = []
    if pipe is None:
        from scipy.ndimage import rotate as nd_rotate

        rng = np.random.default_rng(91)
        for _ in range(int(n_neg)):
            arr = rng.standard_normal((64, 64))
            best = -1.0
            for angle in search_angles():
                view = arr if abs(angle) < 1e-9 else nd_rotate(arr, angle, reshape=False, order=1, mode="nearest")
                best = max(best, score_channel(view, template))
            peaks.append(best)
    else:
        from ringid.pipeline import generate_clean

        for i in range(int(n_neg)):
            img = generate_clean(pipe, prompt, seed=int(seed_base) + i, steps=int(steps))
            peak, _, _, _ = best_peak(pipe, img, template, threshold=None)
            peaks.append(peak)
    return float(np.quantile(peaks, 0.995) + 0.08)


def verify_image(key: str, image: Image.Image, pipe, threshold: float | None = None) -> DetectResult:
    template = build_ring_template(key)
    if threshold is None:
        threshold = calibrate_threshold(key, pipe)
    peak, ch, angle, zero_peak = best_peak(pipe, image, template, threshold)
    return DetectResult(
        verified=bool(peak >= threshold),
        peak=float(peak),
        threshold=float(threshold),
        estimated_deg=float(angle),
        channel=int(ch),
        zero_deg_peak=float(zero_peak),
    )
