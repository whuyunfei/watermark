"""Circ-harmonic any-angle watermark. Embedding strength is fixed at FIELD=1.0.

Only the initial Gaussian latent is modified. Detection uses the key, the RGB
image, and the VAE. There is no angle search.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache

import numpy as np
import torch
from PIL import Image

LATENT_C, LATENT_H, LATENT_W = 4, 64, 64
LATENT_LEN = LATENT_C * LATENT_H * LATENT_W
R_INNER = 10
R_OUTER = 18
WM_CHANNELS = (3,)
M_HARMONICS = 12
M_LO = 3
M_HI = 12
N_ON = 6
FIELD = 1.0
THRESHOLD = 0.50
PR_PASS = 0.99

_KEY_PATH = os.path.join(os.path.dirname(__file__), "key.pem")


def load_key_pem(path: str | None = None) -> str:
    with open(path or _KEY_PATH, encoding="utf-8", newline="") as f:
        return f.read()


def _rng(tag: str) -> np.random.Generator:
    h = hashlib.sha256(tag.encode("utf-8")).hexdigest()[:16]
    return np.random.default_rng(int(h, 16))


@lru_cache(maxsize=1)
def _radius_theta() -> tuple[np.ndarray, np.ndarray]:
    cy, cx = LATENT_H / 2.0, LATENT_W / 2.0
    yy, xx = np.mgrid[0:LATENT_H, 0:LATENT_W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    th = np.arctan2(yy - cy, xx - cx)
    return r, th


@lru_cache(maxsize=1)
def _rings() -> tuple[tuple[np.ndarray, ...], ...]:
    rmap, th = _radius_theta()
    rid = np.rint(rmap).astype(np.int32)
    out: list[tuple[np.ndarray, ...]] = []
    for c in range(LATENT_C):
        base = c * LATENT_H * LATENT_W
        ch: list[np.ndarray] = []
        for rr in range(R_INNER, R_OUTER + 1):
            ys, xs = np.nonzero(rid == rr)
            if ys.size < 8:
                ch.append(np.zeros(0, dtype=np.int32))
                continue
            order = np.argsort(th[ys, xs], kind="stable")
            ch.append((base + ys * LATENT_W + xs)[order].astype(np.int32))
        out.append(tuple(ch))
    return tuple(out)


@lru_cache(maxsize=1)
def _channel_ring_idx() -> tuple[np.ndarray, ...]:
    out = []
    for ch in _rings():
        parts = [idx for idx in ch if idx.size]
        out.append(np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32))
    return tuple(out)


def _payload_ring_idx() -> np.ndarray:
    parts = [_channel_ring_idx()[c] for c in WM_CHANNELS if _channel_ring_idx()[c].size]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)


@lru_cache(maxsize=1)
def _inner_disk_idx() -> np.ndarray:
    rmap, _ = _radius_theta()
    rid = np.rint(rmap).astype(np.int32)
    ys, xs = np.nonzero(rid < R_INNER)
    loc = ys * LATENT_W + xs
    parts = [c * LATENT_H * LATENT_W + loc for c in range(LATENT_C)]
    return np.concatenate(parts).astype(np.int32)


def _flat_theta(idx: np.ndarray) -> np.ndarray:
    _, th = _radius_theta()
    loc = np.asarray(idx, dtype=np.int32) % (LATENT_H * LATENT_W)
    return th[loc // LATENT_W, loc % LATENT_W]


def _ntru_supports(key_pem: str) -> tuple[np.ndarray, np.ndarray]:
    pool = np.arange(M_LO - 1, M_HI, dtype=np.int32)
    rng = _rng(f"{key_pem}|any-ntru|{M_LO}|{M_HI}|{N_ON}")
    perm = rng.permutation(pool)
    n_on = min(max(3, int(N_ON)), max(3, int(pool.size) - 3))
    omega = np.sort(perm[:n_on])
    control = np.sort(perm[n_on:])
    return omega.astype(np.int32), control.astype(np.int32)


def _circ_harm_drive(theta: np.ndarray, amp: np.ndarray, phases: np.ndarray) -> np.ndarray:
    acc = np.zeros(int(np.asarray(theta).size), dtype=np.float64)
    th = np.asarray(theta, dtype=np.float64)
    for i, a in enumerate(np.asarray(amp, dtype=np.float64)):
        if abs(float(a)) < 1e-12:
            continue
        acc += float(a) * np.cos((i + 1) * th + float(phases[i]))
    return acc


def _pm1(flat: np.ndarray) -> np.ndarray:
    s = np.sign(np.asarray(flat, dtype=np.float64).reshape(-1))
    s[s == 0.0] = 1.0
    return s


def _uniform_bits(flat: np.ndarray, pool: np.ndarray) -> float:
    signs = _pm1(flat)
    monobit = abs(float(signs.mean()))
    sp = signs[np.sort(np.asarray(pool, dtype=np.int64))]
    m = int(sp.size)
    rhos = []
    for lag in range(1, 5):
        if m <= lag:
            rhos.append(0.0)
            continue
        rhos.append(abs(float(np.corrcoef(sp[:-lag], sp[lag:])[0, 1])))
    max_lag = float(max(rhos)) if rhos else 0.0
    block = 256
    if m >= block:
        blocks = sp[: (m // block) * block].reshape(-1, block)
        block_bias = float(np.max(np.abs(blocks.mean(axis=1))))
    else:
        block_bias = monobit
    excess = max(0.0, block_bias - 3.0 / np.sqrt(float(block)))
    return float(max(0.0, 1.0 - 10.0 * monobit - 0.5 * max_lag - excess))


def _rebalance_mean(flat: np.ndarray, pool: np.ndarray, rng: np.random.Generator) -> None:
    if pool.size == 0:
        return
    signs = _pm1(flat)
    delta = int(np.sum(signs > 0)) - LATENT_LEN // 2
    if delta > 0:
        cand = pool[signs[pool] > 0]
        n = min(int(delta), int(cand.size))
        if n:
            flat[rng.choice(cand, size=n, replace=False)] *= -1.0
    elif delta < 0:
        cand = pool[signs[pool] < 0]
        n = min(int(-delta), int(cand.size))
        if n:
            flat[rng.choice(cand, size=n, replace=False)] *= -1.0


def _repair_uniform_bits(flat: np.ndarray, pool: np.ndarray, rng: np.random.Generator) -> None:
    if pool.size == 0:
        return
    pool_arr = np.asarray(pool, dtype=np.int64)
    _rebalance_mean(flat, pool, rng)
    if _uniform_bits(flat, pool) >= PR_PASS:
        return
    best = _uniform_bits(flat, pool)
    for _ in range(1200):
        if _uniform_bits(flat, pool) >= PR_PASS:
            return
        idx = int(rng.choice(pool_arr))
        flat[idx] *= -1.0
        sc = _uniform_bits(flat, pool)
        if sc + 1e-12 >= best:
            best = sc
            if abs(float(_pm1(flat).mean())) > 1e-12:
                _rebalance_mean(flat, pool, rng)
        else:
            flat[idx] *= -1.0
    _rebalance_mean(flat, pool, rng)


def embed_latent(z0: np.ndarray, key_pem: str) -> np.ndarray:
    """Return a copy of z0 with the FIELD=1.0 ring pattern written on channel 3."""
    nchw = np.array(z0, dtype=np.float64, copy=True)
    squeeze = nchw.ndim == 3
    if squeeze:
        nchw = nchw[None]
    flat = nchw.reshape(-1)
    omega, _ctrl = _ntru_supports(key_pem)
    amp = np.zeros(M_HARMONICS, dtype=np.float64)
    amp[omega] = 1.0
    used = np.zeros(LATENT_LEN, dtype=bool)
    used[_payload_ring_idx()] = True
    used[_inner_disk_idx()] = True
    for c in WM_CHANNELS:
        phases = _rng(f"{key_pem}|any-phase|{c}").uniform(0.0, 2.0 * np.pi, size=M_HARMONICS)
        for idx in _rings()[c]:
            if idx.size < 16:
                continue
            th = _flat_theta(idx)
            harm = _circ_harm_drive(th, amp, phases)
            rms = float(np.sqrt(np.mean(harm * harm))) + 1e-8
            field = np.clip(harm / rms, -3.0, 3.0)
            flat[idx] = flat[idx] + FIELD * np.abs(flat[idx]) * field
    pr_pool = np.flatnonzero(~used)
    repair_rng = _rng(f"{key_pem}:anyangle:uniform_bits")
    for _ in range(6):
        _repair_uniform_bits(flat, pr_pool, repair_rng)
        if _uniform_bits(flat, pr_pool) >= PR_PASS:
            break
    return nchw[0] if squeeze else nchw


def pseudorandom_score(key_pem: str) -> float:
    rng = np.random.default_rng(0)
    marked = embed_latent(rng.standard_normal((1, LATENT_C, LATENT_H, LATENT_W)), key_pem)
    pool = np.flatnonzero(
        ~(
            np.isin(np.arange(LATENT_LEN), _payload_ring_idx())
            | np.isin(np.arange(LATENT_LEN), _inner_disk_idx())
        )
    )
    return _uniform_bits(marked.reshape(-1), pool)


def _ring_psd(flat: np.ndarray, idx: np.ndarray, *, mag_weight: bool) -> np.ndarray:
    psd = np.zeros(M_HARMONICS, dtype=np.float64)
    if idx.size < 16:
        return psd
    th = _flat_theta(idx)
    z = np.asarray(flat[idx], dtype=np.float64)
    if mag_weight:
        w = np.sqrt(np.abs(z) + 1e-8)
        z = z * (w / (float(np.mean(w)) + 1e-8))
    z = z - float(np.mean(z))
    z = z / (float(np.linalg.norm(z)) + 1e-8)
    for i in range(M_HARMONICS):
        m = i + 1
        re = float(np.dot(z, np.cos(m * th)))
        im = float(np.dot(z, np.sin(m * th)))
        psd[i] = re * re + im * im
    return psd


def _whiten_psd(psd: np.ndarray) -> np.ndarray:
    x = np.asarray(psd, dtype=np.float64)
    ker = np.array([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64)
    ker /= float(np.sum(ker))
    pad = ker.size // 2
    logx = np.log(x + 1e-8)
    sm = np.convolve(np.pad(logx, pad, mode="edge"), ker, mode="valid")
    return x / (np.exp(sm) + 1e-8)


def _support_contrast(psd: np.ndarray, on: np.ndarray, off: np.ndarray) -> float:
    on_e = float(np.mean(psd[on])) if on.size else 0.0
    off_e = float(np.mean(psd[off])) if off.size else 1.0
    return float((on_e - off_e) / (on_e + off_e + 1e-8))


def _template_corr(psd: np.ndarray, on: np.ndarray, off: np.ndarray) -> float:
    pool = np.concatenate([on, off]) if off.size else on
    x = np.asarray(psd[pool], dtype=np.float64)
    g = np.zeros(pool.size, dtype=np.float64)
    g[: on.size] = 1.0
    g[on.size :] = -1.0
    x = x - float(np.mean(x))
    g = g - float(np.mean(g))
    den = float(np.linalg.norm(x) * np.linalg.norm(g)) + 1e-8
    return float(np.dot(x, g) / den)


def score_latent(latent: np.ndarray, key_pem: str, *, whiten: bool = False) -> float:
    flat = np.asarray(latent, dtype=np.float64).reshape(-1)
    omega, control = _ntru_supports(key_pem)
    idx = _payload_ring_idx()
    scores = []
    for mag in (False, True):
        psd = _ring_psd(flat, idx, mag_weight=mag)
        if whiten:
            psd = _whiten_psd(psd)
        scores.append(_support_contrast(psd, omega, control))
        scores.append(_template_corr(psd, omega, control))
    raw = float(max(scores))
    return float(np.clip((raw - 0.15) / 0.70, 0.0, 1.0))


def detect_image(pipe, image: Image.Image, key_pem: str, *, whiten: bool = False) -> float:
    from anyangle.sd import encode_image

    latent = encode_image(pipe, image)
    return score_latent(latent, key_pem, whiten=whiten)


def embed_torch(z0: torch.Tensor, key_pem: str) -> torch.Tensor:
    out = embed_latent(z0.detach().cpu().numpy(), key_pem)
    return torch.from_numpy(np.asarray(out, dtype=np.float32)).to(z0.device)
