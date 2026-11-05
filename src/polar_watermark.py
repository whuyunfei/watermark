"""Polar-sector watermark for the initial diffusion latent.

The inscribed disk of every latent channel is cut into annular sectors. Each
sector sum, normalized by sqrt(pixel count), is one N(0, 1) coordinate of the
latent, orthogonal to all other sectors and to the within-sector detail.
Embedding replaces that coordinate f by b * |f|, where b is a keyed ChaCha20
sign. With uniform signs this leaves the latent exactly i.i.d. N(0, 1).

A rotation about the image centre keeps the disk inside the frame and only
moves sectors along their rings, so detection inverts the image once and
un-rotates that latent over a grid of angles. Every image draws a fresh nonce
that selects its sign pattern; detection searches all nonces.

Without the watermark the signs are independent of the image, so for each
(angle, nonce) pair the normalized correlation T is a weighted Rademacher sum
with P(T >= t) <= exp(-t^2 / 2). The threshold applies a union bound over all
pairs, which bounds the false positive rate for any unwatermarked image.
"""
import math
import secrets

import numpy as np
import torch
import torchvision.transforms.functional as TF
from Crypto.Cipher import ChaCha20
from scipy.special import ndtr, ndtri
from torchvision.transforms import InterpolationMode


def transport(u, b, strength=1.0):
    """Monotone map from N(0, 1) onto a law with P(sign = b) = strength.

    For uniform b the output is again N(0, 1). Among all maps with that
    output law it moves u the least, so it changes far less than b * |u|.
    """
    v = b * u
    right = -ndtri(ndtr(-v) / (2 * strength))
    if strength < 1:
        p = ndtr(v)
        right = np.where(p <= 1 - strength, ndtri(p / (2 * (1 - strength))), right)
    return b * right


def sector_layout(size=64, r_max=30.0, ring_width=3.0, arc=3.0):
    centre = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    radius = np.hypot(yy - centre, xx - centre)
    angle = np.mod(np.arctan2(yy - centre, xx - centre), 2 * np.pi)

    ids = np.full((size, size), -1, dtype=np.int64)
    next_id = 0
    for ring in range(int(r_max // ring_width)):
        r0, r1 = ring * ring_width, (ring + 1) * ring_width
        mask = (radius >= r0) & (radius < r1)
        n_sectors = max(1, round(np.pi * (r0 + r1) / arc))
        ids[mask] = next_id + (angle[mask] / (2 * np.pi) * n_sectors).astype(np.int64)
        next_id += n_sectors

    used, compact = np.unique(ids[ids >= 0], return_inverse=True)
    ids[ids >= 0] = compact
    counts = np.bincount(compact, minlength=len(used))
    return ids, counts


class PolarWatermark:
    def __init__(self, key, nonce_bits=16, channels=4, size=64, r_max=30.0, ring_width=3.0, arc=3.0):
        self.key = key
        self.nonce_bits = nonce_bits
        self.shape = (channels, size, size)

        ids, counts = sector_layout(size, r_max, ring_width, arc)
        pixels = np.flatnonzero(ids >= 0)
        sectors = ids.reshape(-1)[pixels]
        per_channel = len(counts)
        self.num_bits = channels * per_channel
        self.pixel_index = torch.from_numpy(np.concatenate([c * size * size + pixels for c in range(channels)]))
        self.sector_index = torch.from_numpy(np.concatenate([c * per_channel + sectors for c in range(channels)]))
        self.scale = torch.from_numpy(np.tile(1.0 / np.sqrt(counts), channels))
        self.disk = torch.from_numpy(ids >= 0)

        self.row_bytes = (self.num_bits + 7) // 8
        self._table = None

    @classmethod
    def new(cls, **kwargs):
        return cls(secrets.token_bytes(32), **kwargs)

    def random_nonce(self):
        return secrets.randbelow(2 ** self.nonce_bits)

    def _stream(self, offset, length):
        cipher = ChaCha20.new(key=self.key, nonce=bytes(8))
        cipher.seek(offset)
        return np.frombuffer(cipher.encrypt(bytes(length)), dtype=np.uint8)

    def signs(self, nonce):
        bits = np.unpackbits(self._stream(nonce * self.row_bytes, self.row_bytes))[: self.num_bits]
        return torch.from_numpy(1.0 - 2.0 * bits)

    def sign_table(self):
        if self._table is None:
            rows = 2 ** self.nonce_bits
            stream = self._stream(0, rows * self.row_bytes).reshape(rows, self.row_bytes)
            bits = np.unpackbits(stream, axis=1)[:, : self.num_bits]
            self._table = torch.from_numpy(1 - 2 * bits.astype(np.int8))
        return self._table

    def features(self, latents):
        flat = latents.detach().cpu().reshape(latents.shape[0], -1).to(torch.float64)
        out = torch.zeros(flat.shape[0], self.num_bits, dtype=torch.float64)
        out.index_add_(1, self.sector_index, flat[:, self.pixel_index])
        return out * self.scale

    def embed(self, latents, nonce):
        z = latents.detach().cpu().to(torch.float64).reshape(1, *self.shape).clone()
        f = self.features(z)[0]
        target = self.signs(nonce) * f.abs()
        z.view(-1).index_add_(0, self.pixel_index, ((target - f) * self.scale)[self.sector_index])
        return z

    def shift(self, latents, nonce, strength=1.0):
        f = self.features(latents)[0]
        rms = f.pow(2).mean().sqrt()
        target = rms * torch.from_numpy(transport((f / rms).numpy(), self.signs(nonce).numpy(), strength))
        delta = torch.zeros(math.prod(self.shape), dtype=torch.float64)
        delta.index_add_(0, self.pixel_index, ((target - f) * self.scale)[self.sector_index])
        return delta.reshape(1, *self.shape)

    def unrotate(self, latents, angles):
        z = latents.detach().cpu().to(torch.float32).reshape(self.shape)
        return torch.stack([TF.rotate(z, -float(a), interpolation=InterpolationMode.BILINEAR) for a in angles])

    def detect(self, latents, fpr=1e-5, angles=None, chunk=8192, num_tests=None):
        angles = np.arange(-180.0, 180.0, 1.0) if angles is None else np.asarray(angles, dtype=float)
        feats = self.features(self.unrotate(latents, angles))
        feats = (feats / feats.norm(dim=1, keepdim=True)).float()

        table = self.sign_table()
        best_score, best_angle, best_nonce = -math.inf, None, None
        for start in range(0, table.shape[0], chunk):
            scores = feats @ table[start:start + chunk].float().T
            flat = int(torch.argmax(scores))
            score = float(scores.reshape(-1)[flat])
            if score > best_score:
                row, col = divmod(flat, scores.shape[1])
                best_score, best_angle, best_nonce = score, float(angles[row]), start + col

        if num_tests is None:
            num_tests = len(angles) * table.shape[0]
        threshold = math.sqrt(2 * math.log(num_tests / fpr))
        return {
            'detected': best_score >= threshold,
            'score': best_score,
            'threshold': threshold,
            'angle': best_angle,
            'nonce': best_nonce,
        }
