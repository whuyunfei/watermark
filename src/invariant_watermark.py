"""Rotation-invariant watermark in the initial diffusion latent, detected without any angle search.

The inscribed disk of every latent channel is cut into rings of equal angular
sectors. The angular DFT of a ring's normalized sector sums gives coefficients
X[c, k, m] for harmonic m. Rotating the image about its centre by theta shifts
every ring circularly, which multiplies all coefficients of harmonic m, in every
ring and channel, by the same phase exp(-i m theta).

Embedding keeps every magnitude and sets the phase of X[c, k, m] to
alpha[c, k, m] + phi[m]: alpha is a keyed template and phi[m] a fresh uniform
phase per image. Detection forms, for each harmonic, the coherent sum
S[m] = sum_{c,k} Xhat[c, k, m] exp(-i alpha[c, k, m]). The unknown rotation and
phi[m] only change the phase of S[m], so W[m] = |S[m]|^2 / sum |Xhat|^2 and
D = sum_m W[m] are rotation invariant and need one inversion.

Distribution: for an i.i.d. N(0, 1) latent the coefficients are independent
complex Gaussians with uniform phases. With a uniformly random key the new
phases are again i.i.d. uniform and independent of the magnitudes, so a single
watermarked latent is exactly i.i.d. N(0, 1).

False positives: for an image independent of the key, the template phases are
i.i.d. uniform given Xhat. Every projection of S[m] / |Xhat| is then
sub-Gaussian with variance proxy 1/2 (E exp(t cos U) = I0(t) <= exp(t^2 / 4)),
hence E exp(lambda W[m]) <= 1 / (1 - lambda), and harmonics use disjoint
template phases. Chernoff gives P(D >= t) <= (t / M)^M exp(M - t) for any image.
"""
import math
import secrets

import numpy as np
import torch
from Crypto.Cipher import ChaCha20


def chernoff_threshold(num_harmonics, fpr):
    lo, hi = float(num_harmonics), 10.0 * num_harmonics + 100.0
    log_fpr = math.log(fpr)
    for _ in range(100):
        t = (lo + hi) / 2
        if num_harmonics * math.log(t / num_harmonics) + num_harmonics - t > log_fpr:
            lo = t
        else:
            hi = t
    return hi


class InvariantWatermark:
    def __init__(self, key, channels=4, size=64, r_max=30.0, ring_width=2.0, arc=2.0, max_harmonic=16):
        self.key = key
        self.shape = (channels, size, size)
        centre = (size - 1) / 2.0
        yy, xx = np.mgrid[0:size, 0:size]
        radius = np.hypot(yy - centre, xx - centre).reshape(-1)
        angle = np.mod(np.arctan2(yy - centre, xx - centre), 2 * np.pi).reshape(-1)

        self.rings = []
        for ring in range(int(r_max // ring_width)):
            r0, r1 = ring * ring_width, (ring + 1) * ring_width
            pixels = np.flatnonzero((radius >= r0) & (radius < r1))
            n = max(3, round(np.pi * (r0 + r1) / arc))
            sectors = (angle[pixels] / (2 * np.pi) * n).astype(np.int64)
            counts = np.bincount(sectors, minlength=n)
            assert counts.min() > 0, f'empty sector in ring {ring}'
            for c in range(channels):
                self.rings.append((c * size * size + pixels, sectors, counts, n))

        self.harmonics = np.arange(1, max_harmonic + 1)
        # members[m] lists the rings that carry harmonic m (it must lie strictly below Nyquist).
        self.members = {int(m): [j for j, r in enumerate(self.rings) if m < r[3] / 2] for m in self.harmonics}
        self.num_coefficients = sum(len(v) for v in self.members.values())

        cipher = ChaCha20.new(key=key, nonce=bytes(8))
        words = np.frombuffer(cipher.encrypt(bytes(4 * self.num_coefficients)), dtype=np.uint32)
        phases = words.astype(np.float64) / 2.0 ** 32 * 2 * np.pi
        self.template, start = {}, 0
        for m, members in self.members.items():
            self.template[m] = np.exp(1j * phases[start:start + len(members)])
            start += len(members)

    @classmethod
    def new(cls, **kwargs):
        return cls(secrets.token_bytes(32), **kwargs)

    def _spectra(self, latents):
        flat = latents.detach().cpu().to(torch.float64).reshape(-1).numpy().copy()
        sums = [np.bincount(s, weights=flat[p], minlength=n) / np.sqrt(c) for p, s, c, n in self.rings]
        return flat, sums, [np.fft.rfft(f) for f in sums]

    def embed(self, latents, strength=1.0):
        """strength 1 aligns every phase with the template; smaller values only contract toward it."""
        flat, sums, spectra = self._spectra(latents)
        new = [X.copy() for X in spectra]
        rng = np.random.default_rng(secrets.randbits(128))
        for m, members in self.members.items():
            target = self.template[m] * np.exp(1j * rng.uniform(0, 2 * np.pi))
            for j, t in zip(members, target):
                offset = np.angle(spectra[j][m] / t)
                new[j][m] = np.abs(spectra[j][m]) * t * np.exp(1j * (1 - strength) * offset)
        out = flat.copy()
        for (pixels, sectors, counts, n), f, X in zip(self.rings, sums, new):
            out[pixels] += ((np.fft.irfft(X, n) - f) / np.sqrt(counts))[sectors]
        return torch.from_numpy(out).reshape(1, *self.shape)

    def harmonic_scores(self, latents):
        _, _, spectra = self._spectra(latents)
        scores = []
        for m, members in self.members.items():
            x = np.array([spectra[j][m] for j in members])
            scores.append(abs(np.sum(x * np.conj(self.template[m]))) ** 2 / np.sum(np.abs(x) ** 2))
        return np.array(scores)

    def detect(self, latents, fpr=1e-5):
        score = float(self.harmonic_scores(latents).sum())
        threshold = chernoff_threshold(len(self.harmonics), fpr)
        return {'detected': score >= threshold, 'score': score, 'threshold': threshold}
