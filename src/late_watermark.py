"""Polar watermark injected into the noise component at a late denoising step.

By step K of the deterministic sampler the layout and objects are fixed, and
x_t = alpha_t * x0_hat + sigma_t * eps_hat. Only the polar-sector features of
the unconditional noise estimate eps_hat are moved toward the keyed signs, by
the least-change transport map. The image therefore keeps the content of the
unwatermarked generation from the same seed and differs only in fine detail.

Detection inverts the image back to step K, re-estimates eps_hat with the
empty prompt, and correlates its sector features with every nonce. The
inversion of rotated content loses most of the signal at large angles, so the
image is first rotated back to anchor angles every 45 degrees (lossless at
multiples of 90) and each inversion is searched over the residual integer
angles by latent un-rotation. That peak is flat, so a second pass rotates the
image itself back around the best coarse angle on a 0.5 degree grid and
inverts again. Each statistic depends on the image and a fixed angle only, so
the threshold is a union bound over all 360 + 720 angles and all nonces: for
any unwatermarked image P(detect) <= fpr.
"""
import math

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from src.optim_utils import transform_img
from src.polar_watermark import PolarWatermark


def wrap(angle):
    return (angle + 180.0) % 360.0 - 180.0


class LateStepWatermark:
    def __init__(self, pipe, key, step=45, strength=1.0, num_inference_steps=50, anchor_step=45, **polar_kwargs):
        assert 360 % anchor_step == 0
        self.pipe = pipe
        self.device = pipe.unet.device
        self.empty = self.encode('')[1]
        self.polar = PolarWatermark(key, **polar_kwargs)
        self.step = step
        self.strength = strength
        self.num_inference_steps = num_inference_steps
        self.anchors = wrap(np.arange(0.0, 360.0, anchor_step))
        self.residuals = np.arange(anchor_step) - anchor_step // 2
        self.num_tests = (360 + 720) * 2 ** self.polar.nonce_bits

    @torch.no_grad()
    def encode(self, prompt):
        """(prompt, empty prompt) embeddings; the text encoder may live on another device than the UNet."""
        cond, uncond = self.pipe.encode_prompt(prompt, self.pipe.text_encoder.device, 1, True, None)
        return cond.to(self.device), uncond.to(self.device)

    @torch.no_grad()
    def generate(self, prompt, init_latents, nonce=None, guidance_scale=3.0):
        pipe, sched = self.pipe, self.pipe.scheduler
        device = self.device
        cond, uncond = self.encode(prompt)
        embeddings = torch.cat([uncond, cond])
        sched.set_timesteps(self.num_inference_steps, device=device)
        x = init_latents.float().to(device) * sched.init_noise_sigma
        for i, t in enumerate(sched.timesteps):
            if nonce is not None and i == self.step:
                eps = pipe.unet(x, t, encoder_hidden_states=uncond).sample
                delta = self.polar.shift(eps.cpu(), nonce, self.strength)
                x = x + float(sched.sigma_t[int(t)]) * delta.float().to(device)
            eps = pipe.unet(torch.cat([x] * 2), t, encoder_hidden_states=embeddings).sample
            eps_uncond, eps_text = eps.chunk(2)
            x = sched.step(eps_uncond + guidance_scale * (eps_text - eps_uncond), t, x).prev_sample
        return pipe.numpy_to_pil(pipe.decode_latents(x))[0]

    def recover_noise(self, images, batch_size=1):
        """Noise estimate at step K for every image, shape (len(images), 4, 64, 64)."""
        return torch.cat([self._recover_noise(images[i:i + batch_size]) for i in range(0, len(images), batch_size)])

    @torch.no_grad()
    def _recover_noise(self, images):
        pipe, sched = self.pipe, self.pipe.scheduler
        device = self.device
        uncond = self.empty.expand(len(images), -1, -1)
        batch = torch.stack([transform_img(image) for image in images]).to(device)
        x = pipe.get_image_latents(batch, sample=False).float()
        sched.set_timesteps(self.num_inference_steps)
        stop = int(sched.timesteps[self.step])
        stride = sched.config.num_train_timesteps // self.num_inference_steps
        for s in reversed(sched.timesteps.tolist()):
            t = s - stride
            sigma_s, sigma_t = float(sched.sigma_t[s]), float(sched.sigma_t[t])
            phi_1 = math.expm1(float(sched.lambda_t[s]) - float(sched.lambda_t[t]))
            eps = pipe.unet(x, s, encoder_hidden_states=uncond).sample
            x0 = sched.convert_model_output(eps, t, x)
            x = (sigma_s / sigma_t) * (x + float(sched.alpha_t[t]) * phi_1 * x0)
            if s == stop:
                break
        return pipe.unet(x, stop, encoder_hidden_states=uncond).sample.cpu()

    def detect(self, image, fpr=1e-5, polar=None, cache=None, refine=True):
        """`cache` maps realignment angles to recovered noise and can be shared between keys."""
        polar = self.polar if polar is None else polar
        cache = {} if cache is None else cache
        fine = {}

        def noises_at(angles):
            missing = [a for a in dict.fromkeys(angles) if a not in cache]
            if missing:
                rotated = [TF.rotate(image, -a, interpolation=InterpolationMode.BILINEAR) for a in missing]
                cache.update(zip(missing, self.recover_noise(rotated)))
            return [cache[a] for a in angles]

        def best_realigned(angles):
            angles = [wrap(a) for a in angles]
            for angle, noise in zip(angles, noises_at(angles)):
                if angle not in fine:
                    fine[angle] = polar.detect(noise, fpr, [0.0], num_tests=self.num_tests)
            return max(angles, key=lambda a: fine[a]['score'])

        coarse = None
        for anchor, noise in zip(self.anchors, noises_at(list(self.anchors))):
            result = polar.detect(noise, fpr, self.residuals, num_tests=self.num_tests)
            if coarse is None or result['score'] > coarse['score']:
                coarse = {**result, 'angle': wrap(anchor + result['angle'])}
        if not refine:
            return coarse

        angle = best_realigned([coarse['angle'] - 1, coarse['angle'], coarse['angle'] + 1])
        angle = best_realigned([angle - 0.5, angle, angle + 0.5])
        best = fine[angle] if fine[angle]['score'] >= coarse['score'] else coarse
        reported = angle if best is fine[angle] else coarse['angle']
        return {**best, 'angle': reported, 'coarse_score': coarse['score'],
                'fine_score': fine[angle]['score'], 'inversions': len(cache)}
