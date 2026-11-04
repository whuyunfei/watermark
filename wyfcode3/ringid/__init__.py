"""RingID latent v4: SD watermark, VAE detect, 10-degree rotation search."""

from ringid.codec import DetectResult, verify_image
from ringid.key import load_key, rng_from_key
from ringid.pipeline import generate_watermarked, load_pipeline

__all__ = [
    "DetectResult",
    "generate_watermarked",
    "load_key",
    "load_pipeline",
    "rng_from_key",
    "verify_image",
]
