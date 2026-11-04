"""Key file and the only RNGs this scheme uses.

Ring amplitudes come from NumPy Generator seeded by SHA-256(key || label).
The diffusion initial latent comes from torch.Generator(manual_seed=image_seed).
No separate Rust crate is required.
"""

from __future__ import annotations

import hashlib
import os
import secrets

import numpy as np

KEY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "keys")
DEFAULT_KEY_PATH = os.path.join(KEY_DIR, "watermark.key")


def generate_key(path: str = DEFAULT_KEY_PATH) -> str:
    """Write a new 32-byte key. Returns the key text."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = "ringid-v4-key-v1\n" + secrets.token_hex(32) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return text.strip()


def load_key(path: str | None = None) -> str:
    """Load key text. Creates keys/watermark.key on first use."""
    path = path or DEFAULT_KEY_PATH
    if not os.path.isfile(path):
        generate_key(path)
    with open(path, encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        raise ValueError(f"empty key file: {path}")
    return text


def rng_from_key(key: str, label: str = "") -> np.random.Generator:
    digest = hashlib.sha256((key + label).encode("utf-8")).hexdigest()
    return np.random.default_rng(int(digest[:16], 16))
