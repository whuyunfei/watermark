"""Rotation attack used by evaluation: expand, then resize back to 512."""

from __future__ import annotations

import math

from PIL import Image


def rotate(img: Image.Image, deg: float) -> Image.Image:
    if abs(deg) < 1e-9:
        return img.copy()
    spun = img.rotate(deg, expand=True, resample=Image.BICUBIC, fillcolor=(128, 128, 128))
    return spun.resize(img.size, Image.LANCZOS)


def undo_expand_rotation(image: Image.Image, degrees: float) -> Image.Image:
    """Approximate inverse of rotate(expand=True) followed by resize to the original size."""
    if abs(degrees) < 1e-7:
        return image.copy()
    w, h = image.size
    rad = math.radians(degrees)
    expanded_w = max(w, int(math.ceil(abs(w * math.cos(rad)) + abs(h * math.sin(rad)))))
    expanded_h = max(h, int(math.ceil(abs(h * math.cos(rad)) + abs(w * math.sin(rad)))))
    expanded = image.resize((expanded_w, expanded_h), Image.Resampling.LANCZOS)
    restored = expanded.rotate(
        -degrees,
        expand=False,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(128, 128, 128),
    )
    left = (expanded_w - w) // 2
    top = (expanded_h - h) // 2
    return restored.crop((left, top, left + w, top + h))
