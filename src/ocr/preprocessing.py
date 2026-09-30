"""Composable Pillow image preprocessing utilities for OCR experiments."""

from __future__ import annotations

from PIL import Image, ImageEnhance


def resize_image(image: Image.Image, scale: float = 2.0) -> Image.Image:
    """Return a proportionally resized copy of ``image``."""

    if scale <= 0:
        raise ValueError("scale must be greater than zero")
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def grayscale_image(image: Image.Image) -> Image.Image:
    """Return a grayscale copy of ``image`` in mode ``L``."""

    return image.convert("L")


def enhance_contrast(image: Image.Image, factor: float = 2.0) -> Image.Image:
    """Return a copy of ``image`` with configurable contrast enhancement."""

    if factor < 0:
        raise ValueError("factor must be non-negative")
    return ImageEnhance.Contrast(image).enhance(factor)


def threshold_image(image: Image.Image, threshold: int = 128) -> Image.Image:
    """Return a binary copy of a grayscale image using ``threshold``."""

    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be between 0 and 255")
    grayscale = image.convert("L")
    return grayscale.point(lambda pixel: 255 if pixel >= threshold else 0, mode="1")