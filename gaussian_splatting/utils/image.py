import os

import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def is_image(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def stack_images_horizontally(
    left_image: np.ndarray,
    right_image: np.ndarray,
) -> np.ndarray:
    """Stack two HWC images side-by-side after checking shape compatibility."""
    if left_image.ndim != 3 or right_image.ndim != 3:
        raise ValueError("Both images must be HWC tensors with 3 dimensions.")

    if left_image.shape[0] != right_image.shape[0]:
        raise ValueError(f"Image heights must match. Got {left_image.shape[0]} and {right_image.shape[0]}.")

    if left_image.shape[2] != right_image.shape[2]:
        raise ValueError(f"Image channel counts must match. Got {left_image.shape[2]} and {right_image.shape[2]}.")

    if left_image.dtype != right_image.dtype:
        right_image = right_image.astype(left_image.dtype)

    return np.concatenate([left_image, right_image], axis=1)
