"""
Image transformation utilities:
Resizing with aspect ratio preservation and padding, conversion
to normalized PyTorch tensors expected by CRNN and OCR models.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch


def resize_and_pad(
    image: np.ndarray,
    target_height: int = 32,
    max_width: int = 800,
    pad_value: int = 0,
) -> np.ndarray:
    """
    Rescale an image to a fixed height while preserving aspect ratio,
    then pad horizontally to max_width (or truncate if wider).
    Assumes image is binary or grayscale (2D array).
    """
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return np.full((target_height, max_width), pad_value, dtype=np.uint8)

    scale = target_height / float(h)
    new_w = min(int(w * scale), max_width)

    resized = cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_AREA)

    padded = np.full((target_height, max_width), pad_value, dtype=np.uint8)
    padded[:, :new_w] = resized
    return padded


def to_tensor(
    image: np.ndarray,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Convert a 2D numpy grayscale/binary image (H, W) or 3D (H, W, C)
    into a float PyTorch Tensor (C, H, W) normalized to [0, 1] or [-1, 1].
    """
    if image.ndim == 2:
        img_arr = image[np.newaxis, ...]  # (1, H, W)
    elif image.ndim == 3:
        img_arr = np.transpose(image, (2, 0, 1))  # (C, H, W)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    tensor = torch.from_numpy(img_arr).float()
    if normalize:
        tensor = tensor / 255.0
    return tensor
