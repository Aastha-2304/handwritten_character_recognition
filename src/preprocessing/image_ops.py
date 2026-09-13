"""
Low-level image cleanup operations applied before segmentation:
grayscale conversion, denoising, adaptive thresholding, and deskewing.

These work on a single input image (a canvas drawing or an uploaded
photo of a handwritten note) and return a clean binary image ready
for line/word segmentation.
"""
from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR/RGB/RGBA image to single-channel grayscale.
    No-op if already grayscale.
    """
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def denoise(gray: np.ndarray) -> np.ndarray:
    """Remove speckle/scanner noise while preserving stroke edges."""
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


def binarize(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive threshold -> binary image with WHITE ink (255) on
    BLACK background (0). This orientation is what the segmentation
    functions and the CRNN model expect.
    """
    # Normalize illumination if uneven
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Adaptive Gaussian thresholding with tuned block size
    # A block size of 31-41 is optimal for camera phone handwriting photos
    block_size = 35
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # ink -> white, background -> black
        blockSize=block_size,
        C=8,
    )
    return binary


def deskew(binary: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """
    Estimate and correct small rotation angles using the minimum-area
    bounding box of ink pixels. Only corrects subtle slants (< max_angle).
    """
    coords = cv2.findNonZero(binary)
    if coords is None or len(coords) < 50:
        return binary

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    # cv2.minAreaRect angle convention normalization:
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    # Only deskew if the tilt is minor and within realistic notebook skew range
    if abs(angle) < 0.8 or abs(angle) > max_angle:
        return binary  # do not distort large vertical drawings or severe rotations

    (h, w) = binary.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary, matrix, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated


def clean_image(image: np.ndarray) -> np.ndarray:
    """
    Full cleanup pipeline: raw input image -> deskewed binary image
    with white ink on black background, ready for segmentation.
    """
    gray = to_grayscale(image)
    gray = denoise(gray)
    binary = binarize(gray)
    binary = deskew(binary)
    return binary