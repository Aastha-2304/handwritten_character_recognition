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
    # Otsu gives a good global threshold for fairly uniform lighting;
    # adaptive handles uneven lighting/shadows in photographed notes.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # ink -> white, background -> black
        blockSize=25,
        C=10,
    )
    return binary


def deskew(binary: np.ndarray) -> np.ndarray:
    """
    Estimate and correct small rotation angles using the minimum-area
    bounding box of ink pixels. Handwritten lines are rarely perfectly
    horizontal; this measurably helps line segmentation accuracy.
    """
    coords = cv2.findNonZero(binary)
    if coords is None:
        return binary

    angle = cv2.minAreaRect(coords)[-1]
    # cv2.minAreaRect angle convention: normalize to [-45, 45]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5:
        return binary  # not worth correcting

    (h, w) = binary.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary, matrix, (w, h),
        flags=cv2.INTER_CUBIC,
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