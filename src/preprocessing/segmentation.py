"""
Segments a cleaned binary page image (white ink on black background,
see image_ops.clean_image) into individual text lines, and each line
into individual words.

Approach:
- Line segmentation: horizontal projection profile. Sum ink pixels
  per row; rows above a low threshold are "inside a line", runs of
  such rows are line bands.
- Word segmentation: within a line, vertical projection profile with
  a gap-width heuristic to tell inter-word gaps from inter-letter gaps.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.y:self.y + self.h, self.x:self.x + self.w]


def _find_runs(mask: np.ndarray, min_gap: int) -> list[tuple[int, int]]:
    """
    Given a 1D boolean array, find (start, end) index ranges of
    contiguous True runs, merging runs separated by gaps smaller
    than min_gap (handles noisy rows/columns inside a single line/word).
    """
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, val in enumerate(mask):
        if val and not in_run:
            start = i
            in_run = True
        elif not val and in_run:
            runs.append((start, i))
            in_run = False
    if in_run:
        runs.append((start, len(mask)))

    if not runs:
        return runs

    merged = [runs[0]]
    for s, e in runs[1:]:
        prev_s, prev_e = merged[-1]
        if s - prev_e < min_gap:
            merged[-1] = (prev_s, e)
        else:
            merged.append((s, e))
    return merged


def segment_lines(
    binary: np.ndarray,
    row_threshold_ratio: float = 0.02,
    min_line_gap: int = 6,
    min_line_height: int = 8,
) -> list[BoundingBox]:
    """
    Split a page into line-level bounding boxes using a horizontal
    projection profile (row-wise ink pixel counts).
    """
    h, w = binary.shape
    row_sums = binary.sum(axis=1) / 255  # count of ink pixels per row
    threshold = max(1, int(row_threshold_ratio * w))
    row_has_ink = row_sums > threshold

    runs = _find_runs(row_has_ink, min_gap=min_line_gap)

    boxes = []
    for y0, y1 in runs:
        if (y1 - y0) < min_line_height:
            continue
        boxes.append(BoundingBox(x=0, y=y0, w=w, h=y1 - y0))
    return boxes


def segment_words(
    line_image: np.ndarray,
    col_threshold_ratio: float = 0.01,
    word_gap_multiplier: float = 2.5,
    min_word_width: int = 5,
) -> list[BoundingBox]:
    """
    Split a single line image into word-level bounding boxes using a
    vertical projection profile. The gap between words is estimated
    as word_gap_multiplier times the median inter-character gap, so
    this adapts to each writer's spacing instead of a fixed pixel gap.
    """
    h, w = line_image.shape
    col_sums = line_image.sum(axis=0) / 255
    threshold = max(1, int(col_threshold_ratio * h))
    col_has_ink = col_sums > threshold

    # First pass: tight runs (roughly per-character/stroke clusters)
    tight_runs = _find_runs(col_has_ink, min_gap=1)
    if len(tight_runs) < 2:
        gaps = []
    else:
        gaps = [
            tight_runs[i + 1][0] - tight_runs[i][1]
            for i in range(len(tight_runs) - 1)
        ]

    median_gap = float(np.median(gaps)) if gaps else 4.0
    word_gap = max(min_word_width, int(median_gap * word_gap_multiplier))

    runs = _find_runs(col_has_ink, min_gap=word_gap)

    boxes = []
    for x0, x1 in runs:
        if (x1 - x0) < min_word_width:
            continue
        boxes.append(BoundingBox(x=x0, y=0, w=x1 - x0, h=h))
    return boxes


def segment_page(binary: np.ndarray) -> list[list[BoundingBox]]:
    """
    Full segmentation: page -> lines -> words per line.
    Returns a list (one entry per line) of lists of word BoundingBoxes,
    with word boxes given in page coordinates (not line-local).
    """
    line_boxes = segment_lines(binary)
    page_words: list[list[BoundingBox]] = []

    for line_box in line_boxes:
        line_img = line_box.crop(binary)
        word_boxes_local = segment_words(line_img)
        # shift word x/y back into page coordinates
        word_boxes_page = [
            BoundingBox(x=line_box.x + wb.x, y=line_box.y, w=wb.w, h=line_box.h)
            for wb in word_boxes_local
        ]
        page_words.append(word_boxes_page)

    return page_words