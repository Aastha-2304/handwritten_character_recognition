"""
Smoke test: builds a synthetic 'page' with two lines of fake text
(rectangles standing in for words) and checks that segmentation
recovers the right number of lines and words per line.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.image_ops import clean_image
from src.preprocessing.segmentation import segment_page


def draw_word(page: np.ndarray, x: int, y: int, n_letters: int, letter_w: int = 12,
              letter_gap: int = 4, h: int = 25) -> int:
    """Draw a fake 'word' as several small letter-blobs with small
    gaps between them (like real handwriting), returning the x
    position right after the word.
    """
    cx = x
    for _ in range(n_letters):
        cv2.rectangle(page, (cx, y), (cx + letter_w, y + h), (0, 0, 0), -1)
        cx += letter_w + letter_gap
    return cx - letter_gap  # end of last letter, no trailing gap


def make_synthetic_page() -> np.ndarray:
    page = np.full((220, 600, 3), 255, dtype=np.uint8)  # white paper

    word_gap = 35  # much bigger than the 4px inter-letter gap above

    # Line 1: 3 words of varying letter-count
    x = 20
    x = draw_word(page, x, 30, n_letters=4)      # word 1
    x = draw_word(page, x + word_gap, 30, n_letters=5)  # word 2
    draw_word(page, x + word_gap, 30, n_letters=3)      # word 3

    # Line 2: 2 words
    x = 20
    x = draw_word(page, x, 130, n_letters=6)
    draw_word(page, x + word_gap, 130, n_letters=4)

    return page


def main():
    page = make_synthetic_page()
    binary = clean_image(page)
    result = segment_page(binary)

    print(f"Lines detected: {len(result)}")
    for i, words in enumerate(result):
        print(f"  Line {i}: {len(words)} word(s) -> {[ (w.x, w.w) for w in words ]}")

    assert len(result) == 2, f"expected 2 lines, got {len(result)}"
    assert len(result[0]) == 3, f"expected 3 words in line 0, got {len(result[0])}"
    assert len(result[1]) == 2, f"expected 2 words in line 1, got {len(result[1])}"
    print("OK: preprocessing + segmentation smoke test passed.")


if __name__ == "__main__":
    main()