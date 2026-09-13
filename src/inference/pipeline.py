"""
Full Recognition Pipeline:
Raw Input Image -> Clean & Binarize -> Segment Lines & Words -> Recognize Crops -> Full Text
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.inference.predictor import Predictor
from src.preprocessing.image_ops import clean_image
from src.preprocessing.segmentation import BoundingBox, segment_lines, segment_page


@dataclass
class RecognizedWord:
    box: BoundingBox
    text: str
    confidence: float = 1.0


@dataclass
class RecognizedLine:
    line_box: BoundingBox
    words: list[RecognizedWord]
    text: str


@dataclass
class PipelineResult:
    cleaned_binary: np.ndarray
    lines: list[RecognizedLine]
    full_text: str


class OCRPipeline:
    def __init__(self, engine: str = "trocr", checkpoint_path: Optional[str] = None):
        self.predictor = Predictor(engine=engine, checkpoint_path=checkpoint_path)

    def run(self, raw_image: np.ndarray, segment_level: str = "line") -> PipelineResult:
        """
        Runs complete OCR pipeline on an input image.
        Args:
            raw_image: BGR, RGB, or Grayscale image
            segment_level: 'line' or 'word'
        Returns:
            PipelineResult with cleaned image, bounding boxes, predictions, and combined text.
        """
        # 1. Image preprocessing
        binary = clean_image(raw_image)

        recognized_lines: list[RecognizedLine] = []

        if segment_level == "line":
            # Segment into lines and predict each line directly
            line_boxes = segment_lines(binary)
            for lb in line_boxes:
                crop = lb.crop(binary)
                pred_text = self.predictor.predict_crop(crop)
                rw = RecognizedWord(box=lb, text=pred_text)
                recognized_lines.append(RecognizedLine(line_box=lb, words=[rw], text=pred_text))
        else:
            # Segment into lines, then words per line
            page_words = segment_page(binary)
            line_boxes = segment_lines(binary)

            for i, line_words in enumerate(page_words):
                lb = line_boxes[i] if i < len(line_boxes) else BoundingBox(0, 0, binary.shape[1], binary.shape[0])
                words_res = []
                for wb in line_words:
                    crop = wb.crop(binary)
                    word_text = self.predictor.predict_crop(crop)
                    words_res.append(RecognizedWord(box=wb, text=word_text))
                line_text = " ".join([w.text for w in words_res if w.text])
                recognized_lines.append(RecognizedLine(line_box=lb, words=words_res, text=line_text))

        full_text = "\n".join([line.text for line in recognized_lines if line.text.strip()])
        return PipelineResult(cleaned_binary=binary, lines=recognized_lines, full_text=full_text)
