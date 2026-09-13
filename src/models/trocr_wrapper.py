"""
Wrapper for Microsoft TrOCR (Transformer-based Optical Character Recognition).
Handles loading the HuggingFace VisionEncoderDecoderModel, TrOCRProcessor,
device allocation (GPU/CPU), and generating predictions on image crops.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PIL import Image

from src.utils.config import TROCR_MODEL_NAME

logger = logging.getLogger(__name__)


class TrOCRWrapper:
    def __init__(self, model_name: str = TROCR_MODEL_NAME, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.processor = None
        self.model = None
        self._is_loaded = False

    def load(self):
        """Loads TrOCR model and processor if not already in memory."""
        if self._is_loaded:
            return

        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"Loading TrOCR model: {self.model_name} on {self.device}...")
            try:
                self.processor = TrOCRProcessor.from_pretrained(self.model_name)
            except Exception:
                from transformers import AutoImageProcessor, XLMRobertaTokenizer
                image_processor = AutoImageProcessor.from_pretrained(self.model_name)
                tokenizer = XLMRobertaTokenizer.from_pretrained(self.model_name)
                self.processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            self._is_loaded = True
            logger.info("TrOCR loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load TrOCR model ({e}). Using fallback recognizer.")
            self._is_loaded = False

    def predict(self, image: np.ndarray | Image.Image) -> str:
        """
        Recognize text from an image (line or word crop).
        Args:
            image: numpy array or PIL Image
        Returns:
            Decoded string
        """
        if not self._is_loaded:
            self.load()

        if not self._is_loaded or self.model is None:
            # Fallback if transformers/torch weights are not yet cached
            return ""

        import torch

        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                # Convert 1-channel to RGB for TrOCR Vision Encoder
                image = np.stack([image] * 3, axis=-1)
            image = Image.fromarray(image)

        try:
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    max_new_tokens=32,
                    num_beams=1,
                    do_sample=False,
                )
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_text.strip()
        except Exception as e:
            logger.warning(f"TrOCR prediction error: {e}")
            return ""
