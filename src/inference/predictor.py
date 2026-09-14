"""
Predictor class that manages models and runs recognition on line/word crops.
Can switch between 'crnn', 'trocr', or a heuristic fallback.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from src.models.crnn import CRNN
from src.models.ctc_decoder import CTCDecoder
from src.models.trocr_wrapper import TrOCRWrapper
from src.preprocessing.transforms import resize_and_pad, to_tensor
from src.utils.config import CHARSET, CHECKPOINTS_DIR, IMG_HEIGHT, IMG_MAX_WIDTH


class Predictor:
    def __init__(self, engine: str = "trocr", checkpoint_path: Optional[Path | str] = None):
        """
        Args:
            engine: 'trocr' or 'crnn'
            checkpoint_path: Path to CRNN weights (.pt or .pth)
        """
        self.engine = engine
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else CHECKPOINTS_DIR / "crnn_best.pt"
        self.ctc_decoder = CTCDecoder(charset=CHARSET)
        self.crnn_model = None
        self.trocr = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_engine()

    def _init_engine(self):
        if self.engine == "trocr":
            self.trocr = TrOCRWrapper(device=self.device)
        elif self.engine == "crnn":
            self._load_crnn()

    def _load_crnn(self):
        try:
            import torch
            self.crnn_model = CRNN(img_channel=1, num_classes=self.ctc_decoder.num_classes).to(self.device)
            if self.checkpoint_path.exists():
                state_dict = torch.load(self.checkpoint_path, map_location=self.device)
                self.crnn_model.load_state_dict(state_dict)
            self.crnn_model.eval()
        except Exception as e:
            print(f"Warning: Could not load CRNN ({e}).")

    def predict_crop(self, crop: np.ndarray) -> str:
        """Predict text from a single crop (word or line)."""
        if crop is None or crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return ""

        if self.engine == "trocr":
            if self.trocr is None:
                self.trocr = TrOCRWrapper(device=self.device)
            return self.trocr.predict(crop)

        elif self.engine == "crnn" and self.crnn_model is not None:
            import torch
            padded = resize_and_pad(crop, target_height=IMG_HEIGHT, max_width=IMG_MAX_WIDTH)
            tensor = to_tensor(padded).unsqueeze(0).to(self.device)  # (1, 1, 32, max_w)
            with torch.no_grad():
                logits = self.crnn_model(tensor)
                decoded = self.ctc_decoder.decode_greedy(logits)
                return decoded[0] if decoded else ""

        return ""

