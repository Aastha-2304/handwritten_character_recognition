"""
CTC (Connectionist Temporal Classification) Decoders:
Includes greedy best-path decoding and character dictionary mapping.
Handles blank-token collapsing and repeated characters.
"""
from __future__ import annotations

import torch
from src.utils.config import CHARSET


class CTCDecoder:
    def __init__(self, charset: str = CHARSET, blank_idx: int = 0):
        """
        Index 0 is reserved for CTC blank (<blank>).
        Indices 1 .. len(charset) map to characters in charset.
        """
        self.charset = charset
        self.blank_idx = blank_idx
        self.idx2char = {idx + 1: ch for idx, ch in enumerate(charset)}
        self.char2idx = {ch: idx + 1 for idx, ch in enumerate(charset)}

    @property
    def num_classes(self) -> int:
        return len(self.charset) + 1  # includes blank

    def decode_greedy(self, logits: torch.Tensor) -> list[str]:
        """
        Greedy / Best-Path CTC Decoding.
        Args:
            logits: Tensor of shape (T, B, C) or (B, T, C)
        Returns:
            List of decoded text strings of length B.
        """
        # Ensure logits shape has class dimension last: shape becomes (*, C)
        if logits.dim() == 3 and logits.size(1) == self.num_classes and logits.size(-1) != self.num_classes:
            logits = logits.permute(0, 2, 1)

        # In PyTorch, CRNN output is (T, B, C). If dim 0 is time (T) and dim 1 is batch (B),
        # permute to (B, T, C). We know CRNN outputs (T, B, C).
        # If the caller provides (B, T, C), we want (B, T, C).
        # We can accept an explicit convention or infer:
        # If dim(0) > dim(1) and dim(1) <= 64 (typical batch size), or if caller passes CRNN output:
        # Best convention: CRNN outputs (T, B, C).
        if logits.dim() == 3:
            # If shape is (T, B, C) where T != B:
            # We can detect if it's (T, B, C) by checking if T > B or if B is batch
            # If batch is first, (B, T, C).
            pass

        # argmax over class dimension
        best_path = torch.argmax(logits, dim=-1)  # shape could be (B, T) or (T, B)

        # If best_path is (T, B) where T is time and B is batch, we transpose to (B, T)
        # For CRNN output (T, B, C), best_path is (T, B). If B < T (typical in OCR where T ~ 100):
        if best_path.dim() == 2 and best_path.size(0) > best_path.size(1):
            # Transpose to (B, T)
            best_path = best_path.t()

        batch_size = best_path.size(0)
        decoded_texts = []
        for b in range(batch_size):
            seq = best_path[b].tolist()
            res = []
            prev_idx = None
            for idx in seq:
                if idx != prev_idx:
                    if idx != self.blank_idx and idx in self.idx2char:
                        res.append(self.idx2char[idx])
                    prev_idx = idx
            decoded_texts.append("".join(res))

        return decoded_texts

    def text_to_labels(self, text: str) -> list[int]:
        """Convert a ground-truth string into integer labels for CTC loss."""
        return [self.char2idx[ch] for ch in text if ch in self.char2idx]

    def labels_to_text(self, labels: list[int]) -> str:
        """Convert integer labels directly back to text."""
        return "".join([self.idx2char.get(i, "") for i in labels if i != self.blank_idx])
