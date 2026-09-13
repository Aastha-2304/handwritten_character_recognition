"""
Unit test: Verifies CTC Decoder with blank collapsing and text-to-label mappings.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.ctc_decoder import CTCDecoder


def test_ctc_decoder():
    charset = "abcdefghijklmnopqrstuvwxyz"
    decoder = CTCDecoder(charset=charset)

    # Convert text to labels and back
    text = "hello"
    labels = decoder.text_to_labels(text)
    decoded = decoder.labels_to_text(labels)
    assert decoded == text, f"Expected '{text}', got '{decoded}'"

    # Simulate logits with repeated characters and blanks:
    # 'h' (idx 8), blank (0), 'e' (idx 5), 'l' (idx 12), 'l' (idx 12), 'o' (idx 15)
    # CTC repeated 'l' without blank would collapse to single 'l'.
    # With a blank in between: [h, e, l, blank, l, o] -> "hello"
    seq = [decoder.char2idx['h'], decoder.char2idx['e'], decoder.char2idx['l'],
           0, decoder.char2idx['l'], decoder.char2idx['o']]
    
    T = len(seq)
    C = decoder.num_classes
    logits = torch.full((1, T, C), -10.0)
    for t, char_idx in enumerate(seq):
        logits[0, t, char_idx] = 10.0  # high confidence

    decoded_texts = decoder.decode_greedy(logits)
    assert decoded_texts[0] == "hello", f"Expected 'hello', got {decoded_texts[0]}"
    print("OK: CTC Decoder test passed.")


if __name__ == "__main__":
    test_ctc_decoder()
