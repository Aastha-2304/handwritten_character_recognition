"""
Smoke test: Verifies CRNN model shapes and forward pass.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.crnn import CRNN


def test_crnn():
    batch_size = 2
    channels = 1
    height = 32
    width = 400
    num_classes = 80

    model = CRNN(img_channel=channels, num_classes=num_classes)
    dummy_input = torch.randn(batch_size, channels, height, width)

    output = model(dummy_input)

    # Output shape should be (TimeSteps, BatchSize, NumClasses)
    print(f"CRNN Output shape: {output.shape}")
    assert output.dim() == 3
    assert output.size(1) == batch_size
    assert output.size(2) == num_classes
    print("OK: CRNN forward pass test passed.")


if __name__ == "__main__":
    test_crnn()
