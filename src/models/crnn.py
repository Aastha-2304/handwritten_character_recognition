"""
CRNN (Convolutional Recurrent Neural Network) for Handwritten Text Recognition.
Architecture:
1. Standard VGG-style CNN feature extractor: extracts spatial representations.
2. Map-to-Sequence layer: converts CNN features into time-step sequence.
3. Bidirectional GRU/LSTM recurrent layers: models context across sequential characters.
4. Linear projection: outputs logits over CHARSET + CTC blank token.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(
        self,
        img_channel: int = 1,
        num_classes: int = 80,
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.num_classes = num_classes

        # CNN Backbone
        # Input: (B, img_channel, 32, W)
        self.cnn = nn.Sequential(
            # Conv 1
            nn.Conv2d(img_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> (B, 64, 16, W/2)

            # Conv 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> (B, 128, 8, W/4)

            # Conv 3 & 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # Pool: asymmetric (2, 1) to preserve horizontal resolution
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # -> (B, 256, 4, W/4)

            # Conv 5 & 6
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # -> (B, 512, 2, W/4)

            # Conv 7
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),  # -> (B, 512, 1, (W/4) - 1)
        )

        # Recurrent Context Network (Bidirectional GRU)
        self.rnn = nn.GRU(
            input_size=512,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=False,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )

        # Output Classifier
        # Bidirectional GRU produces 2 * rnn_hidden
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (Batch, Channels, Height=32, Width)
        Output: (TimeSteps, Batch, NumClasses) ready for nn.CTCLoss
        """
        # Feature extraction
        features = self.cnn(x)  # (B, 512, 1, T)

        # Map to Sequence
        b, c, h, t = features.size()
        assert h == 1, f"Expected height to be 1 after CNN, got {h}"
        features = features.squeeze(2)          # (B, 512, T)
        features = features.permute(2, 0, 1)    # (T, B, 512) -> Sequence first

        # Recurrent layers
        recurrent_out, _ = self.rnn(features)   # (T, B, 2 * rnn_hidden)

        # Linear classification
        logits = self.fc(recurrent_out)         # (T, B, num_classes)
        return logits
