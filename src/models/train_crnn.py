"""
Training pipeline for the CRNN handwritten text recognition model.
Uses synthetic or IAM line/word datasets with PyTorch CTCLoss and AdamW optimizer.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.crnn import CRNN
from src.models.ctc_decoder import CTCDecoder
from src.preprocessing.transforms import resize_and_pad, to_tensor
from src.utils.config import (
    BATCH_SIZE,
    CHARSET,
    CHECKPOINTS_DIR,
    IMG_HEIGHT,
    IMG_MAX_WIDTH,
    LEARNING_RATE,
    NUM_EPOCHS,
)
from src.utils.metrics import character_error_rate


class DummyOCRDataset(Dataset):
    """
    Synthetic dataset generator for testing training loop without external downloads.
    Generates images of random words.
    """
    def __init__(self, size: int = 100):
        self.size = size
        self.decoder = CTCDecoder(charset=CHARSET)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        import numpy as np
        # random grayscale dummy image
        img = np.random.randint(0, 255, (IMG_HEIGHT, 150), dtype=np.uint8)
        padded = resize_and_pad(img, target_height=IMG_HEIGHT, max_width=IMG_MAX_WIDTH)
        tensor = to_tensor(padded)
        label = "test"
        return tensor, label


def collate_fn(batch):
    decoder = CTCDecoder(charset=CHARSET)
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)

    target_lengths = torch.tensor([len(decoder.text_to_labels(lbl)) for lbl in labels], dtype=torch.long)
    targets = []
    for lbl in labels:
        targets.extend(decoder.text_to_labels(lbl))
    targets = torch.tensor(targets, dtype=torch.long)

    return images, targets, target_lengths, labels


def train(epochs: int = NUM_EPOCHS, lr: float = LEARNING_RATE, batch_size: int = BATCH_SIZE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    decoder = CTCDecoder(charset=CHARSET)
    model = CRNN(img_channel=1, num_classes=decoder.num_classes).to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    dataset = DummyOCRDataset(size=50)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, targets, target_lengths, _ in loader:
            images = images.to(device)
            optimizer.zero_grad()

            logits = model(images)  # (T, B, C)
            T, B, _ = logits.size()
            input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long)

            log_probs = logits.log_softmax(2)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        print(f"Epoch [{epoch+1}/{epochs}] - CTC Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = CHECKPOINTS_DIR / "crnn_best.pt"
            torch.save(model.state_dict(), ckpt_path)

    print("Training finished. Best checkpoint saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()
    train(epochs=args.epochs, lr=args.lr)
