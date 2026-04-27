from __future__ import annotations

import torch
import torch.nn as nn


class FEMNISTCNN(nn.Module):
    """Small CNN for FEMNIST/EMNIST-style 28x28 grayscale character inputs."""

    def __init__(self, num_tasks: int = 1, num_classes: int = 62) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
        )
        self.heads = nn.ModuleList([nn.Linear(512, num_classes) for _ in range(num_tasks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        shared = self.shared(features)
        return torch.stack([head(shared) for head in self.heads], dim=1)
