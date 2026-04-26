from __future__ import annotations

import torch
import torch.nn as nn


class BasicCNNMTL(nn.Module):
    """Compact multi-head CNN for 36x36 MultiMNIST images."""

    def __init__(self, input_channels: int = 1, num_tasks: int = 2,
                 num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
        )
        self.heads = nn.ModuleList([
            nn.Linear(256, num_classes) for _ in range(num_tasks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            side = int(x.shape[1] ** 0.5)
            x = x.view(x.shape[0], 1, side, side)
        features = self.features(x)
        shared = self.shared(features)
        return torch.stack([head(shared) for head in self.heads], dim=1)
