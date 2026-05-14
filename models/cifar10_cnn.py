from __future__ import annotations

import torch
import torch.nn as nn


class FedMGDAPlusCIFAR10CNN(nn.Module):
    """CIFAR-10 CNN from FedMGDA+ Table 2."""

    def __init__(self, num_tasks: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LocalResponseNorm(size=2),
            nn.Conv2d(64, 64, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
        )
        self.heads = nn.ModuleList([nn.Linear(192, num_classes) for _ in range(num_tasks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        shared = self.shared(features)
        return torch.stack([head(shared) for head in self.heads], dim=1)
