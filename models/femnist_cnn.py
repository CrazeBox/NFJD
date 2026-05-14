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


class FedMGDAPlusFEMNISTCNN(nn.Module):
    """Federated EMNIST CNN from Hu et al. FedMGDA+ Table 4.

    The parameter counts match the paper architecture: Conv(32, 3x3),
    Conv(64, 3x3), MaxPool, Dropout(0.25), Dense(128), Dropout(0.5), Dense(62).
    """

    def __init__(self, num_tasks: int = 1, num_classes: int = 62) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.25),
        )
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.heads = nn.ModuleList([nn.Linear(128, num_classes) for _ in range(num_tasks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        shared = self.shared(features)
        return torch.stack([head(shared) for head in self.heads], dim=1)
