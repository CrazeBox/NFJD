from __future__ import annotations

import torch
import torch.nn as nn


class LeNetMTL(nn.Module):
    def __init__(self, input_channels: int = 1, num_tasks: int = 2,
                 num_classes: int = 10) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(input_channels, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten_dim = 64 * 9 * 9
        self.fc_shared = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(256, num_classes) for _ in range(num_tasks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            s = int(x.shape[1] ** 0.5)
            x = x.view(x.shape[0], 1, s, s)
        shared = self.shared(x)
        shared = shared.view(shared.shape[0], -1)
        shared = self.fc_shared(shared)
        outputs = torch.stack([head(shared) for head in self.heads], dim=1)
        return outputs
