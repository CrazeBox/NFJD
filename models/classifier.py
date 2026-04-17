from __future__ import annotations

import torch
from torch import nn


class MultiTaskClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 10, num_tasks: int = 2) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_tasks)
        ])

    def forward(self, inputs):
        shared_features = self.shared(inputs)
        return torch.stack([head(shared_features) for head in self.heads], dim=1)
