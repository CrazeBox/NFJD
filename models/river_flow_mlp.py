from __future__ import annotations

import torch
import torch.nn as nn


class RiverFlowMLP(nn.Module):
    def __init__(self, input_dim: int, num_tasks: int = 8,
                 hidden_dim: int = 128) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(64, 1) for _ in range(num_tasks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared(x)
        outputs = torch.cat([head(shared) for head in self.heads], dim=1)
        return outputs
