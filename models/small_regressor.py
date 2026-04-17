from __future__ import annotations

from torch import nn


class SmallRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16, output_dim: int = 2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs):
        return self.network(inputs)


class MediumRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs):
        return self.network(inputs)


class LargeRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, inputs):
        return self.network(inputs)


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "small": SmallRegressor,
    "medium": MediumRegressor,
    "large": LargeRegressor,
}
