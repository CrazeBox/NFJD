from __future__ import annotations

import torch


def two_objective_regression(predictions: torch.Tensor, targets: torch.Tensor, _inputs: torch.Tensor) -> list[torch.Tensor]:
    errors = predictions - targets
    loss_1 = torch.mean(errors[:, 0] ** 2)
    loss_2 = torch.mean(errors[:, 1] ** 2)
    return [loss_1, loss_2]


def multi_objective_regression(predictions: torch.Tensor, targets: torch.Tensor, _inputs: torch.Tensor) -> list[torch.Tensor]:
    errors = predictions - targets
    num_objectives = predictions.shape[1]
    losses = []
    for i in range(num_objectives):
        losses.append(torch.mean(errors[:, i] ** 2))
    return losses
