from __future__ import annotations

import torch


def make_client_level_next_char_objective(num_clients: int):
    def objective(predictions: torch.Tensor, targets: torch.Tensor, _inputs: torch.Tensor) -> list[torch.Tensor]:
        labels = targets[:, 0].long()
        client_ids = targets[:, 1].long()
        zero = predictions.sum() * 0.0
        losses = []
        for client_id in range(num_clients):
            mask = client_ids == client_id
            if torch.any(mask):
                losses.append(torch.nn.functional.cross_entropy(predictions[mask], labels[mask], reduction="mean"))
            else:
                losses.append(zero)
        return losses

    return objective
