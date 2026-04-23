from __future__ import annotations

import torch
from torch.utils.data import DataLoader


def evaluate_objectives_on_dataset(model, dataset, objective_fn, device, batch_size: int = 256) -> list[float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    running = None
    total_weight = 0

    model.eval()
    with torch.no_grad():
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            predictions = model(batch_inputs)
            values = objective_fn(predictions, batch_targets, batch_inputs)
            stacked = torch.stack([value.detach() for value in values])
            batch_values = stacked.reshape(stacked.shape[0], -1).mean(dim=1)
            batch_weight = int(batch_inputs.shape[0])

            if running is None:
                running = batch_values * batch_weight
            else:
                running.add_(batch_values, alpha=batch_weight)
            total_weight += batch_weight
    model.train()

    if running is None or total_weight == 0:
        raise RuntimeError("No samples available for evaluation.")

    return [float(value.item()) for value in (running / total_weight)]


__all__ = ["evaluate_objectives_on_dataset"]
