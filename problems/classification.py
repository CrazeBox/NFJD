from __future__ import annotations

import torch


def multi_task_classification(predictions: torch.Tensor, targets: torch.Tensor, _inputs: torch.Tensor) -> list[torch.Tensor]:
    num_tasks = predictions.shape[1]
    losses = []
    for t in range(num_tasks):
        task_logits = predictions[:, t]
        task_labels = targets[:, t].long()
        log_probs = torch.log_softmax(task_logits, dim=-1)
        loss = torch.nn.functional.nll_loss(log_probs, task_labels, reduction="mean")
        losses.append(loss)
    return losses
