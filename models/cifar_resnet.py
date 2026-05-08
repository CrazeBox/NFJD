from __future__ import annotations

import torch
import torch.nn as nn


class CIFARResNet18MTL(nn.Module):
    """ResNet-18 adapted for 32x32 CIFAR-style inputs.

    Uses a 3x3 stride-1 first convolution and removes the ImageNet max-pool.
    BatchNorm is replaced with GroupNorm, which is more stable for federated
    non-IID training because it has no unaggregated running statistics.
    The output is shaped as [batch, num_tasks, num_classes] to match the
    repository's multi-task classification loss.
    """

    def __init__(self, num_tasks: int = 1, num_classes: int = 10, group_count: int = 32) -> None:
        super().__init__()
        try:
            from torchvision.models import resnet18
        except ModuleNotFoundError as exc:
            raise ImportError("CIFARResNet18MTL requires torchvision.") from exc

        def norm_layer(channels: int) -> nn.GroupNorm:
            groups = min(group_count, channels)
            while channels % groups != 0:
                groups -= 1
            return nn.GroupNorm(groups, channels)

        backbone = resnet18(weights=None, norm_layer=norm_layer)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.heads = nn.ModuleList([nn.Linear(feature_dim, num_classes) for _ in range(num_tasks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return torch.stack([head(features) for head in self.heads], dim=1)
