from __future__ import annotations

import torch
from torch import nn


class MeanPooledTextClassifier(nn.Module):
    """Lightweight Sent140 classifier for federated sweeps.

    The model intentionally stays small: an embedding table, masked mean pooling,
    and a two-layer MLP binary classifier. It is fast enough for client-level FL
    pilot runs while still learning user-specific vocabulary differences.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        padding_idx: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens.long())
        mask = (tokens != self.padding_idx).unsqueeze(-1).to(embedded.dtype)
        summed = (embedded * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = summed / denom
        return self.classifier(pooled)
