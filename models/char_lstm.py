from __future__ import annotations

import torch
from torch import nn


class CharLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 32, hidden_dim: int = 128) -> None:
        super().__init__()
        self.shared = nn.Sequential()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(inputs.long())
        outputs, _ = self.lstm(embedded)
        last_hidden = outputs[:, -1]
        return self.classifier(last_hidden)
