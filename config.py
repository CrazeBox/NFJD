from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    # Experiment identity
    experiment_id: str = "S1-synth_m2-mlp-m2-K8-C0.5-E1-fulljac-v1"
    seed: int = 7

    # Data
    num_clients: int = 8
    samples_per_client: int = 64
    input_dim: int = 8
    noise_std: float = 0.1

    # Model
    hidden_dim: int = 16
    output_dim: int = 2

    # Training
    num_rounds: int = 30
    batch_size: int = 32
    learning_rate: float = 0.05
    participation_rate: float = 0.5

    # Aggregator
    aggregator: str = "minnorm"  # "minnorm", "mean", "random"
    aggregator_max_iters: int = 250
    aggregator_lr: float = 0.2

    # Output
    output_dir: str = ""
    save_checkpoints: bool = False
    checkpoint_interval: int = 10

    # Device
    device: str = "auto"

    def get_output_dir(self) -> Path:
        if self.output_dir:
            return Path(self.output_dir)
        return Path("results") / self.experiment_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "seed": self.seed,
            "num_clients": self.num_clients,
            "samples_per_client": self.samples_per_client,
            "input_dim": self.input_dim,
            "noise_std": self.noise_std,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_rounds": self.num_rounds,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "participation_rate": self.participation_rate,
            "aggregator": self.aggregator,
            "aggregator_max_iters": self.aggregator_max_iters,
            "aggregator_lr": self.aggregator_lr,
            "output_dir": self.output_dir,
            "save_checkpoints": self.save_checkpoints,
            "checkpoint_interval": self.checkpoint_interval,
            "device": self.device,
        }

    def save_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
