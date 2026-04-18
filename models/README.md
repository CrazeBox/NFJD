# Models

Neural network architectures for federated multi-objective optimization.

## Regression Models

| Model | Hidden Dims | Parameters (input=8, output=2) | Usage |
|-------|-------------|-------------------------------|-------|
| SmallRegressor | 16 | ~162 | Quick experiments, Phase 1-3 |
| MediumRegressor | 64→64 | ~4,290 | Full benchmark, Phase 4 |
| LargeRegressor | 256→256→128 | ~67,074 | Stress testing |

## Classification Model

| Model | Description |
|-------|-------------|
| MultiTaskClassifier | Shared backbone + per-task heads |

## Usage

```python
from fedjd.models import SmallRegressor, MediumRegressor, MultiTaskClassifier

model = SmallRegressor(input_dim=8, output_dim=3)
classifier = MultiTaskClassifier(input_dim=64, num_classes=10, num_tasks=2)
```
