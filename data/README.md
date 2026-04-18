# Data Module

Dataset generators for federated multi-objective optimization experiments.

## Datasets

| Generator | Description | Conflict Level |
|-----------|-------------|----------------|
| `make_synthetic_federated_regression` | Standard synthetic regression with m objectives | Low (default) |
| `make_high_conflict_federated_regression` | High-conflict regression with controllable conflict_strength | Configurable |
| `make_federated_classification` | Multi-task classification with Non-IID support | N/A |

## Usage

```python
from fedjd.data import make_synthetic_federated_regression, make_high_conflict_federated_regression

# Low conflict
data = make_synthetic_federated_regression(num_clients=10, num_objectives=3)

# High conflict (cosine similarity ≈ -0.96)
data = make_high_conflict_federated_regression(num_clients=10, num_objectives=3, conflict_strength=1.0)
```
