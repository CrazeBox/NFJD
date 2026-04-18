# Aggregators

Jacobian aggregation strategies for multi-objective direction finding.

## Available Aggregators

| Aggregator | Description | Direction |
|------------|-------------|-----------|
| MinNormAggregator | MGDA-style minimum-norm direction via projected gradient descent on simplex | `J^T λ*` where `λ* = argmin ||J^T λ||₂` |
| MeanAggregator | Simple average of all objective gradients | `mean(J, dim=0)` |
| RandomAggregator | Random simplex weighting | `J^T w` where `w ~ Dir(1)` |
