# Metrics

Evaluation metrics for multi-objective optimization.

## Available Metrics

| Metric | Description | Usage |
|--------|-------------|-------|
| `hypervolume` | 2D exact / m-D Monte Carlo hypervolume indicator | Pareto front quality |
| `extract_pareto_front` | Extract non-dominated points from objective history | Pareto analysis |
| `pareto_gap` | Distance from final objectives to ideal point | Convergence quality |
| `is_pareto_dominated` | Check if point a is Pareto-dominated by point b | Pareto comparison |
