# NFJD: New Federated Jacobian Descent

Federated multi-objective optimization with communication-efficient Δθ upload, adaptive rescaling, and conflict-aware momentum.

## Project Structure

```
fedjd/
├── aggregators/          # Jacobian aggregation (MinNorm, Mean, Random)
├── compressors/          # Jacobian compression
├── core/                 # Core framework
│   ├── nfjd_client.py    # NFJDClient (Δθ upload, ConflictAwareMomentum, AdaptiveRescaling)
│   ├── nfjd_server.py    # NFJDServer (FedAvg + ConflictAware GlobalMomentum)
│   ├── nfjd_trainer.py   # NFJDTrainer
│   ├── scaling.py        # AdaptiveRescaling, StochasticGramianSolver, ConflictAwareMomentum
│   ├── client.py         # FedJDClient (legacy)
│   ├── server.py         # FedJDServer (legacy)
│   ├── trainer.py        # FedJDTrainer (legacy)
│   └── baselines.py      # FMGDA, WeightedSum, DirectionAvg
├── data/                 # Dataset generators
├── docs/                 # Documentation
│   ├── NFJD_Architecture_Design.md
│   └── NFJD_Experiment_Plans/  # Phase 1-4 plans & checklists
├── experiments/          # Experiment scripts
│   ├── nfjd_phases/      # NFJD Phase 1-4 experiments
│   └── fedjd_legacy/     # Legacy FedJD Stage 1-5 scripts
├── metrics/              # Hypervolume, Pareto front, Pareto gap
├── models/               # Small/Medium/Large regressor, classifier
├── problems/             # Objective functions
├── config.py             # ExperimentConfig
└── visualization.py      # Training curve plotting
```

## Key Innovations (NFJD vs FedJD)

| Feature | FedJD | NFJD |
|---------|-------|------|
| Upload content | Jacobian (m×d) | Δθ (d-dim) |
| Communication vs m | O(m×d) | O(d), decoupled from m |
| Direction finding | Server-side MinNorm | Client-side MinNorm + StochasticGramian |
| Scaling | None | AdaptiveRescaling |
| Momentum | None | ConflictAwareMomentum (auto-adjusts β by gradient consistency) |
| Local epochs | 1 | Configurable (default 3) |

## ConflictAwareMomentum

Dynamic momentum coefficient based on gradient consistency:

```
avg_cosine_sim = mean of pairwise cosine similarities of objective gradients
effective_beta = max(base_beta * (1.0 - avg_cosine_sim), min_beta)
```

- High consistency (sim→1) → low β → less inertia, faster convergence
- High conflict (sim→0) → high β → momentum smooths conflicting directions

## Quick Start

```python
import torch
from fedjd.core import NFJDClient, NFJDServer, NFJDTrainer
from fedjd.data import make_synthetic_federated_regression
from fedjd.problems import multi_objective_regression
from fedjd.models import SmallRegressor

data = make_synthetic_federated_regression(num_clients=10, num_objectives=3)
model = SmallRegressor(input_dim=8, output_dim=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clients = [NFJDClient(client_id=i, dataset=data.client_datasets[i], batch_size=32,
                       device=device, conflict_aware_momentum=True) for i in range(10)]
server = NFJDServer(model=model, clients=clients, objective_fn=multi_objective_regression,
                    participation_rate=0.5, learning_rate=0.01, device=device,
                    conflict_aware_momentum=True)
trainer = NFJDTrainer(server=server, num_rounds=50)
history = trainer.fit()
```
