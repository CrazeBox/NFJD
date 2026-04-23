# NFJD: New Federated Jacobian Descent

Federated multi-objective optimization with client-side `UPGrad`-based Jacobian descent, communication-efficient `delta_theta` upload, adaptive rescaling, and server-side stabilization.

## Current Mainline

The current canonical NFJD implementation is:

- client-side local `UPGrad`-based Jacobian descent
- `recompute_interval` with reuse of the latest local UPGrad weight vector between recomputation steps
- `StochasticGramianSolver` for high-objective settings
- `AdaptiveRescaling` + local momentum on clients
- `delta_theta + align_scores` communication
- alignment-aware weighted averaging + global momentum on the server

For the current implementation state and future work plan, see:

- `docs/NFJD_Method.md`
- `docs/NFJD_Status_and_Roadmap.md`

## Project Structure

```
fedjd/
├── aggregators/          # Jacobian aggregation (UPGrad, MinNorm, Mean, Random)
├── compressors/          # Jacobian compression
├── core/                 # Core framework
│   ├── nfjd_client.py    # NFJDClient (local UPGrad-JD, Δθ upload, local momentum)
│   ├── nfjd_server.py    # NFJDServer (alignment-aware aggregation + global momentum)
│   ├── nfjd_trainer.py   # NFJDTrainer
│   ├── scaling.py        # AdaptiveRescaling, StochasticGramianSolver, momentum modules
│   ├── phase5_official_baselines.py # Paper-sourced Phase 5 baseline wrappers
│   ├── client.py         # FedJDClient (legacy)
│   ├── server.py         # FedJDServer (legacy)
│   ├── trainer.py        # FedJDTrainer (legacy)
│   └── baselines.py      # FMGDA, WeightedSum, DirectionAvg
├── data/                 # Dataset generators
├── docs/                 # Documentation
│   ├── NFJD_Method.md
│   ├── NFJD_Status_and_Roadmap.md
│   └── NFJD_Experiment_Plans/  # Phase 1-5 plans & checklists
├── experiments/          # Experiment scripts
│   ├── nfjd_phases/      # NFJD Phase 1-6 experiments
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
| Local multi-objective core | Server-side MinNorm | Client-side UPGrad-JD + StochasticGramian |
| Scaling | None | AdaptiveRescaling |
| Server stabilization | Mean update | Alignment-aware weighted average + global momentum |
| Local epochs | 1 | Configurable (default 3) |

## Server Stabilization

The current NFJD mainline uses:

```
1. alignment-aware client reweighting using align_scores
2. weighted averaging of delta_theta
3. global momentum on the aggregated update
```

This is the practical federated version currently recommended for experiments.

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
                    global_momentum_beta=0.9)
trainer = NFJDTrainer(server=server, num_rounds=50)
history = trainer.fit()
```
