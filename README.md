# NFJD: New Federated Jacobian Descent

Federated multi-objective optimization framework with communication-efficient Δθ upload, adaptive rescaling, and dual momentum.

## Project Structure

```
fedjd/
├── aggregators/          # Jacobian aggregation strategies (MinNorm, Mean, Random)
├── compressors/          # Jacobian compression for communication efficiency
├── core/                 # Core framework: client, server, trainer, baselines, NFJD modules
│   ├── client.py         # FedJDClient (original, uploads Jacobian)
│   ├── server.py         # FedJDServer (original, server-side MinNorm)
│   ├── trainer.py        # FedJDTrainer
│   ├── baselines.py      # FMGDA, WeightedSum, DirectionAvg servers
│   ├── nfjd_client.py    # NFJDClient (Δθ upload, local momentum, adaptive rescaling)
│   ├── nfjd_server.py    # NFJDServer (FedAvg + global momentum)
│   ├── nfjd_trainer.py   # NFJDTrainer
│   └── scaling.py        # AdaptiveRescaling, StochasticGramianSolver, momentum modules
├── data/                 # Dataset generators (synthetic regression, high-conflict, classification)
├── docs/                 # Documentation and experiment plans
│   ├── NFJD_Experiment_Plans/  # Phase 1-4 experiment plans and checklists
│   ├── experiment_guide/       # Original FedJD experiment guide
│   └── plans/                  # Original FedJD stage plans
├── experiments/          # Experiment scripts
│   ├── s1_baseline/      # Stage 1: Baseline smoke tests
│   ├── s2_profile/       # Stage 2: Communication profiling
│   ├── s3_compress/      # Stage 3: Compression experiments
│   ├── s4_benchmark/     # Stage 4: Full benchmark + high-conflict
│   └── nfjd_phases/      # NFJD Phase 1-4 experiment scripts
├── metrics/              # Evaluation metrics (hypervolume, Pareto front, Pareto gap)
├── models/               # Neural network models (Small/Medium/Large regressor, classifier)
├── problems/             # Objective function definitions (regression, classification)
├── config.py             # ExperimentConfig dataclass
└── visualization.py      # Training curve plotting utilities
```

## Key Innovations (NFJD vs FedJD)

| Feature | FedJD | NFJD |
|---------|-------|------|
| Upload content | Jacobian (m×d) | Δθ (d-dim) |
| Communication vs m | O(m×d) | O(d), decoupled from m |
| Direction finding | Server-side MinNorm | Client-side MinNorm + StochasticGramian |
| Scaling | None | AdaptiveRescaling |
| Momentum | None | Dual (local + global) |
| Local epochs | 1 | Configurable (default 3) |

## Quick Start

```python
from fedjd.core import NFJDClient, NFJDServer, NFJDTrainer
from fedjd.data import make_synthetic_federated_regression
from fedjd.problems import multi_objective_regression
from fedjd.models import SmallRegressor

data = make_synthetic_federated_regression(num_clients=10, num_objectives=3)
model = SmallRegressor(input_dim=8, output_dim=3)
clients = [NFJDClient(client_id=i, dataset=data.client_datasets[i], batch_size=32,
                       device=torch.device("cpu")) for i in range(10)]
server = NFJDServer(model=model, clients=clients, objective_fn=multi_objective_regression,
                    participation_rate=0.5, learning_rate=0.01, device=torch.device("cpu"))
trainer = NFJDTrainer(server=server, num_rounds=50)
history = trainer.fit()
```
