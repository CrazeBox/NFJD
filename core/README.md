# Core Modules

Core implementation of FedJD and NFJD federated multi-objective optimization.

## Files

| File | Description |
|------|-------------|
| `client.py` | FedJDClient - original client that uploads full Jacobian |
| `server.py` | FedJDServer - original server with server-side MinNorm aggregation |
| `trainer.py` | FedJDTrainer - training loop for FedJD/FMGDA/WeightedSum/DirectionAvg |
| `baselines.py` | FMGDAServer, WeightedSumServer, DirectionAvgServer |
| `nfjd_client.py` | NFJDClient - new client with Δθ upload, local momentum, adaptive rescaling |
| `nfjd_server.py` | NFJDServer - new server with FedAvg aggregation + global momentum |
| `nfjd_trainer.py` | NFJDTrainer - training loop for NFJD |
| `scaling.py` | AdaptiveRescaling, StochasticGramianSolver, LocalMomentum, GlobalMomentum |

## Architecture Comparison

**FedJD**: Client computes Jacobian → Upload m×d matrix → Server aggregates Jacobians → Server finds MinNorm direction → Server updates model

**NFJD**: Client computes Jacobian → Client finds MinNorm direction locally → AdaptiveRescaling → Local momentum → Multiple local epochs → Upload only Δθ (d-dim) → Server FedAvg aggregation → Global momentum → Server updates model
