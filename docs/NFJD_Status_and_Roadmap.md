# NFJD Status and Roadmap

## Purpose

This document records the current canonical state of NFJD, the major changes that have already been implemented, and the most relevant next steps. It is intended to make future iteration, version control, and design discussion easier.

## Current Canonical State

As of the current codebase state, the intended mainline definition of NFJD is:

- Local multi-objective core: `UPGrad`-based Jacobian descent on each client
- Local approximation strategy: `recompute_interval` + reuse of the latest UPGrad-derived weight vector between recomputation steps
- High-objective approximation: `StochasticGramianSolver` using sampled sub-Jacobians
- Local stabilization: `AdaptiveRescaling` + local momentum
- Server communication: upload `delta_theta` and lightweight `align_scores`
- Server aggregation: alignment-aware weighted average + global momentum
- Held-out tracking: validation set for round-by-round objective tracking, test set for final task metrics

Main code paths:

- `core/nfjd_client.py`
- `core/nfjd_server.py`
- `core/scaling.py`
- `aggregators/__init__.py`

## Implemented Changes

The following changes are already reflected in the code and should be treated as completed work.

### Algorithm Core

- Replaced the NFJD local Jacobian aggregator from `MinNorm/MGDA-style` to `UPGrad`
- Implemented `UPGradAggregator` in `aggregators/__init__.py`
- Used the paper-aligned Gramian-space dual formulation instead of direct parameter-space projection
- Updated `StochasticGramianSolver` to support `upgrad` mode
- Switched stale local reuse from simplex-style `lambda` semantics to UPGrad-derived weight reuse

### Evaluation and Fairness

- Fixed the initial-objective accounting bug that previously distorted RI-style metrics
- Fixed upload/download accounting to use actual sampled-client communication volume
- Switched round-by-round objective evaluation to held-out validation sets where available
- Kept final task metrics on test sets for Phase 5
- Replaced Phase 5 informal baselines with sourced baselines tied to papers and official repos

### Phase 5 Benchmarking

- Added formal Phase 5 baseline metadata and method provenance fields
- Added `run_phase5_suite.py` as the canonical Phase 5 entrypoint
- Bound the main Phase 5 baselines to:
  - `FedAvg+LS`
  - `FMGDA`
  - `FedAvg+PCGrad`
  - `FedAvg+CAGrad`
- Replaced the earlier Phase 5 `FedAvg+MGDA-UB` placeholder with a paper-aligned `FMGDA` implementation that uses per-objective local trajectories on clients and server-side MGDA aggregation
- Tightened Phase 5 baseline fidelity so that `LS / PCGrad / CAGrad / FMGDA` better match their official paper/repo formulations, including shared-parameter surgery for `PCGrad/CAGrad` and sample-weighted server aggregation for `FMGDA`

## Recommended Current Entry Points

These are the recommended scripts to use for current work.

### Main Experiments

- Phase 3: `experiments/nfjd_phases/run_phase3_highconflict.py`
- Phase 4: `experiments/nfjd_phases/run_phase4_benchmark.py`
- Phase 5: `experiments/nfjd_phases/run_phase5_suite.py`
- Phase 6: `experiments/nfjd_phases/run_phase6_recompute_ablation.py`

### Utility / Profiling

- `experiments/nfjd_phases/profile_nfjd.py`
- `experiments/nfjd_phases/profile_lenet.py`
- `experiments/nfjd_phases/run_recompute_sweep.py`

### Legacy / Historical Only

These should not be treated as the current evidence path for NFJD.

- `experiments/fedjd_legacy/*`
- `docs/fedjd_legacy/*`

## Open Questions

These are the most important unresolved method questions.

### 1. Exact vs Approximate UPGrad

Current NFJD uses approximate local Jacobian descent:

- exact on recomputation steps
- stale-weight approximation on intermediate steps

This is the practical version. A small-scale exact reference variant is still worth implementing for comparison.

### 2. Server Aggregation Rule

Current mainline server update is:

- alignment-aware weighted average
- followed by global momentum

This is empirically reasonable for noisy federated `delta_theta` updates, but it is still heuristic. A cleaner comparison against pure weighted averaging remains useful.

### 3. Alignment Heuristic Strength

The current server-side alignment adjustment is intentionally simple. It is useful in practice but should still be regarded as heuristic unless stronger empirical or theoretical support is added.

### 4. Full Gramian Reverse Accumulation

The JD paper outlines a deeper Gramian-based implementation path that avoids explicit Jacobian storage. This is not yet integrated into the current PyTorch training stack.

## Recommended Next Steps

These are the highest-value next steps, ordered by importance.

1. Add `NFJD-exact-UPGrad` on small tasks
2. Compare `NFJD-exact-UPGrad` vs current approximate NFJD on toy and small synthetic settings
3. Run a clean server ablation: pure weighted average vs alignment-aware average vs alignment-aware average + global momentum
4. Add sensitivity analysis for alignment adjustment strength
5. Add stronger statistical reporting for Phase 5 summaries
6. Decide whether to keep or archive older benchmark scripts that still assume pre-UPGrad NFJD semantics

## Phase 5 Post-Run Follow-Ups

These items should be handled after the current Phase 5 program finishes running. They are intentionally separated from the live experiment run so the current jobs remain stable.

1. Run a small hyperparameter sweep for `FMGDA`, `FedAvg+LS`, `FedAvg+PCGrad`, and `FedAvg+CAGrad` instead of relying on a single shared learning rate
2. For `FMGDA`, test whether separating local step size `eta_L` and server step size `eta_t` changes the ranking or stability
3. Report Phase 5 main-table results as `mean ± std` over seeds, not only raw per-seed rows
4. Add statistical tests for the final Phase 5 summary tables, preferably Wilcoxon for pairwise comparisons and Friedman/Nemenyi for overall ranking
5. Keep manuscript wording precise: treat `FMGDA` as a native federated multi-objective baseline, and treat `FedAvg+LS / PCGrad / CAGrad` as federated adaptations of centralized multi-task optimizers
6. Re-check Phase 5 compute and communication tables after the runs finish so that wall-clock and upload claims are reported conservatively

## Not Recommended Right Now

These ideas are possible but are not the best immediate use of effort.

- Rewriting the whole training core around custom Gramian reverse accumulation
- Replacing all historical experiments immediately
- Claiming exact equivalence between current NFJD and the centralized JD/UPGrad algorithm

## Versioning Guidance

For future iteration, use explicit labels in experiment notes and result summaries.

Suggested tags:

- `nfjd-upgrad-mainline`
- `nfjd-upgrad-exact-ref`
- `nfjd-server-no-momentum`
- `nfjd-server-no-align`
- `nfjd-stochasticgramian-off`

When updating the method, record changes in three places:

1. `README.md` for top-level canonical description
2. `docs/NFJD_Method.md` for algorithm definition
3. this file for design-state tracking and future work decisions
