# Experiments

Experiment scripts for FedJD (Stages 1-4) and NFJD (Phases 1-4).

## FedJD Experiments (Original)

| Directory | Stage | Description |
|-----------|-------|-------------|
| `s1_baseline/` | Stage 1 | Baseline smoke tests, MinNorm verification |
| `s2_profile/` | Stage 2 | Communication profiling (Jacobian size vs gradient size) |
| `s3_compress/` | Stage 3 | Compression experiments (TopK, LowRank, Float16, etc.) |
| `s4_benchmark/` | Stage 4 | Full benchmark + high-conflict experiments |

## NFJD Experiments (New Architecture)

| Directory | Phase | Description |
|-----------|-------|-------------|
| `nfjd_phases/` | Phase 1-4 | NFJD benchmark, ablation, high-conflict, full benchmark |

## Running Experiments

```bash
# FedJD Stage 1
cd experiments/s1_baseline && python run_experiment.py

# NFJD Benchmark
cd experiments/nfjd_phases && python run_nfjd_benchmark.py
```
