# Experiments

## NFJD Experiments

| Script | Phase | Description |
|--------|-------|-------------|
| `nfjd_phases/run_phase1_baseline.py` | Phase 1 | 基线验证（5方法×3m×3seed=45次） |
| `nfjd_phases/run_phase2_ablation.py` | Phase 2 | 消融实验（AR/GM/LM/SG/Epoch=66次） |
| `nfjd_phases/run_nfjd_benchmark.py` | Benchmark | 完整基准测试 |

## Legacy FedJD Experiments

| Directory | Description |
|-----------|-------------|
| `fedjd_legacy/` | 原始FedJD Stage 1-5实验脚本 |

## Running Experiments

```bash
# Phase 1: Baseline
python experiments/nfjd_phases/run_phase1_baseline.py

# Phase 2: Ablation
python experiments/nfjd_phases/run_phase2_ablation.py
```
