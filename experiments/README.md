# Experiments

## NFJD Experiments

| Path | Purpose |
|---|---|
| `nfjd_phases/README.md` | NFJD phase index and quick commands |
| `nfjd_phases/run_phase1_baseline.py` | Phase 1 baseline comparison |
| `nfjd_phases/run_phase2_ablation.py` | Phase 2 ablation study |
| `nfjd_phases/run_phase3_highconflict.py` | Phase 3 high-conflict study |
| `nfjd_phases/run_phase4_benchmark.py` | Phase 4 full benchmark |
| `nfjd_phases/run_phase5_realdata.py` | Phase 5 real-data benchmark |
| `nfjd_phases/run_phase6_recompute_ablation.py` | Phase 6 recompute-interval ablation |
| `nfjd_phases/run_recompute_sweep.py` | Fast recompute-interval sweep |

## Legacy FedJD Experiments

| Directory | Description |
|---|---|
| `fedjd_legacy/` | Original FedJD Stage 1-5 scripts |

## Notes

Generated outputs should live under `results/`, not inside `experiments/`.
Profiling and quick-sweep text outputs are centralized in `results/nfjd_tools/`.
