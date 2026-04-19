# NFJD Phase Scripts

## Main Runs

| Script | Purpose | Default Output |
|---|---|---|
| `run_phase1_baseline.py` | Phase 1 baseline comparison | `results/nfjd_phase1/` |
| `run_phase2_ablation.py` | Phase 2 ablation study | `results/nfjd_phase2/` |
| `run_phase3_highconflict.py` | High-conflict benchmark | `results/nfjd_phase3/` |
| `run_phase4_benchmark.py` | Full benchmark | `results/nfjd_phase4/` |
| `run_phase5_realdata.py` | Real-data benchmark | phase 5 results dir |
| `run_phase6_recompute_ablation.py` | Full recompute-interval ablation | `results/nfjd_phase6_recompute/` |

## Quick Tools

| Script | Purpose | Default Output |
|---|---|---|
| `run_recompute_sweep.py` | Fast recompute-interval sweep with CLI knobs | `results/nfjd_tools/recompute_sweep_results.txt` |
| `profile_nfjd.py` | Small-model NFJD hotspot profile | `results/nfjd_tools/profile_result.txt` |
| `profile_lenet.py` | LeNet NFJD hotspot profile | `results/nfjd_tools/profile_lenet_result.txt` |
| `analyze_nfjd_results.py` | Aggregate NFJD CSV results | report/console |

## Quick Usage

```bash
python experiments/nfjd_phases/run_recompute_sweep.py --rounds 5 --m-values 5 --recompute-intervals 1 2 4 6 --conflicts 0.0 1.0 --seeds 7 42
python experiments/nfjd_phases/profile_nfjd.py
python experiments/nfjd_phases/profile_lenet.py
```
