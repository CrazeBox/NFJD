# NFJD Experiment Plans

Phase-by-phase experiment plans and checklists for NFJD validation.

Current recommended reading order:

1. `../NFJD_Method.md`
2. `../NFJD_Status_and_Roadmap.md`
3. phase-specific plan/checklist documents below

## Phases

| Phase | Directory | Experiments | Focus |
|-------|-----------|-------------|-------|
| Phase 1 | `Phase_1/` | 45 | Baseline verification (5 methods × 3 m × 3 seeds) |
| Phase 2 | `Phase_2/` | 90 | Ablation study (AR, momentum, SG, local epochs) |
| Phase 3 | `Phase_3/` | 135 | High-conflict data verification (3 conflict levels) |
| Phase 4 | `Phase_4/` | 285 | Full benchmark (regression + classification + Non-IID) |
| Phase 5 | `Phase_5/` | 117 | Real-data benchmark with formally sourced baselines |

## Files per Phase

Each phase directory contains:
- `NFJD_Phase_N_Plan.md` - Detailed experiment plan
- `NFJD_Phase_N_Checklist.md` - Verification checklist

Use `docs/NFJD_Status_and_Roadmap.md` to record cross-phase design changes that should not be buried inside a single phase plan.
