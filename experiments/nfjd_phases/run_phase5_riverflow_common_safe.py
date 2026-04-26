from __future__ import annotations

import argparse
import copy
import csv
import logging
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from fedjd.data.river_flow import make_river_flow
from fedjd.experiments.nfjd_phases.phase5_utils import (
    NFJD_VARIANT_CONFIGS,
    cleanup,
    evaluate_model,
    fill_regression_metrics,
    run_experiment,
)
from fedjd.models.river_flow_mlp import RiverFlowMLP
from fedjd.problems import multi_objective_regression


RESULTS_DIR = Path("results/nfjd_phase5/riverflow_common_safe")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(RESULTS_DIR / "p5_riverflow_common_safe.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

DEFAULT_METHODS = ["nfjd_common_safe", "nfjd", "fedavg_cagrad", "fedavg_pcgrad", "fedavg_ls"]


def _write_csv(csv_path: Path, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "exp_id", "method", "dataset", "data_split", "m", "seed", "num_rounds",
        "num_clients", "participation_rate", "learning_rate", "local_epochs",
        "avg_mse", "max_mse", "mse_std", "avg_r2",
        "elapsed_time", "avg_upload_bytes", "avg_round_time", "upload_per_client",
        "public_preprocess_alpha", "public_preprocess_recompute_interval",
        "public_preprocess_steps", "public_preprocess_probe_batch_size",
        "fedclient_update_scale", "fedclient_normalize_updates",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_riverflow_common_safe(
    method: str,
    seed: int,
    iid: bool,
    num_rounds: int,
    num_clients: int,
    participation_rate: float,
    learning_rate: float,
    num_tasks: int,
    public_preprocess_alpha: float | None,
    public_preprocess_recompute_interval: int | None,
    public_preprocess_steps: int | None,
    public_preprocess_probe_batch_size: int | None,
    local_epochs: int,
    fedclient_update_scale: float | None,
    fedclient_normalize_updates: bool | None,
) -> dict:
    split_name = "iid" if iid else "noniid"
    suffix_parts = []
    if method == "nfjd_common_safe":
        suffix_parts.append(f"a{public_preprocess_alpha:g}")
        suffix_parts.append(f"R{public_preprocess_recompute_interval}")
        suffix_parts.append(f"s{public_preprocess_steps}")
        suffix_parts.append(f"pb{public_preprocess_probe_batch_size}")
    if method == "fedclient_upgrad":
        suffix_parts.append(f"le{local_epochs}")
        suffix_parts.append(f"scale{fedclient_update_scale:g}")
        suffix_parts.append(f"norm{int(bool(fedclient_normalize_updates))}")
    suffix = "-" + "-".join(suffix_parts) if suffix_parts else ""
    exp_id = f"P5-rf-common-safe-{method}-{split_name}-m{num_tasks}{suffix}-seed{seed}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)

    original_cfg = None
    if method == "nfjd_common_safe":
        original_cfg = copy.deepcopy(NFJD_VARIANT_CONFIGS[method])
        NFJD_VARIANT_CONFIGS[method]["public_preprocess_alpha"] = float(public_preprocess_alpha)
        NFJD_VARIANT_CONFIGS[method]["public_preprocess_recompute_interval"] = int(public_preprocess_recompute_interval)
        NFJD_VARIANT_CONFIGS[method]["public_preprocess_steps"] = int(public_preprocess_steps)
        NFJD_VARIANT_CONFIGS[method]["public_preprocess_probe_batch_size"] = int(public_preprocess_probe_batch_size)

    try:
        data = make_river_flow(num_clients=num_clients, iid=iid, seed=seed, num_tasks=num_tasks)
        model = RiverFlowMLP(input_dim=data["input_dim"], num_tasks=num_tasks)
        row = run_experiment(
            exp_id=exp_id,
            method=method,
            model=model,
            client_datasets=data["client_datasets"],
            objective_fn=multi_objective_regression,
            m=num_tasks,
            seed=seed,
            device=device,
            num_rounds=num_rounds,
            num_clients=num_clients,
            participation_rate=participation_rate,
            learning_rate=learning_rate,
            model_arch="river_flow_mlp",
            dataset="riverflow",
            data_split=split_name,
            local_epochs=local_epochs,
            eval_dataset=data["val_dataset"],
            fedclient_update_scale=float(fedclient_update_scale or 1.0),
            fedclient_normalize_updates=bool(fedclient_normalize_updates),
        )
        preds, targets = evaluate_model(model, data["test_dataset"], device)
        row = fill_regression_metrics(row, preds, targets, num_tasks)
        row["public_preprocess_alpha"] = public_preprocess_alpha if method == "nfjd_common_safe" else ""
        row["public_preprocess_recompute_interval"] = public_preprocess_recompute_interval if method == "nfjd_common_safe" else ""
        row["public_preprocess_steps"] = public_preprocess_steps if method == "nfjd_common_safe" else ""
        row["public_preprocess_probe_batch_size"] = public_preprocess_probe_batch_size if method == "nfjd_common_safe" else ""
        row["fedclient_update_scale"] = fedclient_update_scale if method == "fedclient_upgrad" else ""
        row["fedclient_normalize_updates"] = fedclient_normalize_updates if method == "fedclient_upgrad" else ""
        return row
    finally:
        if original_cfg is not None:
            NFJD_VARIANT_CONFIGS[method].update(original_cfg)


def parse_args():
    parser = argparse.ArgumentParser(description="Run RiverFlow nfjd_common_safe sweep with baselines.")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--tasks", nargs="+", type=int, default=[8])
    parser.add_argument("--splits", nargs="+", choices=["iid", "noniid"], default=["noniid"])
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--participation-rate", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.2, 0.25])
    parser.add_argument("--recompute-intervals", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--probe-batch-sizes", nargs="+", type=int, default=[32])
    parser.add_argument("--local-epochs", nargs="+", type=int, default=[3])
    parser.add_argument("--baseline-local-epochs", nargs="+", type=int, default=[3])
    parser.add_argument("--fedclient-update-scales", nargs="+", type=float, default=[1.0])
    parser.add_argument("--fedclient-normalize-updates", nargs="+", type=int, choices=[0, 1], default=[0])
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    experiments = []

    for num_tasks in args.tasks:
        for split in args.splits:
            iid = split == "iid"
            for method in args.methods:
                if method == "nfjd_common_safe":
                    for alpha in args.alphas:
                        for interval in args.recompute_intervals:
                            for steps in args.steps:
                                for probe_batch_size in args.probe_batch_sizes:
                                    for seed in args.seeds:
                                        experiments.append(dict(
                                            method=method,
                                            seed=seed,
                                            iid=iid,
                                            num_rounds=args.rounds,
                                            num_clients=args.num_clients,
                                            participation_rate=args.participation_rate,
                                            learning_rate=args.learning_rate,
                                            num_tasks=num_tasks,
                                            public_preprocess_alpha=alpha,
                                            public_preprocess_recompute_interval=interval,
                                            public_preprocess_steps=steps,
                                            public_preprocess_probe_batch_size=probe_batch_size,
                                            local_epochs=3,
                                            fedclient_update_scale=None,
                                            fedclient_normalize_updates=None,
                                        ))
                elif method == "fedclient_upgrad":
                    for local_epochs in args.local_epochs:
                        for update_scale in args.fedclient_update_scales:
                            for normalize in args.fedclient_normalize_updates:
                                for seed in args.seeds:
                                    experiments.append(dict(
                                        method=method,
                                        seed=seed,
                                        iid=iid,
                                        num_rounds=args.rounds,
                                        num_clients=args.num_clients,
                                        participation_rate=args.participation_rate,
                                        learning_rate=args.learning_rate,
                                        num_tasks=num_tasks,
                                        public_preprocess_alpha=None,
                                        public_preprocess_recompute_interval=None,
                                        public_preprocess_steps=None,
                                        public_preprocess_probe_batch_size=None,
                                        local_epochs=local_epochs,
                                        fedclient_update_scale=update_scale,
                                        fedclient_normalize_updates=bool(normalize),
                                    ))
                else:
                    for local_epochs in args.baseline_local_epochs:
                        for seed in args.seeds:
                            experiments.append(dict(
                                method=method,
                                seed=seed,
                                iid=iid,
                                num_rounds=args.rounds,
                                num_clients=args.num_clients,
                                participation_rate=args.participation_rate,
                                learning_rate=args.learning_rate,
                                num_tasks=num_tasks,
                                public_preprocess_alpha=None,
                                public_preprocess_recompute_interval=None,
                                public_preprocess_steps=None,
                                public_preprocess_probe_batch_size=None,
                                local_epochs=local_epochs,
                                fedclient_update_scale=None,
                                fedclient_normalize_updates=None,
                            ))

    logger.info("Starting RiverFlow common-safe sweep: %d experiments", len(experiments))
    for idx, exp in enumerate(experiments):
        logger.info("[%d/%d] Running %s", idx + 1, len(experiments), exp)
        try:
            rows.append(run_riverflow_common_safe(**exp))
        except Exception as exc:
            logger.exception("Experiment failed: %s", exc)
        finally:
            cleanup()

    csv_path = RESULTS_DIR / "phase5_riverflow_common_safe.csv"
    _write_csv(csv_path, rows)
    logger.info("RiverFlow common-safe sweep complete: %d/%d saved to %s", len(rows), len(experiments), csv_path)


if __name__ == "__main__":
    main()
