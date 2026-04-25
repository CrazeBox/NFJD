from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from fedjd.aggregators import MinNormAggregator
from fedjd.core import (
    DirectionAvgServer,
    FMGDAClient,
    FMGDAServer,
    FedJDClient,
    FedJDServer,
    FedJDTrainer,
    NFJDClient,
    NFJDServer,
    NFJDTrainer,
    Phase5OfficialBaselineClient,
    Phase5OfficialBaselineServer,
    WeightedSumServer,
)
from fedjd.data import make_high_conflict_federated_regression, make_synthetic_federated_regression
from fedjd.experiments.nfjd_phases.metric_utils import summarize_round_history
from fedjd.experiments.nfjd_phases.phase5_utils import evaluate_model, fill_regression_metrics
from fedjd.experiments.nfjd_phases.phase5_utils import NFJD_VARIANT_CONFIGS, build_trainer as build_phase5_trainer
from fedjd.models import SmallRegressor
from fedjd.problems import multi_objective_regression


RESULTS_DIR = Path("results/nfjd_cone_prototype")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(RESULTS_DIR / "cone_prototype.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

FIELDNAMES = [
    "exp_id", "method", "dataset", "seed", "m", "num_rounds", "num_clients",
    "participation_rate", "learning_rate", "local_epochs", "conflict_strength",
    "cone_align_alpha", "cone_reference_mode", "cone_basis_size", "cone_align_positive_only", "public_preprocess_alpha", "public_preprocess_mode", "public_preprocess_positive_only", "public_preprocess_center_mode", "public_preprocess_trim_k", "public_preprocess_adaptive_mode", "elapsed_time",
    "avg_upload_bytes", "avg_round_time", "upload_per_client",
    "avg_mse", "max_mse", "mse_std", "avg_r2",
]


def _write_csv(rows, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_data(dataset: str, m: int, seed: int, num_clients: int, conflict_strength: float):
    if dataset == "highconflict_regression":
        return make_high_conflict_federated_regression(
            num_clients=num_clients,
            samples_per_client=100,
            input_dim=8,
            num_objectives=m,
            conflict_strength=conflict_strength,
            seed=seed,
        )
    if dataset == "synthetic_regression":
        return make_synthetic_federated_regression(
            num_clients=num_clients,
            samples_per_client=100,
            input_dim=8,
            num_objectives=m,
            seed=seed,
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def _build_nfjd_trainer(method: str, data, device, seed, num_rounds, num_clients, participation_rate, learning_rate, local_epochs, cone_align_alpha, cone_reference_mode, cone_align_positive_only, cone_basis_size, public_preprocess_alpha, public_preprocess_mode, public_preprocess_positive_only, public_preprocess_center_mode, public_preprocess_trim_k, public_preprocess_adaptive_mode):
    clients = [
        NFJDClient(
            client_id=i,
            dataset=data.client_datasets[i],
            batch_size=32,
            device=device,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            local_momentum_beta=0.0,
            use_adaptive_rescaling=False,
            use_stochastic_gramian=False,
            stochastic_subset_size=4,
            stochastic_seed=seed + i,
            conflict_aware_momentum=False,
            momentum_min_beta=0.1,
            recompute_interval=1,
            exact_upgrad=True,
            use_objective_normalization=True,
            upload_align_scores=False,
            cone_align_positive_only=cone_align_positive_only,
        )
        for i in range(num_clients)
    ]
    server = NFJDServer(
        model=SmallRegressor(input_dim=data.input_dim, output_dim=data.num_objectives),
        clients=clients,
        objective_fn=multi_objective_regression,
        participation_rate=participation_rate,
        learning_rate=learning_rate,
        device=device,
        global_momentum_beta=0.0,
        conflict_aware_momentum=False,
        momentum_min_beta=0.1,
        parallel_clients=False,
        eval_dataset=data.val_dataset,
        use_global_progress_weights=True,
        progress_beta=2.0,
        progress_min_weight=0.5,
        progress_max_weight=2.0,
        method_name=method,
        cone_align_alpha=cone_align_alpha if method in {"nfjd_cone", "nfjd_cone_basis"} else 0.0,
        cone_reference_mode=cone_reference_mode,
        cone_basis_size=cone_basis_size if method == "nfjd_cone_basis" else 0,
        public_preprocess_alpha=public_preprocess_alpha if method == "nfjd_common_safe" else 0.0,
        public_preprocess_mode=public_preprocess_mode if method == "nfjd_common_safe" else "",
        public_preprocess_positive_only=public_preprocess_positive_only if method == "nfjd_common_safe" else False,
        public_preprocess_center_mode=public_preprocess_center_mode if method == "nfjd_common_safe" else "mean",
        public_preprocess_trim_k=public_preprocess_trim_k if method == "nfjd_common_safe" else 0,
        public_preprocess_adaptive_mode=public_preprocess_adaptive_mode if method == "nfjd_common_safe" else "fixed",
    )
    return NFJDTrainer(server=server, num_rounds=num_rounds)


def _build_phase5_variant_trainer(method: str, data, device, seed, num_rounds, num_clients, participation_rate, learning_rate, local_epochs):
    model = SmallRegressor(input_dim=data.input_dim, output_dim=data.num_objectives)
    return build_phase5_trainer(
        method=method,
        model=model,
        client_datasets=data.client_datasets,
        objective_fn=multi_objective_regression,
        m=data.num_objectives,
        seed=seed,
        device=device,
        num_rounds=num_rounds,
        num_clients=num_clients,
        participation_rate=participation_rate,
        learning_rate=learning_rate,
        local_epochs=local_epochs,
        eval_dataset=data.val_dataset,
    )


def _build_legacy_trainer(method: str, data, device, num_rounds, num_clients, participation_rate, learning_rate, local_epochs):
    model = SmallRegressor(input_dim=data.input_dim, output_dim=data.num_objectives)
    if method == "fedjd":
        clients = [FedJDClient(client_id=i, dataset=data.client_datasets[i], batch_size=32, device=device, use_full_loader=True, local_epochs=local_epochs) for i in range(num_clients)]
        aggregator = MinNormAggregator(max_iters=250, lr=0.1, max_direction_norm=0.0)
        server = FedJDServer(model=model, clients=clients, aggregator=aggregator, objective_fn=multi_objective_regression, participation_rate=participation_rate, learning_rate=learning_rate, device=device, eval_dataset=data.val_dataset)
    elif method == "fmgda":
        clients = [FMGDAClient(client_id=i, dataset=data.client_datasets[i], batch_size=32, device=device, learning_rate=learning_rate, local_epochs=local_epochs) for i in range(num_clients)]
        server = FMGDAServer(model=model, clients=clients, objective_fn=multi_objective_regression, participation_rate=participation_rate, learning_rate=learning_rate, device=device, eval_dataset=data.val_dataset, num_objectives=data.num_objectives)
    elif method == "weighted_sum":
        clients = [FedJDClient(client_id=i, dataset=data.client_datasets[i], batch_size=32, device=device, use_full_loader=True, local_epochs=local_epochs) for i in range(num_clients)]
        server = WeightedSumServer(model=model, clients=clients, objective_fn=multi_objective_regression, participation_rate=participation_rate, learning_rate=learning_rate, device=device, eval_dataset=data.val_dataset)
    elif method == "direction_avg":
        clients = [FedJDClient(client_id=i, dataset=data.client_datasets[i], batch_size=32, device=device, use_full_loader=True, local_epochs=local_epochs) for i in range(num_clients)]
        server = DirectionAvgServer(model=model, clients=clients, objective_fn=multi_objective_regression, participation_rate=participation_rate, learning_rate=learning_rate, device=device, eval_dataset=data.val_dataset)
    else:
        raise ValueError(f"Unsupported legacy method: {method}")
    return FedJDTrainer(server=server, num_rounds=num_rounds)


def _build_official_trainer(method: str, data, device, seed, num_rounds, num_clients, participation_rate, learning_rate, local_epochs):
    model = SmallRegressor(input_dim=data.input_dim, output_dim=data.num_objectives)
    clients = [
        Phase5OfficialBaselineClient(
            client_id=i,
            dataset=data.client_datasets[i],
            batch_size=32,
            device=device,
            learning_rate=learning_rate,
            local_epochs=local_epochs,
            method=method,
            seed=seed + i,
        )
        for i in range(num_clients)
    ]
    server = Phase5OfficialBaselineServer(
        model=model,
        clients=clients,
        objective_fn=multi_objective_regression,
        participation_rate=participation_rate,
        device=device,
        method_name=method,
        eval_dataset=data.val_dataset,
    )
    return FedJDTrainer(server=server, num_rounds=num_rounds)


def _run_single(method: str, dataset: str, seed: int, m: int, num_rounds: int, num_clients: int, participation_rate: float, learning_rate: float, local_epochs: int, conflict_strength: float, cone_align_alpha: float, cone_reference_mode: str, cone_align_positive_only: bool, cone_basis_size: int, public_preprocess_alpha: float, public_preprocess_mode: str, public_preprocess_positive_only: bool, public_preprocess_center_mode: str, public_preprocess_trim_k: int, public_preprocess_adaptive_mode: str) -> dict:
    if method == "nfjd_cone":
        exp_suffix = f"-a{cone_align_alpha:g}-{cone_reference_mode}"
    elif method == "nfjd_cone_basis":
        gate = "-pos" if cone_align_positive_only else ""
        exp_suffix = f"-a{cone_align_alpha:g}-basis{cone_basis_size}{gate}"
    elif method == "nfjd_common_safe":
        gate = "-pos" if public_preprocess_positive_only else ""
        trim = f"-trim{public_preprocess_trim_k}" if public_preprocess_center_mode == "trimmed_mean" else ""
        adaptive = f"-{public_preprocess_adaptive_mode}"
        exp_suffix = f"-a{public_preprocess_alpha:g}-{public_preprocess_mode}-{public_preprocess_center_mode}{trim}{adaptive}{gate}"
    else:
        exp_suffix = ""
    exp_id = f"cone-proto-{dataset}-{method}{exp_suffix}-m{m}-seed{seed}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)

    data = _make_data(dataset, m, seed, num_clients, conflict_strength)

    if method in {"nfjd_cone", "nfjd_cone_basis", "nfjd_common_safe"}:
        trainer = _build_nfjd_trainer(method, data, device, seed, num_rounds, num_clients, participation_rate, learning_rate, local_epochs, cone_align_alpha, cone_reference_mode, cone_align_positive_only, cone_basis_size, public_preprocess_alpha, public_preprocess_mode, public_preprocess_positive_only, public_preprocess_center_mode, public_preprocess_trim_k, public_preprocess_adaptive_mode)
    elif method in NFJD_VARIANT_CONFIGS:
        trainer = _build_phase5_variant_trainer(method, data, device, seed, num_rounds, num_clients, participation_rate, learning_rate, local_epochs)
    elif method in {"fedjd", "fmgda", "weighted_sum", "direction_avg"}:
        trainer = _build_legacy_trainer(method, data, device, num_rounds, num_clients, participation_rate, learning_rate, local_epochs)
    elif method in {"fedavg_ls", "fedavg_pcgrad", "fedavg_cagrad"}:
        trainer = _build_official_trainer(method, data, device, seed, num_rounds, num_clients, participation_rate, learning_rate, local_epochs)
    else:
        raise ValueError(f"Unknown method: {method}")

    start = time.time()
    history = trainer.fit()
    elapsed = time.time() - start
    round_summary = summarize_round_history(history)
    predictions, targets = evaluate_model(trainer.server.model, data.test_dataset, device, batch_size=256)

    row = {
        "exp_id": exp_id,
        "method": method,
        "dataset": dataset,
        "seed": seed,
        "m": m,
        "num_rounds": num_rounds,
        "num_clients": num_clients,
        "participation_rate": participation_rate,
        "learning_rate": learning_rate,
        "local_epochs": local_epochs,
        "conflict_strength": conflict_strength,
        "cone_align_alpha": cone_align_alpha if method in {"nfjd_cone", "nfjd_cone_basis"} else "",
        "cone_reference_mode": cone_reference_mode if method in {"nfjd_cone", "nfjd_cone_basis"} else "",
        "cone_basis_size": cone_basis_size if method == "nfjd_cone_basis" else "",
        "cone_align_positive_only": cone_align_positive_only if method in {"nfjd_cone", "nfjd_cone_basis"} else "",
        "public_preprocess_alpha": public_preprocess_alpha if method == "nfjd_common_safe" else "",
        "public_preprocess_mode": public_preprocess_mode if method == "nfjd_common_safe" else "",
        "public_preprocess_positive_only": public_preprocess_positive_only if method == "nfjd_common_safe" else "",
        "public_preprocess_center_mode": public_preprocess_center_mode if method == "nfjd_common_safe" else "",
        "public_preprocess_trim_k": public_preprocess_trim_k if method == "nfjd_common_safe" else "",
        "public_preprocess_adaptive_mode": public_preprocess_adaptive_mode if method == "nfjd_common_safe" else "",
        "elapsed_time": round(elapsed, 2),
        "avg_upload_bytes": round(float(round_summary["avg_upload_bytes"]), 0),
        "avg_round_time": round(float(round_summary["avg_round_time"]), 4),
        "upload_per_client": round(float(round_summary["upload_per_client"]), 0),
        "avg_mse": "",
        "max_mse": "",
        "mse_std": "",
        "avg_r2": "",
    }
    row = fill_regression_metrics(row, predictions, targets, data.num_objectives)

    logger.info(
        "[%s] %s MSE=%.6f maxMSE=%.6f R2=%.6f time=%.1fs",
        exp_id,
        method,
        float(row["avg_mse"]),
        float(row["max_mse"]),
        float(row["avg_r2"]),
        elapsed,
    )
    return row


def parse_args():
    parser = argparse.ArgumentParser(description="Run Cone-Aligned NFJD synthetic prototype benchmark.")
    parser.add_argument("--dataset", choices=["synthetic_regression", "highconflict_regression"], default="highconflict_regression")
    parser.add_argument("--methods", nargs="+", default=["nfjd_common_safe", "nfjd", "nfjd_fast", "fedjd", "fmgda", "weighted_sum", "direction_avg", "fedavg_ls", "fedavg_pcgrad", "fedavg_cagrad"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 123])
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--participation-rate", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--conflict-strength", type=float, default=1.0)
    parser.add_argument("--cone-align-alphas", nargs="+", type=float, default=[0.25])
    parser.add_argument("--cone-reference-mode", choices=["delta", "validation_gradient", "probe_basis"], default="delta")
    parser.add_argument("--cone-align-positive-only", action="store_true")
    parser.add_argument("--cone-basis-size", type=int, default=2)
    parser.add_argument("--public-preprocess-alphas", nargs="+", type=float, default=[0.1])
    parser.add_argument("--public-preprocess-mode", choices=["common_safe_ray"], default="common_safe_ray")
    parser.add_argument("--public-preprocess-positive-only", action="store_true")
    parser.add_argument("--public-preprocess-center-mode", choices=["mean", "trimmed_mean", "geometric_median"], default="mean")
    parser.add_argument("--public-preprocess-trim-k", type=int, default=1)
    parser.add_argument("--public-preprocess-adaptive-mode", choices=["fixed", "cosine"], default="fixed")
    parser.add_argument("--output-name", default="cone_prototype_results.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    experiments = []
    for method in args.methods:
        cone_alphas = args.cone_align_alphas if method in {"nfjd_cone", "nfjd_cone_basis"} else [0.0]
        public_alphas = args.public_preprocess_alphas if method == "nfjd_common_safe" else [0.0]
        for cone_alpha in cone_alphas:
            for public_alpha in public_alphas:
                for seed in args.seeds:
                    experiments.append(dict(
                        method=method,
                        dataset=args.dataset,
                        seed=seed,
                        m=args.m,
                        num_rounds=args.rounds,
                        num_clients=args.num_clients,
                        participation_rate=args.participation_rate,
                        learning_rate=args.learning_rate,
                        local_epochs=args.local_epochs,
                        conflict_strength=args.conflict_strength,
                        cone_align_alpha=cone_alpha,
                        cone_reference_mode=args.cone_reference_mode,
                        cone_align_positive_only=args.cone_align_positive_only,
                        cone_basis_size=args.cone_basis_size,
                        public_preprocess_alpha=public_alpha,
                        public_preprocess_mode=args.public_preprocess_mode,
                        public_preprocess_positive_only=args.public_preprocess_positive_only,
                        public_preprocess_center_mode=args.public_preprocess_center_mode,
                        public_preprocess_trim_k=args.public_preprocess_trim_k,
                        public_preprocess_adaptive_mode=args.public_preprocess_adaptive_mode,
                    ))

    logger.info("Starting Cone-Aligned NFJD prototype benchmark: %d experiments", len(experiments))
    for idx, exp in enumerate(experiments):
        logger.info("[%d/%d] Running %s", idx + 1, len(experiments), exp)
        try:
            rows.append(_run_single(**exp))
        except Exception as exc:
            logger.exception("Experiment failed: %s", exc)

    csv_path = RESULTS_DIR / args.output_name
    _write_csv(rows, csv_path)
    logger.info("Prototype benchmark complete: %d/%d results saved to %s", len(rows), len(experiments), csv_path)


if __name__ == "__main__":
    main()
