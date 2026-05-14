from __future__ import annotations

import csv
import gc
import logging
import time

import numpy as np
import torch

from fedjd.core import (
    FMGDAClient, FMGDAServer, FedAvgUPGradServer,
    FedAvgServer, FedClientUPGradServer, FedJDClient, FedLocalTrainClient, FedMGDAPlusServer,
    QFedAvgServer,
    NFJDClient, NFJDServer, NFJDTrainer,
    PHASE5_FORMAL_BASELINES, Phase5OfficialBaselineClient,
    Phase5OfficialBaselineServer, FedJDTrainer, get_phase5_method_spec,
)
from fedjd.experiments.nfjd_phases.metric_utils import summarize_round_history
from fedjd.metrics import (
    compute_f1_scores,
    compute_accuracy, compute_mse_per_task, compute_mae_per_task, compute_r2_per_task,
)

logger = logging.getLogger(__name__)

NFJD_VARIANT_CONFIGS = {
    "nfjd": {
        "use_adaptive_rescaling": False,
        "use_stochastic_gramian": False,
        "stochastic_subset_size": None,
        "recompute_interval": 1,
        "exact_upgrad": True,
        "use_objective_normalization": True,
        "use_global_progress_weights": True,
        "progress_beta": 2.0,
        "progress_min_weight": 0.5,
        "progress_max_weight": 2.0,
        "progress_ema_beta": 0.0,
        "progress_max_change": 0.0,
        "local_momentum_beta": 0.0,
        "global_momentum_beta": 0.0,
        "shared_prox_mu": 0.0,
        "conflict_aware_momentum": False,
        "upload_align_scores": False,
    },
    "nfjd_fast": {
        "use_adaptive_rescaling": True,
        "use_stochastic_gramian": True,
        "stochastic_subset_size": 4,
        "recompute_interval": 4,
        "exact_upgrad": False,
        "use_objective_normalization": True,
        "use_global_progress_weights": True,
        "progress_beta": 0.75,
        "progress_min_weight": 0.7,
        "progress_max_weight": 1.5,
        "progress_ema_beta": 0.8,
        "progress_max_change": 0.1,
        "local_momentum_beta": 0.5,
        "global_momentum_beta": 0.5,
        "shared_prox_mu": 0.0,
        "conflict_aware_momentum": False,
        "upload_align_scores": False,
    },
    "nfjd_noweight": {
        "use_adaptive_rescaling": False,
        "use_stochastic_gramian": False,
        "stochastic_subset_size": None,
        "recompute_interval": 1,
        "exact_upgrad": True,
        "use_objective_normalization": True,
        "use_global_progress_weights": False,
        "progress_beta": 0.0,
        "progress_min_weight": 1.0,
        "progress_max_weight": 1.0,
        "progress_ema_beta": 0.0,
        "progress_max_change": 0.0,
        "local_momentum_beta": 0.0,
        "global_momentum_beta": 0.0,
        "shared_prox_mu": 0.0,
        "conflict_aware_momentum": False,
        "upload_align_scores": False,
    },
    "nfjd_cached": {
        "use_adaptive_rescaling": False,
        "use_stochastic_gramian": False,
        "stochastic_subset_size": None,
        "recompute_interval": 4,
        "exact_upgrad": False,
        "use_objective_normalization": True,
        "use_global_progress_weights": True,
        "progress_beta": 2.0,
        "progress_min_weight": 0.5,
        "progress_max_weight": 2.0,
        "progress_ema_beta": 0.0,
        "progress_max_change": 0.0,
        "local_momentum_beta": 0.0,
        "global_momentum_beta": 0.0,
        "shared_prox_mu": 0.0,
        "conflict_aware_momentum": False,
        "upload_align_scores": False,
    },
    "nfjd_momentum": {
        "use_adaptive_rescaling": False,
        "use_stochastic_gramian": False,
        "stochastic_subset_size": None,
        "recompute_interval": 1,
        "exact_upgrad": True,
        "use_objective_normalization": True,
        "use_global_progress_weights": True,
        "progress_beta": 2.0,
        "progress_min_weight": 0.5,
        "progress_max_weight": 2.0,
        "progress_ema_beta": 0.0,
        "progress_max_change": 0.0,
        "local_momentum_beta": 0.5,
        "global_momentum_beta": 0.5,
        "shared_prox_mu": 0.0,
        "conflict_aware_momentum": False,
        "upload_align_scores": False,
    },
    "nfjd_rescale": {
        "use_adaptive_rescaling": True,
        "use_stochastic_gramian": False,
        "stochastic_subset_size": None,
        "recompute_interval": 1,
        "exact_upgrad": True,
        "use_objective_normalization": True,
        "use_global_progress_weights": True,
        "progress_beta": 2.0,
        "progress_min_weight": 0.5,
        "progress_max_weight": 2.0,
        "progress_ema_beta": 0.0,
        "progress_max_change": 0.0,
        "local_momentum_beta": 0.0,
        "global_momentum_beta": 0.0,
        "shared_prox_mu": 0.0,
        "conflict_aware_momentum": False,
        "upload_align_scores": False,
    },
    "nfjd_softweight": {
        "use_adaptive_rescaling": False,
        "use_stochastic_gramian": False,
        "stochastic_subset_size": None,
        "recompute_interval": 1,
        "exact_upgrad": True,
        "use_objective_normalization": True,
        "use_global_progress_weights": True,
        "progress_beta": 0.75,
        "progress_min_weight": 0.7,
        "progress_max_weight": 1.5,
        "progress_ema_beta": 0.8,
        "progress_max_change": 0.1,
        "local_momentum_beta": 0.0,
        "global_momentum_beta": 0.0,
        "shared_prox_mu": 0.0,
        "conflict_aware_momentum": False,
        "upload_align_scores": False,
    },
    "nfjd_hybrid": {
        "use_adaptive_rescaling": False,
        "use_stochastic_gramian": True,
        "stochastic_subset_size": 4,
        "recompute_interval": 2,
        "exact_upgrad": False,
        "use_objective_normalization": True,
        "use_global_progress_weights": True,
        "progress_beta": 0.5,
        "progress_min_weight": 0.75,
        "progress_max_weight": 1.4,
        "progress_ema_beta": 0.8,
        "progress_max_change": 0.1,
        "local_momentum_beta": 0.2,
        "global_momentum_beta": 0.2,
        "shared_prox_mu": 0.0,
        "conflict_aware_momentum": False,
        "upload_align_scores": False,
    },
    "nfjd_fedprox_shared": {
        "use_adaptive_rescaling": False,
        "use_stochastic_gramian": False,
        "stochastic_subset_size": None,
        "recompute_interval": 1,
        "exact_upgrad": True,
        "use_objective_normalization": True,
        "use_global_progress_weights": True,
        "progress_beta": 2.0,
        "progress_min_weight": 0.5,
        "progress_max_weight": 2.0,
        "progress_ema_beta": 0.0,
        "progress_max_change": 0.0,
        "local_momentum_beta": 0.0,
        "global_momentum_beta": 0.0,
        "shared_prox_mu": 0.001,
        "use_shared_scaffold": False,
        "conflict_aware_momentum": False,
        "upload_align_scores": False,
    },
    "nfjd_scaffold_shared": {
        "use_adaptive_rescaling": False,
        "use_stochastic_gramian": False,
        "stochastic_subset_size": None,
        "recompute_interval": 1,
        "exact_upgrad": True,
        "use_objective_normalization": True,
        "use_global_progress_weights": True,
        "progress_beta": 2.0,
        "progress_min_weight": 0.5,
        "progress_max_weight": 2.0,
        "progress_ema_beta": 0.0,
        "progress_max_change": 0.0,
        "local_momentum_beta": 0.0,
        "global_momentum_beta": 0.0,
        "shared_prox_mu": 0.0,
        "use_shared_scaffold": True,
        "conflict_aware_momentum": False,
        "upload_align_scores": False,
    },
    "nfjd_common_safe": {
        "use_adaptive_rescaling": False,
        "use_stochastic_gramian": False,
        "stochastic_subset_size": None,
        "recompute_interval": 1,
        "exact_upgrad": True,
        "use_objective_normalization": True,
        "use_global_progress_weights": True,
        "progress_beta": 2.0,
        "progress_min_weight": 0.5,
        "progress_max_weight": 2.0,
        "progress_ema_beta": 0.0,
        "progress_max_change": 0.0,
        "local_momentum_beta": 0.0,
        "global_momentum_beta": 0.0,
        "shared_prox_mu": 0.0,
        "conflict_aware_momentum": False,
        "upload_align_scores": False,
        "public_preprocess_alpha": 0.25,
        "public_preprocess_mode": "common_safe_ray",
        "public_preprocess_positive_only": False,
        "public_preprocess_center_mode": "mean",
        "public_preprocess_trim_k": 0,
        "public_preprocess_adaptive_mode": "fixed",
        "public_preprocess_recompute_interval": 5,
        "public_preprocess_probe_batch_size": 32,
        "public_preprocess_steps": 2,
    },
}

ALL_FIELDNAMES = [
    "exp_id", "method", "dataset", "data_split", "m", "seed", "num_rounds",
    "num_clients", "participation_rate", "learning_rate", "local_epochs",
    "use_adaptive_rescaling", "use_stochastic_gramian", "conflict_aware_momentum",
    "use_objective_normalization", "exact_upgrad", "use_global_progress_weights",
    "progress_beta", "progress_min_weight", "progress_max_weight",
    "progress_ema_beta", "progress_max_change", "stochastic_subset_size",
    "recompute_interval", "local_momentum_beta", "global_momentum_beta",
    "shared_prox_mu",
    "use_shared_scaffold",
    "model_arch", "total_local_steps",
    "elapsed_time",
    "avg_upload_bytes", "avg_round_time", "upload_per_client",
    "avg_accuracy", "avg_f1", "min_task_acc", "min_task_f1",
    "avg_mse", "max_mse", "mse_std", "avg_r2",
]


def build_trainer(method, model, client_datasets, objective_fn, m, seed,
                  device, num_rounds, num_clients, participation_rate,
                  learning_rate, local_epochs=1, eval_dataset=None,
                  local_batch_size: int = 256,
                  fedclient_update_scale: float = 1.0,
                  fedclient_normalize_updates: bool = False,
                  fmgda_update_scale: float = 1.0,
                  fedmgda_plus_update_scale: float = 1.0,
                  fedmgda_plus_update_decay: float | None = None,
                  fedmgda_plus_normalize_updates: bool = False,
                  qfedavg_q: float = 0.5,
                  qfedavg_update_scale: float = 1.0,
                  qfedavg_lipschitz: float | None = None,
                  qfedavg_mode: str = "official_delta"):
    batch_size = local_batch_size if local_batch_size and local_batch_size > 0 else 1_000_000_000
    if method in NFJD_VARIANT_CONFIGS:
        cfg = NFJD_VARIANT_CONFIGS[method]
        subset_size = cfg["stochastic_subset_size"] or min(4, m)
        clients = [NFJDClient(
            client_id=i, dataset=client_datasets[i], batch_size=batch_size,
            device=device, local_epochs=local_epochs, learning_rate=learning_rate,
            local_momentum_beta=cfg["local_momentum_beta"],
            use_adaptive_rescaling=cfg["use_adaptive_rescaling"],
            use_stochastic_gramian=cfg["use_stochastic_gramian"],
            stochastic_subset_size=min(subset_size, m),
            stochastic_seed=seed + i,
            conflict_aware_momentum=cfg["conflict_aware_momentum"],
            momentum_min_beta=0.1,
            recompute_interval=cfg["recompute_interval"],
            exact_upgrad=cfg["exact_upgrad"],
            use_objective_normalization=cfg["use_objective_normalization"],
            upload_align_scores=cfg["upload_align_scores"],
            shared_prox_mu=cfg["shared_prox_mu"],
            cone_align_positive_only=False,
        ) for i in range(num_clients)]
        server = NFJDServer(
            model=model, clients=clients, objective_fn=objective_fn,
            participation_rate=participation_rate, learning_rate=learning_rate,
            device=device, global_momentum_beta=cfg["global_momentum_beta"],
            conflict_aware_momentum=cfg["conflict_aware_momentum"], momentum_min_beta=0.1,
            parallel_clients=False, eval_dataset=eval_dataset,
            use_global_progress_weights=cfg["use_global_progress_weights"],
            progress_beta=cfg["progress_beta"],
            progress_min_weight=cfg["progress_min_weight"],
            progress_max_weight=cfg["progress_max_weight"],
            progress_ema_beta=cfg["progress_ema_beta"],
            progress_max_change=cfg["progress_max_change"],
            method_name=method,
            use_shared_scaffold=cfg.get("use_shared_scaffold", False),
            public_preprocess_alpha=cfg.get("public_preprocess_alpha", 0.0),
            public_preprocess_mode=cfg.get("public_preprocess_mode", ""),
            public_preprocess_positive_only=cfg.get("public_preprocess_positive_only", False),
            public_preprocess_center_mode=cfg.get("public_preprocess_center_mode", "mean"),
            public_preprocess_trim_k=cfg.get("public_preprocess_trim_k", 0),
            public_preprocess_adaptive_mode=cfg.get("public_preprocess_adaptive_mode", "fixed"),
            public_preprocess_recompute_interval=cfg.get("public_preprocess_recompute_interval", 1),
            public_preprocess_probe_batch_size=cfg.get("public_preprocess_probe_batch_size"),
            public_preprocess_steps=cfg.get("public_preprocess_steps", 0),
        )
        return NFJDTrainer(server=server, num_rounds=num_rounds)

    if method == "fmgda":
        clients = [
            FMGDAClient(
                client_id=i,
                dataset=client_datasets[i],
                batch_size=batch_size,
                device=device,
                learning_rate=learning_rate,
                local_epochs=local_epochs,
                objective_indices=getattr(client_datasets[i], "objective_indices", None),
            )
            for i in range(num_clients)
        ]
        server = FMGDAServer(
            model=model,
            clients=clients,
            objective_fn=objective_fn,
            participation_rate=participation_rate,
            learning_rate=learning_rate,
            device=device,
            eval_dataset=eval_dataset,
            num_objectives=m,
            update_scale=fmgda_update_scale,
        )
        return FedJDTrainer(server=server, num_rounds=num_rounds)

    if method == "fedmgda_plus":
        clients = [
            FedLocalTrainClient(
                client_id=i,
                dataset=client_datasets[i],
                batch_size=batch_size,
                device=device,
                learning_rate=learning_rate,
                local_epochs=local_epochs,
            )
            for i in range(num_clients)
        ]
        server = FedMGDAPlusServer(
            model=model,
            clients=clients,
            objective_fn=objective_fn,
            participation_rate=participation_rate,
            learning_rate=learning_rate,
            device=device,
            eval_dataset=eval_dataset,
            update_scale=fedmgda_plus_update_scale,
            update_decay=fedmgda_plus_update_decay,
            total_rounds=num_rounds,
            normalize_client_updates=fedmgda_plus_normalize_updates,
        )
        return FedJDTrainer(server=server, num_rounds=num_rounds)

    if method == "fedavg":
        clients = [
            FedLocalTrainClient(
                client_id=i,
                dataset=client_datasets[i],
                batch_size=batch_size,
                device=device,
                learning_rate=learning_rate,
                local_epochs=local_epochs,
            )
            for i in range(num_clients)
        ]
        server = FedAvgServer(
            model=model,
            clients=clients,
            objective_fn=objective_fn,
            participation_rate=participation_rate,
            learning_rate=learning_rate,
            device=device,
            eval_dataset=eval_dataset,
        )
        return FedJDTrainer(server=server, num_rounds=num_rounds)

    if method == "fedavg_upgrad":
        clients = [
            FedJDClient(
                client_id=i,
                dataset=client_datasets[i],
                batch_size=batch_size,
                device=device,
                use_full_loader=True,
                local_epochs=local_epochs,
            )
            for i in range(num_clients)
        ]
        server = FedAvgUPGradServer(
            model=model,
            clients=clients,
            objective_fn=objective_fn,
            participation_rate=participation_rate,
            learning_rate=learning_rate,
            device=device,
            eval_dataset=eval_dataset,
        )
        return FedJDTrainer(server=server, num_rounds=num_rounds)

    if method == "fedclient_upgrad":
        clients = [
            FedLocalTrainClient(
                client_id=i,
                dataset=client_datasets[i],
                batch_size=batch_size,
                device=device,
                learning_rate=learning_rate,
                local_epochs=local_epochs,
            )
            for i in range(num_clients)
        ]
        server = FedClientUPGradServer(
            model=model,
            clients=clients,
            objective_fn=objective_fn,
            participation_rate=participation_rate,
            learning_rate=learning_rate,
            device=device,
            eval_dataset=eval_dataset,
            update_scale=fedclient_update_scale,
            normalize_client_updates=fedclient_normalize_updates,
        )
        return FedJDTrainer(server=server, num_rounds=num_rounds)

    if method == "qfedavg":
        clients = [
            FedLocalTrainClient(
                client_id=i,
                dataset=client_datasets[i],
                batch_size=batch_size,
                device=device,
                learning_rate=learning_rate,
                local_epochs=local_epochs,
            )
            for i in range(num_clients)
        ]
        server = QFedAvgServer(
            model=model,
            clients=clients,
            objective_fn=objective_fn,
            participation_rate=participation_rate,
            learning_rate=learning_rate,
            device=device,
            eval_dataset=eval_dataset,
            q=qfedavg_q,
            update_scale=qfedavg_update_scale,
            qffl_lipschitz=qfedavg_lipschitz,
            mode=qfedavg_mode,
        )
        return FedJDTrainer(server=server, num_rounds=num_rounds)

    if method in PHASE5_FORMAL_BASELINES:
        clients = [
            Phase5OfficialBaselineClient(
                client_id=i,
                dataset=client_datasets[i],
                batch_size=batch_size,
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
            objective_fn=objective_fn,
            participation_rate=participation_rate,
            device=device,
            method_name=method,
            eval_dataset=eval_dataset,
        )
        return FedJDTrainer(server=server, num_rounds=num_rounds)

    raise ValueError(f"Unknown Phase 5 method: {method}")


def run_experiment(exp_id, method, model, client_datasets, objective_fn, m, seed,
                   device, num_rounds, num_clients, participation_rate, learning_rate,
                   model_arch, dataset, data_split, local_epochs=1,
                   eval_dataset=None, fedclient_update_scale: float = 1.0,
                   fedclient_normalize_updates: bool = False,
                   fmgda_update_scale: float = 1.0,
                   fedmgda_plus_update_scale: float = 1.0,
                   qfedavg_q: float = 0.5,
                   qfedavg_update_scale: float = 1.0,
                   qfedavg_mode: str = "official_delta"):

    trainer = build_trainer(
        method=method, model=model, client_datasets=client_datasets,
        objective_fn=objective_fn, m=m, seed=seed, device=device,
        num_rounds=num_rounds, num_clients=num_clients,
        participation_rate=participation_rate, learning_rate=learning_rate,
        local_epochs=local_epochs, eval_dataset=eval_dataset,
        fedclient_update_scale=fedclient_update_scale,
        fedclient_normalize_updates=fedclient_normalize_updates,
        fmgda_update_scale=fmgda_update_scale,
        fedmgda_plus_update_scale=fedmgda_plus_update_scale,
        qfedavg_q=qfedavg_q,
        qfedavg_update_scale=qfedavg_update_scale,
        qfedavg_mode=qfedavg_mode,
    )

    start = time.time()
    history = trainer.fit()
    elapsed = time.time() - start

    round_summary = summarize_round_history(history)
    avg_upload = round_summary["avg_upload_bytes"]
    avg_round_time = round_summary["avg_round_time"]
    upload_per_client = round_summary["upload_per_client"]

    total_local_steps = local_epochs * sum(max(int(getattr(s, "num_sampled_clients", 0)), 0) for s in history)
    spec = get_phase5_method_spec(method)
    nfjd_cfg = NFJD_VARIANT_CONFIGS.get(method)

    row = {
        "exp_id": exp_id, "method": method, "dataset": dataset,
        "data_split": data_split, "m": m, "seed": seed,
        "num_rounds": num_rounds, "num_clients": num_clients,
        "participation_rate": participation_rate,
        "learning_rate": learning_rate, "local_epochs": local_epochs,
        "use_adaptive_rescaling": nfjd_cfg["use_adaptive_rescaling"] if nfjd_cfg else False,
        "use_stochastic_gramian": nfjd_cfg["use_stochastic_gramian"] if nfjd_cfg else False,
        "conflict_aware_momentum": nfjd_cfg["conflict_aware_momentum"] if nfjd_cfg else False,
        "use_objective_normalization": nfjd_cfg["use_objective_normalization"] if nfjd_cfg else False,
        "exact_upgrad": nfjd_cfg["exact_upgrad"] if nfjd_cfg else False,
        "use_global_progress_weights": nfjd_cfg["use_global_progress_weights"] if nfjd_cfg else False,
        "progress_beta": nfjd_cfg["progress_beta"] if nfjd_cfg else "",
        "progress_min_weight": nfjd_cfg["progress_min_weight"] if nfjd_cfg else "",
        "progress_max_weight": nfjd_cfg["progress_max_weight"] if nfjd_cfg else "",
        "progress_ema_beta": nfjd_cfg["progress_ema_beta"] if nfjd_cfg else "",
        "progress_max_change": nfjd_cfg["progress_max_change"] if nfjd_cfg else "",
        "stochastic_subset_size": (nfjd_cfg["stochastic_subset_size"] or min(4, m)) if nfjd_cfg else "",
        "recompute_interval": nfjd_cfg["recompute_interval"] if nfjd_cfg else "",
        "local_momentum_beta": nfjd_cfg["local_momentum_beta"] if nfjd_cfg else "",
        "global_momentum_beta": nfjd_cfg["global_momentum_beta"] if nfjd_cfg else "",
        "shared_prox_mu": nfjd_cfg["shared_prox_mu"] if nfjd_cfg else "",
        "use_shared_scaffold": nfjd_cfg.get("use_shared_scaffold", False) if nfjd_cfg else False,
        "model_arch": model_arch,
        "total_local_steps": total_local_steps,
        "elapsed_time": round(elapsed, 2),
        "avg_upload_bytes": round(avg_upload, 0),
        "avg_round_time": round(avg_round_time, 4),
        "upload_per_client": round(upload_per_client, 0),
        "avg_accuracy": "", "avg_f1": "", "min_task_acc": "", "min_task_f1": "",
        "avg_mse": "", "max_mse": "", "mse_std": "", "avg_r2": "",
        "fedclient_update_scale": fedclient_update_scale if method == "fedclient_upgrad" else "",
        "fedclient_normalize_updates": fedclient_normalize_updates if method == "fedclient_upgrad" else "",
        "fmgda_update_scale": fmgda_update_scale if method == "fmgda" else "",
        "fedmgda_plus_update_scale": fedmgda_plus_update_scale if method == "fedmgda_plus" else "",
        "qfedavg_q": qfedavg_q if method == "qfedavg" else "",
        "qfedavg_update_scale": qfedavg_update_scale if method == "qfedavg" else "",
        "qfedavg_mode": qfedavg_mode if method == "qfedavg" else "",
    }
    logger.info("[%s] %s (%s): steps=%d time=%.1fs", exp_id, spec.display_name, spec.family, total_local_steps, elapsed)
    return row


def evaluate_model(model, test_dataset, device, batch_size=256):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            by = by.to(device)
            pred = model(bx)
            all_preds.append(pred.cpu())
            all_targets.append(by.cpu())
    return torch.cat(all_preds), torch.cat(all_targets)


def fill_classification_metrics(row, predictions, targets, m):
    accs = compute_accuracy(predictions, targets, m)
    f1s = compute_f1_scores(predictions, targets, m)
    row["avg_accuracy"] = round(sum(accs) / m, 4)
    row["avg_f1"] = round(sum(f1s) / m, 4)
    row["min_task_acc"] = round(min(accs), 4)
    row["min_task_f1"] = round(min(f1s), 4)
    return row


def fill_regression_metrics(row, predictions, targets, m):
    mses = compute_mse_per_task(predictions, targets, m)
    r2s = compute_r2_per_task(predictions, targets, m)
    row["avg_mse"] = round(sum(mses) / m, 6)
    row["max_mse"] = round(max(mses), 6)
    row["mse_std"] = round(float(np.std(mses)), 6)
    row["avg_r2"] = round(sum(r2s) / m, 6)
    return row


def write_csv(csv_path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
