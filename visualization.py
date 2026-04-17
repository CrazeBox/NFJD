from __future__ import annotations

import os
from pathlib import Path

from .core.server import RoundStats


def plot_training_curves(
    history: list[RoundStats],
    output_dir: Path,
    experiment_id: str = "",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not installed, skipping plots.")
        return

    os.makedirs(output_dir / "plots", exist_ok=True)

    if not history:
        return

    num_objectives = len(history[0].objective_values)
    rounds = [s.round_idx for s in history]

    # --- Objective curves ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(num_objectives):
        values = [s.objective_values[i] for s in history]
        ax.plot(rounds, values, label=f"Objective {i}", linewidth=1.5)
    ax.set_xlabel("Round")
    ax.set_ylabel("Objective Value")
    ax.set_title(f"Training Objectives{(' — ' + experiment_id) if experiment_id else ''}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "plots" / "objectives.png", dpi=150)
    plt.close(fig)

    # --- Jacobian norm and direction norm ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    jacobian_norms = [s.jacobian_norm for s in history]
    axes[0].plot(rounds, jacobian_norms, color="tab:blue", linewidth=1.5)
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("||J||_F")
    axes[0].set_title("Jacobian Frobenius Norm")
    axes[0].grid(True, alpha=0.3)

    direction_norms = [s.direction_norm for s in history]
    axes[1].plot(rounds, direction_norms, color="tab:orange", linewidth=1.5)
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("||d||_2")
    axes[1].set_title("Direction Norm")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "plots" / "norms.png", dpi=150)
    plt.close(fig)

    # --- Communication cost ---
    fig, ax = plt.subplots(figsize=(8, 5))
    upload_bytes = [s.upload_bytes for s in history]
    download_bytes = [s.download_bytes for s in history]
    ax.plot(rounds, upload_bytes, label="Upload (bytes)", linewidth=1.5)
    ax.plot(rounds, download_bytes, label="Download (bytes)", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Round")
    ax.set_ylabel("Bytes")
    ax.set_title("Per-Round Communication Cost")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "plots" / "communication.png", dpi=150)
    plt.close(fig)

    # --- Round time breakdown ---
    fig, ax = plt.subplots(figsize=(8, 5))
    client_times = [s.client_compute_time for s in history]
    dir_times = [s.direction_time for s in history]
    update_times = [s.update_time for s in history]
    other_times = [
        max(0, s.round_time - s.client_compute_time - s.direction_time - s.update_time)
        for s in history
    ]
    ax.stackplot(
        rounds,
        client_times,
        dir_times,
        update_times,
        other_times,
        labels=["Client Compute", "Direction", "Update", "Other"],
        alpha=0.8,
    )
    ax.set_xlabel("Round")
    ax.set_ylabel("Time (s)")
    ax.set_title("Per-Round Time Breakdown")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "plots" / "time_breakdown.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved to {output_dir / 'plots'}")
