from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

from .nfjd_server import NFJDServer, RoundStats

logger = logging.getLogger(__name__)


class NFJDTrainer:
    def __init__(
        self,
        server: NFJDServer,
        num_rounds: int,
        output_dir: str | None = None,
    ) -> None:
        self.server = server
        self.num_rounds = num_rounds
        self.output_dir = Path(output_dir) if output_dir else None
        self.initial_objectives: list[float] | None = None

    def fit(self) -> list[RoundStats]:
        history: list[RoundStats] = []
        self.initial_objectives = self.server.evaluate_global_objectives()
        self.server.set_initial_objectives(self.initial_objectives)
        logger.info("Initial objectives: %s", ", ".join(f"{v:.4f}" for v in self.initial_objectives))
        for round_idx in range(self.num_rounds):
            stats = self.server.run_round(round_idx)
            history.append(stats)
            obj_str = ", ".join(f"{v:.4f}" for v in stats.objective_values)
            logger.info(
                "Round %d | sampled=%s | obj=[%s] | ||Δθ||=%.4f | ||v||=%.4f | "
                "scale=%.2f | q=%s | time=%.3fs | upload=%d B",
                round_idx,
                stats.sampled_client_ids,
                obj_str,
                stats.delta_norm,
                stats.global_momentum_norm,
                stats.avg_rescale_factor,
                [round(v, 3) for v in stats.task_weights],
                stats.round_time,
                stats.upload_bytes,
            )

        if self.output_dir:
            self._save_results(history)
        return history

    def _save_results(self, history: list[RoundStats]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = self.output_dir / "metrics.csv"
        fieldnames = [
            "round_idx", "num_sampled_clients", "objective_values",
            "delta_norm", "global_momentum_norm", "round_time",
            "upload_bytes", "download_bytes", "client_compute_time",
            "aggregation_time", "update_time", "avg_rescale_factor",
            "avg_local_epochs", "task_weights", "task_weight_gap", "method_name",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in history:
                row = {
                    "round_idx": s.round_idx,
                    "num_sampled_clients": s.num_sampled_clients,
                    "objective_values": str(s.objective_values),
                    "delta_norm": round(s.delta_norm, 6),
                    "global_momentum_norm": round(s.global_momentum_norm, 6),
                    "round_time": round(s.round_time, 6),
                    "upload_bytes": s.upload_bytes,
                    "download_bytes": s.download_bytes,
                    "client_compute_time": round(s.client_compute_time, 6),
                    "aggregation_time": round(s.aggregation_time, 6),
                    "update_time": round(s.update_time, 6),
                    "avg_rescale_factor": round(s.avg_rescale_factor, 4),
                    "avg_local_epochs": s.avg_local_epochs,
                    "task_weights": str([round(v, 6) for v in s.task_weights]),
                    "task_weight_gap": round(s.task_weight_gap, 6),
                    "method_name": s.method_name,
                }
                writer.writerow(row)

        summary_path = self.output_dir / "summary.md"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"# NFJD Training Summary\n\n")
            f.write(f"- Method: {history[0].method_name}\n")
            f.write(f"- Rounds: {len(history)}\n")
            initial = self.initial_objectives if self.initial_objectives is not None else history[0].objective_values
            f.write(f"- Initial objectives: {initial}\n")
            f.write(f"- Final objectives: {history[-1].objective_values}\n")
            m = len(history[0].objective_values)
            init = initial
            final = history[-1].objective_values
            for j in range(m):
                delta = final[j] - init[j]
                ri = (init[j] - final[j]) / abs(init[j]) if abs(init[j]) > 1e-10 else 0.0
                f.write(f"  - obj_{j}: {init[j]:.6f} → {final[j]:.6f} (Δ={delta:.6f}, RI={ri:.4f})\n")
            f.write(f"- Total time: {sum(s.round_time for s in history):.2f}s\n")
            f.write(f"- Avg rescale factor: {sum(s.avg_rescale_factor for s in history)/len(history):.4f}\n")
