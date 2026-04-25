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
        logger.info("Training started for %d rounds", self.num_rounds)
        for round_idx in range(self.num_rounds):
            stats = self.server.run_round(round_idx)
            history.append(stats)
            logger.info(
                "Round %d | sampled=%s | time=%.3fs | upload=%d B | download=%d B",
                round_idx,
                stats.sampled_client_ids,
                stats.round_time,
                stats.upload_bytes,
                stats.download_bytes,
            )

        if self.output_dir:
            self._save_results(history)
        return history

    def _save_results(self, history: list[RoundStats]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = self.output_dir / "metrics.csv"
        fieldnames = [
            "round_idx", "num_sampled_clients", "round_time",
            "upload_bytes", "download_bytes", "client_compute_time",
            "aggregation_time", "update_time", "avg_local_epochs", "method_name",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in history:
                row = {
                    "round_idx": s.round_idx,
                    "num_sampled_clients": s.num_sampled_clients,
                    "round_time": round(s.round_time, 6),
                    "upload_bytes": s.upload_bytes,
                    "download_bytes": s.download_bytes,
                    "client_compute_time": round(s.client_compute_time, 6),
                    "aggregation_time": round(s.aggregation_time, 6),
                    "update_time": round(s.update_time, 6),
                    "avg_local_epochs": s.avg_local_epochs,
                    "method_name": s.method_name,
                }
                writer.writerow(row)

        summary_path = self.output_dir / "summary.md"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"# NFJD Training Summary\n\n")
            f.write(f"- Method: {history[0].method_name}\n")
            f.write(f"- Rounds: {len(history)}\n")
            f.write(f"- Total time: {sum(s.round_time for s in history):.2f}s\n")
            f.write(f"- Avg round time: {sum(s.round_time for s in history)/len(history):.4f}s\n")
            f.write(f"- Avg upload bytes: {sum(s.upload_bytes for s in history)/len(history):.0f}\n")
            f.write(f"- Avg download bytes: {sum(s.download_bytes for s in history)/len(history):.0f}\n")
