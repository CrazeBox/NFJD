from __future__ import annotations

import csv
import logging
import os
import time
from pathlib import Path

import torch

from .server import RoundStats

logger = logging.getLogger(__name__)


class FedJDTrainer:
    def __init__(
        self,
        server,
        num_rounds: int,
        output_dir: str | None = None,
        save_checkpoints: bool = False,
        checkpoint_interval: int = 10,
    ) -> None:
        self.server = server
        self.num_rounds = num_rounds
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_checkpoints = save_checkpoints
        self.checkpoint_interval = checkpoint_interval

    def fit(self) -> list[RoundStats]:
        if self.output_dir:
            self._setup_output_dir()

        history: list[RoundStats] = []
        total_start = time.time()

        logger.info("Starting FedJD training for %d rounds", self.num_rounds)
        initial_objectives = self.server.evaluate_global_objectives()
        logger.info("Initial objectives: %s", _fmt_objectives(initial_objectives))

        for round_idx in range(self.num_rounds):
            stats = self.server.run_round(round_idx)
            history.append(stats)

            self._log_round(stats)

            if self.save_checkpoints and (round_idx + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(round_idx)

            if stats.nan_inf_count > 0:
                logger.warning(
                    "Round %d: %d NaN/Inf values detected!",
                    round_idx,
                    stats.nan_inf_count,
                )

        total_time = time.time() - total_start
        logger.info("Training complete. Total time: %.2f seconds", total_time)

        if self.output_dir:
            self._save_metrics_csv(history)
            self._save_summary(history, total_time)

        return history

    def _setup_output_dir(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_dir / "checkpoints", exist_ok=True)
        os.makedirs(self.output_dir / "plots", exist_ok=True)

    def _log_round(self, stats: RoundStats) -> None:
        objectives = _fmt_objectives(stats.objective_values)
        logger.info(
            "Round %02d | sampled=%s | obj=[%s] | ||J||=%.4f | ||d||=%.4f | "
            "time=%.3fs | upload=%d B | J/grad=%.1fx | nan/inf=%d",
            stats.round_idx,
            stats.sampled_client_ids,
            objectives,
            stats.jacobian_norm,
            stats.direction_norm,
            stats.round_time,
            stats.upload_bytes,
            stats.jacobian_vs_gradient_ratio,
            stats.nan_inf_count,
        )

    def _save_checkpoint(self, round_idx: int) -> None:
        path = self.output_dir / "checkpoints" / f"round_{round_idx:04d}.pt"
        torch.save(self.server.model.state_dict(), path)
        logger.debug("Checkpoint saved: %s", path)

    def _save_metrics_csv(self, history: list[RoundStats]) -> None:
        if not history:
            return
        num_objectives = len(history[0].objective_values)
        fieldnames = [
            "round",
            "sampled_clients",
            "num_sampled",
        ]
        for i in range(num_objectives):
            fieldnames.append(f"objective_{i}")
        fieldnames.extend([
            "direction_norm",
            "jacobian_norm",
            "round_time",
            "upload_bytes",
            "download_bytes",
            "nan_inf_count",
            "client_compute_time",
            "client_serialize_time",
            "aggregation_time",
            "direction_time",
            "update_time",
            "client_peak_memory_mb",
            "server_peak_memory_mb",
            "jacobian_upload_per_client",
            "gradient_upload_per_client",
            "jacobian_vs_gradient_ratio",
            "compressor_name",
            "compressed_upload_per_client",
            "compression_ratio",
            "is_full_sync_round",
            "local_steps",
            "method_name",
        ])

        csv_path = self.output_dir / "metrics.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in history:
                row = {
                    "round": s.round_idx,
                    "sampled_clients": s.sampled_client_ids,
                    "num_sampled": s.num_sampled_clients,
                }
                for i, v in enumerate(s.objective_values):
                    row[f"objective_{i}"] = v
                row.update({
                    "direction_norm": s.direction_norm,
                    "jacobian_norm": s.jacobian_norm,
                    "round_time": s.round_time,
                    "upload_bytes": s.upload_bytes,
                    "download_bytes": s.download_bytes,
                    "nan_inf_count": s.nan_inf_count,
                    "client_compute_time": s.client_compute_time,
                    "client_serialize_time": s.client_serialize_time,
                    "aggregation_time": s.aggregation_time,
                    "direction_time": s.direction_time,
                    "update_time": s.update_time,
                    "client_peak_memory_mb": s.client_peak_memory_mb,
                    "server_peak_memory_mb": s.server_peak_memory_mb,
                    "jacobian_upload_per_client": s.jacobian_upload_per_client,
                    "gradient_upload_per_client": s.gradient_upload_per_client,
                    "jacobian_vs_gradient_ratio": s.jacobian_vs_gradient_ratio,
                    "compressor_name": s.compressor_name,
                    "compressed_upload_per_client": s.compressed_upload_per_client,
                    "compression_ratio": s.compression_ratio,
                    "is_full_sync_round": s.is_full_sync_round,
                    "local_steps": s.local_steps,
                    "method_name": s.method_name,
                })
                writer.writerow(row)
        logger.info("Metrics saved to %s", csv_path)

    def _save_summary(self, history: list[RoundStats], total_time: float) -> None:
        num_objectives = len(history[0].objective_values) if history else 0
        initial = history[0].objective_values if history else []
        final = history[-1].objective_values if history else []

        total_upload = sum(s.upload_bytes for s in history)
        total_download = sum(s.download_bytes for s in history)
        total_nan_inf = sum(s.nan_inf_count for s in history)

        agg_name = getattr(self.server, "aggregator", None)
        agg_str = type(agg_name).__name__ if agg_name else "N/A"

        lines = [
            "# FedJD Experiment Summary",
            "",
            "## Configuration",
            f"- Number of rounds: {self.num_rounds}",
            f"- Number of clients: {len(self.server.clients)}",
            f"- Participation rate: {self.server.participation_rate}",
            f"- Learning rate: {self.server.learning_rate}",
            f"- Aggregator: {agg_str}",
            f"- Device: {self.server.device}",
            "",
            "## Results",
            f"- Total training time: {total_time:.2f}s",
            f"- Total upload bytes: {total_upload}",
            f"- Total download bytes: {total_download}",
            f"- Total NaN/Inf count: {total_nan_inf}",
            "",
            "## Objective Values",
            f"- Number of objectives: {num_objectives}",
        ]
        for i in range(num_objectives):
            init_v = initial[i] if i < len(initial) else float("nan")
            final_v = final[i] if i < len(final) else float("nan")
            delta = final_v - init_v
            lines.append(f"  - Objective {i}: initial={init_v:.6f}, final={final_v:.6f}, delta={delta:.6f}")

        if history:
            avg_jac_grad_ratio = sum(s.jacobian_vs_gradient_ratio for s in history) / len(history)
            lines.extend([
                "",
                "## Per-Round Averages",
                f"- Avg round time: {sum(s.round_time for s in history) / len(history):.4f}s",
                f"- Avg client compute time: {sum(s.client_compute_time for s in history) / len(history):.4f}s",
                f"- Avg direction time: {sum(s.direction_time for s in history) / len(history):.4f}s",
                f"- Avg direction norm: {sum(s.direction_norm for s in history) / len(history):.4f}",
                f"- Avg Jacobian norm: {sum(s.jacobian_norm for s in history) / len(history):.4f}",
                f"- Avg upload bytes: {sum(s.upload_bytes for s in history) / len(history):.0f}",
                f"- Avg Jacobian/Gradient upload ratio: {avg_jac_grad_ratio:.2f}x",
                f"- Avg client peak memory: {sum(s.client_peak_memory_mb for s in history) / len(history):.1f} MB",
                f"- Avg server peak memory: {sum(s.server_peak_memory_mb for s in history) / len(history):.1f} MB",
            ])

        summary_path = self.output_dir / "summary.md"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        logger.info("Summary saved to %s", summary_path)


def _fmt_objectives(values: list[float]) -> str:
    return ", ".join(f"{v:.4f}" for v in values)
