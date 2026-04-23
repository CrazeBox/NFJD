from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from fedjd.experiments.nfjd_phases import run_phase5_celeba, run_phase5_multimnist, run_phase5_riverflow


RESULTS_DIR = Path("results/nfjd_phase5")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(RESULTS_DIR / "phase5_suite.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def main() -> None:
    start = time.time()
    jobs = [
        ("MultiMNIST", run_phase5_multimnist.main),
        ("CelebA", run_phase5_celeba.main),
        ("RiverFlow", run_phase5_riverflow.main),
    ]

    logger.info("Starting Phase 5 formal benchmark suite")
    for name, entrypoint in jobs:
        logger.info("Running %s benchmark", name)
        entrypoint()
        logger.info("Completed %s benchmark", name)

    elapsed = time.time() - start
    logger.info("Phase 5 suite completed in %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
