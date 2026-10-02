#!/usr/bin/env python3
"""Unified MARL training entrypoint.

Examples:
  python train.py algorithm=masac runtime=kaggle_2xt4 experiment=paper_50k
  python train.py algorithm=masac runtime=vast_1gpu experiment=paper_50k
  python train.py algorithm=matd3 runtime=vast_1gpu experiment=paper_50k
  python train.py algorithm=masac algorithm.actor_lr=1e-4
  python train.py runtime=vast_1gpu dry_run=true
"""

from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig

from uav_marl.configuration import describe_run
from uav_marl.runner import run_experiment


@hydra.main(
    version_base="1.3",
    config_path="configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    summary = describe_run(cfg)
    print("RESOLVED_RUN")
    print(json.dumps(summary, indent=2, sort_keys=True))
    result = run_experiment(cfg)
    print("RUN_RESULT")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
