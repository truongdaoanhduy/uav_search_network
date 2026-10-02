#!/usr/bin/env python3
"""CPU-only post-training visualization entrypoint.

Examples:
  python visualize.py --handoff /kaggle/input/<gpu-kernel-output>
  python visualize.py --handoff ./uav_training_handoff
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from uav_marl.handoff import (
    find_handoff,
    run_cpu_postprocess_from_handoff,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--handoff",
        type=Path,
        required=True,
        help=(
            "Path to uav_training_handoff.json or a directory containing "
            "exactly one handoff file."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("post_train_visualization"),
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--no-verify-checkpoint",
        action="store_true",
    )
    parser.add_argument(
        "--log-wandb",
        action="store_true",
        help="Optional: log CPU evaluation media to W&B.",
    )
    parser.add_argument(
        "--upload-huggingface",
        action="store_true",
        help="Optional: mirror the checkpoint/manifest to Hugging Face.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    handoff_path = find_handoff(args.handoff)
    result = run_cpu_postprocess_from_handoff(
        handoff_path,
        repo=args.repo,
        output_dir=args.output_dir,
        verify_checkpoint=not args.no_verify_checkpoint,
        log_wandb=bool(args.log_wandb),
        upload_huggingface=bool(args.upload_huggingface),
    )
    print("CPU_VISUALIZATION_COMPLETE")
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
