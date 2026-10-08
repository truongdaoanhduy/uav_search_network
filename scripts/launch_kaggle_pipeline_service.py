#!/usr/bin/env python3
"""Launch kaggle_pipeline.py as a persistent user systemd service.

This keeps the GPU->CPU orchestration alive even when the invoking terminal,
agent tool session, or SSH connection disappears. The Kaggle GPU and CPU jobs
remain separate notebook sessions; this service only watches the GPU job and
submits the CPU visualization job after the GPU job reaches COMPLETE.
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "scripts" / "kaggle_pipeline.py"


def _safe_unit_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "_.-" else "-" for ch in value)
    return cleaned.strip("-.") or "uav-kaggle-pipeline"


def build_systemd_command(
    *,
    unit: str,
    log_file: Path,
    pipeline_args: list[str],
) -> list[str]:
    command = [sys.executable, str(PIPELINE), *pipeline_args]
    shell = (
        f"mkdir -p {shlex.quote(str(log_file.parent))} && "
        f"cd {shlex.quote(str(ROOT))} && "
        f"exec {shlex.join(command)} >> {shlex.quote(str(log_file))} 2>&1"
    )
    return [
        "systemd-run",
        "--user",
        f"--unit={_safe_unit_name(unit)}",
        f"--setenv=PATH={os.environ.get('PATH', '')}",
        "--collect",
        "--property=Type=exec",
        "--property=Restart=no",
        "/bin/bash",
        "-lc",
        shell,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run scripts/kaggle_pipeline.py under a persistent user systemd "
            "service so GPU->CPU chaining survives caller disconnects."
        )
    )
    parser.add_argument("--unit", default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument(
        "pipeline_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed verbatim to kaggle_pipeline.py; prefix with --.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if shutil.which("systemd-run") is None:
        raise RuntimeError("systemd-run is unavailable on this host")

    suffix = time.strftime("%Y%m%d-%H%M%S")
    unit = _safe_unit_name(args.unit or f"uav-kaggle-pipeline-{suffix}")
    log_file = (
        args.log_file
        if args.log_file is not None
        else ROOT / "pipeline_logs" / f"{unit}.log"
    ).resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    pipeline_args = list(args.pipeline_args)
    if pipeline_args and pipeline_args[0] == "--":
        pipeline_args = pipeline_args[1:]
    if not pipeline_args:
        raise ValueError("pipeline arguments are required after --")

    command = build_systemd_command(
        unit=unit,
        log_file=log_file,
        pipeline_args=pipeline_args,
    )
    subprocess.run(command, check=True)

    print(f"PIPELINE_SERVICE_UNIT={unit}")
    print(f"PIPELINE_SERVICE_LOG={log_file}")
    print(f"STATUS_COMMAND=systemctl --user status {shlex.quote(unit)}")
    print(f"LOG_COMMAND=tail -f {shlex.quote(str(log_file))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
