#!/usr/bin/env python3
"""Launch the six non-ReLU architecture ablations on separate Kaggle accounts."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from .credential_manager import check_accounts, child_env, load_config
    from .kaggle_pipeline import resolve_kaggle_username
    from .launch_kaggle_pipeline_service import build_systemd_command
except ImportError:
    from credential_manager import check_accounts, child_env, load_config
    from kaggle_pipeline import resolve_kaggle_username
    from launch_kaggle_pipeline_service import build_systemd_command

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KEY_FILE = ROOT / "key.txt"

ABLATION_RUNS = [
    {"account": "account_01", "architecture": "leaky_default"},
    {"account": "account_03", "architecture": "prelu_default"},
    {"account": "account_04", "architecture": "leaky_kaiming"},
    {"account": "account_05", "architecture": "prelu_kaiming"},
    {"account": "account_06", "architecture": "leaky_kaiming_ln"},
    {"account": "account_07", "architecture": "prelu_kaiming_ln"},
]


def _slug(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def credential_dataset_ref(username: str) -> str:
    """Private per-account dataset used to inject the W&B API key."""
    owner = str(username).strip()
    if not owner:
        raise ValueError("username must not be empty")
    return f"{owner}/uav-wandb-credential-arch-ablation"


def credential_dataset_metadata(username: str) -> dict:
    return {
        "title": "uav-wandb-credential-arch-ablation",
        "id": credential_dataset_ref(username),
        "licenses": [{"name": "other"}],
        "isPrivate": True,
    }


def credential_datasets_for_runs(
    usernames: dict[str, str],
    *,
    override: str | None = None,
) -> dict[str, str]:
    """Map each logical account to its own private W&B credential dataset."""
    if override:
        return {name: str(override) for name in usernames}
    return {
        name: credential_dataset_ref(username)
        for name, username in usernames.items()
    }


def build_pipeline_args(
    *,
    account: str,
    architecture: str,
    algorithm: str,
    seed: int,
    credential_dataset: str,
    required_gpu_hours: float,
) -> list[str]:
    algorithm = str(algorithm).strip().lower()
    variant = _slug(architecture)
    prefix = f"uav-{algorithm}-{variant}-h512x256-seed{int(seed)}"
    cuda_graph = "true" if algorithm == "masac" else "false"
    return [
        "--account", str(account),
        "--algorithm", algorithm,
        "--runtime", "kaggle_2xt4",
        "--experiment", "paper_20k",
        "--seed", str(int(seed)),
        "--required-gpu-hours", str(float(required_gpu_hours)),
        "--machine-shape", "NvidiaTeslaT4",
        "--gpu-kernel-slug", f"{prefix}-gpu",
        "--cpu-kernel-slug", f"{prefix}-cpu-viz",
        "--gpu-credential-dataset", str(credential_dataset),
        "--cpu-log-wandb",
        "--cpu-credential-dataset", str(credential_dataset),
        "--gpu-session-timeout-seconds", "43200",
        "--cpu-session-timeout-seconds", "10800",
        "--gpu-wait-timeout-seconds", "43200",
        "--cpu-wait-timeout-seconds", "10800",
        "--output-dir", "pipeline_outputs",
        f"architecture={architecture}",
        "runtime.num_envs=4096",
        f"runtime.cuda_graph_policy_actions={cuda_graph}",
        "runtime.max_gpus=2",
        "runtime.auto_multi_gpu=true",
        "task.scenario.max_steps=3000",
        "experiment.total_episodes=20000",
        f"experiment.name=arch_ablation_{architecture}",
    ]


def _account_map(data: dict) -> dict[str, dict]:
    return {
        str(account.get("name") or f"account_{index + 1:02d}"): account
        for index, account in enumerate(data["kaggle"]["accounts"])
    }


def _validate_quota(data: dict, required_gpu_hours: float) -> None:
    statuses = {row["name"]: row for row in check_accounts(data)}
    for run in ABLATION_RUNS:
        name = run["account"]
        row = statuses.get(name)
        if row is None or not row.get("ok"):
            raise RuntimeError(f"{name} is unavailable")
        remaining = row["resources"]["GPU"].get("remaining_hours")
        if remaining is None or float(remaining) <= float(required_gpu_hours):
            raise RuntimeError(
                f"{name} needs > {required_gpu_hours} GPU-hours, has {remaining}"
            )
        print(f"QUOTA_OK {name} remaining={remaining}h")


def _validate_credential_dataset(
    data: dict,
    *,
    credential_dataset: str,
) -> None:
    accounts = _account_map(data)
    for run in ABLATION_RUNS:
        name = run["account"]
        account = accounts[name]
        env = child_env(data, str(account["token"]))
        proc = subprocess.run(
            [
                "kaggle", "datasets", "files", credential_dataset,
                "--page-size", "1",
            ],
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=90,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"{name} cannot access W&B credential dataset {credential_dataset!r}"
            )
        print(f"WANDB_DATASET_OK {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--algorithm", choices=("masac", "matd3"), default="masac")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--required-gpu-hours", type=float, default=12.01)
    parser.add_argument(
        "--credential-dataset",
        default=None,
        help=(
            "Optional shared W&B credential dataset override. By default each "
            "Kaggle account uses its own private credential dataset ref."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="Run controllers in this process instead of persistent systemd services.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_config(args.key_file)
    _validate_quota(data, float(args.required_gpu_hours))
    accounts = _account_map(data)
    usernames = {}
    for run in ABLATION_RUNS:
        name = run["account"]
        account = accounts[name]
        env = child_env(data, str(account["token"]))
        usernames[name] = resolve_kaggle_username(env=env, explicit=None)
    credential_refs = credential_datasets_for_runs(
        usernames,
        override=args.credential_dataset,
    )
    for name, credential_ref in credential_refs.items():
        _validate_credential_dataset(
            data={
                **data,
                "kaggle": {
                    **data["kaggle"],
                    "accounts": [accounts[name]],
                },
            },
            credential_dataset=credential_ref,
        )

    commands = []
    for run in ABLATION_RUNS:
        pipeline_args = build_pipeline_args(
            account=run["account"],
            architecture=run["architecture"],
            algorithm=args.algorithm,
            seed=args.seed,
            credential_dataset=credential_refs[run["account"]],
            required_gpu_hours=args.required_gpu_hours,
        )
        unit = f"uav-{args.algorithm}-{_slug(run['architecture'])}-seed{args.seed}"
        log_file = ROOT / "pipeline_logs" / f"{unit}.log"
        if args.foreground:
            command = [sys.executable, str(ROOT / "scripts" / "kaggle_pipeline.py"), *pipeline_args]
        else:
            command = build_systemd_command(
                unit=unit,
                log_file=log_file,
                pipeline_args=pipeline_args,
            )
        commands.append((run, command, log_file))

    for run, command, log_file in commands:
        print(
            f"PLAN {run['account']} architecture={run['architecture']} "
            f"log={log_file}"
        )
        if args.dry_run:
            continue
        if args.foreground:
            subprocess.Popen(command, cwd=ROOT)
        else:
            subprocess.run(command, check=True, cwd=ROOT)

    print(f"ABLATION_CONTROLLERS={len(commands)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
