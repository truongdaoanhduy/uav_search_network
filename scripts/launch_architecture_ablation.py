#!/usr/bin/env python3
"""Launch the six non-ReLU architecture ablations on separate Kaggle accounts."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
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
ARCHITECTURES = [item["architecture"] for item in ABLATION_RUNS]


def select_ablation_runs_from_statuses(
    statuses: list[dict],
    *,
    required_gpu_hours: float,
) -> list[dict]:
    """Select six Kaggle accounts with strictly more than the required GPU quota."""
    eligible = []
    for row in statuses:
        if not row.get("ok"):
            continue
        remaining = (
            row.get("resources", {})
            .get("GPU", {})
            .get("remaining_hours")
        )
        if remaining is None or float(remaining) <= float(required_gpu_hours):
            continue
        eligible.append((str(row["name"]), float(remaining)))
    eligible.sort(key=lambda item: (-item[1], item[0]))
    if len(eligible) < len(ARCHITECTURES):
        details = ", ".join(f"{name}={hours}h" for name, hours in eligible) or "none"
        raise RuntimeError(
            f"Need {len(ARCHITECTURES)} Kaggle accounts with GPU quota > "
            f"{required_gpu_hours}h; eligible: {details}"
        )
    selected = sorted(eligible[: len(ARCHITECTURES)], key=lambda item: item[0])
    return [
        {
            "account": selected[index][0],
            "remaining_gpu_hours": selected[index][1],
            "architecture": architecture,
        }
        for index, architecture in enumerate(ARCHITECTURES)
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


def write_service_credential_bundle(
    folder: Path,
    data: dict,
    *,
    kaggle_username: str,
    hf_repo_id: str,
) -> None:
    """Write shared W&B/HF service credentials without copying Kaggle tokens."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    files = {
        "wandb_api_key.txt": str(data["wandb"]["api_key"]),
        "hf_token.txt": str(data["huggingface"]["token"]),
        "hf_repo_id.txt": str(hf_repo_id),
    }
    for filename, value in files.items():
        if not value.strip():
            raise RuntimeError(f"shared credential {filename} is empty")
        path = folder / filename
        path.write_text(value.strip())
        path.chmod(0o600)
    (folder / "dataset-metadata.json").write_text(
        json.dumps(
            credential_dataset_metadata(kaggle_username),
            indent=2,
            sort_keys=True,
        )
    )


def resolve_shared_hf_repo_id(data: dict) -> str:
    explicit = str(data.get("huggingface", {}).get("repo_id") or "").strip()
    if explicit:
        return explicit
    from huggingface_hub import HfApi

    info = HfApi(token=str(data["huggingface"]["token"])).whoami()
    username = str(info.get("name") if isinstance(info, dict) else "").strip()
    if not username:
        raise RuntimeError("could not resolve Hugging Face username")
    return f"{username}/uav-search-target-checkpoints"


def provision_service_credential_dataset(
    data: dict,
    *,
    account: dict,
    kaggle_username: str,
    hf_repo_id: str,
) -> str:
    """Create or refresh a private credential dataset owned by one Kaggle account."""
    ref = credential_dataset_ref(kaggle_username)
    env = child_env(data, str(account["token"]))
    with tempfile.TemporaryDirectory(prefix="uav_shared_credentials_") as tmp:
        folder = Path(tmp)
        write_service_credential_bundle(
            folder,
            data,
            kaggle_username=kaggle_username,
            hf_repo_id=hf_repo_id,
        )
        exists = subprocess.run(
            [
                "kaggle", "datasets", "files", ref,
                "--page-size", "1", "--format", "json",
            ],
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=90,
        ).returncode == 0
        command = (
            [
                "kaggle", "datasets", "version", "-p", str(folder),
                "-m", "Refresh shared service credentials", "-q",
            ]
            if exists
            else ["kaggle", "datasets", "create", "-p", str(folder), "-q"]
        )
        proc = subprocess.run(
            command,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=300,
        )
        if proc.returncode != 0:
            message = proc.stderr or proc.stdout or "credential dataset update failed"
            for value in (
                str(account.get("token", "")),
                str(data["wandb"].get("api_key", "")),
                str(data["huggingface"].get("token", "")),
            ):
                if value:
                    message = message.replace(value, "<redacted>")
            raise RuntimeError(message[-2000:])
    return ref


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
        "experiment.postprocess.upload_huggingface=true",
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
    for name, account in accounts.items():
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
                f"{name} cannot access service credential dataset {credential_dataset!r}"
            )
        print(f"SERVICE_DATASET_OK {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--algorithm", choices=("masac", "matd3"), default="masac")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--required-gpu-hours", type=float, default=12.0)
    parser.add_argument(
        "--credential-dataset",
        default=None,
        help=(
            "Optional shared W&B credential dataset override. By default each "
            "Kaggle account uses its own private credential dataset ref."
        ),
    )
    parser.add_argument(
        "--no-provision-credentials",
        action="store_true",
        help="Validate existing per-account credential datasets without creating/updating them.",
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
    runs = select_ablation_runs_from_statuses(
        check_accounts(data),
        required_gpu_hours=float(args.required_gpu_hours),
    )
    for run in runs:
        print(
            f"QUOTA_OK {run['account']} remaining={run['remaining_gpu_hours']}h "
            f"> {float(args.required_gpu_hours)}h"
        )

    accounts = _account_map(data)
    usernames = {}
    for run in runs:
        name = run["account"]
        account = accounts[name]
        env = child_env(data, str(account["token"]))
        usernames[name] = resolve_kaggle_username(env=env, explicit=None)

    hf_repo_id = resolve_shared_hf_repo_id(data)
    if args.credential_dataset:
        credential_refs = credential_datasets_for_runs(
            usernames,
            override=args.credential_dataset,
        )
    else:
        credential_refs = {}
        for run in runs:
            name = run["account"]
            ref = credential_dataset_ref(usernames[name])
            if not args.no_provision_credentials:
                ref = provision_service_credential_dataset(
                    data,
                    account=accounts[name],
                    kaggle_username=usernames[name],
                    hf_repo_id=hf_repo_id,
                )
                print(f"SERVICE_CREDENTIAL_DATASET_READY {name} {ref}")
            credential_refs[name] = ref

    for run in runs:
        name = run["account"]
        _validate_credential_dataset(
            data={
                **data,
                "kaggle": {
                    **data["kaggle"],
                    "accounts": [accounts[name]],
                },
            },
            credential_dataset=credential_refs[name],
        )

    commands = []
    for run in runs:
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
            command = [
                sys.executable,
                str(ROOT / "scripts" / "kaggle_pipeline.py"),
                *pipeline_args,
            ]
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
            f"quota={run['remaining_gpu_hours']}h log={log_file}"
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
