#!/usr/bin/env python3
"""Automate Kaggle GPU training -> GPU shutdown -> CPU visualization.

The GPU kernel only trains and writes a checkpoint handoff package. Once Kaggle
reports that kernel COMPLETE, this launcher starts a second CPU-only kernel with
the GPU kernel output attached through kernel_sources. The CPU kernel verifies
the checkpoint SHA256, runs the authoritative CPU post-process path, and renders
PNG/MP4 without consuming GPU quota.

Kaggle API/CLI pushes do not propagate interactive Notebook Secrets. The GPU
stage therefore still requires an approved W&B credential mechanism available
inside the remote kernel. This launcher deliberately keeps credential transport
separate from the checkpoint handoff; the CPU stage does not need W&B access to
consume the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

try:
    from .credential_manager import (
        child_env,
        load_config,
        pick_account,
    )
except ImportError:
    from credential_manager import (
        child_env,
        load_config,
        pick_account,
    )

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KEY_FILE = ROOT / "key.txt"
TERMINAL_STATES = {"complete", "error", "cancelled", "failed"}


def _run(
    args: list[str],
    *,
    env: dict[str, str],
    cwd: Path | None = None,
    timeout: int = 300,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        args,
        env=env,
        cwd=None if cwd is None else str(cwd),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if check and proc.returncode != 0:
        message = (
            f"command failed rc={proc.returncode}: {' '.join(args)}\n"
            f"stdout:\n{proc.stdout[-4000:]}\n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )
        raise RuntimeError(message)
    return proc


def _git_commit(repo: Path) -> str:
    proc = _run(
        ["git", "rev-parse", "HEAD"],
        env=os.environ.copy(),
        cwd=repo,
    )
    return proc.stdout.strip()


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return text.strip("-")


def _parse_ref(rows: Any) -> str | None:
    if not isinstance(rows, list) or not rows:
        return None
    for row in rows:
        if not isinstance(row, dict):
            continue
        ref = row.get("ref") or row.get("id")
        if isinstance(ref, str) and "/" in ref:
            return ref.split("/", 1)[0]
    return None


def resolve_kaggle_username(
    *,
    env: dict[str, str],
    explicit: str | None,
) -> str:
    if explicit:
        return explicit.strip()
    if env.get("KAGGLE_USERNAME"):
        return str(env["KAGGLE_USERNAME"]).strip()

    probes = [
        [
            "kaggle",
            "kernels",
            "list",
            "--mine",
            "--page-size",
            "1",
            "--format",
            "json",
        ],
        [
            "kaggle",
            "datasets",
            "list",
            "--mine",
            "--page-size",
            "1",
            "--format",
            "json",
        ],
    ]
    for command in probes:
        proc = _run(command, env=env, check=False)
        if proc.returncode != 0:
            continue
        try:
            username = _parse_ref(json.loads(proc.stdout))
        except json.JSONDecodeError:
            username = None
        if username:
            return username

    raise RuntimeError(
        "could not resolve Kaggle username for the selected account; "
        "pass --username explicitly"
    )


def _kernel_status(ref: str, *, env: dict[str, str]) -> str:
    proc = _run(
        ["kaggle", "kernels", "status", ref],
        env=env,
        check=False,
    )
    text = (proc.stdout + "\n" + proc.stderr).strip().lower()
    for state in (
        "complete",
        "error",
        "cancelled",
        "failed",
        "running",
        "queued",
        "pending",
    ):
        if state in text:
            return state
    if proc.returncode != 0:
        raise RuntimeError(
            f"could not read Kaggle kernel status for {ref}: {text[-2000:]}"
        )
    return "unknown"


def _kernel_logs(ref: str, *, env: dict[str, str]) -> str:
    proc = _run(
        ["kaggle", "kernels", "logs", ref],
        env=env,
        check=False,
        timeout=180,
    )
    return (proc.stdout + "\n" + proc.stderr)[-12000:]


def wait_for_kernel(
    ref: str,
    *,
    env: dict[str, str],
    poll_seconds: int,
    timeout_seconds: int,
) -> str:
    deadline = time.monotonic() + timeout_seconds
    previous = None
    while True:
        status = _kernel_status(ref, env=env)
        if status != previous:
            print(f"{ref}: {status}", flush=True)
            previous = status
        if status in TERMINAL_STATES:
            if status != "complete":
                print(_kernel_logs(ref, env=env), flush=True)
                raise RuntimeError(
                    f"Kaggle kernel {ref} ended with status {status}"
                )
            return status
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for Kaggle kernel {ref}"
            )
        time.sleep(max(5, int(poll_seconds)))


def _write_metadata(
    folder: Path,
    *,
    ref: str,
    title: str,
    code_file: str,
    enable_gpu: bool,
    machine_shape: str,
    kernel_sources: list[str] | None = None,
    dataset_sources: list[str] | None = None,
) -> None:
    metadata = {
        "id": ref,
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": bool(enable_gpu),
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": machine_shape if enable_gpu else "",
        "dataset_sources": list(dataset_sources or []),
        "competition_sources": [],
        "kernel_sources": list(kernel_sources or []),
        "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )


def _gpu_script(
    *,
    commit: str,
    algorithm: str,
    runtime: str,
    experiment: str,
    seed: int,
    overrides: list[str],
    wandb_credential_dataset: str | None = None,
) -> str:
    override_literals = json.dumps(list(overrides))
    return textwrap.dedent(
        f"""
        import json
        import os
        import subprocess
        import sys
        from pathlib import Path

        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        COMMIT = {commit!r}
        WANDB_CREDENTIAL_DATASET = {wandb_credential_dataset!r}
        REPO_URL = "https://github.com/truongdaoanhduy/uav_search_network.git"
        REPO = Path("/tmp/uav_search_network")
        if REPO.exists():
            subprocess.run(["rm", "-rf", str(REPO)], check=True)
        subprocess.run(["git", "clone", "--quiet", REPO_URL, str(REPO)], check=True)
        subprocess.run(["git", "checkout", "--quiet", COMMIT], cwd=REPO, check=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO / "requirements.txt")],
            check=True,
        )

        if WANDB_CREDENTIAL_DATASET:
            dataset_slug = str(WANDB_CREDENTIAL_DATASET).split("/", 1)[-1]
            preferred = Path("/kaggle/input") / dataset_slug / "wandb_api_key.txt"
            if preferred.is_file():
                key_path = preferred
            else:
                matches = sorted(Path("/kaggle/input").rglob("wandb_api_key.txt"))
                if len(matches) != 1:
                    raise RuntimeError(
                        "expected exactly one W&B credential file, found "
                        + repr([str(path) for path in matches])
                    )
                key_path = matches[0]
            api_key = key_path.read_text().strip()
            if not api_key:
                raise RuntimeError("W&B credential file is empty")
            os.environ["WANDB_API_KEY"] = api_key
            os.environ.setdefault("WANDB_SILENT", "true")

        command = [
            sys.executable,
            str(REPO / "train.py"),
            "algorithm={algorithm}",
            "runtime={runtime}",
            "experiment={experiment}",
            "seed={int(seed)}",
        ] + {override_literals}
        print("GPU_TRAIN_COMMAND", " ".join(command), flush=True)
        subprocess.run(command, cwd=REPO, check=True)

        handoffs = sorted(Path("/kaggle/working/uav_training_handoff").rglob("uav_training_handoff.json"))
        if len(handoffs) != 1:
            raise RuntimeError(f"expected exactly one GPU handoff, found {{handoffs}}")
        print("GPU_TRAINING_HANDOFF_READY", handoffs[0], flush=True)
        """
    ).strip() + "\n"


def _cpu_script(
    *,
    commit: str,
    log_wandb: bool = False,
    wandb_credential_dataset: str | None = None,
) -> str:
    if log_wandb and not wandb_credential_dataset:
        raise ValueError(
            "wandb_credential_dataset is required when CPU W&B logging is enabled"
        )
    return textwrap.dedent(
        f"""
        import os
        import subprocess
        import sys
        from pathlib import Path

        COMMIT = {commit!r}
        LOG_WANDB = {bool(log_wandb)!r}
        WANDB_CREDENTIAL_DATASET = {wandb_credential_dataset!r}
        REPO_URL = "https://github.com/truongdaoanhduy/uav_search_network.git"
        REPO = Path("/tmp/uav_search_network")
        if REPO.exists():
            subprocess.run(["rm", "-rf", str(REPO)], check=True)
        subprocess.run(["git", "clone", "--quiet", REPO_URL, str(REPO)], check=True)
        subprocess.run(["git", "checkout", "--quiet", COMMIT], cwd=REPO, check=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO / "requirements.txt")],
            check=True,
        )

        import torch
        if torch.cuda.is_available() or torch.cuda.device_count() != 0:
            raise RuntimeError(
                f"CPU visualization kernel unexpectedly has CUDA: "
                f"available={{torch.cuda.is_available()}} count={{torch.cuda.device_count()}}"
            )

        if LOG_WANDB:
            dataset_slug = str(WANDB_CREDENTIAL_DATASET).split("/", 1)[-1]
            preferred = Path("/kaggle/input") / dataset_slug / "wandb_api_key.txt"
            if preferred.is_file():
                key_path = preferred
            else:
                matches = sorted(Path("/kaggle/input").rglob("wandb_api_key.txt"))
                if len(matches) != 1:
                    raise RuntimeError(
                        "expected exactly one W&B credential file, found "
                        + repr([str(path) for path in matches])
                    )
                key_path = matches[0]
            api_key = key_path.read_text().strip()
            if not api_key:
                raise RuntimeError("W&B credential file is empty")
            os.environ["WANDB_API_KEY"] = api_key
            os.environ.setdefault("WANDB_SILENT", "true")

        command = [
            sys.executable,
            str(REPO / "visualize.py"),
            "--handoff",
            "/kaggle/input",
            "--output-dir",
            "/kaggle/working/post_train_visualization",
            "--repo",
            str(REPO),
        ]
        if LOG_WANDB:
            command.append("--log-wandb")
        print("CPU_VISUALIZATION_COMMAND", " ".join(command), flush=True)
        subprocess.run(command, cwd=REPO, check=True)

        completion = Path("/kaggle/working/post_train_visualization/cpu_visualization_complete.json")
        if not completion.is_file() or completion.stat().st_size <= 0:
            raise RuntimeError("CPU visualization completion manifest is missing")
        print("CPU_SESSION_VISUALIZATION_COMPLETE", completion, flush=True)
        """
    ).strip() + "\n"


def build_kernels(
    build_root: Path,
    *,
    username: str,
    commit: str,
    algorithm: str,
    runtime: str,
    experiment: str,
    seed: int,
    gpu_kernel_slug: str,
    cpu_kernel_slug: str,
    machine_shape: str,
    overrides: list[str],
    gpu_dataset_sources: list[str] | None = None,
    cpu_log_wandb: bool = False,
    cpu_dataset_sources: list[str] | None = None,
) -> tuple[Path, Path, str, str]:
    gpu_ref = f"{username}/{gpu_kernel_slug}"
    cpu_ref = f"{username}/{cpu_kernel_slug}"

    gpu_dir = build_root / "gpu"
    cpu_dir = build_root / "cpu"
    gpu_dir.mkdir(parents=True, exist_ok=True)
    cpu_dir.mkdir(parents=True, exist_ok=True)

    gpu_code = f"{gpu_kernel_slug}.py"
    cpu_code = f"{cpu_kernel_slug}.py"
    gpu_dataset_sources = list(gpu_dataset_sources or [])
    if len(gpu_dataset_sources) > 1:
        raise ValueError(
            "GPU W&B credential supports at most one dataset source"
        )
    (gpu_dir / gpu_code).write_text(
        _gpu_script(
            commit=commit,
            algorithm=algorithm,
            runtime=runtime,
            experiment=experiment,
            seed=seed,
            overrides=overrides,
            wandb_credential_dataset=(
                gpu_dataset_sources[0] if gpu_dataset_sources else None
            ),
        )
    )
    cpu_dataset_sources = list(cpu_dataset_sources or [])
    if cpu_log_wandb and len(cpu_dataset_sources) != 1:
        raise ValueError(
            "CPU W&B logging requires exactly one credential dataset source"
        )
    (cpu_dir / cpu_code).write_text(
        _cpu_script(
            commit=commit,
            log_wandb=bool(cpu_log_wandb),
            wandb_credential_dataset=(
                cpu_dataset_sources[0] if cpu_log_wandb else None
            ),
        )
    )

    _write_metadata(
        gpu_dir,
        ref=gpu_ref,
        title=gpu_kernel_slug.replace("-", " ").title(),
        code_file=gpu_code,
        enable_gpu=True,
        machine_shape=machine_shape,
        dataset_sources=gpu_dataset_sources,
    )
    _write_metadata(
        cpu_dir,
        ref=cpu_ref,
        title=cpu_kernel_slug.replace("-", " ").title(),
        code_file=cpu_code,
        enable_gpu=False,
        machine_shape="",
        kernel_sources=[gpu_ref],
        dataset_sources=cpu_dataset_sources,
    )
    return gpu_dir, cpu_dir, gpu_ref, cpu_ref


def _push_kernel(
    folder: Path,
    *,
    env: dict[str, str],
    accelerator: str | None,
    timeout_seconds: int,
) -> None:
    command = [
        "kaggle",
        "kernels",
        "push",
        "-p",
        str(folder),
        "--timeout",
        str(int(timeout_seconds)),
    ]
    if accelerator:
        command += ["--accelerator", accelerator]
    proc = _run(
        command,
        env=env,
        timeout=min(int(timeout_seconds) + 120, 1800),
    )
    print(proc.stdout.strip(), flush=True)


def _download_cpu_outputs(
    cpu_ref: str,
    *,
    env: dict[str, str],
    destination: Path,
) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "kaggle",
            "kernels",
            "output",
            cpu_ref,
            "-p",
            str(destination),
            "-o",
            "-q",
        ],
        env=env,
        timeout=600,
    )

    completion = list(destination.rglob("cpu_visualization_complete.json"))
    pngs = list(destination.rglob("*.png"))
    mp4s = list(destination.rglob("*.mp4"))
    if len(completion) != 1:
        raise RuntimeError(
            f"expected one CPU completion manifest, found {completion}"
        )
    if not pngs:
        raise RuntimeError("CPU kernel output contains no PNG visualization")
    if not mp4s:
        raise RuntimeError("CPU kernel output contains no MP4 visualization")
    return {
        "completion_manifest": str(completion[0]),
        "png": str(pngs[0]),
        "mp4": str(mp4s[0]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--account", type=str, default=None)
    parser.add_argument("--username", type=str, default=None)
    parser.add_argument("--algorithm", choices=("masac", "matd3"), default="masac")
    parser.add_argument("--runtime", default="kaggle_2xt4")
    parser.add_argument("--experiment", default="paper_50k")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--required-gpu-hours", type=float, default=0.5)
    parser.add_argument("--machine-shape", default="NvidiaTeslaT4")
    parser.add_argument("--gpu-kernel-slug", default="uav-marl-gpu-train")
    parser.add_argument("--cpu-kernel-slug", default="uav-marl-cpu-visualize")
    parser.add_argument(
        "--gpu-credential-dataset",
        default=None,
        help=(
            "Optional private Kaggle dataset ref containing wandb_api_key.txt "
            "for GPU training when Kaggle Secrets are unavailable."
        ),
    )
    parser.add_argument(
        "--cpu-log-wandb",
        action="store_true",
        help="Log authoritative CPU evaluation metrics/media to W&B online.",
    )
    parser.add_argument(
        "--cpu-credential-dataset",
        default=None,
        help=(
            "Private Kaggle dataset ref containing wandb_api_key.txt. "
            "Required with --cpu-log-wandb."
        ),
    )
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--gpu-timeout-seconds", type=int, default=12 * 3600)
    parser.add_argument("--cpu-timeout-seconds", type=int, default=3 * 3600)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "pipeline_outputs",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-build-dir", action="store_true")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional Hydra overrides, e.g. algorithm.actor_lr=1e-4",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_config(args.key_file)
    account, account_meta = pick_account(
        data,
        purpose="train",
        required_gpu_hours=float(args.required_gpu_hours),
        requested_name=args.account,
    )
    env = child_env(data, str(account["token"]))
    username = resolve_kaggle_username(
        env=env,
        explicit=args.username,
    )
    commit = _git_commit(ROOT)

    run_tag = (
        f"{args.algorithm}-seed{args.seed}-"
        f"{time.strftime('%Y%m%d-%H%M%S')}"
    )
    build_root = Path(
        tempfile.mkdtemp(prefix=f"uav_kaggle_pipeline_{run_tag}_")
    )
    gpu_slug = _slugify(args.gpu_kernel_slug)
    cpu_slug = _slugify(args.cpu_kernel_slug)

    gpu_dir, cpu_dir, gpu_ref, cpu_ref = build_kernels(
        build_root,
        username=username,
        commit=commit,
        algorithm=args.algorithm,
        runtime=args.runtime,
        experiment=args.experiment,
        seed=args.seed,
        gpu_kernel_slug=gpu_slug,
        cpu_kernel_slug=cpu_slug,
        machine_shape=args.machine_shape,
        overrides=list(args.overrides),
        gpu_dataset_sources=(
            [str(args.gpu_credential_dataset)]
            if args.gpu_credential_dataset
            else []
        ),
        cpu_log_wandb=bool(args.cpu_log_wandb),
        cpu_dataset_sources=(
            [str(args.cpu_credential_dataset)]
            if args.cpu_credential_dataset
            else []
        ),
    )

    plan = {
        "commit": commit,
        "account": account_meta["name"],
        "kaggle_username": username,
        "gpu_kernel": gpu_ref,
        "cpu_kernel": cpu_ref,
        "gpu_enable_gpu": True,
        "gpu_credential_dataset": (
            str(args.gpu_credential_dataset)
            if args.gpu_credential_dataset
            else None
        ),
        "cpu_enable_gpu": False,
        "cpu_log_wandb": bool(args.cpu_log_wandb),
        "cpu_credential_dataset": (
            str(args.cpu_credential_dataset)
            if args.cpu_credential_dataset
            else None
        ),
        "algorithm": args.algorithm,
        "runtime": args.runtime,
        "experiment": args.experiment,
        "seed": args.seed,
        "build_root": str(build_root),
    }
    print("KAGGLE_PIPELINE_PLAN")
    print(json.dumps(plan, indent=2, sort_keys=True))

    if args.dry_run:
        if not args.keep_build_dir:
            shutil.rmtree(build_root, ignore_errors=True)
        return 0

    try:
        print("STEP 1/4: launch GPU training kernel", flush=True)
        _push_kernel(
            gpu_dir,
            env=env,
            accelerator=args.machine_shape,
            timeout_seconds=args.gpu_timeout_seconds,
        )
        wait_for_kernel(
            gpu_ref,
            env=env,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.gpu_timeout_seconds,
        )

        print(
            "STEP 2/4: GPU kernel COMPLETE; GPU session is no longer needed",
            flush=True,
        )
        print("STEP 3/4: launch CPU-only visualization kernel", flush=True)
        _push_kernel(
            cpu_dir,
            env=env,
            accelerator=None,
            timeout_seconds=args.cpu_timeout_seconds,
        )
        wait_for_kernel(
            cpu_ref,
            env=env,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.cpu_timeout_seconds,
        )

        print("STEP 4/4: download and verify CPU outputs", flush=True)
        output_dir = (
            args.output_dir
            / username
            / run_tag
        )
        outputs = _download_cpu_outputs(
            cpu_ref,
            env=env,
            destination=output_dir,
        )
        print("KAGGLE_PIPELINE_COMPLETE")
        print(
            json.dumps(
                {
                    **plan,
                    "local_outputs": outputs,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    finally:
        if not args.keep_build_dir:
            shutil.rmtree(build_root, ignore_errors=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, TimeoutError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
