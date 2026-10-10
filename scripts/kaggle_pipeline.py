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
import csv
import io
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
    requested = (
        explicit.strip()
        if explicit
        else str(env.get("KAGGLE_USERNAME", "")).strip() or None
    )

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
            if requested and username != requested:
                raise RuntimeError(
                    f"authenticated Kaggle account {username!r} does not match "
                    f"requested username {requested!r}"
                )
            return requested or username

    if requested:
        return requested
    raise RuntimeError(
        "could not resolve Kaggle username for the selected account; "
        "pass --username explicitly"
    )


def _parse_kernel_refs(output: str) -> set[str]:
    # Kaggle may prepend a version warning before the CSV header.
    lines = output.splitlines()
    header = next((i for i, line in enumerate(lines) if line.startswith("ref,")), None)
    if header is None:
        if lines and lines[-1].strip() == "Not found":
            return set()
        raise RuntimeError("Kaggle kernel listing did not contain a CSV ref header")
    return {
        row["ref"].strip()
        for row in csv.DictReader(io.StringIO("\n".join(lines[header:])))
        if row.get("ref")
    }


def _kernel_status(ref: str, *, env: dict[str, str]) -> str:
    proc = _run(
        ["kaggle", "kernels", "status", ref],
        env=env,
        check=False,
    )
    text = (proc.stdout + "\n" + proc.stderr).strip()
    if proc.returncode != 0:
        raise RuntimeError(
            f"could not read Kaggle kernel status for {ref}: {text[-2000:]}"
        )
    match = re.search(
        r'has status [\"\'](?:KernelWorkerStatus\.)?([a-z_]+)[\"\']',
        proc.stdout,
        re.IGNORECASE,
    )
    if match is None:
        return "unknown"
    state = match.group(1).lower()
    return {
        "cancel_acknowledged": "cancelled",
        "cancel_requested": "running",
        "new_script": "pending",
    }.get(state, state if state in TERMINAL_STATES | {"running", "queued", "pending"} else "unknown")



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
    cpu_ref: str,
    config_name: str = "config",
    wandb_credential_dataset: str | None = None,
) -> str:
    override_literals = json.dumps(list(overrides))
    cpu_url = f"https://www.kaggle.com/code/{cpu_ref}"
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
        CPU_KERNEL_REF = {cpu_ref!r}
        CPU_KERNEL_URL = {cpu_url!r}
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
            dataset_root = Path("/kaggle/input") / dataset_slug
            preferred_txt = dataset_root / "wandb_api_key.txt"
            preferred_json = dataset_root / "wandb_secret.json"
            if preferred_txt.is_file():
                api_key = preferred_txt.read_text().strip()
            elif preferred_json.is_file():
                credential_payload = json.loads(preferred_json.read_text())
                api_key = str(credential_payload["api_key"]).strip()
            else:
                txt_matches = sorted(Path("/kaggle/input").rglob("wandb_api_key.txt"))
                json_matches = sorted(Path("/kaggle/input").rglob("wandb_secret.json"))
                if len(txt_matches) == 1:
                    api_key = txt_matches[0].read_text().strip()
                elif len(json_matches) == 1:
                    credential_payload = json.loads(json_matches[0].read_text())
                    api_key = str(credential_payload["api_key"]).strip()
                else:
                    raise RuntimeError(
                        "expected exactly one W&B credential file, found "
                        + repr([str(path) for path in txt_matches + json_matches])
                    )
            if not api_key:
                raise RuntimeError("W&B credential file is empty")
            os.environ["WANDB_API_KEY"] = api_key
            os.environ.setdefault("WANDB_SILENT", "true")

            hf_token_path = dataset_root / "hf_token.txt"
            hf_repo_path = dataset_root / "hf_repo_id.txt"
            if not hf_token_path.is_file():
                matches = sorted(Path("/kaggle/input").rglob("hf_token.txt"))
                if len(matches) == 1:
                    hf_token_path = matches[0]
            if not hf_repo_path.is_file():
                matches = sorted(Path("/kaggle/input").rglob("hf_repo_id.txt"))
                if len(matches) == 1:
                    hf_repo_path = matches[0]
            if hf_token_path.is_file() and hf_repo_path.is_file():
                hf_token = hf_token_path.read_text().strip()
                hf_repo_id = hf_repo_path.read_text().strip()
                if not hf_token or not hf_repo_id:
                    raise RuntimeError("Hugging Face credential files are empty")
                os.environ["HF_TOKEN"] = hf_token
                os.environ["HF_REPO_ID"] = hf_repo_id

        command = [
            sys.executable,
            str(REPO / "train.py"),
            "--config-name",
            {config_name!r},
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
        next_stage = {{
            "stage": "cpu_visualization",
            "source_git_commit": COMMIT,
            "algorithm": {algorithm!r},
            "runtime": {runtime!r},
            "experiment": {experiment!r},
            "seed": {int(seed)},
            "cpu_kernel_ref": CPU_KERNEL_REF,
            "cpu_kernel_url": CPU_KERNEL_URL,
            "note": (
                "CPU visualization runs in a separate Kaggle CPU kernel. "
                "PNG/MP4 will appear in that kernel's Output tab, not in this GPU Output tab."
            ),
        }}
        next_stage_path = Path("/kaggle/working/CPU_VISUALIZATION_NEXT_STAGE.json")
        next_stage_path.write_text(json.dumps(next_stage, indent=2))
        print("GPU_TRAINING_HANDOFF_READY", handoffs[0], flush=True)
        print("CPU_VISUALIZATION_NEXT_STAGE", CPU_KERNEL_URL, flush=True)
        """
    ).strip() + "\n"


def _cpu_script(
    *,
    commit: str,
    log_wandb: bool = False,
    wandb_credential_dataset: str | None = None,
    source_gpu_ref: str | None = None,
    watch_timeout_seconds: int = 12 * 3600,
) -> str:
    if (log_wandb or source_gpu_ref) and not wandb_credential_dataset:
        raise ValueError(
            "wandb_credential_dataset is required for W&B logging or GPU watching"
        )
    return textwrap.dedent(
        f"""
        import json
        import os
        import re
        import subprocess
        import sys
        import time
        from pathlib import Path

        COMMIT = {commit!r}
        LOG_WANDB = {bool(log_wandb)!r}
        WANDB_CREDENTIAL_DATASET = {wandb_credential_dataset!r}
        SOURCE_GPU_REF = {source_gpu_ref!r}
        WATCH_TIMEOUT_SECONDS = {int(watch_timeout_seconds)}
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
            dataset_root = Path("/kaggle/input") / dataset_slug
            preferred_txt = dataset_root / "wandb_api_key.txt"
            preferred_json = dataset_root / "wandb_secret.json"
            if preferred_txt.is_file():
                api_key = preferred_txt.read_text().strip()
            elif preferred_json.is_file():
                credential_payload = json.loads(preferred_json.read_text())
                api_key = str(credential_payload["api_key"]).strip()
            else:
                txt_matches = sorted(Path("/kaggle/input").rglob("wandb_api_key.txt"))
                json_matches = sorted(Path("/kaggle/input").rglob("wandb_secret.json"))
                if len(txt_matches) == 1:
                    api_key = txt_matches[0].read_text().strip()
                elif len(json_matches) == 1:
                    credential_payload = json.loads(json_matches[0].read_text())
                    api_key = str(credential_payload["api_key"]).strip()
                else:
                    raise RuntimeError(
                        "expected exactly one W&B credential file, found "
                        + repr([str(path) for path in txt_matches + json_matches])
                    )
            if not api_key:
                raise RuntimeError("W&B credential file is empty")
            os.environ["WANDB_API_KEY"] = api_key
            os.environ.setdefault("WANDB_SILENT", "true")

            hf_token_path = dataset_root / "hf_token.txt"
            hf_repo_path = dataset_root / "hf_repo_id.txt"
            if not hf_token_path.is_file():
                matches = sorted(Path("/kaggle/input").rglob("hf_token.txt"))
                if len(matches) == 1:
                    hf_token_path = matches[0]
            if not hf_repo_path.is_file():
                matches = sorted(Path("/kaggle/input").rglob("hf_repo_id.txt"))
                if len(matches) == 1:
                    hf_repo_path = matches[0]
            if hf_token_path.is_file() and hf_repo_path.is_file():
                hf_token = hf_token_path.read_text().strip()
                hf_repo_id = hf_repo_path.read_text().strip()
                if not hf_token or not hf_repo_id:
                    raise RuntimeError("Hugging Face credential files are empty")
                os.environ["HF_TOKEN"] = hf_token
                os.environ["HF_REPO_ID"] = hf_repo_id

        handoff_root = Path("/kaggle/input")
        if SOURCE_GPU_REF:
            deadline = time.monotonic() + WATCH_TIMEOUT_SECONDS
            while True:
                status_proc = subprocess.run(
                    ["kaggle", "kernels", "status", SOURCE_GPU_REF],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                status_text = status_proc.stdout + "\\n" + status_proc.stderr
                match = re.search(r"KernelWorkerStatus\\.([A-Z_]+)", status_text)
                status = match.group(1) if match else "UNKNOWN"
                print("CPU_WATCH_GPU_STATUS", SOURCE_GPU_REF, status, flush=True)
                if status == "COMPLETE":
                    break
                if status in {"ERROR", "FAILED", "CANCELLED"}:
                    raise RuntimeError("source GPU kernel failed before CPU visualization: " + status)
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for source GPU kernel: " + SOURCE_GPU_REF)
                time.sleep(20)

            source_output = Path("/kaggle/working/source_gpu_output")
            source_output.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["kaggle", "kernels", "output", SOURCE_GPU_REF, "-p", str(source_output), "-o", "-q"],
                check=True,
            )
            handoff_root = source_output
            print("CPU_WATCH_GPU_COMPLETE", SOURCE_GPU_REF, flush=True)

        command = [
            sys.executable,
            str(REPO / "visualize.py"),
            "--handoff",
            str(handoff_root),
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
    config_name: str = "config",
    gpu_dataset_sources: list[str] | None = None,
    cpu_log_wandb: bool = False,
    cpu_dataset_sources: list[str] | None = None,
    cpu_watch_gpu: bool = False,
    cpu_session_timeout_seconds: int = 3 * 3600,
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
            cpu_ref=cpu_ref,
            config_name=config_name,
            wandb_credential_dataset=(
                gpu_dataset_sources[0] if gpu_dataset_sources else None
            ),
        )
    )
    cpu_dataset_sources = list(cpu_dataset_sources or [])
    if (cpu_log_wandb or cpu_watch_gpu) and len(cpu_dataset_sources) != 1:
        raise ValueError(
            "CPU W&B logging or GPU watching requires exactly one credential dataset source"
        )
    (cpu_dir / cpu_code).write_text(
        _cpu_script(
            commit=commit,
            log_wandb=bool(cpu_log_wandb),
            wandb_credential_dataset=(
                cpu_dataset_sources[0]
                if (cpu_log_wandb or cpu_watch_gpu) and cpu_dataset_sources
                else None
            ),
            source_gpu_ref=(gpu_ref if cpu_watch_gpu else None),
            watch_timeout_seconds=int(cpu_session_timeout_seconds),
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
        kernel_sources=([] if cpu_watch_gpu else [gpu_ref]),
        dataset_sources=cpu_dataset_sources,
    )
    return gpu_dir, cpu_dir, gpu_ref, cpu_ref


def _push_kernel(
    folder: Path,
    *,
    env: dict[str, str],
    accelerator: str | None,
    session_timeout_seconds: int,
    reuse_existing: bool = False,
) -> None:
    if reuse_existing:
        ref = json.loads((folder / "kernel-metadata.json").read_text())["id"]
        existing = _run(
            ["kaggle", "kernels", "list", "--mine", "--search",
             ref.split("/", 1)[1], "--page-size", "100", "-v"],
            env=env,
        )
        if ref in _parse_kernel_refs(existing.stdout):
            existing_status = _kernel_status(ref, env=env)
            if existing_status not in {"error", "cancelled", "failed"}:
                print(
                    f"KAGGLE_KERNEL_REUSED {ref} status={existing_status}",
                    flush=True,
                )
                return
            print(
                f"KAGGLE_KERNEL_RETRY {ref} status={existing_status}",
                flush=True,
            )
    command = [
        "kaggle",
        "kernels",
        "push",
        "-p",
        str(folder),
        "--timeout",
        str(int(session_timeout_seconds)),
    ]
    if accelerator:
        command += ["--accelerator", accelerator]
    proc = _run(
        command,
        env=env,
        timeout=min(int(session_timeout_seconds) + 120, 1800),
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
    parser.add_argument("--config-name", default="config")
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
            "or wandb_secret.json for GPU training when Kaggle Secrets are unavailable."
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
            "Private Kaggle dataset ref containing wandb_api_key.txt or "
            "wandb_secret.json. Required with --cpu-log-wandb."
        ),
    )
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--gpu-session-timeout-seconds", type=int, default=12 * 3600)
    parser.add_argument("--cpu-session-timeout-seconds", type=int, default=3 * 3600)
    parser.add_argument("--gpu-wait-timeout-seconds", type=int, default=12 * 3600)
    parser.add_argument("--cpu-wait-timeout-seconds", type=int, default=3 * 3600)
    parser.add_argument(
        "--launch-only",
        action="store_true",
        help="Submit the GPU kernel, persist a launch manifest, and return without polling.",
    )
    parser.add_argument(
        "--cpu-watch-gpu",
        action="store_true",
        help=(
            "Launch a detached CPU watcher alongside the GPU job so visualization "
            "continues on Kaggle without a local controller."
        ),
    )
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
    cloud_token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if cloud_token:
        # Cloud-controller mode (for example GitHub Actions). The Kaggle token
        # comes from the runner secret store, so no local key.txt is needed.
        env = os.environ.copy()
        env["KAGGLE_API_TOKEN"] = cloud_token
        account_meta = {
            "name": os.environ.get(
                "KAGGLE_ACCOUNT_NAME",
                "cloud_controller",
            ),
        }
    else:
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
    cpu_runtime_timeout = (
        max(
            int(args.cpu_session_timeout_seconds),
            int(args.gpu_session_timeout_seconds),
        )
        if args.cpu_watch_gpu
        else int(args.cpu_session_timeout_seconds)
    )

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
        config_name=args.config_name,
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
        cpu_watch_gpu=bool(args.cpu_watch_gpu),
        cpu_session_timeout_seconds=cpu_runtime_timeout,
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
        "cpu_watch_gpu": bool(args.cpu_watch_gpu),
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
            session_timeout_seconds=args.gpu_session_timeout_seconds,
            reuse_existing=bool(args.launch_only),
        )
        if args.cpu_watch_gpu:
            print("STEP 1B/4: launch detached CPU watcher kernel", flush=True)
            _push_kernel(
                cpu_dir,
                env=env,
                accelerator=None,
                session_timeout_seconds=cpu_runtime_timeout,
                reuse_existing=True,
            )
        if args.launch_only:
            output_dir = args.output_dir / username / run_tag
            output_dir.mkdir(parents=True, exist_ok=True)
            launch_manifest = output_dir / "kaggle_pipeline_launch.json"
            launch_manifest.write_text(
                json.dumps(
                    {**plan, "stage": "gpu_submitted", "status": "submitted"},
                    indent=2,
                    sort_keys=True,
                )
            )
            print("KAGGLE_GPU_SUBMITTED")
            print(launch_manifest)
            return 0
        wait_for_kernel(
            gpu_ref,
            env=env,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.gpu_wait_timeout_seconds,
        )

        print(
            "STEP 2/4: GPU kernel COMPLETE; GPU session is no longer needed",
            flush=True,
        )
        if args.cpu_watch_gpu:
            print("STEP 3/4: CPU watcher already submitted; wait for visualization", flush=True)
        else:
            print("STEP 3/4: launch CPU-only visualization kernel", flush=True)
            _push_kernel(
                cpu_dir,
                env=env,
                accelerator=None,
                session_timeout_seconds=args.cpu_session_timeout_seconds,
            )
        wait_for_kernel(
            cpu_ref,
            env=env,
            poll_seconds=args.poll_seconds,
            timeout_seconds=(
                cpu_runtime_timeout
                if args.cpu_watch_gpu
                else args.cpu_wait_timeout_seconds
            ),
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
