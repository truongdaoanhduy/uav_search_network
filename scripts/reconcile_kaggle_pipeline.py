#!/usr/bin/env python3
"""Reconcile detached Kaggle GPU training into CPU visualization.

This script is intentionally short-lived so it can run from a scheduled
GitHub Actions job. Kaggle owns the long GPU/CPU sessions; each invocation only
inspects current state and launches the next missing stage.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

try:
    from .kaggle_pipeline import (
        _kernel_status,
        _parse_kernel_refs,
        _push_kernel,
        _run,
        build_kernels,
    )
except ImportError:
    from kaggle_pipeline import (
        _kernel_status,
        _parse_kernel_refs,
        _push_kernel,
        _run,
        build_kernels,
    )

GHA_GPU_REF_RE = re.compile(
    r"^(?P<user>[^/]+)/(?P<slug>uav-(?P<algorithm>masac|matd3)-gha-\d+-\d+-gpu)$"
)
ABLATION_GPU_REF_RE = re.compile(
    r"^(?P<user>[^/]+)/(?P<slug>uav-(?P<algorithm>masac|matd3)-"
    r"(?P<architecture>leaky-default|prelu-default|leaky-kaiming|prelu-kaiming|"
    r"leaky-kaiming-ln|prelu-kaiming-ln)-h\d+x\d+-seed\d+-gpu)$"
)
COMMIT_RE = re.compile(r"^\s*COMMIT\s*=\s*['\"](?P<sha>[0-9a-f]{40})['\"]\s*$", re.MULTILINE)
ACTIVE_STATES = {"queued", "pending", "running", "unknown"}
FAILURE_STATES = {"error", "cancelled", "failed"}


def _owned_kernel_refs(*, env: dict[str, str], username: str, limit: int) -> set[str]:
    proc = _run(
        [
            "kaggle",
            "kernels",
            "list",
            "--mine",
            "--sort-by",
            "dateRun",
            "--page-size",
            str(int(limit)),
            "-v",
        ],
        env=env,
    )
    return {
        ref for ref in _parse_kernel_refs(proc.stdout)
        if ref.startswith(f"{username}/")
    }



def _parse_gpu_ref(ref: str):
    for kind, pattern in (
        ("gha", GHA_GPU_REF_RE),
        ("ablation", ABLATION_GPU_REF_RE),
    ):
        parsed = pattern.match(ref)
        if parsed is not None:
            return kind, parsed
    return None


def _gpu_refs(refs: set[str], username: str) -> list[str]:
    matches = []
    for ref in refs:
        parsed = _parse_gpu_ref(ref)
        if parsed is not None and parsed[1].group("user") == username:
            matches.append(ref)
    return sorted(matches)


def _kernel_source_commit(ref: str, *, env: dict[str, str], temp_root: Path) -> str:
    pull_dir = temp_root / (ref.split("/", 1)[1] + "-source")
    pull_dir.mkdir(parents=True, exist_ok=True)
    _run(
        ["kaggle", "kernels", "pull", ref, "-p", str(pull_dir), "-m"],
        env=env,
        timeout=180,
    )
    for source in sorted(pull_dir.glob("*.py")):
        match = COMMIT_RE.search(source.read_text(errors="replace"))
        if match:
            return match.group("sha")
    raise RuntimeError(f"could not recover source git commit from {ref}")


def _cpu_ref_for_gpu(gpu_ref: str) -> str:
    parsed = _parse_gpu_ref(gpu_ref)
    if parsed is None:
        raise ValueError(f"not a GPU pipeline ref: {gpu_ref}")
    kind, _match = parsed
    return gpu_ref[:-4] + ("-cpu" if kind == "gha" else "-cpu-viz")


def reconcile_one(
    gpu_ref: str,
    *,
    refs: set[str],
    env: dict[str, str],
    temp_root: Path,
    cpu_credential_dataset: str | None,
    cpu_log_wandb: bool,
    cpu_session_timeout_seconds: int,
) -> str:
    status = _kernel_status(gpu_ref, env=env)
    print(f"RECONCILE_GPU {gpu_ref} status={status}", flush=True)
    if status in ACTIVE_STATES:
        return "gpu_active"
    if status in FAILURE_STATES:
        return "gpu_failed"
    if status != "complete":
        return "gpu_unhandled"

    cpu_ref = _cpu_ref_for_gpu(gpu_ref)
    if cpu_ref in refs:
        cpu_status = _kernel_status(cpu_ref, env=env)
        print(f"RECONCILE_CPU {cpu_ref} status={cpu_status}", flush=True)
        if cpu_status == "complete":
            return "cpu_complete"
        if cpu_status in ACTIVE_STATES:
            return "cpu_active"
        if cpu_status not in FAILURE_STATES:
            return "cpu_unhandled"
        print(f"RECONCILE_CPU_RETRY {cpu_ref} status={cpu_status}", flush=True)

    parsed_info = _parse_gpu_ref(gpu_ref)
    assert parsed_info is not None
    _kind, parsed = parsed_info
    commit = _kernel_source_commit(gpu_ref, env=env, temp_root=temp_root)
    gpu_slug = parsed.group("slug")
    cpu_slug = cpu_ref.split("/", 1)[1]
    build_root = temp_root / (gpu_slug + "-reconcile")
    _gpu_dir, cpu_dir, rebuilt_gpu_ref, rebuilt_cpu_ref = build_kernels(
        build_root,
        username=parsed.group("user"),
        commit=commit,
        algorithm=parsed.group("algorithm"),
        runtime="kaggle_2xt4",
        experiment="paper_20k",
        seed=44,
        gpu_kernel_slug=gpu_slug,
        cpu_kernel_slug=cpu_slug,
        machine_shape="NvidiaTeslaT4",
        overrides=[],
        cpu_log_wandb=bool(cpu_log_wandb),
        cpu_dataset_sources=(
            [cpu_credential_dataset]
            if cpu_log_wandb and cpu_credential_dataset
            else []
        ),
    )
    if rebuilt_gpu_ref != gpu_ref or rebuilt_cpu_ref != cpu_ref:
        raise RuntimeError("reconciler rebuilt unexpected Kaggle refs")
    _push_kernel(
        cpu_dir,
        env=env,
        accelerator=None,
        session_timeout_seconds=int(cpu_session_timeout_seconds),
        reuse_existing=True,
    )
    print(f"RECONCILE_CPU_SUBMITTED {cpu_ref}", flush=True)
    return "cpu_submitted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--cpu-session-timeout-seconds", type=int, default=3 * 3600)
    parser.add_argument("--cpu-log-wandb", action="store_true")
    parser.add_argument("--cpu-credential-dataset", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("KAGGLE_API_TOKEN is required")
    if args.cpu_log_wandb and not args.cpu_credential_dataset:
        raise RuntimeError("--cpu-log-wandb requires --cpu-credential-dataset")

    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token
    refs = _owned_kernel_refs(env=env, username=args.username, limit=args.limit)
    gpu_refs = _gpu_refs(refs, args.username)
    print(f"RECONCILE_FOUND_GPU_PIPELINES {len(gpu_refs)}", flush=True)

    temp_root = Path(tempfile.mkdtemp(prefix="uav_kaggle_reconcile_"))
    try:
        outcomes: dict[str, str] = {}
        for gpu_ref in gpu_refs:
            try:
                outcomes[gpu_ref] = reconcile_one(
                    gpu_ref,
                    refs=refs,
                    env=env,
                    temp_root=temp_root,
                    cpu_credential_dataset=args.cpu_credential_dataset,
                    cpu_log_wandb=bool(args.cpu_log_wandb),
                    cpu_session_timeout_seconds=args.cpu_session_timeout_seconds,
                )
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                outcomes[gpu_ref] = "reconcile_error"
                print(f"RECONCILE_ERROR {gpu_ref}: {exc}", flush=True)
        for ref, outcome in outcomes.items():
            print(f"RECONCILE_RESULT {ref} {outcome}", flush=True)
        return int("reconcile_error" in outcomes.values())
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
