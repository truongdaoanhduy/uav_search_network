"""Regression cases from Kaggle controller failures."""
import subprocess

import pytest

from scripts import kaggle_pipeline as pipeline


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("CANCEL_ACKNOWLEDGED", "cancelled"),
        ("CANCEL_REQUESTED", "running"),
        ("COMPLETE", "complete"),
        ("ERROR", "error"),
        ("RUNNING", "running"),
        ("QUEUED", "queued"),
    ],
)
def test_status_reads_enum_from_kaggle_response(monkeypatch, raw, expected):
    # A slug may contain status words; only the returned status is authoritative.
    ref = "user/error-complete-running"
    monkeypatch.setattr(
        pipeline, "_run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, f'{ref} has status "KernelWorkerStatus.{raw}"\n', ""
        ),
    )
    assert pipeline._kernel_status(ref, env={}) == expected


def test_status_api_error_is_not_a_terminal_training_failure(monkeypatch):
    monkeypatch.setattr(
        pipeline, "_run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 1, "", "ConnectionError: temporary network error"
        ),
    )
    with pytest.raises(RuntimeError, match="could not read"):
        pipeline._kernel_status("user/gpu", env={})


def test_wait_exits_on_acknowledged_cancellation(monkeypatch):
    monkeypatch.setattr(
        pipeline, "_run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, 'user/gpu has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"', ""
        ),
    )
    monkeypatch.setattr(pipeline, "_kernel_logs", lambda *args, **kwargs: "")
    with pytest.raises(RuntimeError, match="ended with status cancelled"):
        pipeline.wait_for_kernel("user/gpu", env={}, poll_seconds=5, timeout_seconds=0)


@pytest.mark.parametrize("failure", [RuntimeError("submission rejected"), subprocess.TimeoutExpired("kaggle", 1)])
def test_reconciler_reports_failure_and_continues_other_jobs(monkeypatch, failure):
    from argparse import Namespace

    from scripts import reconcile_kaggle_pipeline as reconcile

    refs = {"user/uav-masac-gha-1-1-gpu", "user/uav-matd3-gha-2-1-gpu"}
    monkeypatch.setenv("KAGGLE_API_TOKEN", "test-token")
    monkeypatch.setattr(reconcile, "parse_args", lambda: Namespace(
        username="user", limit=100, cpu_log_wandb=False,
        cpu_credential_dataset=None, cpu_session_timeout_seconds=10,
    ))
    monkeypatch.setattr(reconcile, "_owned_kernel_refs", lambda **kwargs: refs)
    visited = []

    def one(ref, **kwargs):
        visited.append(ref)
        if "masac" in ref:
            raise failure
        return "gpu_failed"  # Historical cancelled kernels are not controller errors.

    monkeypatch.setattr(reconcile, "reconcile_one", one)
    assert reconcile.main() == 1
    assert set(visited) == refs


def test_detached_retry_preserves_active_existing_gpu_kernel(tmp_path, monkeypatch):
    import json

    folder = tmp_path / "gpu"
    folder.mkdir()
    ref = "user/uav-masac-gha-123-1-gpu"
    (folder / "kernel-metadata.json").write_text(json.dumps({"id": ref}))
    calls = []

    def command(args, **kwargs):
        calls.append(args)
        if args[1:3] == ["kernels", "list"]:
            return subprocess.CompletedProcess(
                args, 0,
                f"ref,title,author,lastRunTime,totalVotes\n{ref},GPU,user,2026-10-07,0\n", "",
            )
        if args[1:3] == ["kernels", "status"]:
            return subprocess.CompletedProcess(
                args, 0,
                f'{ref} has status "KernelWorkerStatus.RUNNING"\n', "",
            )
        raise AssertionError("Active retry must not submit another GPU session")

    monkeypatch.setattr(pipeline, "_run", command)
    pipeline._push_kernel(
        folder, env={}, accelerator="NvidiaTeslaT4",
        session_timeout_seconds=43200, reuse_existing=True,
    )
    assert [args[1:3] for args in calls] == [
        ["kernels", "list"],
        ["kernels", "status"],
    ]


def test_detached_retry_resubmits_cancelled_existing_gpu_kernel(tmp_path, monkeypatch):
    import json

    folder = tmp_path / "gpu"
    folder.mkdir()
    ref = "user/uav-masac-gha-123-1-gpu"
    (folder / "kernel-metadata.json").write_text(json.dumps({"id": ref}))
    calls = []

    def command(args, **kwargs):
        calls.append(args)
        if args[1:3] == ["kernels", "list"]:
            return subprocess.CompletedProcess(
                args, 0,
                f"ref,title,author,lastRunTime,totalVotes\n{ref},GPU,user,2026-10-07,0\n", "",
            )
        if args[1:3] == ["kernels", "status"]:
            return subprocess.CompletedProcess(
                args, 0,
                f'{ref} has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"\n', "",
            )
        if args[1:3] == ["kernels", "push"]:
            return subprocess.CompletedProcess(args, 0, "pushed\n", "")
        raise AssertionError(args)

    monkeypatch.setattr(pipeline, "_run", command)
    pipeline._push_kernel(
        folder, env={}, accelerator="NvidiaTeslaT4",
        session_timeout_seconds=43200, reuse_existing=True,
    )
    assert [args[1:3] for args in calls] == [
        ["kernels", "list"],
        ["kernels", "status"],
        ["kernels", "push"],
    ]


def test_kernel_list_warning_does_not_hide_existing_pipeline():
    import unittest.mock

    from scripts.reconcile_kaggle_pipeline import _owned_kernel_refs

    output = ("Warning: old CLI version\n"
              "ref,title,author,lastRunTime,totalVotes\n"
              "user/uav-masac-gha-1-1-gpu,Title,user,2026-10-07,0\n")
    with unittest.mock.patch(
        "scripts.reconcile_kaggle_pipeline._run",
        return_value=subprocess.CompletedProcess([], 0, output, ""),
    ):
        assert _owned_kernel_refs(env={}, username="user", limit=100) == {
            "user/uav-masac-gha-1-1-gpu"
        }


@pytest.mark.parametrize("output", ["Not found\n", "Warning: old CLI version\nNot found\n"])
def test_new_detached_kernel_is_submitted_when_search_is_empty(tmp_path, monkeypatch, output):
    import json

    (tmp_path / "kernel-metadata.json").write_text(json.dumps({"id": "user/new-gpu"}))
    calls = []

    def command(args, **kwargs):
        calls.append(args)
        text = output if args[1:3] == ["kernels", "list"] else "pushed"
        return subprocess.CompletedProcess(args, 0, text, "")

    monkeypatch.setattr(pipeline, "_run", command)
    pipeline._push_kernel(
        tmp_path, env={}, accelerator="NvidiaTeslaT4",
        session_timeout_seconds=43200, reuse_existing=True,
    )
    assert [args[1:3] for args in calls] == [["kernels", "list"], ["kernels", "push"]]


def test_reconciler_retries_failed_existing_cpu_stage(tmp_path, monkeypatch):
    from scripts import reconcile_kaggle_pipeline as reconcile

    gpu_ref = "user/uav-masac-gha-123-1-gpu"
    cpu_ref = "user/uav-masac-gha-123-1-cpu"
    statuses = {gpu_ref: "complete", cpu_ref: "failed"}
    monkeypatch.setattr(reconcile, "_kernel_status", lambda ref, env: statuses[ref])
    monkeypatch.setattr(
        reconcile,
        "_kernel_source_commit",
        lambda *args, **kwargs: "c" * 40,
    )
    submitted = []
    monkeypatch.setattr(
        reconcile,
        "_push_kernel",
        lambda folder, **kwargs: submitted.append(folder),
    )

    result = reconcile.reconcile_one(
        gpu_ref,
        refs={gpu_ref, cpu_ref},
        env={},
        temp_root=tmp_path,
        cpu_credential_dataset=None,
        cpu_log_wandb=False,
        cpu_session_timeout_seconds=10_800,
    )

    assert result == "cpu_submitted"
    assert len(submitted) == 1
