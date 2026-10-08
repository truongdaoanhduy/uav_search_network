from pathlib import Path

from scripts.launch_kaggle_pipeline_service import (
    build_systemd_command,
    _safe_unit_name,
)


def test_safe_unit_name():
    assert _safe_unit_name("UAV pipeline / run 1") == "UAV-pipeline---run-1"


def test_build_systemd_command_wraps_pipeline_and_log():
    command = build_systemd_command(
        unit="uav-test",
        log_file=Path("/tmp/uav pipeline.log"),
        pipeline_args=["--algorithm", "masac", "runtime.num_envs=4096"],
    )
    joined = " ".join(command)
    assert "systemd-run" in command[0]
    assert "--user" in command
    assert "--unit=uav-test" in command
    assert "kaggle_pipeline.py" in joined
    assert "runtime.num_envs=4096" in joined
    assert "uav pipeline.log" in joined


def test_architecture_ablation_maps_six_non_relu_variants_to_six_gpu_accounts():
    from scripts.launch_architecture_ablation import ABLATION_RUNS

    assert [item["account"] for item in ABLATION_RUNS] == [
        "account_01", "account_03", "account_04",
        "account_05", "account_06", "account_07",
    ]
    assert [item["architecture"] for item in ABLATION_RUNS] == [
        "leaky_default", "prelu_default", "leaky_kaiming",
        "prelu_kaiming", "leaky_kaiming_ln", "prelu_kaiming_ln",
    ]
    assert len({item["architecture"] for item in ABLATION_RUNS}) == 6
    assert all(item["architecture"] != "relu" for item in ABLATION_RUNS)


def test_architecture_ablation_pipeline_args_are_self_describing_and_chain_cpu_wandb():
    from scripts.launch_architecture_ablation import build_pipeline_args

    args = build_pipeline_args(
        account="account_06",
        architecture="leaky_kaiming_ln",
        algorithm="masac",
        seed=44,
        credential_dataset="demo/private-wandb",
        required_gpu_hours=12.01,
    )
    joined = " ".join(args)
    assert "--account account_06" in joined
    assert "--required-gpu-hours 12.01" in joined
    assert "--cpu-log-wandb" in args
    assert "--gpu-credential-dataset demo/private-wandb" in joined
    assert "--cpu-credential-dataset demo/private-wandb" in joined
    assert "architecture=leaky_kaiming_ln" in args
    assert "uav-masac-leaky-kaiming-ln-h512x256-seed44-gpu" in joined
    assert "uav-masac-leaky-kaiming-ln-h512x256-seed44-cpu-viz" in joined
    assert "--launch-only" in args
    assert "--cpu-watch-gpu" in args


def test_architecture_ablation_credential_dataset_is_account_private():
    from scripts.launch_architecture_ablation import (
        credential_dataset_metadata,
        credential_dataset_ref,
    )

    ref = credential_dataset_ref("duytrngoanh")
    assert ref == "duytrngoanh/uav-wandb-credential-arch-ablation"
    metadata = credential_dataset_metadata("duytrngoanh")
    assert metadata["id"] == ref
    assert metadata["isPrivate"] is True
    assert metadata["licenses"] == [{"name": "other"}]


def test_architecture_ablation_defaults_to_each_accounts_private_credential_dataset():
    from scripts.launch_architecture_ablation import credential_datasets_for_runs

    usernames = {
        "account_01": "user01",
        "account_03": "user03",
        "account_04": "user04",
        "account_05": "user05",
        "account_06": "user06",
        "account_07": "user07",
    }
    refs = credential_datasets_for_runs(usernames)
    assert refs == {
        account: f"{username}/uav-wandb-credential-arch-ablation"
        for account, username in usernames.items()
    }
    shared = credential_datasets_for_runs(usernames, override="shared/wandb")
    assert set(shared.values()) == {"shared/wandb"}


def test_ablation_selects_six_highest_quota_accounts_strictly_above_12h():
    from scripts.launch_architecture_ablation import select_ablation_runs_from_statuses

    statuses = [
        {"name": "account_01", "ok": True, "resources": {"GPU": {"remaining_hours": 29.0}}},
        {"name": "account_02", "ok": True, "resources": {"GPU": {"remaining_hours": 12.0}}},
        {"name": "account_03", "ok": True, "resources": {"GPU": {"remaining_hours": 28.0}}},
        {"name": "account_04", "ok": True, "resources": {"GPU": {"remaining_hours": 27.0}}},
        {"name": "account_05", "ok": True, "resources": {"GPU": {"remaining_hours": 26.0}}},
        {"name": "account_06", "ok": True, "resources": {"GPU": {"remaining_hours": 25.0}}},
        {"name": "account_07", "ok": True, "resources": {"GPU": {"remaining_hours": 24.0}}},
    ]
    runs = select_ablation_runs_from_statuses(statuses, required_gpu_hours=12.0)
    assert [item["account"] for item in runs] == [
        "account_01", "account_03", "account_04",
        "account_05", "account_06", "account_07",
    ]
    assert all(item["remaining_gpu_hours"] > 12.0 for item in runs)
    assert len({item["architecture"] for item in runs}) == 6


def test_shared_service_credentials_bundle_contains_one_wandb_and_hf_identity(tmp_path):
    from scripts.launch_architecture_ablation import write_service_credential_bundle

    data = {
        "kaggle": {
            "accounts": [
                {"name": "account_01", "token": "kaggle-secret-1"},
                {"name": "account_02", "token": "kaggle-secret-2"},
            ]
        },
        "wandb": {"api_key": "wandb-shared-key", "entity": "entity", "project": "project"},
        "huggingface": {"token": "hf-shared-token"},
    }
    write_service_credential_bundle(
        tmp_path,
        data,
        kaggle_username="demo-user",
        hf_repo_id="hf-user/uav-search-target-checkpoints",
    )

    assert (tmp_path / "wandb_api_key.txt").read_text() == "wandb-shared-key"
    assert (tmp_path / "hf_token.txt").read_text() == "hf-shared-token"
    assert (tmp_path / "hf_repo_id.txt").read_text() == "hf-user/uav-search-target-checkpoints"
    metadata = __import__("json").loads((tmp_path / "dataset-metadata.json").read_text())
    assert metadata["id"] == "demo-user/uav-wandb-credential-arch-ablation"
    assert metadata["isPrivate"] is True
    all_text = "\n".join(path.read_text() for path in tmp_path.iterdir() if path.is_file())
    assert "kaggle-secret-1" not in all_text
    assert "kaggle-secret-2" not in all_text


def test_cpu_kernel_reads_shared_huggingface_credentials_from_dataset(tmp_path):
    from scripts.kaggle_pipeline import build_kernels

    _gpu_dir, cpu_dir, _gpu_ref, _cpu_ref = build_kernels(
        tmp_path,
        username="demo-user",
        commit="f" * 40,
        algorithm="masac",
        runtime="kaggle_2xt4",
        experiment="paper_20k",
        seed=44,
        gpu_kernel_slug="gpu-shared-services",
        cpu_kernel_slug="cpu-shared-services",
        machine_shape="NvidiaTeslaT4",
        overrides=[],
        gpu_dataset_sources=["demo-user/uav-wandb-credential-arch-ablation"],
        cpu_log_wandb=True,
        cpu_dataset_sources=["demo-user/uav-wandb-credential-arch-ablation"],
    )
    cpu_script = next(cpu_dir.glob("*.py")).read_text()
    assert "hf_token.txt" in cpu_script
    assert "hf_repo_id.txt" in cpu_script
    assert 'os.environ["HF_TOKEN"] = hf_token' in cpu_script
    assert 'os.environ["HF_REPO_ID"] = hf_repo_id' in cpu_script


def test_build_systemd_command_creates_log_parent_before_redirect():
    command = build_systemd_command(
        unit="uav-test-log-parent",
        log_file=Path("/tmp/nonexistent-uav-log-dir/run.log"),
        pipeline_args=["--algorithm", "masac"],
    )
    shell = command[-1]
    assert "mkdir -p" in shell
    assert "/tmp/nonexistent-uav-log-dir" in shell
    assert shell.index("mkdir -p") < shell.index(">>")


def test_build_systemd_command_propagates_current_path(monkeypatch):
    monkeypatch.setenv("PATH", "/opt/custom/bin:/usr/bin")
    command = build_systemd_command(
        unit="uav-test-path",
        log_file=Path("/tmp/uav-test-path/run.log"),
        pipeline_args=["--algorithm", "masac"],
    )
    assert "--setenv=PATH=/opt/custom/bin:/usr/bin" in command
