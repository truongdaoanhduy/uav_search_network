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
    assert "--launch-only" not in args


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
