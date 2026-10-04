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
