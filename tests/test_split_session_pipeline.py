from __future__ import annotations

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.kaggle_pipeline import build_kernels
from uav_marl.handoff import (
    CHECKPOINT_FILENAME,
    HANDOFF_FILENAME,
    create_training_handoff,
    load_training_handoff,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = str((ROOT / "configs").resolve())


def _cfg():
    with initialize_config_dir(
        version_base="1.3",
        config_dir=CONFIG_DIR,
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "algorithm=masac",
                "runtime=local_debug",
                "experiment=smoke",
            ],
        )
    return OmegaConf.to_container(cfg, resolve=True)


def test_training_handoff_copies_and_verifies_checkpoint(tmp_path):
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"model-checkpoint-bytes")

    result = create_training_handoff(
        {
            "final_checkpoint_path": str(checkpoint),
            "wandb_run_id": "abc123",
            "episodes_completed": 10,
            "global_step": 42,
            "resolved_config_sha256": "cfg-hash",
        },
        _cfg(),
        repo=tmp_path,
    )
    assert result["visualization_complete"] is False
    assert result["visualization_pending"] is True
    assert result["postprocess_mode"] == "separate_cpu"

    handoff_path = Path(result["handoff_path"])
    copied_checkpoint = Path(result["handoff_checkpoint_path"])
    assert handoff_path.name == HANDOFF_FILENAME
    assert copied_checkpoint.name == CHECKPOINT_FILENAME
    assert copied_checkpoint.read_bytes() == checkpoint.read_bytes()

    payload, loaded_checkpoint = load_training_handoff(handoff_path)
    assert payload["algorithm"] == "masac"
    assert payload["seed"] == 44
    assert loaded_checkpoint == copied_checkpoint


def test_training_handoff_detects_checkpoint_tampering(tmp_path):
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"good")

    result = create_training_handoff(
        {"final_checkpoint_path": str(checkpoint)},
        _cfg(),
        repo=tmp_path,
    )
    copied_checkpoint = Path(result["handoff_checkpoint_path"])
    copied_checkpoint.write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="size mismatch|SHA256 mismatch"):
        load_training_handoff(result["handoff_path"])


def test_kaggle_pipeline_builds_gpu_then_cpu_kernel(tmp_path):
    gpu_dir, cpu_dir, gpu_ref, cpu_ref = build_kernels(
        tmp_path,
        username="demo-user",
        commit="a" * 40,
        algorithm="masac",
        runtime="kaggle_2xt4",
        experiment="smoke",
        seed=44,
        gpu_kernel_slug="uav-marl-gpu-train",
        cpu_kernel_slug="uav-marl-cpu-visualize",
        machine_shape="NvidiaTeslaT4",
        overrides=["experiment.total_episodes=10"],
    )

    gpu_meta = json.loads((gpu_dir / "kernel-metadata.json").read_text())
    cpu_meta = json.loads((cpu_dir / "kernel-metadata.json").read_text())

    assert gpu_ref == "demo-user/uav-marl-gpu-train"
    assert cpu_ref == "demo-user/uav-marl-cpu-visualize"
    assert gpu_meta["enable_gpu"] is True
    assert gpu_meta["machine_shape"] == "NvidiaTeslaT4"
    assert gpu_meta["kernel_sources"] == []

    assert cpu_meta["enable_gpu"] is False
    assert cpu_meta["machine_shape"] == ""
    assert cpu_meta["kernel_sources"] == [gpu_ref]

    gpu_script = next(gpu_dir.glob("*.py")).read_text()
    cpu_script = next(cpu_dir.glob("*.py")).read_text()
    compile(gpu_script, "<gpu-kernel>", "exec")
    compile(cpu_script, "<cpu-kernel>", "exec")

    assert "GPU_TRAINING_HANDOFF_READY" in gpu_script
    assert "torch.cuda.is_available()" in cpu_script
    assert "--handoff" in cpu_script
    assert "/kaggle/input" in cpu_script


def test_default_experiment_uses_separate_cpu_session():
    cfg = _cfg()
    assert cfg["experiment"]["visualization"]["session"] == "separate_cpu"


def test_generated_kernels_keep_repo_checkout_out_of_kaggle_working(tmp_path):
    gpu_dir, cpu_dir, _, _ = build_kernels(
        tmp_path,
        username="demo-user",
        commit="b" * 40,
        algorithm="masac",
        runtime="kaggle_2xt4",
        experiment="smoke",
        seed=44,
        gpu_kernel_slug="gpu-clean-output",
        cpu_kernel_slug="cpu-clean-output",
        machine_shape="NvidiaTeslaT4",
        overrides=[],
    )
    gpu_script = next(gpu_dir.glob("*.py")).read_text()
    cpu_script = next(cpu_dir.glob("*.py")).read_text()
    assert 'Path("/tmp/uav_search_network")' in gpu_script
    assert 'Path("/tmp/uav_search_network")' in cpu_script
    assert 'Path("/kaggle/working/uav_search_network")' not in gpu_script
    assert 'Path("/kaggle/working/uav_search_network")' not in cpu_script
