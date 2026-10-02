from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from config import CONFIG as BASE_CONFIG
from uav_marl.configuration import (
    apply_to_legacy_config,
    derive_transition_budget,
    validate_config,
)
from uav_marl.runner import run_experiment

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = str((ROOT / "configs").resolve())


def compose_cfg(*overrides):
    with initialize_config_dir(
        version_base="1.3",
        config_dir=CONFIG_DIR,
    ):
        return compose(
            config_name="config",
            overrides=list(overrides),
        )


def test_kaggle_budget_is_derived_from_episodes():
    cfg = compose_cfg(
        "runtime=kaggle_2xt4",
        "algorithm=masac",
    )
    assert cfg.experiment.total_episodes == 50_000
    assert cfg.runtime.num_envs == 4096
    assert derive_transition_budget(
        total_episodes=cfg.experiment.total_episodes,
        num_envs=cfg.runtime.num_envs,
        max_steps=cfg.task.scenario.max_steps,
    ) == 53_248_000


def test_vast_profile_changes_runtime_not_task_or_algorithm():
    kaggle = compose_cfg(
        "runtime=kaggle_2xt4",
        "algorithm=masac",
    )
    vast = compose_cfg(
        "runtime=vast_1gpu",
        "algorithm=masac",
    )
    assert kaggle.task == vast.task
    assert kaggle.reward == vast.reward
    assert kaggle.algorithm == vast.algorithm
    assert kaggle.runtime.provider == "kaggle"
    assert vast.runtime.provider == "vast"
    assert vast.runtime.max_gpus == 1


def test_algorithm_override_isolated_from_task():
    masac = compose_cfg("algorithm=masac")
    matd3 = compose_cfg("algorithm=matd3")
    assert masac.task == matd3.task
    assert masac.reward == matd3.reward
    assert masac.algorithm.name == "masac"
    assert matd3.algorithm.name == "matd3"
    assert "alpha_lr" in masac.algorithm
    assert "policy_delay" in matd3.algorithm


def test_reward_override_isolated_from_algorithm():
    base = compose_cfg("algorithm=masac", "reward=paper_v1")
    changed = compose_cfg("algorithm=masac", "reward=search_heavy")
    assert base.algorithm == changed.algorithm
    assert base.task == changed.task
    assert base.reward.info_gain == 10.0
    assert changed.reward.info_gain == 20.0


def test_legacy_bridge_maps_selected_profile():
    cfg = compose_cfg(
        "runtime=vast_1gpu",
        "algorithm=matd3",
        "algorithm.actor_lr=0.0001",
    )
    legacy = dict(BASE_CONFIG)
    apply_to_legacy_config(legacy, cfg)
    assert legacy["seed"] == 44
    assert legacy["num_targets"] == 50
    assert legacy["reward_blocked"] == 1.0
    assert legacy["matd3_actor_lr"] == pytest.approx(0.0001)
    assert legacy["training_num_envs"] == 2048
    assert legacy["full_gpu_max_gpus"] == 1
    assert legacy["network_backend"] == "uavnetsim_gpu"
    assert legacy["_runtime_provider"] == "vast"
    assert legacy["training_wandb_mode"] == "online"


def test_offline_wandb_is_rejected_for_full_flow():
    cfg = compose_cfg(
        "experiment.wandb.mode=offline",
    )
    with pytest.raises(ValueError, match="W&B|wandb"):
        validate_config(cfg)

def test_runner_hides_transition_ceiling_from_user_api(monkeypatch, tmp_path):
    cfg = compose_cfg(
        "runtime=vast_1gpu",
        "algorithm=masac",
        "experiment.total_episodes=123",
    )
    captured = {}
    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_bytes(b"checkpoint")

    def fake_train(**kwargs):
        captured.update(kwargs)
        return {
            "wandb_run_id": "fake",
            "final_checkpoint_path": str(checkpoint),
        }

    monkeypatch.setattr(
        "uav_marl.runner._load_train_full_gpu_auto",
        lambda: fake_train,
    )

    result = run_experiment(cfg, repo=ROOT)
    assert captured["target_episodes"] == 123
    assert captured["num_envs"] == 2048
    assert captured["max_gpus"] == 1
    assert captured["device"] == "cuda:0"
    assert captured["total_transitions"] == 2_048_000
    assert result["derived_transition_ceiling"] == 2_048_000


def test_multi_gpu_rejects_nonzero_preferred_device():
    cfg = compose_cfg(
        "runtime=kaggle_2xt4",
        "runtime.device=cuda:1",
    )
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES|cuda:0"):
        validate_config(cfg)


def test_multi_gpu_rejects_unsupported_training_options():
    cfg = compose_cfg(
        "runtime=kaggle_2xt4",
        "runtime.multi_gpu_strategy=env_shard_single_learner",
    )
    with pytest.raises(ValueError, match="multi_gpu_strategy=ddp"):
        validate_config(cfg)

    cfg = compose_cfg(
        "runtime=kaggle_2xt4",
        "runtime.compile_mode=default",
    )
    with pytest.raises(ValueError, match="compile_mode"):
        validate_config(cfg)

    cfg = compose_cfg(
        "runtime=kaggle_2xt4",
        "runtime.fused_adam=true",
    )
    with pytest.raises(ValueError, match="fused_adam"):
        validate_config(cfg)


def test_single_gpu_full_flow_rejects_unsupported_runtime_optimizer_switches():
    cfg = compose_cfg(
        "runtime=vast_1gpu",
        "runtime.compile_mode=default",
    )
    with pytest.raises(ValueError, match="compile_mode"):
        validate_config(cfg)

    cfg = compose_cfg(
        "runtime=vast_1gpu",
        "runtime.fused_adam=true",
    )
    with pytest.raises(ValueError, match="fused_adam"):
        validate_config(cfg)

    cfg = compose_cfg(
        "runtime=vast_1gpu",
        "algorithm.gradient_steps=0",
    )
    with pytest.raises(ValueError, match="gradient_steps"):
        validate_config(cfg)

    cfg = compose_cfg(
        "runtime=vast_1gpu",
        "algorithm.train_freq=0",
    )
    with pytest.raises(ValueError, match="train_freq"):
        validate_config(cfg)
