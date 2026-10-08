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
    assert cfg.experiment.name == "paper_20k"
    assert cfg.experiment.total_episodes == 20_000
    assert cfg.runtime.num_envs == 4096
    assert derive_transition_budget(
        total_episodes=cfg.experiment.total_episodes,
        num_envs=cfg.runtime.num_envs,
        max_steps=cfg.task.scenario.max_steps,
    ) == 61_440_000


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
    assert legacy["reward_blocked"] == 0.2
    assert legacy["_reward_name"] == "paper_v2"
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
    assert captured["total_transitions"] == 6_144_000
    assert result["derived_transition_ceiling"] == 6_144_000


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



def test_matd3_rejects_masac_only_cuda_graph_policy_path_early():
    cfg = compose_cfg(
        "algorithm=matd3",
        "runtime=kaggle_2xt4",
        "runtime.cuda_graph_policy_actions=true",
    )
    with pytest.raises(ValueError, match="CUDA Graph|cuda_graph_policy_actions|MASAC"):
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


def test_risk_aware_apf_config_is_minimal_and_maps_to_legacy_bridge():
    cfg = compose_cfg("algorithm=masac")

    assert cfg.task.scenario.map_size == 3000
    assert cfg.task.scenario.num_targets == 50
    assert cfg.task.scenario.num_obstacles == 30
    assert cfg.task.scenario.max_steps == 3000
    assert BASE_CONFIG["max_steps"] == cfg.task.scenario.max_steps
    assert list(cfg.task.scenario.gcs_position) == [1500, 0, 0.0]

    assert cfg.task.safety.peer_distance_m == pytest.approx(30.0)
    assert cfg.task.safety.obstacle_clearance_m == pytest.approx(30.0)
    assert cfg.task.spawn.launch_radius_m == pytest.approx(300.0)
    assert cfg.task.spawn.launch_min_spacing_m == pytest.approx(60.0)

    assert cfg.task.apf.enabled is True
    assert cfg.task.apf.soft_gain == pytest.approx(1.5)
    assert cfg.task.apf.lookahead_s == pytest.approx(3.0)
    assert cfg.task.apf.braking_margin == pytest.approx(1.5)

    assert "apf_peer_influence_m" not in cfg.task.scenario
    assert "apf_obstacle_influence_m" not in cfg.task.scenario
    assert "apf_emergency_gain" not in cfg.task.scenario

    legacy = dict(BASE_CONFIG)
    apply_to_legacy_config(legacy, cfg)
    assert legacy["apf_enabled"] is True
    assert legacy["apf_soft_gain"] == pytest.approx(1.5)
    assert legacy["apf_lookahead_s"] == pytest.approx(3.0)
    assert legacy["apf_braking_margin"] == pytest.approx(1.5)
    assert legacy["safety_distance"] == pytest.approx(30.0)
    assert legacy["launch_min_spacing_m"] == pytest.approx(60.0)


def test_paper_20k_persists_periodic_checkpoints_for_cloud_recovery():
    cfg = compose_cfg("experiment=paper_20k")
    assert cfg.experiment.checkpoint.interval_episodes == 5000
    assert cfg.experiment.checkpoint.upload_periodic_to_wandb is True

    source = (ROOT / "uav_marl" / "training" / "gpu.py").read_text()
    assert 'base_path=str(Path(saved_path).parent)' in source


def test_kaggle_profile_preserves_masac_entropy_stability_setting():
    cfg = compose_cfg("runtime=kaggle_2xt4", "algorithm=masac")
    legacy = dict(BASE_CONFIG)
    apply_to_legacy_config(legacy, cfg)
    assert legacy["masac_discrete_target_entropy_ratio"] == pytest.approx(
        BASE_CONFIG["masac_discrete_target_entropy_ratio"]
    )


@pytest.mark.parametrize(
    "variant,activation,weight_init,layer_norm",
    [
        ("leaky_default", "leaky_relu", "default", False),
        ("prelu_default", "prelu", "default", False),
        ("leaky_kaiming", "leaky_relu", "kaiming", False),
        ("prelu_kaiming", "prelu", "kaiming", False),
        ("leaky_kaiming_ln", "leaky_relu", "kaiming", True),
        ("prelu_kaiming_ln", "prelu", "kaiming", True),
    ],
)
def test_ablation_architecture_variants_are_explicit_and_relu_free(
    variant, activation, weight_init, layer_norm
):
    cfg = compose_cfg(f"architecture={variant}", "algorithm=masac")
    assert cfg.architecture.name == variant
    assert cfg.architecture.activation == activation
    assert cfg.architecture.weight_init == weight_init
    assert cfg.architecture.layer_norm is layer_norm
    assert list(cfg.algorithm.hidden_dims) == [512, 256]
    assert activation != "relu"


def test_architecture_profile_maps_into_legacy_runtime_config():
    cfg = compose_cfg("architecture=prelu_kaiming_ln", "algorithm=masac")
    legacy = dict(BASE_CONFIG)
    apply_to_legacy_config(legacy, cfg)
    assert legacy["model_architecture_name"] == "prelu_kaiming_ln"
    assert legacy["model_activation"] == "prelu"
    assert legacy["model_weight_init"] == "kaiming"
    assert legacy["model_layer_norm"] is True
    assert legacy["model_prelu_init"] == pytest.approx(0.25)
    assert legacy["masac_hidden_dims"] == (512, 256)


def test_wandb_identity_exposes_full_model_variant():
    from uav_marl.configuration import make_wandb_run_identity

    legacy = dict(BASE_CONFIG)
    legacy.update(
        {
            "model_architecture_name": "prelu_kaiming_ln",
            "model_activation": "prelu",
            "model_weight_init": "kaiming",
            "model_layer_norm": True,
            "model_prelu_init": 0.25,
            "masac_hidden_dims": (512, 256),
        }
    )
    identity = make_wandb_run_identity(
        legacy,
        algorithm="masac",
        seed=44,
        stage="gpu_training",
        run_id="deadbeef",
    )
    assert identity["group"] == "MASAC-PReLU-Kaiming-LN-H512x256-seed44"
    assert identity["name"] == "MASAC-PReLU-Kaiming-LN-H512x256-seed44-GPU-deadbeef"
    assert identity["config"]["activation"] == "prelu"
    assert identity["config"]["weight_init"] == "kaiming"
    assert identity["config"]["layer_norm"] is True
    assert identity["config"]["hidden_dims"] == [512, 256]
