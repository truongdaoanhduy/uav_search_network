"""Behavioral checks for the shared hybrid-action learner contract."""

from pathlib import Path

import numpy as np
import pytest
import torch

from config import CONFIG
from uav_marl.algorithms.masac import HybridMASACActor, clip_grad_norm_finite
from uav_marl.algorithms.matd3 import HybridMATD3Actor
from uav_marl.training.gpu import _trainer, train_full_gpu


def _actor_observation_layout():
    prefix = (
        9
        + 3
        + 5 * (int(CONFIG["num_uavs"]) - 1)
        + 5 * int(CONFIG["observation_nearest_obstacles"])
    )
    patch_cells = int(CONFIG["belief_patch_cells"]) ** 2
    coarse_cells = int(CONFIG["belief_coarse_cells"]) ** 2
    patch = slice(prefix, prefix + patch_cells)
    coarse = slice(patch.stop, patch.stop + coarse_cells)
    observation_dim = coarse.stop + 5 + int(CONFIG["num_uavs"]) + 1
    return observation_dim, patch, coarse


@pytest.mark.parametrize(
    "actor_type",
    [HybridMASACActor, HybridMATD3Actor],
)
def test_actor_centers_uninformative_belief_before_encoder(actor_type):
    observation_dim, patch, coarse = _actor_observation_layout()
    actor = actor_type(
        observation_dim=observation_dim,
        continuous_dim=4,
        discrete_dim=int(CONFIG["num_uavs"]) + 1,
        hidden_dims=(8, 8),
    )
    observations = torch.full((2, observation_dim), 0.25)
    observations[0, patch] = float(CONFIG["belief_prior"])
    observations[0, coarse] = 1.0
    observations[1, patch] = float(CONFIG["belief_prior"])
    observations[1, coarse] = 1.0
    observations[1, patch.start] = 0.0
    observations[1, patch.start + 1] = 1.0
    observations[1, coarse.start] = 0.0

    encoder_inputs = []
    hook = actor.encoder[0].register_forward_pre_hook(
        lambda _module, args: encoder_inputs.append(args[0].detach().clone())
    )
    try:
        actor(observations)
    finally:
        hook.remove()

    transformed = encoder_inputs[0]
    torch.testing.assert_close(
        transformed[0, patch],
        torch.zeros(patch.stop - patch.start),
    )
    torch.testing.assert_close(
        transformed[0, coarse],
        torch.zeros(coarse.stop - coarse.start),
    )
    assert transformed[1, patch.start].item() == pytest.approx(-1.0)
    assert transformed[1, patch.start + 1].item() == pytest.approx(1.0)
    assert transformed[1, coarse.start].item() == pytest.approx(1.0)
    assert transformed[0, 0].item() == pytest.approx(0.25)


@pytest.mark.parametrize(
    "actor_type",
    [HybridMASACActor, HybridMATD3Actor],
)
def test_actor_encoder_keeps_gradient_path_for_negative_preactivations(actor_type):
    observation_dim, _, _ = _actor_observation_layout()
    actor = actor_type(
        observation_dim=observation_dim,
        continuous_dim=4,
        discrete_dim=int(CONFIG["num_uavs"]) + 1,
        hidden_dims=(8, 8),
    )
    first = actor.encoder[0]
    second = actor.encoder[2]
    with torch.no_grad():
        first.weight.zero_()
        first.bias.fill_(-1.0)
        second.weight.zero_()
        second.bias.zero_()
        second.weight.copy_(torch.eye(8))

    features = actor.encoder(torch.zeros(1, observation_dim))
    assert torch.count_nonzero(features).item() == features.numel()
    assert (features < 0).all()


def test_masac_rejects_nonfinite_gradients_before_optimizer_step():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([float("inf")])

    with pytest.raises(FloatingPointError, match="non-finite MASAC actor gradient"):
        clip_grad_norm_finite(
            [parameter],
            max_norm=10.0,
            label="MASAC actor",
        )

    assert parameter.item() == pytest.approx(1.0)


def test_masac_default_discrete_entropy_target_avoids_near_uniform_runaway():
    # Long-horizon discrete SAC is more stable below the historical 0.98*Hmax
    # target. The failed paper run drove alpha_discrete from ~0.5 to >127.
    assert float(CONFIG["masac_discrete_target_entropy_ratio"]) <= 0.89


def _small_learner_config(monkeypatch, algorithm):
    for key, value in {
        "hidden_dims": (32, 32), "batch_size": 4, "replay_capacity": 64,
        "learning_starts": 1, "train_freq": 1, "gradient_steps": 1,
    }.items():
        monkeypatch.setitem(CONFIG, f"{algorithm}_{key}", value)
    monkeypatch.setitem(CONFIG, "matd3_policy_delay", 2)
    monkeypatch.setitem(CONFIG, "training_auto_scale_gradient_steps", False)
    monkeypatch.setitem(CONFIG, "training_inline_evaluation", False)
    monkeypatch.setitem(CONFIG, "training_checkpoint_interval_episodes", 1000)


def _parameters(module):
    return torch.cat([p.detach().flatten().clone() for p in module.parameters()])


@pytest.mark.parametrize("algorithm", ["masac", "matd3"])
def test_hybrid_policy_respects_single_legal_destination_and_motion_ball(monkeypatch, algorithm):
    _small_learner_config(monkeypatch, algorithm)
    trainer = _trainer(Path.cwd(), algorithm, "cpu", 44)
    observations = torch.zeros(8, trainer.num_agents, trainer.observation_dim)
    masks = torch.zeros(8, trainer.num_agents, trainer.discrete_dim)
    legal = torch.arange(trainer.num_agents) % trainer.discrete_dim
    masks.scatter_(-1, legal.view(1, -1, 1).expand(8, -1, -1), 1)
    if algorithm == "masac":
        actions = trainer._sample_joint_policy(observations, masks, deterministic=False)
        assert torch.isfinite(actions["continuous_log_probability"]).all()
        assert torch.allclose(actions["destination_entropy"], torch.zeros(8, 1), atol=1e-6)
    else:
        actions = trainer._actor_joint_actions(observations, masks)
    assert torch.equal(actions["destination_indices"], legal.expand(8, -1))
    continuous = actions["continuous"]
    assert torch.isfinite(continuous).all()
    assert (torch.linalg.vector_norm(continuous[..., :3], dim=-1) <= 1 + 1e-6).all()
    assert (continuous[..., 3].abs() <= 1).all()


@pytest.mark.parametrize("algorithm", ["masac", "matd3"])
def test_terminal_targets_gradient_updates_and_delayed_target_updates(monkeypatch, algorithm):
    _small_learner_config(monkeypatch, algorithm)
    trainer = _trainer(Path.cwd(), algorithm, "cpu", 44)
    replay = trainer.make_replay_buffer(seed=44)
    observations = np.ones((trainer.num_agents, trainer.observation_dim), dtype=np.float32) * 0.1
    state = np.ones(trainer.state_dim, dtype=np.float32) * 0.1
    actions = np.zeros((trainer.num_agents, trainer.continuous_dim), dtype=np.float32)
    destinations = np.zeros(trainer.num_agents, dtype=np.int64)
    masks = np.ones((trainer.num_agents, trainer.discrete_dim), dtype=np.float32)
    for _ in range(4):
        replay.add(observations, state, actions, destinations, masks, 1.5,
                   observations, state, masks, True, False)
    actor_before = _parameters(trainer.actor)
    critic_before = _parameters(trainer.critic_1)
    target_before = _parameters(trainer.target_critic_1)
    for index in range(1, 3):
        metrics = trainer.update(replay, batch_size=4)
        assert metrics["target_q_mean"] == pytest.approx(1.5)
        assert metrics["critic_grad_norm"] > 0
        assert not torch.equal(_parameters(trainer.critic_1), critic_before)
        actor_due = algorithm == "masac" or index == 2
        if actor_due:
            assert not torch.equal(_parameters(trainer.actor), actor_before)
            expected = target_before * (1 - trainer.tau) + _parameters(trainer.critic_1) * trainer.tau
            torch.testing.assert_close(_parameters(trainer.target_critic_1), expected)
        else:
            assert torch.equal(_parameters(trainer.actor), actor_before)
            assert torch.equal(_parameters(trainer.target_critic_1), target_before)
        assert all(p.grad is None for p in trainer.target_critic_1.parameters())
        target_before = _parameters(trainer.target_critic_1)
    if algorithm == "masac":
        assert torch.isfinite(trainer.alpha_continuous) and trainer.alpha_continuous > 0
        assert torch.isfinite(trainer.alpha_discrete) and trainer.alpha_discrete > 0


def _assert_identical(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_identical(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_identical(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("algorithm", ["masac", "matd3"])
@pytest.mark.parametrize("backend", ["simple", "uavnetsim_gpu"])
def test_training_same_seed_reproduces_weights_optimizer_and_episode_budget(monkeypatch, tmp_path, algorithm, backend):
    _small_learner_config(monkeypatch, algorithm)
    monkeypatch.setitem(CONFIG, "max_steps", 8)
    runs = []
    for repeat in range(2):
        result = train_full_gpu(
            repo=tmp_path / str(repeat), algorithm=algorithm, num_envs=2,
            total_transitions=40, seed=44, device="cpu", network_backend=backend,
            strict_cuda=False, enable_wandb=False, enable_evaluation=False, target_episodes=3,
        )
        assert result["episodes_completed"] == 3
        assert result["transitions"] == 32
        assert result["updates"] > 0
        runs.append(result)
    assert runs[0]["model_fingerprint_sha256"] == runs[1]["model_fingerprint_sha256"]
    assert runs[0]["latest_update_metrics"] == runs[1]["latest_update_metrics"]
    saved = [torch.load(r["final_checkpoint_path"], map_location="cpu", weights_only=True) for r in runs]
    _assert_identical(saved[0]["trainer_state"], saved[1]["trainer_state"])


@pytest.mark.parametrize("alpha", [127.0, 129.0, 1000.0])
def test_masked_entropy_large_temperature_has_finite_correct_gradients(alpha):
    from uav_marl.envs.uav_search import straight_through_categorical_sample

    # The failed 2-GPU run used 128 replay rows per rank and six agents.
    logits = torch.zeros(128 * 6, 7, requires_grad=True)
    with torch.no_grad():
        logits[:, 0] = np.log(3.0)
    masks = torch.zeros_like(logits)
    masks[:, :2] = 1
    masks[::2, 1] = 0  # Include silent-only rows.
    sampled = straight_through_categorical_sample(logits, masks, 1.0)
    entropy = sampled["entropy"].reshape(128, 6, 1).sum(dim=1)
    (-alpha * entropy.mean()).backward()

    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[::2]) == 0
    assert torch.count_nonzero(logits.grad[:, 2:]) == 0
    # d(-alpha*H)/dx0 = alpha/B * p0*p1*log(p0/p1).
    expected = alpha / 128 * 0.75 * 0.25 * np.log(3.0)
    torch.testing.assert_close(
        logits.grad[1::2, 0], torch.full_like(logits.grad[1::2, 0], expected)
    )
    torch.testing.assert_close(logits.grad[1::2, 1], -logits.grad[1::2, 0])


def test_radial_squash_keeps_entropy_gradient_when_motion_saturates():
    from uav_marl.envs.uav_search import radial_squash_motion_action

    latent = torch.tensor([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]], requires_grad=True)
    actions, log_det = radial_squash_motion_action(latent)
    log_det.sum().backward()
    assert torch.isfinite(latent.grad).all()
    assert log_det[0].item() == pytest.approx(0.0, abs=1e-6)
    # At r=20: log(sech(r)^2) + 2*log(tanh(r)/r).
    assert log_det[1].item() == pytest.approx(-44.605170186, abs=1e-5)
    assert latent.grad[1, 0].item() == pytest.approx(-2.1, abs=1e-5)
    assert torch.linalg.vector_norm(actions[1]).item() <= 1.0


@pytest.mark.parametrize("ddp_forward", [False, True])
def test_masac_saturated_power_keeps_entropy_gradient(ddp_forward):
    from uav_marl.training.gpu import _MASACDDPActorForward
    from uav_marl.envs.uav_search import (
        radial_squash_motion_action, straight_through_categorical_sample,
    )

    observation_dim, _, _ = _actor_observation_layout()
    actor = HybridMASACActor(observation_dim, 4, 7, (8, 8))
    with torch.no_grad():
        actor.mean_head.weight.zero_()
        actor.mean_head.bias.zero_()
        actor.mean_head.bias[3] = 20.0
        actor.log_std_head.weight.zero_()
        actor.log_std_head.bias.zero_()
    observations = torch.zeros(2, observation_dim)
    masks = torch.ones(2, 7)
    if ddp_forward:
        wrapper = _MASACDDPActorForward(actor, {
            "radial_squash_motion_action": radial_squash_motion_action,
            "straight_through_categorical_sample": straight_through_categorical_sample,
        })
        log_probability = wrapper(observations, masks)[1]
    else:
        log_probability = actor.sample(observations, masks, deterministic=True)[
            "continuous_log_probability"
        ]
    log_probability.mean().backward()
    assert actor.mean_head.bias.grad[3].item() == pytest.approx(2.0, abs=1e-5)


def test_masked_categorical_log_probability_ignores_invalid_half_precision_logits():
    from uav_marl.envs.uav_search import straight_through_categorical_sample

    logits = torch.tensor([[64.0, 0.0, -5.0]], dtype=torch.float16, requires_grad=True)
    masks = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float16)
    sampled = straight_through_categorical_sample(logits, masks, 1.0)
    assert sampled["log_probability"].item() == 0.0
    assert sampled["entropy"].item() == 0.0
    (-sampled["log_probability"] - 10.0 * sampled["entropy"]).sum().backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))
