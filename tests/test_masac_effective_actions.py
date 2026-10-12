"""SAC must optimize entropy and Q only over controls the mission can execute."""
from pathlib import Path
import copy

import numpy as np
import pytest
import torch
import torch.distributed as dist

from config import CONFIG
from uav_marl.envs.gpu import FullGpuUAVBatchEnv
from uav_marl.training.gpu import (
    _trainer, _MASACDDPActorForward, _ddp_update_masac,
)


@pytest.fixture
def process_group(tmp_path):
    dist.init_process_group("gloo", init_method=(tmp_path / "gloo").as_uri(),
                            rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def make_case(monkeypatch, active_agents=2):
    monkeypatch.setitem(CONFIG, "masac_hidden_dims", (16, 16))
    monkeypatch.setitem(CONFIG, "masac_replay_capacity", 16)
    trainer = _trainer(Path.cwd(), "masac", "cpu", 44)
    obs = torch.full((trainer.num_agents, trainer.observation_dim), 0.1)
    obs[:, 7] = 0
    obs[:active_agents, 7] = 1
    mask = torch.zeros(trainer.num_agents, trainer.discrete_dim)
    mask[:, 0] = 1
    return trainer, obs, mask


def sample(trainer, obs, mask, ddp_path):
    if ddp_path:
        return _MASACDDPActorForward(trainer.actor, trainer._project_namespace)(obs, mask)
    result = trainer.actor.sample(obs, mask)
    return result["continuous"], result["continuous_log_probability"]


@pytest.mark.parametrize("ddp_path", [False, True])
def test_inactive_uavs_have_no_continuous_entropy_gradient(monkeypatch, ddp_path):
    trainer, obs, mask = make_case(monkeypatch, active_agents=0)
    obs.requires_grad_(True)
    _, logp, *_ = sample(trainer, obs, mask, ddp_path)
    assert torch.equal(logp, torch.zeros_like(logp))
    logp.sum().backward()
    assert torch.equal(obs.grad, torch.zeros_like(obs.grad))


@pytest.mark.parametrize("ddp_path", [False, True])
def test_silent_power_head_cannot_change_continuous_entropy(monkeypatch, ddp_path):
    trainer, obs, mask = make_case(monkeypatch, active_agents=int(CONFIG["num_uavs"]))
    torch.manual_seed(123)
    actions_before, logp_before, *_ = sample(trainer, obs, mask, ddp_path)
    with torch.no_grad():
        trainer.actor.mean_head.bias[3] += 8.0
        trainer.actor.log_std_head.bias[3] -= 3.0
    torch.manual_seed(123)
    actions_after, logp_after, *_ = sample(trainer, obs, mask, ddp_path)
    torch.testing.assert_close(actions_before[:, :3], actions_after[:, :3], rtol=0, atol=0)
    torch.testing.assert_close(logp_before, logp_after, rtol=0, atol=0)


def test_critic_identifies_equivalent_ignored_controls(monkeypatch):
    trainer, obs, mask = make_case(monkeypatch)
    obs = obs.unsqueeze(0)
    dest = mask.unsqueeze(0)
    left = torch.zeros(1, trainer.num_agents, 4)
    right = left.clone()
    right[:, 2:, :3] = 0.57
    right[:, :, 3] = 0.9
    first = trainer._joint_action_tensor(left, dest, observations=obs)
    second = trainer._joint_action_tensor(right, dest, observations=obs)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    right.requires_grad_(True)
    trainer._joint_action_tensor(right, dest, observations=obs).sum().backward()
    assert torch.count_nonzero(right.grad[:, 2:, :]) == 0
    assert torch.count_nonzero(right.grad[:, :, 3]) == 0
    assert torch.count_nonzero(right.grad[:, :2, :3]) == 6


def make_replay(trainer, obs, mask):
    replay = trainer.make_replay_buffer(seed=44)
    state = np.full(trainer.state_dim, 0.1, np.float32)
    actions = np.full((trainer.num_agents, 4), 0.1, np.float32)
    destinations = np.zeros(trainer.num_agents, np.int64)
    for _ in range(4):
        replay.add(obs.numpy(), state, actions, destinations, mask.numpy(),
                   0., obs.numpy(), state, mask.numpy(), True, False)
    return replay


def update(trainer, replay, ddp_path):
    if ddp_path:
        forward = _MASACDDPActorForward(trainer.actor, trainer._project_namespace)
        return _ddp_update_masac(trainer, replay, 4, forward,
                                trainer.critic_1, trainer.critic_2)
    return trainer.update(replay, batch_size=4)


@pytest.mark.parametrize("ddp_path", [False, True])
def test_temperature_target_counts_only_effective_controls(monkeypatch, ddp_path, process_group):
    trainer, obs, mask = make_case(monkeypatch)
    metrics = update(trainer, make_replay(trainer, obs, mask), ddp_path)
    assert metrics.get("continuous_target_entropy") == pytest.approx(-6.0)
    assert metrics.get("continuous_action_dimensions") == pytest.approx(6.0)


@pytest.mark.parametrize("ddp_path", [False, True])
def test_no_effective_control_preserves_temperature_adam_momentum(monkeypatch, ddp_path, process_group):
    trainer, obs, mask = make_case(monkeypatch, active_agents=0)
    optimizer = trainer.alpha_continuous_optimizer
    optimizer.zero_grad()
    (-trainer.log_alpha_continuous).backward()
    optimizer.step()
    before = trainer.log_alpha_continuous.detach().clone()
    step_before = optimizer.state[trainer.log_alpha_continuous]["step"].clone()
    update(trainer, make_replay(trainer, obs, mask), ddp_path)
    assert torch.equal(trainer.log_alpha_continuous.detach(), before)
    assert torch.equal(optimizer.state[trainer.log_alpha_continuous]["step"], step_before)


def test_mission_transition_ignores_inactive_motion_and_silent_power():
    env = FullGpuUAVBatchEnv(1, device="cpu", seed=44, strict_cuda=False)
    env.active[:, 2:] = False
    env.battery[:, 2:] = 0
    env.velocities[:, 2:] = 0
    other = copy.deepcopy(env)
    left = torch.zeros(1, env.num_uavs, 4)
    right = left.clone()
    right[:, 2:, :3] = 0.57
    right[:, :, 3] = 0.9
    destinations = torch.zeros(1, env.num_uavs, dtype=torch.long)
    first = env.step(left, destinations)
    second = other.step(right, destinations)
    for index in (0, 1, 2, 3, 4):
        torch.testing.assert_close(first[index], second[index], rtol=0, atol=0)


def test_transmitting_controls_retain_power_dimension_and_gradients(monkeypatch):
    trainer, obs, mask = make_case(monkeypatch, active_agents=int(CONFIG["num_uavs"]))
    mask.zero_()
    mask[:, 1] = 1
    policy = trainer._sample_joint_policy(obs.unsqueeze(0), mask.unsqueeze(0))
    assert policy["continuous_action_dimensions"].item() == 4 * trainer.num_agents
    policy["continuous_log_probability"].sum().backward()
    assert abs(trainer.actor.mean_head.bias.grad[3].item()) > 1e-6
    assert abs(trainer.actor.log_std_head.bias.grad[3].item()) > 1e-6


def test_conditional_power_entropy_preserves_categorical_gradient(monkeypatch):
    trainer, obs, mask = make_case(monkeypatch, active_agents=int(CONFIG["num_uavs"]))
    mask[:, 1] = 1
    with torch.no_grad():
        trainer.actor.mean_head.bias[3] = 2.0
    trainer.actor.sample(obs, mask)["continuous_log_probability"].sum().backward()
    assert torch.linalg.vector_norm(trainer.actor.discrete_head.bias.grad[:2]) > 1e-6


def test_depleted_destination_is_canonical_silent(monkeypatch):
    trainer, obs, mask = make_case(monkeypatch)
    dest = mask.unsqueeze(0)
    non_silent = dest.clone()
    non_silent[:, 2:, 0] = 0
    non_silent[:, 2:, 1] = 1
    actions = torch.ones(1, trainer.num_agents, 4) * 0.1
    left = trainer._joint_action_tensor(actions, dest, observations=obs.unsqueeze(0))
    right = trainer._joint_action_tensor(actions, non_silent, observations=obs.unsqueeze(0))
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    non_silent.requires_grad_(True)
    trainer._joint_action_tensor(actions, non_silent, observations=obs.unsqueeze(0)).sum().backward()
    assert torch.count_nonzero(non_silent.grad[:, 2:]) == 0


def _continuous_two_rank_worker(rank, init_file, output_dir):
    from torch.nn.parallel import DistributedDataParallel as DDP
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=Path(init_file).as_uri(),
                            rank=rank, world_size=2)
    try:
        CONFIG["masac_hidden_dims"] = (16, 16)
        CONFIG["masac_replay_capacity"] = 16
        trainer = _trainer(Path.cwd(), "masac", "cpu", 44)
        obs = torch.full((trainer.num_agents, trainer.observation_dim), 0.1)
        obs[:, 7] = 0
        if rank == 1:
            obs[:2, 7] = 1
        mask = torch.zeros(trainer.num_agents, trainer.discrete_dim)
        mask[:, 0] = 1
        actor = DDP(_MASACDDPActorForward(trainer.actor, trainer._project_namespace))
        critic1, critic2 = DDP(trainer.critic_1), DDP(trainer.critic_2)
        metrics = _ddp_update_masac(trainer, make_replay(trainer, obs, mask),
                                   4, actor, critic1, critic2)
        before = trainer.log_alpha_continuous.detach().clone()
        state_before = copy.deepcopy(trainer.alpha_continuous_optimizer.state_dict())
        obs[:, 7] = 0
        _ddp_update_masac(trainer, make_replay(trainer, obs, mask),
                         4, actor, critic1, critic2)
        torch.save({"alpha_before": before,
                    "alpha_after": trainer.log_alpha_continuous.detach().clone(),
                    "state_before": state_before,
                    "state_after": trainer.alpha_continuous_optimizer.state_dict(),
                    "metrics": metrics}, Path(output_dir) / f"continuous-{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_two_rank_continuous_temperature_and_metrics_share_effective_objective(tmp_path):
    torch.multiprocessing.spawn(
        _continuous_two_rank_worker,
        args=(str(tmp_path / "continuous-rendezvous"), str(tmp_path)),
        nprocs=2, join=True,
    )
    rows = [torch.load(tmp_path / f"continuous-{rank}.pt", weights_only=True)
            for rank in range(2)]
    torch.testing.assert_close(rows[0]["alpha_before"], rows[1]["alpha_before"], rtol=0, atol=0)
    for row in rows:
        assert torch.equal(row["alpha_before"], row["alpha_after"])
        for key in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(row["state_before"]["state"][0][key],
                               row["state_after"]["state"][0][key])
        assert row["metrics"]["continuous_action_dimensions"] == pytest.approx(3.0)
        assert row["metrics"]["continuous_target_entropy"] == pytest.approx(-3.0)
    for key in ("continuous_entropy", "continuous_target_entropy",
                "continuous_action_dimensions", "continuous_entropy_error"):
        assert rows[0]["metrics"][key] == pytest.approx(rows[1]["metrics"][key])
