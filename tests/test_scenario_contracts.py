"""Mission edge cases beyond the short training smoke tests."""

import numpy as np
import pytest
import torch

from config import CONFIG
from uav_marl.envs.gpu import FullGpuUAVBatchEnv, GpuReplayBuffer
from uav_marl.network.backends import SimpleNetworkBackend
from uav_marl.network.communication import GCS_NODE, TransmissionIntent
from uav_marl.world.entities import UAV, Report
from uav_marl.world.reports import (
    enqueue_pending_report,
    enqueue_report,
    remove_expired_reports,
)
from uav_marl.world.sensing import bayes_update, check_confirmation, sensing_profile


@pytest.mark.parametrize("altitude,pd,pf,radius", [(0, .99, .01, 0), (50, .9, .1, 50), (100, .8, .2, 100), (150, .7, .3, 150)])
def test_sensor_altitude_changes_noise_and_footprint(altitude, pd, pf, radius):
    assert sensing_profile(altitude) == pytest.approx((pd, pf, radius))


def test_bayes_requires_repeated_evidence_at_confirmation_altitude():
    posterior = bayes_update(.5, 1, .9, .1)
    assert posterior == pytest.approx(.9)
    assert bayes_update(.5, 0, .9, .1) == pytest.approx(.1)
    uav = UAV(0, np.array([100., 100., 50.]), np.zeros(3), 277200.)
    record = {"posterior": posterior, "observation": 1, "target_ids": [0]}
    assert check_confirmation(uav, record) == ("none", [])
    record["posterior"] = bayes_update(bayes_update(posterior, 1, .9, .1), 1, .9, .1)
    assert check_confirmation(uav, record) == ("true_confirmation", [0])


@pytest.mark.parametrize("active,altitude,belief,observation,targets,expected", [
    (True, 50., .99, 1, [0], "true_confirmation"),
    (True, 50., .99, 1, [], "false_confirmation"),
    (False, 50., .99, 1, [0], "none"),
    (True, 50.01, .99, 1, [0], "none"),
    (True, 50., .9899, 1, [0], "none"),
    (True, 50., .999, 0, [0], "none"),
])
def test_confirmation_gate(active, altitude, belief, observation, targets, expected):
    uav = UAV(0, np.array([100., 100., altitude]), np.zeros(3), 277200., active)
    assert check_confirmation(uav, {"posterior": belief, "observation": observation, "target_ids": targets})[0] == expected


def test_report_capacity_duplicate_and_exact_ttl_boundary(monkeypatch):
    monkeypatch.setitem(CONFIG, "dt", 1.)
    monkeypatch.setitem(CONFIG, "buffer_bytes", 3000000)
    monkeypatch.setitem(CONFIG, "pending_buffer_bytes", 3000000)
    reports = [Report(i, 0, 0, 1000000, 300.) for i in range(4)]
    buffer, pending = [], []
    for report in reports[:3]:
        assert enqueue_report(buffer, report) == (True, "enqueued")
    assert enqueue_report(buffer, reports[0]) == (False, "duplicate")
    assert enqueue_report(buffer, reports[3]) == (False, "buffer_full")
    for report in reports[:3]:
        assert enqueue_pending_report(pending, report) == (True, "enqueued")
    assert enqueue_pending_report(pending, reports[3]) == (False, "pending_full")
    assert enqueue_pending_report(pending, Report(4, 1, 0, 1000000, 300.)) == (True, "enqueued")
    assert remove_expired_reports(buffer, 299) == []
    assert len(remove_expired_reports(buffer, 300)) == 3
    assert buffer == []


def test_simple_network_copy_forward_and_complete_gcs_delivery():
    gcs = np.asarray(CONFIG["gcs_position"], dtype=float)
    uavs = [UAV(i, gcs + [0., 100. + 50. * i, 10.], np.zeros(3), 277200.) for i in range(int(CONFIG["num_uavs"]))]
    buffers = [[] for _ in uavs]
    buffers[0].append(Report(0, 0, 0, 1000000, 300.))
    delivered = set()
    backend = SimpleNetworkBackend()
    backend.reset(uavs=uavs, gcs_position=gcs, obstacles=[], report_buffers=buffers, pending_reports=[], gcs_received_target_ids=delivered)
    for step in range(1, 5):
        result = backend.step([TransmissionIntent(0, 1, 0, 1000000, .4)], dt=1., current_step=step)[0]
        assert result["tx_bytes"] == 250000
        assert len(buffers[0]) == 1
        assert len(buffers[1]) == int(step == 4)
    assert buffers[1][0].source_uav == 0
    assert buffers[0][0] is not buffers[1][0]
    for step in range(5, 9):
        result = backend.step([TransmissionIntent(1, GCS_NODE, 0, 1000000, .4)], dt=1., current_step=step)[0]
        assert result["tx_bytes"] == 250000
        assert (0 in delivered) == (step == 8)
    assert all(not buffer for buffer in buffers)


def test_partial_vector_reset_preserves_unfinished_environment():
    env = FullGpuUAVBatchEnv(2, device="cpu", seed=44, strict_cuda=False)
    try:
        env.step_count[:] = 7
        env.coverage_seen[:, 0] = True
        fields = [name for name, value in vars(env).items() if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == 2]
        before = {name: getattr(env, name)[1].clone() for name in fields}
        env.reset(mask=torch.tensor([True, False]))
        for name, value in before.items():
            assert torch.equal(getattr(env, name)[1], value), name
        assert env.step_count.tolist() == [0, 7]
        assert not bool(env.coverage_seen[0].any())
    finally:
        env.close()


def test_replay_oversized_batch_keeps_latest_rows_and_restores_sampling_rng():
    replay = GpuReplayBuffer(4, 1, 1, 1, 1, 2, "cpu", strict_cuda=False)
    def insert(start, size):
        value = torch.arange(start, start + size, dtype=torch.float32)
        obs = value.view(size, 1, 1)
        state = value.view(size, 1)
        mask = torch.ones(size, 1, 2)
        replay.add_batch(obs, state, obs, torch.zeros(size, 1, dtype=torch.long), mask,
                         value, obs, state, mask, torch.zeros(size))
    insert(0, 2)
    insert(2, 7)
    assert replay.size == 4 and replay.position == 1
    assert sorted(replay.rewards.flatten().tolist()) == [5., 6., 7., 8.]
    restored = GpuReplayBuffer(4, 1, 1, 1, 1, 2, "cpu", seed=999, strict_cuda=False)
    restored.load_state_dict(replay.state_dict())
    for _ in range(3):
        first, second = replay.sample(4, "cpu"), restored.sample(4, "cpu")
        assert all(torch.equal(first[name], second[name]) for name in first)


def test_gpu_replay_can_store_float16_but_sample_float32():
    replay = GpuReplayBuffer(
        8, 2, 3, 5, 4, 3, "cpu", strict_cuda=False,
        storage_dtype=torch.float16,
    )
    batch = 4
    replay.add_batch(
        observations=torch.rand(batch, 2, 3),
        states=torch.rand(batch, 5),
        continuous_actions=torch.rand(batch, 2, 4),
        destination_indices=torch.zeros(batch, 2, dtype=torch.long),
        destination_masks=torch.ones(batch, 2, 3),
        rewards=torch.rand(batch, 1),
        next_observations=torch.rand(batch, 2, 3),
        next_states=torch.rand(batch, 5),
        next_destination_masks=torch.ones(batch, 2, 3),
        terminated=torch.zeros(batch, 1),
        truncated=torch.zeros(batch, 1),
    )
    assert replay.observations.dtype == torch.float16
    assert replay.states.dtype == torch.float16
    assert replay.destination_masks.dtype == torch.float16
    assert replay.destination_indices.dtype == torch.long

    sampled = replay.sample(2, "cpu")
    for name, value in sampled.items():
        if name == "destination_indices":
            assert value.dtype == torch.long
        else:
            assert value.dtype == torch.float32
