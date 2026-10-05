from __future__ import annotations

import numpy as np
import torch

from config import CONFIG
from uav_marl.envs.gpu import FullGpuUAVBatchEnv
from uav_marl.world.entities import UAV
from uav_marl.world.motion import apply_swarm_motion


def _set_apf_config(monkeypatch):
    monkeypatch.setitem(CONFIG, "apf_enabled", True)
    monkeypatch.setitem(CONFIG, "apf_soft_gain", 1.5)
    monkeypatch.setitem(CONFIG, "apf_lookahead_s", 3.0)
    monkeypatch.setitem(CONFIG, "apf_braking_margin", 1.5)
    monkeypatch.setitem(CONFIG, "safety_distance", 30.0)
    monkeypatch.setitem(CONFIG, "obstacle_clearance_m", 30.0)
    monkeypatch.setitem(CONFIG, "max_accel", 2.0)
    monkeypatch.setitem(CONFIG, "max_speed", 20.0)


def test_apply_swarm_motion_uses_risk_aware_apf_without_hard_blocking(monkeypatch):
    _set_apf_config(monkeypatch)
    uavs = [
        UAV(
            id=0,
            position=np.asarray([100.0, 100.0, 50.0]),
            velocity=np.zeros(3, dtype=np.float64),
            battery_j=float(CONFIG["battery_j"]),
            active=True,
        ),
        UAV(
            id=1,
            position=np.asarray([135.0, 100.0, 50.0]),
            velocity=np.zeros(3, dtype=np.float64),
            battery_j=float(CONFIG["battery_j"]),
            active=True,
        ),
    ]
    result = apply_swarm_motion(
        uavs,
        np.asarray(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        obstacles=[],
        dt=1.0,
    )

    assert not np.any(result["blocked"])
    assert not np.any(result["blocked_by_peer"])
    assert np.all(result["apf_peer_active"])
    assert np.all(result["apf_emergency"])
    assert uavs[0].position[0] <= 100.0
    assert uavs[1].position[0] >= 135.0


def test_gpu_motion_uses_same_risk_aware_apf_safety_layer(monkeypatch):
    _set_apf_config(monkeypatch)
    env = FullGpuUAVBatchEnv(
        num_envs=1,
        device="cpu",
        seed=44,
        strict_cuda=False,
    )
    env.active[:] = False
    env.active[0, 0] = True
    env.active[0, 1] = True
    env.positions[0, 0] = torch.tensor([100.0, 100.0, 50.0])
    env.positions[0, 1] = torch.tensor([135.0, 100.0, 50.0])
    env.velocities.zero_()
    env.obstacle_xy[:] = torch.tensor([2500.0, 2500.0])
    env.obstacle_radius[:] = 50.0
    env.obstacle_height[:] = 30.0

    motion = torch.zeros(
        (1, env.num_uavs, 3),
        dtype=torch.float32,
    )
    motion[0, 0, 0] = 1.0
    motion[0, 1, 0] = -1.0

    result = env._motion(motion)

    assert not bool(result["blocked"].any())
    assert not bool(result["blocked_by_peer"].any())
    assert bool(result["apf_peer_active"][0, 0])
    assert bool(result["apf_peer_active"][0, 1])
    assert bool(result["apf_emergency"][0, 0])
    assert bool(result["apf_emergency"][0, 1])
    assert float(env.positions[0, 0, 0]) <= 100.0
    assert float(env.positions[0, 1, 0]) >= 135.0


def test_gpu_motion_preserves_speed_cap_after_position_round_trip(monkeypatch):
    _set_apf_config(monkeypatch)
    monkeypatch.setitem(CONFIG, "apf_enabled", False)
    env = FullGpuUAVBatchEnv(
        num_envs=1,
        device="cpu",
        seed=44,
        strict_cuda=False,
    )
    env.active[:] = False
    env.active[0, 0] = True
    env.positions[0, 0] = torch.tensor(
        [2745.913818359375, 2325.429931640625, 101.22919464111328],
        dtype=torch.float32,
    )
    env.velocities.zero_()
    env.velocities[0, 0] = torch.tensor(
        [-11.362462043762207, 14.687152862548828, -7.428459644317627],
        dtype=torch.float32,
    )
    env.obstacle_xy[:] = torch.tensor([100.0, 100.0])
    env.obstacle_radius[:] = 50.0
    env.obstacle_height[:] = 30.0

    motion = torch.zeros((1, env.num_uavs, 3), dtype=torch.float32)
    env._motion(motion)

    speed = torch.linalg.vector_norm(env.velocities[0, 0])
    assert float(speed) <= float(CONFIG["max_speed"]) + 1e-5
