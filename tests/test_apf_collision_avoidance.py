from __future__ import annotations

import numpy as np
import torch

from config import CONFIG
from uav_marl.common import (
    apf_repulsion_numpy,
    apf_repulsion_torch,
)
from uav_marl.envs.gpu import FullGpuUAVBatchEnv
from uav_marl.world.entities import Obstacle, UAV
from uav_marl.world.motion import apply_swarm_motion


def _set_apf_test_config(monkeypatch):
    monkeypatch.setitem(CONFIG, "apf_peer_influence_m", 100.0)
    monkeypatch.setitem(CONFIG, "apf_obstacle_influence_m", 60.0)
    monkeypatch.setitem(CONFIG, "apf_peer_gain", 4.0)
    monkeypatch.setitem(CONFIG, "apf_obstacle_gain", 4.0)
    monkeypatch.setitem(CONFIG, "apf_lookahead_s", 2.0)
    monkeypatch.setitem(CONFIG, "apf_emergency_gain", 4.0)
    monkeypatch.setitem(CONFIG, "apf_braking_margin", 1.5)


def test_numpy_apf_peer_repulsion_is_symmetric_and_predictive(monkeypatch):
    _set_apf_test_config(monkeypatch)
    positions = np.asarray(
        [
            [100.0, 100.0, 50.0],
            [160.0, 100.0, 50.0],
        ],
        dtype=np.float64,
    )
    velocities = np.asarray(
        [
            [10.0, 0.0, 0.0],
            [-10.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    active = np.asarray([True, True])

    result = apf_repulsion_numpy(
        positions,
        velocities,
        active,
        obstacle_xy=np.zeros((0, 2), dtype=np.float64),
        obstacle_radius=np.zeros((0,), dtype=np.float64),
        obstacle_height=np.zeros((0,), dtype=np.float64),
        max_accel=float(CONFIG["max_accel"]),
        safety_distance=float(CONFIG["safety_distance"]),
        obstacle_clearance=float(CONFIG["obstacle_clearance_m"]),
        peer_influence=float(CONFIG["apf_peer_influence_m"]),
        obstacle_influence=float(CONFIG["apf_obstacle_influence_m"]),
        peer_gain=float(CONFIG["apf_peer_gain"]),
        obstacle_gain=float(CONFIG["apf_obstacle_gain"]),
        lookahead_s=float(CONFIG["apf_lookahead_s"]),
    )

    peer = result["peer_acceleration_mps2"]
    assert bool(result["peer_active"][0])
    assert bool(result["peer_active"][1])
    assert peer[0, 0] < 0.0
    assert peer[1, 0] > 0.0
    assert np.allclose(peer[0], -peer[1], atol=1e-12)


def test_numpy_and_torch_apf_match_for_peer_and_obstacle(monkeypatch):
    _set_apf_test_config(monkeypatch)
    positions = np.asarray(
        [
            [130.0, 100.0, 50.0],
            [180.0, 100.0, 50.0],
        ],
        dtype=np.float64,
    )
    velocities = np.asarray(
        [
            [-5.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    active = np.asarray([True, True])
    obstacle_xy = np.asarray([[100.0, 100.0]], dtype=np.float64)
    obstacle_radius = np.asarray([20.0], dtype=np.float64)
    obstacle_height = np.asarray([80.0], dtype=np.float64)

    kwargs = dict(
        max_accel=float(CONFIG["max_accel"]),
        safety_distance=float(CONFIG["safety_distance"]),
        obstacle_clearance=float(CONFIG["obstacle_clearance_m"]),
        peer_influence=float(CONFIG["apf_peer_influence_m"]),
        obstacle_influence=float(CONFIG["apf_obstacle_influence_m"]),
        peer_gain=float(CONFIG["apf_peer_gain"]),
        obstacle_gain=float(CONFIG["apf_obstacle_gain"]),
        lookahead_s=float(CONFIG["apf_lookahead_s"]),
    )
    np_result = apf_repulsion_numpy(
        positions,
        velocities,
        active,
        obstacle_xy,
        obstacle_radius,
        obstacle_height,
        **kwargs,
    )
    torch_result = apf_repulsion_torch(
        torch.tensor(positions[None, ...], dtype=torch.float64),
        torch.tensor(velocities[None, ...], dtype=torch.float64),
        torch.tensor(active[None, ...]),
        torch.tensor(obstacle_xy[None, ...], dtype=torch.float64),
        torch.tensor(obstacle_radius[None, ...], dtype=torch.float64),
        torch.tensor(obstacle_height[None, ...], dtype=torch.float64),
        **kwargs,
    )

    assert np.allclose(
        np_result["acceleration_mps2"],
        torch_result["acceleration_mps2"][0].numpy(),
        atol=1e-9,
    )
    assert np.array_equal(
        np_result["peer_active"],
        torch_result["peer_active"][0].numpy(),
    )
    assert np.array_equal(
        np_result["obstacle_active"],
        torch_result["obstacle_active"][0].numpy(),
    )


def test_apply_swarm_motion_uses_apf_instead_of_hard_blocking(monkeypatch):
    _set_apf_test_config(monkeypatch)
    monkeypatch.setitem(CONFIG, "safety_distance", 30.0)
    monkeypatch.setitem(CONFIG, "obstacle_clearance_m", 10.0)

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
            position=np.asarray([132.0, 100.0, 50.0]),
            velocity=np.zeros(3, dtype=np.float64),
            battery_j=float(CONFIG["battery_j"]),
            active=True,
        ),
    ]
    actions = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    result = apply_swarm_motion(
        uavs,
        actions,
        obstacles=[],
        dt=1.0,
    )

    assert not np.any(result["blocked"])
    assert not np.any(result["blocked_by_peer"])
    assert np.all(result["apf_peer_active"])
    assert np.all(result["apf_active"])
    assert uavs[0].position[0] <= 100.0
    assert uavs[1].position[0] >= 132.0


def test_gpu_motion_uses_same_apf_safety_layer_on_cpu_device(monkeypatch):
    _set_apf_test_config(monkeypatch)
    monkeypatch.setitem(CONFIG, "safety_distance", 30.0)
    monkeypatch.setitem(CONFIG, "obstacle_clearance_m", 10.0)

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
    env.positions[0, 1] = torch.tensor([132.0, 100.0, 50.0])
    env.velocities.zero_()
    env.obstacle_xy[:] = torch.tensor([4500.0, 4500.0])
    env.obstacle_radius[:] = 80.0
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
    assert float(env.positions[0, 0, 0]) <= 100.0
    assert float(env.positions[0, 1, 0]) >= 132.0


def test_apf_brakes_max_speed_uav_before_static_obstacle(monkeypatch):
    _set_apf_test_config(monkeypatch)
    monkeypatch.setitem(CONFIG, "apf_peer_influence_m", 200.0)
    monkeypatch.setitem(CONFIG, "apf_obstacle_influence_m", 120.0)
    monkeypatch.setitem(CONFIG, "apf_peer_gain", 1.5)
    monkeypatch.setitem(CONFIG, "apf_obstacle_gain", 1.5)
    monkeypatch.setitem(CONFIG, "apf_emergency_gain", 4.0)
    monkeypatch.setitem(CONFIG, "apf_braking_margin", 1.5)

    obstacle = Obstacle(
        np.asarray([500.0, 500.0]),
        radius=100.0,
        height=100.0,
    )
    uav = UAV(
        0,
        np.asarray([750.0, 500.0, 50.0]),
        np.asarray([-20.0, 0.0, 0.0]),
        float(CONFIG["battery_j"]),
        True,
    )

    min_surface_clearance = float("inf")
    for _ in range(20):
        apply_swarm_motion(
            [uav],
            np.asarray([[-1.0, 0.0, 0.0]]),
            [obstacle],
            dt=1.0,
        )
        surface_clearance = (
            np.linalg.norm(
                uav.position[:2] - obstacle.position
            )
            - obstacle.radius
        )
        min_surface_clearance = min(
            min_surface_clearance,
            float(surface_clearance),
        )

    assert min_surface_clearance >= float(
        CONFIG["obstacle_clearance_m"]
    ) - 1e-6


def test_apf_brakes_head_on_max_speed_peers_before_safety_distance(
    monkeypatch,
):
    _set_apf_test_config(monkeypatch)
    monkeypatch.setitem(CONFIG, "apf_peer_influence_m", 200.0)
    monkeypatch.setitem(CONFIG, "apf_obstacle_influence_m", 120.0)
    monkeypatch.setitem(CONFIG, "apf_peer_gain", 1.5)
    monkeypatch.setitem(CONFIG, "apf_obstacle_gain", 1.5)
    monkeypatch.setitem(CONFIG, "apf_emergency_gain", 4.0)
    monkeypatch.setitem(CONFIG, "apf_braking_margin", 1.5)

    uavs = [
        UAV(
            0,
            np.asarray([250.0, 500.0, 50.0]),
            np.asarray([20.0, 0.0, 0.0]),
            float(CONFIG["battery_j"]),
            True,
        ),
        UAV(
            1,
            np.asarray([750.0, 500.0, 50.0]),
            np.asarray([-20.0, 0.0, 0.0]),
            float(CONFIG["battery_j"]),
            True,
        ),
    ]

    min_distance = float("inf")
    for _ in range(30):
        apply_swarm_motion(
            uavs,
            np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                ]
            ),
            [],
            dt=1.0,
        )
        distance = np.linalg.norm(
            uavs[0].position - uavs[1].position
        )
        min_distance = min(
            min_distance,
            float(distance),
        )

    assert min_distance >= float(
        CONFIG["safety_distance"]
    ) - 1e-6
