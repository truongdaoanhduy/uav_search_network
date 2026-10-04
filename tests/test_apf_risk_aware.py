from __future__ import annotations

import numpy as np

from uav_marl.world.apf import apf_repulsion_numpy


def _empty_obstacles():
    return (
        np.zeros((0, 2), dtype=np.float64),
        np.zeros((0,), dtype=np.float64),
        np.zeros((0,), dtype=np.float64),
    )


def _kwargs():
    return dict(
        max_accel=2.0,
        safety_distance=30.0,
        obstacle_clearance=30.0,
        soft_gain=1.5,
        lookahead_s=3.0,
        braking_margin=1.5,
        enabled=True,
    )


def test_risk_aware_apf_does_not_push_same_direction_peers():
    obstacle_xy, obstacle_radius, obstacle_height = _empty_obstacles()
    positions = np.array([[0.0, 0.0, 50.0], [80.0, 0.0, 50.0]])
    velocities = np.array([[15.0, 0.0, 0.0], [15.0, 0.0, 0.0]])
    result = apf_repulsion_numpy(
        positions,
        velocities,
        np.array([True, True]),
        obstacle_xy,
        obstacle_radius,
        obstacle_height,
        nominal_velocities=velocities,
        **_kwargs(),
    )
    assert not result["peer_active"].any()
    assert np.allclose(result["peer_acceleration_mps2"], 0.0)


def test_risk_aware_apf_uses_nominal_action_to_detect_new_head_on_conflict():
    obstacle_xy, obstacle_radius, obstacle_height = _empty_obstacles()
    positions = np.array([[0.0, 0.0, 50.0], [38.0, 0.0, 50.0]])
    old_velocities = np.zeros((2, 3), dtype=np.float64)
    nominal_velocities = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    result = apf_repulsion_numpy(
        positions,
        old_velocities,
        np.array([True, True]),
        obstacle_xy,
        obstacle_radius,
        obstacle_height,
        nominal_velocities=nominal_velocities,
        **_kwargs(),
    )
    assert result["peer_active"].all()
    assert result["peer_emergency"].all()
    assert result["emergency_acceleration_mps2"][0, 0] < 0.0
    assert result["emergency_acceleration_mps2"][1, 0] > 0.0


def test_risk_aware_apf_does_not_interfere_with_safe_tangential_obstacle_pass():
    positions = np.array([[0.0, 160.0, 50.0]])
    velocities = np.array([[10.0, 0.0, 0.0]])
    result = apf_repulsion_numpy(
        positions,
        velocities,
        np.array([True]),
        np.array([[100.0, 0.0]]),
        np.array([100.0]),
        np.array([100.0]),
        nominal_velocities=velocities,
        **_kwargs(),
    )
    assert not result["obstacle_active"][0]
    assert np.allclose(result["obstacle_acceleration_mps2"][0], 0.0)


def test_obstacle_soft_avoidance_contains_tangential_component():
    positions = np.array([[0.0, 0.0, 50.0]])
    velocities = np.array([[20.0, 0.0, 0.0]])
    result = apf_repulsion_numpy(
        positions,
        velocities,
        np.array([True]),
        np.array([[180.0, 0.0]]),
        np.array([30.0]),
        np.array([100.0]),
        nominal_velocities=velocities,
        **_kwargs(),
    )
    correction = result["obstacle_acceleration_mps2"][0]
    assert result["obstacle_active"][0]
    assert not result["obstacle_emergency"][0]
    assert abs(correction[1]) > 1e-6


def test_risk_aware_numpy_and_torch_match():
    import torch

    from uav_marl.world.apf import apf_repulsion_torch

    positions = np.array(
        [[0.0, 0.0, 50.0], [100.0, 0.0, 50.0]],
        dtype=np.float32,
    )
    velocities = np.array(
        [[12.0, 0.0, 0.0], [-8.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    nominal = np.array(
        [[14.0, 0.0, 0.0], [-10.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    active = np.array([True, True])
    obstacle_xy = np.array([[60.0, 50.0]], dtype=np.float32)
    obstacle_radius = np.array([20.0], dtype=np.float32)
    obstacle_height = np.array([80.0], dtype=np.float32)

    np_result = apf_repulsion_numpy(
        positions,
        velocities,
        active,
        obstacle_xy,
        obstacle_radius,
        obstacle_height,
        nominal_velocities=nominal,
        **_kwargs(),
    )
    torch_result = apf_repulsion_torch(
        torch.tensor(positions[None]),
        torch.tensor(velocities[None]),
        torch.tensor(active[None]),
        torch.tensor(obstacle_xy[None]),
        torch.tensor(obstacle_radius[None]),
        torch.tensor(obstacle_height[None]),
        nominal_velocities=torch.tensor(nominal[None]),
        **_kwargs(),
    )

    for key in (
        "soft_acceleration_mps2",
        "emergency_acceleration_mps2",
        "peer_acceleration_mps2",
        "obstacle_acceleration_mps2",
    ):
        assert np.allclose(
            np_result[key],
            torch_result[key][0].numpy(),
            atol=1e-5,
        )
    for key in (
        "active",
        "peer_active",
        "obstacle_active",
        "emergency",
        "peer_emergency",
        "obstacle_emergency",
    ):
        assert np.array_equal(
            np_result[key],
            torch_result[key][0].numpy(),
        )


def test_risk_aware_apf_can_be_disabled():
    obstacle_xy, obstacle_radius, obstacle_height = _empty_obstacles()
    result = apf_repulsion_numpy(
        np.array([[0.0, 0.0, 50.0], [35.0, 0.0, 50.0]]),
        np.array([[20.0, 0.0, 0.0], [-20.0, 0.0, 0.0]]),
        np.array([True, True]),
        obstacle_xy,
        obstacle_radius,
        obstacle_height,
        nominal_velocities=np.array(
            [[20.0, 0.0, 0.0], [-20.0, 0.0, 0.0]]
        ),
        **{**_kwargs(), "enabled": False},
    )
    assert not result["active"].any()
    assert not result["emergency"].any()
    assert np.allclose(result["soft_acceleration_mps2"], 0.0)
    assert np.allclose(result["emergency_acceleration_mps2"], 0.0)


def test_production_motion_preserves_peer_safety_under_head_on_max_speed(monkeypatch):
    from config import CONFIG
    from uav_marl.world.entities import UAV
    from uav_marl.world.motion import apply_swarm_motion

    monkeypatch.setitem(CONFIG, "apf_enabled", True)
    monkeypatch.setitem(CONFIG, "apf_soft_gain", 1.5)
    monkeypatch.setitem(CONFIG, "apf_lookahead_s", 3.0)
    monkeypatch.setitem(CONFIG, "apf_braking_margin", 1.5)
    monkeypatch.setitem(CONFIG, "safety_distance", 30.0)
    monkeypatch.setitem(CONFIG, "max_accel", 2.0)
    monkeypatch.setitem(CONFIG, "max_speed", 20.0)

    uavs = [
        UAV(
            0,
            np.array([250.0, 500.0, 50.0]),
            np.array([20.0, 0.0, 0.0]),
            float(CONFIG["battery_j"]),
            True,
        ),
        UAV(
            1,
            np.array([750.0, 500.0, 50.0]),
            np.array([-20.0, 0.0, 0.0]),
            float(CONFIG["battery_j"]),
            True,
        ),
    ]

    minimum_distance = float("inf")
    emergency_count = 0
    for _ in range(30):
        result = apply_swarm_motion(
            uavs,
            np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
            [],
            dt=1.0,
        )
        minimum_distance = min(
            minimum_distance,
            float(np.linalg.norm(uavs[0].position - uavs[1].position)),
        )
        emergency_count += int(np.count_nonzero(result["apf_emergency"]))

    assert minimum_distance >= 30.0
    assert emergency_count > 0


def test_production_motion_preserves_obstacle_clearance_at_max_speed(monkeypatch):
    from config import CONFIG
    from uav_marl.world.entities import Obstacle, UAV
    from uav_marl.world.motion import apply_swarm_motion

    monkeypatch.setitem(CONFIG, "apf_enabled", True)
    monkeypatch.setitem(CONFIG, "apf_soft_gain", 1.5)
    monkeypatch.setitem(CONFIG, "apf_lookahead_s", 3.0)
    monkeypatch.setitem(CONFIG, "apf_braking_margin", 1.5)
    monkeypatch.setitem(CONFIG, "obstacle_clearance_m", 30.0)
    monkeypatch.setitem(CONFIG, "max_accel", 2.0)
    monkeypatch.setitem(CONFIG, "max_speed", 20.0)

    obstacle = Obstacle(
        np.array([500.0, 500.0]),
        radius=100.0,
        height=100.0,
    )
    uav = UAV(
        0,
        np.array([750.0, 500.0, 50.0]),
        np.array([-20.0, 0.0, 0.0]),
        float(CONFIG["battery_j"]),
        True,
    )

    minimum_surface_clearance = float("inf")
    lateral_motion_seen = False
    for _ in range(30):
        result = apply_swarm_motion(
            [uav],
            np.array([[-1.0, 0.0, 0.0]]),
            [obstacle],
            dt=1.0,
        )
        surface_clearance = (
            np.linalg.norm(uav.position[:2] - obstacle.position)
            - obstacle.radius
        )
        minimum_surface_clearance = min(
            minimum_surface_clearance,
            float(surface_clearance),
        )
        lateral_motion_seen |= abs(float(uav.velocity[1])) > 1e-6
        assert np.all(np.isfinite(result["apf_acceleration_mps2"]))

    assert minimum_surface_clearance >= 30.0
    assert lateral_motion_seen
