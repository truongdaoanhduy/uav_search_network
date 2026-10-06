from __future__ import annotations

import numpy as np
import torch

from config import CONFIG
from uav_marl.envs.gpu import FullGpuUAVBatchEnv
from uav_marl.world.entities import UAV, Obstacle
from uav_marl.world.motion import apply_swarm_motion, segment_intersects_obstacle


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


def test_cpu_apf_accounts_for_altitude_clipping_before_predicting_peer_motion(monkeypatch):
    _set_apf_config(monkeypatch)
    positions = np.array([[518.07788086, 292.40258789, 120.85519409],
                          [516.22949219, 284.47509766, 150.0]])
    velocities = np.array([[7.4861908, -2.2599511, -1.00809634],
                           [2.63695002, 4.33430529, 0.0]])
    actions = np.array([[0.72794116, -0.57554078, 0.91753817],
                        [-0.93766785, -0.9618572, 0.70207584]])
    uavs = [UAV(i, p.copy(), v.copy(), 277200.0, True)
            for i, (p, v) in enumerate(zip(positions, velocities))]
    result = apply_swarm_motion(uavs, actions, [], dt=1.0)
    after = np.stack([uav.position for uav in uavs])
    relative = positions[0] - positions[1]
    delta = (after[0] - positions[0]) - (after[1] - positions[1])
    closest_time = np.clip(-np.dot(relative, delta) / np.dot(delta, delta), 0, 1)
    assert np.linalg.norm(relative + closest_time * delta) >= 30.0 - 1e-3
    assert result['apf_peer_active'].all()
    assert not result['blocked'].any()


def test_tensor_apf_accounts_for_altitude_clipping_before_predicting_peer_motion(monkeypatch):
    _set_apf_config(monkeypatch)
    env = FullGpuUAVBatchEnv(1, device='cpu', seed=44, strict_cuda=False)
    try:
        env.active.zero_()
        env.active[0, :2] = True
        before = torch.tensor([[518.07788086, 292.40258789, 120.85519409],
                               [516.22949219, 284.47509766, 150.0]])
        env.positions[0, :2] = before
        env.velocities.zero_()
        env.velocities[0, :2] = torch.tensor([[7.4861908, -2.2599511, -1.00809634],
                                             [2.63695002, 4.33430529, 0.0]])
        env.obstacle_xy.fill_(2500.0)
        motion = torch.zeros((1, env.num_uavs, 3))
        motion[0, :2] = torch.tensor([[0.72794116, -0.57554078, 0.91753817],
                                     [-0.93766785, -0.9618572, 0.70207584]])
        result = env._motion(motion)
        after = env.positions[0, :2]
        relative = before[0] - before[1]
        delta = (after[0] - before[0]) - (after[1] - before[1])
        closest_time = (-torch.dot(relative, delta) / torch.dot(delta, delta)).clamp(0, 1)
        assert float(torch.linalg.vector_norm(relative + closest_time * delta)) >= 30.0 - 1e-3
        assert bool(result['apf_peer_active'][0, :2].all())
        assert not bool(result['blocked'].any())
    finally:
        env.close()


def test_cpu_apf_catches_obstacle_tunneling_between_lookahead_endpoints(monkeypatch):
    _set_apf_config(monkeypatch)
    monkeypatch.setitem(CONFIG, "obstacle_clearance_m", 0.0)
    uav = UAV(
        id=0,
        position=np.array([0.0, 0.0, 11.5], dtype=np.float64),
        velocity=np.array([6.0, 0.0, -1.8], dtype=np.float64),
        battery_j=float(CONFIG["battery_j"]),
        active=True,
    )
    obstacle = Obstacle(
        position=np.array([0.0, 0.0], dtype=np.float64),
        radius=10.0,
        height=10.0,
    )

    result = apply_swarm_motion(
        [uav],
        np.zeros((1, 3), dtype=np.float64),
        [obstacle],
        dt=1.0,
    )

    assert bool(result["apf_obstacle_emergency"][0])
    assert uav.position[2] >= obstacle.height


def test_tensor_apf_catches_obstacle_tunneling_between_lookahead_endpoints(monkeypatch):
    _set_apf_config(monkeypatch)
    monkeypatch.setitem(CONFIG, "obstacle_clearance_m", 0.0)
    env = FullGpuUAVBatchEnv(1, device="cpu", seed=44, strict_cuda=False)
    try:
        env.active.zero_()
        env.active[0, 0] = True
        env.positions[0, 0] = torch.tensor([0.0, 0.0, 11.5])
        env.velocities.zero_()
        env.velocities[0, 0] = torch.tensor([6.0, 0.0, -1.8])
        env.obstacle_xy.fill_(2500.0)
        env.obstacle_radius.fill_(10.0)
        env.obstacle_height.fill_(10.0)
        env.obstacle_xy[0, 0] = torch.tensor([0.0, 0.0])

        old = env.positions.clone()
        result = env._motion(torch.zeros((1, env.num_uavs, 3)))
        intersects = env._segment_cylinder_blocked(
            old,
            env.positions,
            obstacle_radius=env.obstacle_radius,
            obstacle_height=env.obstacle_height,
        )

        assert bool(result["apf_obstacle_emergency"][0, 0])
        assert not bool(intersects[0, 0])
    finally:
        env.close()


def _two_obstacle_correction_regression_fixture():
    position = np.array(
        [1099.7215576171875, 939.3732299804688, 37.208377838134766],
        dtype=np.float64,
    )
    velocity = np.array(
        [3.9203343391418457, 9.753774642944336, -5.933643817901611],
        dtype=np.float64,
    )
    action = np.array(
        [-0.467210054397583, 0.17470765113830566, 0.952881932258606],
        dtype=np.float64,
    )
    obstacles = [
        Obstacle(
            position=np.array([995.9379272460938, 1054.376953125]),
            radius=110.82035827636719,
            height=30.582183837890625,
        ),
        Obstacle(
            position=np.array([1207.0615234375, 923.1096801757812]),
            radius=76.06255340576172,
            height=62.57680892944336,
        ),
    ]
    return position, velocity, action, obstacles


def test_cpu_apf_correction_does_not_create_new_obstacle_collision(monkeypatch):
    _set_apf_config(monkeypatch)
    position, velocity, action, obstacles = _two_obstacle_correction_regression_fixture()
    uav = UAV(0, position.copy(), velocity.copy(), float(CONFIG["battery_j"]), True)

    old = uav.position.copy()
    result = apply_swarm_motion([uav], action[None, :], obstacles, dt=1.0)

    assert bool(result["apf_obstacle_active"][0])
    assert not segment_intersects_obstacle(
        old,
        uav.position,
        obstacles,
        margin=float(CONFIG["obstacle_clearance_m"]) - 1e-3,
    )


def test_tensor_apf_correction_does_not_create_new_obstacle_collision(monkeypatch):
    _set_apf_config(monkeypatch)
    position, velocity, action, obstacles = _two_obstacle_correction_regression_fixture()
    env = FullGpuUAVBatchEnv(1, device="cpu", seed=44, strict_cuda=False)
    try:
        env.active.zero_()
        env.active[0, 0] = True
        env.positions[0, 0] = torch.tensor(position, dtype=torch.float32)
        env.velocities.zero_()
        env.velocities[0, 0] = torch.tensor(velocity, dtype=torch.float32)
        env.obstacle_xy.fill_(2500.0)
        env.obstacle_radius.fill_(10.0)
        env.obstacle_height.fill_(10.0)
        for index, obstacle in enumerate(obstacles):
            env.obstacle_xy[0, index] = torch.tensor(obstacle.position, dtype=torch.float32)
            env.obstacle_radius[0, index] = obstacle.radius
            env.obstacle_height[0, index] = obstacle.height

        old = env.positions.clone()
        motion = torch.zeros((1, env.num_uavs, 3), dtype=torch.float32)
        motion[0, 0] = torch.tensor(action, dtype=torch.float32)
        result = env._motion(motion)
        intersects = env._segment_cylinder_blocked(
            old,
            env.positions,
            obstacle_radius=env.obstacle_radius + float(CONFIG["obstacle_clearance_m"]) - 1e-3,
            obstacle_height=env.obstacle_height + float(CONFIG["obstacle_clearance_m"]) - 1e-3,
        )

        assert bool(result["apf_obstacle_active"][0, 0])
        assert not bool(intersects[0, 0])
    finally:
        env.close()


def _opposing_obstacle_emergency_regression_fixture():
    position = np.array(
        [2680.84765625, 1677.55908203125, 7.905452251434326],
        dtype=np.float64,
    )
    velocity = np.array(
        [2.8924901485443115, 6.3287153244018555, -2.48482084274292],
        dtype=np.float64,
    )
    action = np.array(
        [0.7058200836181641, -0.5517627000808716, 0.08406412601470947],
        dtype=np.float64,
    )
    obstacles = [
        Obstacle(
            position=np.array([2802.6318359375, 1611.4656982421875]),
            radius=108.17269897460938,
            height=98.94354248046875,
        ),
        Obstacle(
            position=np.array([2562.670654296875, 1731.8486328125]),
            radius=98.65826416015625,
            height=103.9888916015625,
        ),
    ]
    return position, velocity, action, obstacles


def test_cpu_apf_keeps_safe_motion_when_emergency_directions_oppose(monkeypatch):
    _set_apf_config(monkeypatch)
    position, velocity, action, obstacles = _opposing_obstacle_emergency_regression_fixture()
    uav = UAV(0, position.copy(), velocity.copy(), float(CONFIG["battery_j"]), True)

    old = uav.position.copy()
    result = apply_swarm_motion([uav], action[None, :], obstacles, dt=1.0)

    assert not segment_intersects_obstacle(
        old,
        uav.position,
        obstacles,
        margin=float(CONFIG["obstacle_clearance_m"]) - 1e-3,
    )
    assert np.linalg.norm(result["realized_acceleration_mps2"][0]) <= float(CONFIG["max_accel"]) + 1e-6
    assert not bool(result["blocked"][0])


def test_tensor_apf_keeps_safe_motion_when_emergency_directions_oppose(monkeypatch):
    _set_apf_config(monkeypatch)
    position, velocity, action, obstacles = _opposing_obstacle_emergency_regression_fixture()
    env = FullGpuUAVBatchEnv(1, device="cpu", seed=44, strict_cuda=False)
    try:
        env.active.zero_()
        env.active[0, 0] = True
        env.positions[0, 0] = torch.tensor(position, dtype=torch.float32)
        env.velocities.zero_()
        env.velocities[0, 0] = torch.tensor(velocity, dtype=torch.float32)
        env.obstacle_xy.fill_(2500.0)
        env.obstacle_radius.fill_(10.0)
        env.obstacle_height.fill_(10.0)
        for index, obstacle in enumerate(obstacles):
            env.obstacle_xy[0, index] = torch.tensor(obstacle.position, dtype=torch.float32)
            env.obstacle_radius[0, index] = obstacle.radius
            env.obstacle_height[0, index] = obstacle.height

        old = env.positions.clone()
        motion = torch.zeros((1, env.num_uavs, 3), dtype=torch.float32)
        motion[0, 0] = torch.tensor(action, dtype=torch.float32)
        result = env._motion(motion)
        intersects = env._segment_cylinder_blocked(
            old,
            env.positions,
            obstacle_radius=env.obstacle_radius + float(CONFIG["obstacle_clearance_m"]) - 1e-3,
            obstacle_height=env.obstacle_height + float(CONFIG["obstacle_clearance_m"]) - 1e-3,
        )

        assert not bool(intersects[0, 0])
        assert float(torch.linalg.vector_norm(result["realized_accel"][0, 0])) <= float(CONFIG["max_accel"]) + 1e-5
        assert not bool(result["blocked"][0, 0])
    finally:
        env.close()


def test_cpu_obstacle_control_step_margin_is_emergency(monkeypatch):
    _set_apf_config(monkeypatch)
    monkeypatch.setitem(CONFIG, "obstacle_clearance_m", 0.0)
    uav = UAV(
        0,
        np.array([142.0, 100.0, 5.0], dtype=np.float64),
        np.array([-10.0, 0.0, 0.0], dtype=np.float64),
        float(CONFIG["battery_j"]),
        True,
    )
    obstacle = Obstacle(np.array([100.0, 100.0]), 10.0, 10.0)

    result = apply_swarm_motion(
        [uav],
        np.zeros((1, 3), dtype=np.float64),
        [obstacle],
        dt=1.0,
    )

    assert bool(result["apf_obstacle_emergency"][0])


def test_tensor_obstacle_control_step_margin_is_emergency(monkeypatch):
    _set_apf_config(monkeypatch)
    monkeypatch.setitem(CONFIG, "obstacle_clearance_m", 0.0)
    env = FullGpuUAVBatchEnv(1, device="cpu", seed=44, strict_cuda=False)
    try:
        env.active.zero_()
        env.active[0, 0] = True
        env.positions[0, 0] = torch.tensor([142.0, 100.0, 5.0])
        env.velocities.zero_()
        env.velocities[0, 0] = torch.tensor([-10.0, 0.0, 0.0])
        env.obstacle_xy.fill_(2500.0)
        env.obstacle_radius.fill_(10.0)
        env.obstacle_height.fill_(10.0)
        env.obstacle_xy[0, 0] = torch.tensor([100.0, 100.0])

        result = env._motion(torch.zeros((1, env.num_uavs, 3)))

        assert bool(result["apf_obstacle_emergency"][0, 0])
    finally:
        env.close()


def test_cpu_peer_control_step_margin_is_emergency(monkeypatch):
    _set_apf_config(monkeypatch)
    uavs = [
        UAV(0, np.array([100.0, 100.0, 50.0]), np.array([10.0, 0.0, 0.0]), float(CONFIG["battery_j"]), True),
        UAV(1, np.array([195.0, 100.0, 50.0]), np.array([-10.0, 0.0, 0.0]), float(CONFIG["battery_j"]), True),
    ]

    result = apply_swarm_motion(
        uavs,
        np.zeros((2, 3), dtype=np.float64),
        [],
        dt=1.0,
    )

    assert bool(result["apf_peer_emergency"].all())


def test_tensor_peer_control_step_margin_is_emergency(monkeypatch):
    _set_apf_config(monkeypatch)
    env = FullGpuUAVBatchEnv(1, device="cpu", seed=44, strict_cuda=False)
    try:
        env.active.zero_()
        env.active[0, :2] = True
        env.positions[0, 0] = torch.tensor([100.0, 100.0, 50.0])
        env.positions[0, 1] = torch.tensor([195.0, 100.0, 50.0])
        env.velocities.zero_()
        env.velocities[0, 0] = torch.tensor([10.0, 0.0, 0.0])
        env.velocities[0, 1] = torch.tensor([-10.0, 0.0, 0.0])
        env.obstacle_xy.fill_(2500.0)

        result = env._motion(torch.zeros((1, env.num_uavs, 3)))

        assert bool(result["apf_peer_emergency"][0, :2].all())
    finally:
        env.close()


def _fallback_peer_conflict_fixture():
    position, velocity, action, obstacles = _opposing_obstacle_emergency_regression_fixture()
    positions = np.array([position, [2702.1959846752284, 1710.0303951405206, 6.029592326287824]])
    velocities = np.array([velocity, [-3.4222400930935732, -3.276138919572057, -1.9299507300833851]])
    actions = np.array([action, [-0.5226757382415363, 0.7987210328552949, -0.6483089039756245]])
    return positions, velocities, actions, obstacles


def _assert_safe_fallback_paths(old_positions, new_positions, obstacles):
    relative = old_positions[0] - old_positions[1]
    relative_move = (new_positions[0] - old_positions[0]) - (new_positions[1] - old_positions[1])
    t = np.clip(-np.dot(relative, relative_move) / max(np.dot(relative_move, relative_move), 1e-12), 0.0, 1.0)
    distance = np.linalg.norm(relative + t * relative_move)
    assert distance >= float(CONFIG["safety_distance"]) - 1e-3
    for start, end in zip(old_positions, new_positions):
        assert not segment_intersects_obstacle(start, end, obstacles, margin=float(CONFIG["obstacle_clearance_m"]) - 1e-3)


def test_cpu_obstacle_fallback_preserves_peer_separation(monkeypatch):
    _set_apf_config(monkeypatch)
    positions, velocities, actions, obstacles = _fallback_peer_conflict_fixture()
    uavs = [UAV(i, positions[i].copy(), velocities[i].copy(), float(CONFIG["battery_j"]), True) for i in range(2)]

    result = apply_swarm_motion(uavs, actions, obstacles, dt=1.0)

    _assert_safe_fallback_paths(positions, np.stack([uav.position for uav in uavs]), obstacles)
    assert np.max(np.linalg.norm(result["realized_acceleration_mps2"], axis=-1)) <= float(CONFIG["max_accel"]) + 1e-6
    assert not result["blocked"].any()


def test_tensor_obstacle_fallback_preserves_peer_separation(monkeypatch):
    _set_apf_config(monkeypatch)
    positions, velocities, actions, obstacles = _fallback_peer_conflict_fixture()
    env = FullGpuUAVBatchEnv(1, device="cpu", seed=44, strict_cuda=False)
    try:
        env.active.zero_()
        env.active[0, :2] = True
        env.positions[0, :2] = torch.tensor(positions, dtype=torch.float32)
        env.velocities.zero_()
        env.velocities[0, :2] = torch.tensor(velocities, dtype=torch.float32)
        env.obstacle_xy.fill_(2500.0)
        env.obstacle_radius.fill_(10.0)
        env.obstacle_height.fill_(10.0)
        for index, obstacle in enumerate(obstacles):
            env.obstacle_xy[0, index] = torch.tensor(obstacle.position, dtype=torch.float32)
            env.obstacle_radius[0, index] = obstacle.radius
            env.obstacle_height[0, index] = obstacle.height
        motion = torch.zeros((1, env.num_uavs, 3))
        motion[0, :2] = torch.tensor(actions, dtype=torch.float32)

        result = env._motion(motion)

        _assert_safe_fallback_paths(positions, env.positions[0, :2].numpy(), obstacles)
        assert float(torch.linalg.vector_norm(result["realized_accel"][0, :2], dim=-1).max()) <= float(CONFIG["max_accel"]) + 1e-4
        assert not result["blocked"].any()
    finally:
        env.close()
