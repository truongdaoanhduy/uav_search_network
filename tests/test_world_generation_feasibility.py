import numpy as np
import torch

from config import CONFIG
from uav_marl.envs.gpu import FullGpuUAVBatchEnv
from uav_marl.world.generation import create_world


def test_cpu_world_targets_keep_obstacle_safety_clearance():
    required = float(CONFIG["obstacle_clearance_m"])

    for seed in range(20):
        _, _, targets, obstacles = create_world(seed)
        for target in targets:
            for obstacle in obstacles:
                surface_gap = (
                    np.linalg.norm(target.position - obstacle.position)
                    - float(obstacle.radius)
                )
                assert surface_gap >= required


def test_tensor_world_targets_keep_obstacle_safety_clearance():
    env = FullGpuUAVBatchEnv(
        num_envs=32,
        device="cpu",
        seed=44,
        strict_cuda=False,
    )
    try:
        env.reset()
        distances = torch.linalg.vector_norm(
            env.target_xy[:, :, None, :] - env.obstacle_xy[:, None, :, :],
            dim=-1,
        )
        surface_gap = distances - env.obstacle_radius[:, None, :]
        assert float(surface_gap.min()) >= float(CONFIG["obstacle_clearance_m"])
    finally:
        env.close()


def test_cpu_world_targets_have_direct_safe_confirmation_viewpoint():
    from uav_marl.world.entities import UAV
    from uav_marl.world.generation import point_inside_obstacle
    from uav_marl.world.sensing import camera_target_has_los

    altitude = float(CONFIG["fine_altitude"])
    clearance = float(CONFIG["obstacle_clearance_m"])

    for seed in range(20):
        _, _, targets, obstacles = create_world(seed)
        for target in targets:
            position = np.array(
                [target.position[0], target.position[1], altitude],
                dtype=np.float64,
            )
            uav = UAV(
                id=0,
                position=position,
                velocity=np.zeros(3, dtype=np.float64),
                battery_j=float(CONFIG["battery_j"]),
                active=True,
            )
            assert not point_inside_obstacle(
                position,
                obstacles,
                margin=clearance,
            )
            assert camera_target_has_los(uav, target, obstacles)


def test_cpu_obstacles_leave_room_for_configured_clearance():
    required_gap = (
        2.0 * float(CONFIG["obstacle_clearance_m"])
        + float(CONFIG["safety_distance"])
    )

    for seed in range(20):
        _, _, _, obstacles = create_world(seed)
        for left in range(len(obstacles)):
            for right in range(left + 1, len(obstacles)):
                center_distance = np.linalg.norm(
                    obstacles[left].position - obstacles[right].position
                )
                surface_gap = (
                    center_distance
                    - float(obstacles[left].radius)
                    - float(obstacles[right].radius)
                )
                assert surface_gap >= required_gap


def test_tensor_obstacles_leave_room_for_configured_clearance():
    env = FullGpuUAVBatchEnv(32, device="cpu", seed=44, strict_cuda=False)
    try:
        env.reset()
        center_distance = torch.linalg.vector_norm(
            env.obstacle_xy[:, :, None, :] - env.obstacle_xy[:, None, :, :],
            dim=-1,
        )
        surface_gap = (
            center_distance
            - env.obstacle_radius[:, :, None]
            - env.obstacle_radius[:, None, :]
        )
        diagonal = torch.eye(env.num_obstacles, dtype=torch.bool)[None]
        surface_gap = surface_gap.masked_fill(diagonal, float("inf"))
        required_gap = (
            2.0 * float(CONFIG["obstacle_clearance_m"])
            + float(CONFIG["safety_distance"])
        )
        assert float(surface_gap.min()) >= required_gap
    finally:
        env.close()


def test_cpu_obstacle_clearance_stays_inside_map_bounds():
    clearance = float(CONFIG["obstacle_clearance_m"])
    map_size = float(CONFIG["map_size"])

    for seed in range(20):
        _, _, _, obstacles = create_world(seed)
        for obstacle in obstacles:
            extent = float(obstacle.radius) + clearance
            assert extent <= obstacle.position[0] <= map_size - extent
            assert extent <= obstacle.position[1] <= map_size - extent


def test_tensor_obstacle_clearance_stays_inside_map_bounds():
    env = FullGpuUAVBatchEnv(32, device="cpu", seed=44, strict_cuda=False)
    try:
        env.reset()
        clearance = float(CONFIG["obstacle_clearance_m"])
        extent = env.obstacle_radius + clearance
        assert bool((env.obstacle_xy[..., 0] >= extent).all())
        assert bool((env.obstacle_xy[..., 0] <= env.map_size - extent).all())
        assert bool((env.obstacle_xy[..., 1] >= extent).all())
        assert bool((env.obstacle_xy[..., 1] <= env.map_size - extent).all())
    finally:
        env.close()
