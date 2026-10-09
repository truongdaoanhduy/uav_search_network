"""Production source migrated from frozen fix_test_gpu.ipynb.

Do not import the notebook at runtime. The notebook is retained only as a
reference/regression artifact. Production changes belong in Python modules.
Migrated from notebook cells 16..20.
"""

from .generation import *  # noqa: F401,F403
from .apf import apf_repulsion_numpy, _peer_path_conflicts_numpy

# --- frozen notebook cell 16 ---
def segment_intersects_obstacle(
    start_position,
    end_position,
    obstacles,
    margin=0.0,
):
    start = np.asarray(
        start_position,
        dtype=np.float64,
    )

    end = np.asarray(
        end_position,
        dtype=np.float64,
    )

    if start.shape != (3,) or end.shape != (3,):
        raise ValueError(
            "start_position and end_position must have shape (3,)"
        )

    if (
        not np.all(np.isfinite(start))
        or not np.all(np.isfinite(end))
    ):
        raise ValueError(
            "start_position and end_position "
            "must contain only finite values"
        )

    margin = float(margin)

    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError(
            "margin must be finite and >= 0"
        )

    direction = end - start
    eps = 1e-12

    for obs in obstacles:

        center = np.asarray(obs.position,dtype=np.float64,)

        radius = float(obs.radius) + float(margin)
        height = float(obs.height) + float(margin)

        dz = float(direction[2])

        if abs(dz) <= eps:

            if not (
                0.0 <= start[2] <= height
            ):
                continue

            z_enter = 0.0
            z_exit = 1.0

        else:

            t_ground = (0.0 - start[2]) / dz
            t_top = (height - start[2]) / dz

            z_enter = max(0.0,min(t_ground, t_top))
            z_exit = min(1.0,max(t_ground, t_top))

            if z_enter > z_exit:
                continue

        relative_xy = start[:2] - center[:2]
        direction_xy = direction[:2]

        a = direction_xy@direction_xy
        c = (relative_xy@relative_xy)- radius * radius

        if a <= eps:

            if c > 0:
                continue

            xy_enter = 0
            xy_exit = 1

        else:

            b = 2 * relative_xy@direction_xy
            discriminant = (b * b- 4.0 * a * c)

            if discriminant < 0.0:
                continue

            root = np.sqrt(max(discriminant, 0.0))

            t1 = (-b - root) / (2.0 * a)
            t2 = (-b + root) / (2.0 * a)

            xy_enter = max(0.0,min(t1, t2))
            xy_exit = min(1.0,max(t1, t2))

            if xy_enter > xy_exit:
                continue

        enter = max(z_enter,xy_enter)
        exit_ = min(z_exit,xy_exit)

        if enter <= exit_:
            return True

    return False


# --- frozen notebook cell 17 ---
def project_motion_action_np(action):
    """Project one or more 3-D motion actions onto the unit ball."""
    action = np.asarray(
        action,
        dtype=np.float64,
    )

    if action.ndim < 1 or action.shape[-1] != 3:
        raise ValueError(
            "motion action must have trailing shape (3,)"
        )

    if not np.all(np.isfinite(action)):
        raise ValueError(
            "motion action must contain only finite values"
        )

    clipped = np.clip(
        action,
        -1.0,
        1.0,
    )
    norms = np.linalg.norm(
        clipped,
        axis=-1,
        keepdims=True,
    )
    safe_norms = np.maximum(
        norms,
        1.0,
    )

    return (
        clipped / safe_norms
    ).astype(
        np.float64,
        copy=False,
    )


# --- frozen notebook cell 18 ---
def compute_motion_candidate(
    uav,
    action,
    dt=None,
):
    if dt is None:
        dt = CONFIG["dt"]

    dt = float(dt)

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(
            "dt must be finite and > 0"
        )

    action = np.asarray(
        action,
        dtype=np.float64,
    )

    if action.shape != (3,):
        raise ValueError("action must have shape (3,)")

    if not np.all(np.isfinite(action)):
        raise ValueError("action must contain only finite values")

    if not uav.active:
        return (
            uav.position.copy(),
            np.zeros(3, dtype=np.float64),
            False,
            False,
            False,
        )

    action = project_motion_action_np(
        action
    )

    acceleration = action * CONFIG["max_accel"]

    candidate_velocity = (
        uav.velocity
        + acceleration * dt
    )

    speed = np.linalg.norm(candidate_velocity)

    if speed > CONFIG["max_speed"]:
        candidate_velocity = (candidate_velocity/ speed* CONFIG["max_speed"])

    old_position = uav.position.copy()

    raw_candidate_position = old_position+ candidate_velocity * dt

    candidate_position = raw_candidate_position.copy()

    candidate_position[0] = np.clip(
        candidate_position[0],
        0.0,
        CONFIG["map_size"],
    )

    candidate_position[1] = np.clip(
        candidate_position[1],
        0.0,
        CONFIG["map_size"],
    )

    candidate_position[2] = np.clip(
        candidate_position[2],
        CONFIG["altitude_min"],
        CONFIG["altitude_max"],
    )

    horizontal_boundary_clipped = not np.allclose(
        raw_candidate_position[:2],
        candidate_position[:2],
    )
    altitude_clipped = not np.isclose(
        raw_candidate_position[2],
        candidate_position[2],
    )
    boundary_clipped = bool(
        horizontal_boundary_clipped
        or altitude_clipped
    )

    actual_velocity = (
        candidate_position
        - old_position
    ) / dt

    return (
        candidate_position,
        actual_velocity,
        boundary_clipped,
        bool(horizontal_boundary_clipped),
        bool(altitude_clipped),
    )


# --- frozen notebook cell 20 ---
def apply_swarm_motion(
    uavs,
    actions,
    obstacles=None,
    dt=None,
):
    """Advance the swarm with an APF safety layer instead of hard blocking.

    The MARL action remains the task-directed acceleration command. A bounded,
    predictive Artificial Potential Field adds repulsive acceleration for
    nearby peers and finite-cylinder obstacles, after which the combined
    acceleration is projected back to the physical max-acceleration ball.

    Legacy blocked outputs are retained as all-false compatibility fields;
    no motion is cancelled by the collision-avoidance layer anymore.
    """
    if obstacles is None:
        obstacles = []

    if dt is None:
        dt = CONFIG["dt"]
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and > 0")

    num_uavs = len(uavs)
    if num_uavs == 0:
        raise ValueError("uavs must not be empty")

    actions = np.asarray(actions, dtype=np.float64)
    if actions.shape != (num_uavs, 3):
        raise ValueError(
            f"actions must have shape ({num_uavs}, 3)"
        )
    if not np.all(np.isfinite(actions)):
        raise ValueError(
            "actions must contain only finite values"
        )

    old_positions = np.stack(
        [uav.position.copy() for uav in uavs]
    )
    old_velocities = np.stack(
        [uav.velocity.copy() for uav in uavs]
    )
    active = np.asarray(
        [bool(uav.active) for uav in uavs],
        dtype=bool,
    )

    if obstacles:
        obstacle_xy = np.stack(
            [
                np.asarray(
                    obstacle.position,
                    dtype=np.float64,
                )
                for obstacle in obstacles
            ],
            axis=0,
        )
        obstacle_radius = np.asarray(
            [float(obstacle.radius) for obstacle in obstacles],
            dtype=np.float64,
        )
        obstacle_height = np.asarray(
            [float(obstacle.height) for obstacle in obstacles],
            dtype=np.float64,
        )
    else:
        obstacle_xy = np.zeros((0, 2), dtype=np.float64)
        obstacle_radius = np.zeros((0,), dtype=np.float64)
        obstacle_height = np.zeros((0,), dtype=np.float64)

    projected_actions = project_motion_action_np(actions)
    max_accel = float(CONFIG["max_accel"])
    max_speed = float(CONFIG["max_speed"])
    nominal_acceleration = projected_actions * max_accel
    nominal_velocities = (
        old_velocities + nominal_acceleration * dt
    )
    nominal_speed = np.linalg.norm(
        nominal_velocities,
        axis=-1,
        keepdims=True,
    )
    nominal_velocities = nominal_velocities * np.minimum(
        1.0,
        max_speed / np.maximum(nominal_speed, 1e-12),
    )

    # Quantize APF inputs to float32 so the CPU reference uses the same
    # safety-field state precision as the tensor/CUDA training environment.
    # The physical CPU integrator remains float64.
    # Predict the path the boundary-limited motion can actually execute.
    # An outward velocity at the ceiling/map edge must not hide a peer conflict.
    nominal_positions = old_positions + nominal_velocities * dt
    bounded_positions = np.clip(
        nominal_positions,
        [0.0, 0.0, float(CONFIG["altitude_min"])],
        [float(CONFIG["map_size"]), float(CONFIG["map_size"]), float(CONFIG["altitude_max"])],
    )
    nominal_velocities = np.where(
        nominal_positions != bounded_positions,
        (bounded_positions - old_positions) / dt,
        nominal_velocities,
    )
    apf = apf_repulsion_numpy(
        old_positions.astype(np.float32),
        old_velocities.astype(np.float32),
        active,
        obstacle_xy.astype(np.float32),
        obstacle_radius.astype(np.float32),
        obstacle_height.astype(np.float32),
        nominal_velocities=nominal_velocities.astype(np.float32),
        max_accel=max_accel,
        safety_distance=float(CONFIG["safety_distance"]),
        obstacle_clearance=float(
            CONFIG["obstacle_clearance_m"]
        ),
        soft_gain=float(CONFIG.get("apf_soft_gain", 1.5)),
        lookahead_s=float(CONFIG.get("apf_lookahead_s", 3.0)),
        braking_margin=float(
            CONFIG.get("apf_braking_margin", 1.5)
        ),
        control_dt=dt,
        enabled=bool(CONFIG.get("apf_enabled", True)),
        boundary_lower=[0.0, 0.0, float(CONFIG["altitude_min"])],
        boundary_upper=[
            float(CONFIG["map_size"]),
            float(CONFIG["map_size"]),
            float(CONFIG["altitude_max"]),
        ],
    )

    soft_desired_acceleration = (
        nominal_acceleration
        + apf["soft_acceleration_mps2"]
    )
    desired_acceleration = np.where(
        apf["emergency"][:, None],
        apf["emergency_acceleration_mps2"],
        soft_desired_acceleration,
    )
    combined_actions = project_motion_action_np(
        desired_acceleration / max(max_accel, 1e-12)
    )
    applied_acceleration = combined_actions * max_accel
    apf_correction_acceleration = (
        applied_acceleration - nominal_acceleration
    )

    candidate_positions = []
    candidate_velocities = []
    boundary_clipped = np.zeros(
        num_uavs,
        dtype=bool,
    )
    horizontal_boundary_clipped = np.zeros(
        num_uavs,
        dtype=bool,
    )
    altitude_clipped = np.zeros(
        num_uavs,
        dtype=bool,
    )

    for i, (uav, action) in enumerate(
        zip(uavs, combined_actions)
    ):
        (
            position,
            velocity,
            clipped,
            horizontal_clipped,
            vertical_clipped,
        ) = compute_motion_candidate(
            uav,
            action,
            dt=dt,
        )
        candidate_positions.append(position)
        candidate_velocities.append(velocity)
        boundary_clipped[i] = clipped
        horizontal_boundary_clipped[i] = (
            horizontal_clipped
        )
        altitude_clipped[i] = vertical_clipped

    candidate_positions = np.stack(candidate_positions)
    candidate_velocities = np.stack(candidate_velocities)

    if obstacles and bool(CONFIG.get("apf_enabled", True)):
        clearance = float(CONFIG["obstacle_clearance_m"])
        needs_refinement = np.asarray(
            [
                active[i]
                and segment_intersects_obstacle(
                    old_positions[i],
                    candidate_positions[i],
                    obstacles,
                    margin=clearance,
                )
                for i in range(num_uavs)
            ],
            dtype=bool,
        )
        if np.any(needs_refinement):
            refined_apf = apf_repulsion_numpy(
                old_positions.astype(np.float32),
                old_velocities.astype(np.float32),
                active,
                obstacle_xy.astype(np.float32),
                obstacle_radius.astype(np.float32),
                obstacle_height.astype(np.float32),
                nominal_velocities=candidate_velocities.astype(np.float32),
                max_accel=max_accel,
                safety_distance=float(CONFIG["safety_distance"]),
                obstacle_clearance=clearance,
                soft_gain=float(CONFIG.get("apf_soft_gain", 1.5)),
                lookahead_s=float(CONFIG.get("apf_lookahead_s", 3.0)),
                braking_margin=float(CONFIG.get("apf_braking_margin", 1.5)),
                control_dt=dt,
                enabled=True,
                boundary_lower=[0.0, 0.0, float(CONFIG["altitude_min"])],
                boundary_upper=[
                    float(CONFIG["map_size"]),
                    float(CONFIG["map_size"]),
                    float(CONFIG["altitude_max"]),
                ],
            )
            refined_desired = np.where(
                refined_apf["emergency"][:, None],
                refined_apf["emergency_acceleration_mps2"],
                nominal_acceleration + refined_apf["soft_acceleration_mps2"],
            )
            refined_actions = project_motion_action_np(
                refined_desired / max(max_accel, 1e-12)
            )
            for i in np.flatnonzero(needs_refinement):
                (
                    candidate_positions[i],
                    candidate_velocities[i],
                    boundary_clipped[i],
                    horizontal_boundary_clipped[i],
                    altitude_clipped[i],
                ) = compute_motion_candidate(
                    uavs[i],
                    refined_actions[i],
                    dt=dt,
                )
                combined_actions[i] = refined_actions[i]
            for key, refined_value in refined_apf.items():
                current_value = apf.get(key)
                if isinstance(current_value, np.ndarray) and current_value.shape[:1] == (num_uavs,):
                    selector = needs_refinement.reshape(
                        (num_uavs,) + (1,) * (current_value.ndim - 1)
                    )
                    apf[key] = np.where(selector, refined_value, current_value)

    if bool(CONFIG.get("apf_enabled", True)):
        clearance = float(CONFIG["obstacle_clearance_m"])
        safety_distance = float(CONFIG["safety_distance"])
        # A replacement must be safe against both cylinders and the other
        # executed paths. Accept sequentially so later replacements also see
        # every path already accepted earlier in this step.
        for i in range(num_uavs):
            if not active[i]:
                continue
            peer_unsafe = _peer_path_conflicts_numpy(
                old_positions, candidate_positions, active, safety_distance,
            )[i].any()
            obstacle_unsafe = segment_intersects_obstacle(
                old_positions[i], candidate_positions[i], obstacles, margin=clearance,
            )
            if not (peer_unsafe or obstacle_unsafe):
                continue
            brake_action = project_motion_action_np(
                -old_velocities[i] / max(max_accel * dt, 1e-12)
            )
            for fallback_action in (np.zeros(3, dtype=np.float64), brake_action):
                fallback = compute_motion_candidate(uavs[i], fallback_action, dt=dt)
                trial_positions = candidate_positions.copy()
                trial_positions[i] = fallback[0]
                if segment_intersects_obstacle(
                    old_positions[i], fallback[0], obstacles, margin=clearance,
                ) or _peer_path_conflicts_numpy(
                    old_positions, trial_positions, active, safety_distance,
                )[i].any():
                    continue
                (
                    candidate_positions[i], candidate_velocities[i],
                    boundary_clipped[i], horizontal_boundary_clipped[i], altitude_clipped[i],
                ) = fallback
                combined_actions[i] = fallback_action
                break

    applied_acceleration = combined_actions * max_accel
    apf_correction_acceleration = applied_acceleration - nominal_acceleration

    for i, uav in enumerate(uavs):
        if not uav.active:
            uav.velocity = np.zeros(
                3,
                dtype=np.float64,
            )
            continue
        uav.position = candidate_positions[i].copy()
        uav.velocity = candidate_velocities[i].copy()

    realized_acceleration_mps2 = (
        np.stack(
            [uav.velocity.copy() for uav in uavs]
        )
        - old_velocities
    ) / dt

    blocked = np.zeros(num_uavs, dtype=bool)
    blocked_by_obstacle = np.zeros(
        num_uavs,
        dtype=bool,
    )
    blocked_by_peer = np.zeros(
        num_uavs,
        dtype=bool,
    )

    return {
        "blocked": blocked,
        "blocked_by_obstacle": blocked_by_obstacle,
        "blocked_by_peer": blocked_by_peer,
        "apf_active": np.asarray(
            apf["active"],
            dtype=bool,
        ),
        "apf_peer_active": np.asarray(
            apf["peer_active"],
            dtype=bool,
        ),
        "apf_obstacle_active": np.asarray(
            apf["obstacle_active"],
            dtype=bool,
        ),
        "apf_boundary_active": np.asarray(
            apf["boundary_active"],
            dtype=bool,
        ),
        "apf_emergency": np.asarray(
            apf["emergency"],
            dtype=bool,
        ),
        "apf_peer_emergency": np.asarray(
            apf["peer_emergency"],
            dtype=bool,
        ),
        "apf_obstacle_emergency": np.asarray(
            apf["obstacle_emergency"],
            dtype=bool,
        ),
        "apf_boundary_emergency": np.asarray(
            apf["boundary_emergency"],
            dtype=bool,
        ),
        "apf_acceleration_mps2": np.asarray(
            apf_correction_acceleration,
            dtype=np.float64,
        ),
        "apf_peer_acceleration_mps2": np.asarray(
            apf["peer_acceleration_mps2"],
            dtype=np.float64,
        ),
        "apf_obstacle_acceleration_mps2": np.asarray(
            apf["obstacle_acceleration_mps2"],
            dtype=np.float64,
        ),
        "apf_boundary_acceleration_mps2": np.asarray(
            apf["boundary_acceleration_mps2"],
            dtype=np.float64,
        ),
        "apf_min_peer_clearance_m": np.asarray(
            apf["min_peer_clearance_m"],
            dtype=np.float64,
        ),
        "apf_min_obstacle_clearance_m": np.asarray(
            apf["min_obstacle_clearance_m"],
            dtype=np.float64,
        ),
        "apf_min_boundary_clearance_m": np.asarray(
            apf["min_boundary_clearance_m"],
            dtype=np.float64,
        ),
        "boundary_clipped": boundary_clipped,
        "horizontal_boundary_clipped": (
            horizontal_boundary_clipped
        ),
        "altitude_clipped": altitude_clipped,
        "realized_acceleration_mps2": (
            realized_acceleration_mps2
        ),
    }


# Export public and internal helpers required by downstream production layers.
__all__ = [name for name in globals() if not name.startswith("__")]
