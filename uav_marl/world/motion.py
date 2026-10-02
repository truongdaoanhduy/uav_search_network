"""Production source migrated from frozen fix_test_gpu.ipynb.

Do not import the notebook at runtime. The notebook is retained only as a
reference/regression artifact. Production changes belong in Python modules.
Migrated from notebook cells 16..20.
"""

from .generation import *  # noqa: F401,F403

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


# --- frozen notebook cell 19 ---
def minimum_distance_during_motion(
    start_a,
    end_a,
    start_b,
    end_b,
):
    start_a = np.asarray(start_a,dtype=np.float64,)
    end_a = np.asarray(end_a,dtype=np.float64,)
    start_b = np.asarray(start_b,dtype=np.float64,)
    end_b = np.asarray(end_b,dtype=np.float64,)
    for point in (
        start_a,
        end_a,
        start_b,
        end_b,
    ):
        if point.shape != (3,):
            raise ValueError(
                "all positions must have shape (3,)"
            )

        if not np.all(np.isfinite(point)):
            raise ValueError(
                "all positions must contain only finite values"
            )
    relative_start = start_a - start_b
    displacement_a = end_a - start_a
    displacement_b = end_b - start_b

    relative_motion = displacement_a - displacement_b

    denominator = relative_motion@relative_motion
    if denominator <= 1e-12:
        return np.linalg.norm(
                relative_start
            )


    t_closest = -(relative_start@relative_motion)/ denominator
    t_closest = np.clip(t_closest,0.0,1.0,)

    relative_at_closest = (
        relative_start
        + t_closest * relative_motion
    )

    return float(
        np.linalg.norm(
            relative_at_closest
        )
    )


# --- frozen notebook cell 20 ---
def apply_swarm_motion(
    uavs,
    actions,
    obstacles=None,
    dt=None,
):
    if obstacles is None:
        obstacles = []

    if dt is None:
        dt = CONFIG["dt"]
    dt = float(dt)


    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(
            "dt must be finite and > 0"
        )

    num_uavs = len(uavs)

    if num_uavs == 0:
        raise ValueError(
            "uavs must not be empty"
        )

    actions = np.asarray(actions,dtype=np.float64)

    if actions.shape != (num_uavs, 3):
        raise ValueError(
            f"actions must have shape "
            f"({num_uavs}, 3)"
        )

    if not np.all(np.isfinite(actions)):
        raise ValueError(
            "actions must contain only finite values"
        )

    old_positions = np.stack([
        uav.position.copy()
        for uav in uavs
    ])
    old_velocities = np.stack([
        uav.velocity.copy()
        for uav in uavs
    ])

    safety_distance = float(CONFIG["safety_distance"])

    for i in range(num_uavs):
        if not uavs[i].active:
            continue

        for j in range(i + 1,num_uavs):
            if not uavs[j].active:
                continue

            distance = np.linalg.norm(old_positions[i]- old_positions[j])

            if distance < safety_distance:
                raise ValueError(
                    "initial active UAV positions "
                    "violate safety_distance"
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

    for i, (uav, action) in enumerate(zip(uavs, actions)):

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
        altitude_clipped[i] = (
            vertical_clipped
        )

    candidate_positions = np.stack(candidate_positions)
    candidate_velocities = np.stack(candidate_velocities)

    blocked_by_obstacle = np.zeros(
        num_uavs,
        dtype=bool,
    )

    for i, uav in enumerate(uavs):
        if not uav.active:
            continue

        if segment_intersects_obstacle(
            old_positions[i],
            candidate_positions[i],
            obstacles,
            margin=float(
                CONFIG[
                    "obstacle_clearance_m"
                ]
            ),
        ):
            blocked_by_obstacle[i] = True

    blocked_by_peer = np.zeros(num_uavs,dtype=bool)
    blocked = blocked_by_obstacle.copy()


    while True:
        effective_positions = candidate_positions.copy()
        effective_positions[blocked] = old_positions[blocked]

        next_blocked = blocked.copy()

        for i in range(num_uavs):
            if not uavs[i].active:
                continue

            for j in range(i + 1,num_uavs):
                if not uavs[j].active:
                    continue

                distance = minimum_distance_during_motion(
                        old_positions[i],
                        effective_positions[i],
                        old_positions[j],
                        effective_positions[j],
                    )

                if distance < safety_distance:
                    next_blocked[i] = True
                    next_blocked[j] = True

                    blocked_by_peer[i] = True
                    blocked_by_peer[j] = True

        if np.array_equal(next_blocked,blocked):
            break

        blocked = next_blocked

    for i, uav in enumerate(uavs):
        if not uav.active:
            uav.velocity = np.zeros(3, dtype=np.float64)
            continue

        if blocked[i]:
            uav.position = old_positions[i].copy()
            uav.velocity = np.zeros(3,dtype=np.float64)

        else:
            uav.position = candidate_positions[i].copy()
            uav.velocity = candidate_velocities[i].copy()

    realized_acceleration_mps2 = (
        np.stack([
            uav.velocity.copy()
            for uav in uavs
        ])
        - old_velocities
    ) / dt

    return {
        "blocked": blocked,
        "blocked_by_obstacle":blocked_by_obstacle,
        "blocked_by_peer":blocked_by_peer,
        "boundary_clipped":boundary_clipped,
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
