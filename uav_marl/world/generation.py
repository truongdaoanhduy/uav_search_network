"""Production source migrated from frozen fix_test_gpu.ipynb.

Do not import the notebook at runtime. The notebook is retained only as a
reference/regression artifact. Production changes belong in Python modules.
Migrated from notebook cells 11..15.
"""

from .entities import *  # noqa: F401,F403

# --- frozen notebook cell 11 ---
def create_uav(rng):
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")

    num_uavs = int(CONFIG["num_uavs"])
    gcs = np.asarray(CONFIG["gcs_position"], dtype=np.float64)
    launch_radius = float(CONFIG["launch_radius_m"])
    safety_distance = float(CONFIG["safety_distance"])
    launch_min_spacing = float(CONFIG["launch_min_spacing_m"])
    map_size = float(CONFIG["map_size"])
    altitude = float(CONFIG["altitude_min"])

    if num_uavs < 1:
        raise ValueError("num_uavs must be >= 1")

    if gcs.shape != (3,) or not np.all(np.isfinite(gcs)):
        raise ValueError("gcs_position must be a finite 3D position")

    if not (
        0.0 <= gcs[0] <= map_size
        and 0.0 <= gcs[1] <= map_size
    ):
        raise ValueError("gcs_position must lie inside the map")

    if not np.isfinite(launch_radius) or launch_radius <= 0.0:
        raise ValueError("launch_radius_m must be finite and > 0")

    if not np.isfinite(safety_distance) or safety_distance < 0.0:
        raise ValueError("safety_distance must be finite and >= 0")

    if (
        not np.isfinite(launch_min_spacing)
        or launch_min_spacing < safety_distance
    ):
        raise ValueError(
            "launch_min_spacing_m must be finite and >= safety_distance"
        )

    if not np.isfinite(map_size) or map_size <= 0.0:
        raise ValueError("map_size must be finite and > 0")

    if not np.isfinite(altitude):
        raise ValueError("altitude_min must be finite")

    uavs = []
    max_attempts = max(10_000, 1_000 * num_uavs)
    attempts = 0
    launch_radius_sq = launch_radius * launch_radius

    while len(uavs) < num_uavs:
        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                "Could not place all UAVs inside the launch region while "
                "respecting map bounds and launch_min_spacing_m. "
                "Increase launch_radius_m or reduce num_uavs/launch_min_spacing_m."
            )

        offset_xy = rng.uniform(
            -launch_radius,
            launch_radius,
            size=2,
        ).astype(np.float64)

        if float(offset_xy @ offset_xy) > launch_radius_sq:
            continue

        candidate_xy = gcs[:2] + offset_xy

        if not (
            0.0 <= candidate_xy[0] <= map_size
            and 0.0 <= candidate_xy[1] <= map_size
        ):
            continue

        too_close = any(
            np.linalg.norm(candidate_xy - uav.position[:2])
            < launch_min_spacing
            for uav in uavs
        )

        if too_close:
            continue

        position = np.array(
            [candidate_xy[0], candidate_xy[1], altitude],
            dtype=np.float64,
        )

        uavs.append(
            UAV(
                id=len(uavs),
                position=position,
                velocity=np.zeros(3, dtype=np.float64),
                battery_j=CONFIG["battery_j"],
                active=True,
            )
        )

    return uavs


# --- frozen notebook cell 12 ---
def point_inside_obstacle(point, obstacles, margin=0.0):
    point = np.asarray(point, dtype=np.float64)
    margin = float(margin)

    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("margin must be finite and >= 0")

    if point.shape == (2,):
        point_xy = point
        z = 0.0
    elif point.shape == (3,):
        point_xy = point[:2]
        z = float(point[2])
    else:
        raise ValueError("point must have shape (2,) or (3,)")

    if not np.all(np.isfinite(point)):
        raise ValueError("point must contain only finite values")

    for obs in obstacles:
        horizontal_distance = np.linalg.norm(
            point_xy - obs.position
        )

        inside_horizontal = (
            horizontal_distance
            <= obs.radius + margin
        )
        inside_vertical = (
            0.0 <= z <= obs.height + margin
        )

        if inside_horizontal and inside_vertical:
            return True

    return False


# --- frozen notebook cell 13 ---
def create_obstacle(rng, uavs):
    obstacles = []

    map_size = float(CONFIG["map_size"])
    r_min = float(CONFIG["obstacle_radius_min_m"])
    r_max = float(CONFIG["obstacle_radius_max_m"])
    h_min = float(CONFIG["obstacle_height_min_m"])
    h_max = float(CONFIG["obstacle_height_max_m"])

    safety_margin = float(CONFIG["safety_distance"])
    gcs_xy = np.asarray(CONFIG["gcs_position"][:2], dtype=np.float64)
    gcs_exclusion = float(CONFIG["gcs_exclusion_radius_m"])

    if r_min <= 0.0 or r_max < r_min:
        raise ValueError("obstacle radius range is invalid")
    if 2.0 * r_max > map_size:
        raise ValueError(
            "obstacle_radius_max_m must be <= map_size / 2"
        )

    max_attempts = 100000
    attempts = 0

    while len(obstacles) < CONFIG["num_obstacles"]:
        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                "Could not place all obstacles with the current constraints."
            )

        radius = float(rng.uniform(r_min, r_max))
        height = float(rng.uniform(h_min, h_max))

        x = float(rng.uniform(radius, map_size - radius))
        y = float(rng.uniform(radius, map_size - radius))
        center = np.array([x, y], dtype=np.float64)

        distance_to_gcs = np.linalg.norm(center - gcs_xy)
        if distance_to_gcs <= gcs_exclusion + radius:
            continue

        overlaps_launch = any(
            np.linalg.norm(center - uav.position[:2])
            <= radius + safety_margin
            for uav in uavs
        )
        if overlaps_launch:
            continue

        overlaps_obstacle = any(
            np.linalg.norm(center - other.position)
            <= radius + other.radius
            for other in obstacles
        )
        if overlaps_obstacle:
            continue

        obstacles.append(
            Obstacle(
                position=center,
                radius=radius,
                height=height,
            )
        )

    return obstacles


# --- frozen notebook cell 14 ---
def create_target(rng, obstacles):
    targets = []
    occupied_cells = set()

    map_size = float(CONFIG["map_size"])
    cell_size = float(CONFIG["grid_cell_m"])
    gcs_xy = np.asarray(CONFIG["gcs_position"][:2], dtype=np.float64)
    target_exclusion = float(CONFIG["target_exclusion_radius_m"])

    max_attempts = 100000
    attempts = 0

    while len(targets) < CONFIG["num_targets"]:
        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                "Could not place all targets with the current constraints."
            )

        position = rng.uniform(0.0,map_size,size=2,).astype(np.float64)

        if point_inside_obstacle(position, obstacles):
            continue

        if np.linalg.norm(position - gcs_xy) <= target_exclusion:
            continue

        gx = int(np.floor(position[0] / cell_size))
        gy = int(np.floor(position[1] / cell_size))
        cell = (gx, gy)

        if cell in occupied_cells:
            continue

        targets.append(
            Target(
                id=len(targets),
                position=position,
                confirmed=False,
            )
        )
        occupied_cells.add(cell)

    return targets


# --- frozen notebook cell 15 ---
def create_world(seed):
    rng = np.random.default_rng(seed)
    uavs = create_uav(rng)
    obstacles = create_obstacle(rng, uavs)
    targets = create_target(rng, obstacles)

    return rng, uavs, targets, obstacles


# Export public and internal helpers required by downstream production layers.
__all__ = [name for name in globals() if not name.startswith("__")]
