"""Production source migrated from frozen fix_test_gpu.ipynb.

Do not import the notebook at runtime. The notebook is retained only as a
reference/regression artifact. Production changes belong in Python modules.
Migrated from notebook cells 21..39.
"""

from .motion import *  # noqa: F401,F403

# --- frozen notebook cell 21 ---
def sensing_profile(altitude):
    altitude = float(altitude)

    if not np.isfinite(altitude):
        raise ValueError("altitude must be finite")

    altitude = np.clip(altitude,CONFIG["altitude_min"],CONFIG["altitude_max"])
    altitude_anchors = np.asarray(CONFIG["altitude_anchors"])

    pd_anchors = np.asarray(CONFIG["pd"])
    pf_anchors = np.asarray(CONFIG["pf"])

    pd = np.interp(altitude,altitude_anchors,pd_anchors)
    pf = np.interp(altitude,altitude_anchors,pf_anchors)

    full_fov = CONFIG["camera_full_fov_deg"]
    half_fov_rad = np.deg2rad(full_fov / 2.0)
    fov_radius = altitude* np.tan(half_fov_rad)

    return (
        float(pd),
        float(pf),
        float(fov_radius)
    )


# --- frozen notebook cell 22 ---
def create_belief_maps():
    grid_n = int(np.ceil(CONFIG["map_size"] / CONFIG["grid_cell_m"]))
    belief_maps = np.full((CONFIG["num_uavs"],grid_n,grid_n),CONFIG["belief_prior"])
    return belief_maps


# --- frozen notebook cell 23 ---
def world_to_grid(position_xy):
    position_xy = np.asarray(position_xy, dtype=np.float64)

    if position_xy.ndim != 1 or position_xy.size < 2:
        raise ValueError(
            "position_xy must be a 1D array containing at least x and y"
        )

    x = float(position_xy[0])
    y = float(position_xy[1])

    if not np.isfinite(x) or not np.isfinite(y):
        raise ValueError("position must be finite")

    map_size = float(CONFIG["map_size"])
    cell_size = float(CONFIG["grid_cell_m"])

    if not np.isfinite(map_size) or map_size <= 0.0:
        raise ValueError("map_size must be finite and > 0")

    if not np.isfinite(cell_size) or cell_size <= 0.0:
        raise ValueError("grid_cell_m must be finite and > 0")

    if not (
        0.0 <= x <= map_size
        and 0.0 <= y <= map_size
    ):
        raise ValueError("position is outside the map")

    grid_n = int(np.ceil(map_size / cell_size))

    gx = int(np.floor(x / cell_size))
    gy = int(np.floor(y / cell_size))

    gx = min(gx, grid_n - 1)
    gy = min(gy, grid_n - 1)

    return gx, gy


# --- frozen notebook cell 24 ---
def cell_intersects_fov(
    gx,
    gy,
    uav_xy,
    fov_radius,
):
    cell_size = float(CONFIG["grid_cell_m"])
    map_size = float(CONFIG["map_size"])

    x_min = gx * cell_size
    x_max = min((gx + 1) * cell_size,map_size)

    y_min = gy * cell_size
    y_max = min((gy + 1) * cell_size,map_size)

    closest_x = np.clip(uav_xy[0],x_min,x_max)
    closest_y = np.clip(uav_xy[1],y_min,y_max)

    dx = float(uav_xy[0] - closest_x)
    dy = float(uav_xy[1] - closest_y)

    return (
        dx * dx + dy * dy <= fov_radius * fov_radius
    )


# --- frozen notebook cell 25 ---
import math


# --- frozen notebook cell 26 ---
def _circle_sqrt_integral(x, radius):
    """Antiderivative of sqrt(radius^2 - x^2) on [-radius, radius]."""
    x = max(-radius, min(radius, float(x)))
    root = math.sqrt(max(0.0, radius * radius - x * x))
    return 0.5 * (
        x * root
        + radius * radius * math.asin(x / radius)
    )


# --- frozen notebook cell 27 ---
def _circle_rectangle_intersection_area(
    circle_x,
    circle_y,
    radius,
    x_min,
    x_max,
    y_min,
    y_max,
):
    """Exact area of a circle intersected with an axis-aligned rectangle."""
    radius = float(radius)
    if radius <= 0.0 or x_max <= x_min or y_max <= y_min:
        return 0.0

    # Translate the circle center to the origin.
    x_min = float(x_min) - float(circle_x)
    x_max = float(x_max) - float(circle_x)
    y_min = float(y_min) - float(circle_y)
    y_max = float(y_max) - float(circle_y)

    left = max(x_min, -radius)
    right = min(x_max, radius)

    if right <= left or y_max <= -radius or y_min >= radius:
        return 0.0

    # The integrand changes form only where a horizontal rectangle edge
    # intersects the circle. Split at those x values, then integrate the
    # circle arc analytically on each interval.
    cuts = [left, right]
    for y_edge in (y_min, y_max):
        if abs(y_edge) < radius:
            x_cross = math.sqrt(
                max(0.0, radius * radius - y_edge * y_edge)
            )
            for cut in (-x_cross, x_cross):
                if left < cut < right:
                    cuts.append(cut)

    cuts = sorted(set(cuts))
    area = 0.0

    for a, b in pairwise(cuts):
        if b <= a:
            continue

        midpoint = 0.5 * (a + b)
        half_height = math.sqrt(
            max(0.0, radius * radius - midpoint * midpoint)
        )

        upper = min(y_max, half_height)
        lower = max(y_min, -half_height)
        if upper <= lower:
            continue

        arc_integral = (
            _circle_sqrt_integral(b, radius)
            - _circle_sqrt_integral(a, radius)
        )

        if y_max < half_height:
            upper_integral = y_max * (b - a)
        else:
            upper_integral = arc_integral

        if y_min > -half_height:
            lower_integral = y_min * (b - a)
        else:
            lower_integral = -arc_integral

        area += upper_integral - lower_integral

    return max(0.0, float(area))


# --- frozen notebook cell 28 ---
def cell_fov_coverage_fraction(gx, gy, uav_xy, fov_radius):
    """Return the fraction of a belief cell covered by the circular FOV.

    Geometry depends only on UAV/FOV and grid cell; target ground truth never
    changes measurement support. The circle/rectangle area is analytic, which
    is both deterministic and much cheaper than per-cell numerical quadrature.
    """
    if isinstance(gx, (bool, np.bool_)) or not isinstance(gx, (int, np.integer)):
        raise TypeError("gx must be an integer")
    if isinstance(gy, (bool, np.bool_)) or not isinstance(gy, (int, np.integer)):
        raise TypeError("gy must be an integer")

    gx = int(gx)
    gy = int(gy)
    uav_xy = np.asarray(uav_xy, dtype=np.float64)
    fov_radius = float(fov_radius)

    if uav_xy.shape != (2,) or not np.all(np.isfinite(uav_xy)):
        raise ValueError("uav_xy must be a finite shape-(2,) position")
    if not np.isfinite(fov_radius) or fov_radius < 0.0:
        raise ValueError("fov_radius must be finite and >= 0")
    if fov_radius == 0.0:
        return 0.0

    cell_size = float(CONFIG["grid_cell_m"])
    map_size = float(CONFIG["map_size"])
    grid_n = int(np.ceil(map_size / cell_size))

    if not (0 <= gx < grid_n and 0 <= gy < grid_n):
        raise ValueError("grid cell index out of range")

    x_min = gx * cell_size
    x_max = min((gx + 1) * cell_size, map_size)
    y_min = gy * cell_size
    y_max = min((gy + 1) * cell_size, map_size)

    if not cell_intersects_fov(gx, gy, uav_xy, fov_radius):
        return 0.0

    intersection_area = _circle_rectangle_intersection_area(
        circle_x=float(uav_xy[0]),
        circle_y=float(uav_xy[1]),
        radius=fov_radius,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )
    cell_area = (x_max - x_min) * (y_max - y_min)

    if cell_area <= 0.0:
        return 0.0

    return float(np.clip(intersection_area / cell_area, 0.0, 1.0))


# --- frozen notebook cell 29 ---
def cells_with_fov_coverage(uav):
    if not uav.active:
        return []

    altitude = float(uav.position[2])

    if altitude <= 0.0:
        return []

    _, _, fov_radius = sensing_profile(altitude)
    cell_size = float(CONFIG["grid_cell_m"])
    grid_n = int(np.ceil(CONFIG["map_size"] / cell_size))
    uav_xy = np.asarray(uav.position[:2], dtype=np.float64)
    uav_x = float(uav_xy[0])
    uav_y = float(uav_xy[1])

    gx_min = max(0, int(np.floor((uav_x - fov_radius) / cell_size)))
    gx_max = min(grid_n - 1, int(np.floor((uav_x + fov_radius) / cell_size)))
    gy_min = max(0, int(np.floor((uav_y - fov_radius) / cell_size)))
    gy_max = min(grid_n - 1, int(np.floor((uav_y + fov_radius) / cell_size)))

    visible = []
    for gy in range(gy_min, gy_max + 1):
        for gx in range(gx_min, gx_max + 1):
            coverage = cell_fov_coverage_fraction(
                gx,
                gy,
                uav_xy,
                fov_radius,
            )
            if coverage > 0.0:
                visible.append((gx, gy, coverage))

    return visible


# --- frozen notebook cell 30 ---
def cells_in_fov(uav):
    return [
        (gx, gy)
        for gx, gy, _ in cells_with_fov_coverage(uav)
    ]


# --- frozen notebook cell 31 ---
def get_target_cells(targets):
    target_cells = {}

    for target in targets:
        gx, gy = world_to_grid(target.position)
        target_cells.setdefault((gx, gy), []).append(target.id)

    return target_cells


# --- frozen notebook cell 32 ---
def sample_sensor_measurement(has_target,pd,pf,rng):
    if has_target:
        positive_probability = pd

    else:
        positive_probability = pf
    observation = (rng.random()< positive_probability)

    return int(observation)


# --- frozen notebook cell 33 ---
def bayes_update(prior, observation, pd, pf, eps=1e-8):
    eps = float(eps)

    if (
        not np.isfinite(eps)
        or not 0.0 < eps < 0.5
    ):
        raise ValueError(
            "eps must be finite and in (0, 0.5)"
        )

    prior = float(prior)
    pd = float(pd)
    pf = float(pf)

    for name, value in (
        ("prior", prior),
        ("pd", pd),
        ("pf", pf),
    ):
        if not np.isfinite(value):
            raise ValueError(
                f"{name} must be finite"
            )

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} must be in [0, 1]"
            )

    prior = np.clip(prior,eps,1.0 - eps)

    if observation == 1:
        numerator = pd * prior
        denominator = pd * prior + pf * (1.0 - prior)
    elif observation == 0:
        numerator = (1.0 - pd) * prior
        denominator = (
            (1.0 - pd) * prior
            + (1.0 - pf) * (1.0 - prior)
        )
    else:
        raise ValueError("observation must be 0 or 1")

    denominator = max(float(denominator), eps)
    posterior = numerator / denominator

    return float(np.clip(posterior, eps, 1.0 - eps))


# --- frozen notebook cell 34 ---
def camera_ground_point_has_los(
    uav,
    ground_xy,
    obstacles=None,
):
    if obstacles is None:
        obstacles = []

    ground_xy = np.asarray(
        ground_xy,
        dtype=np.float64,
    )

    if (
        ground_xy.shape != (2,)
        or not np.all(
            np.isfinite(
                ground_xy
            )
        )
    ):
        raise ValueError(
            "ground_xy must be a "
            "finite shape-(2,) point"
        )

    ground_position = np.array(
        [
            float(
                ground_xy[0]
            ),
            float(
                ground_xy[1]
            ),
            0.0,
        ],
        dtype=np.float64,
    )

    return not segment_intersects_obstacle(
        uav.position,
        ground_position,
        obstacles,
    )


# --- frozen notebook cell 35 ---
def camera_target_has_los(
    uav,
    target,
    obstacles=None,
):
    return camera_ground_point_has_los(
        uav,
        target.position,
        obstacles=obstacles,
    )


# --- frozen notebook cell 36 ---
def sense_and_update(
    uav,
    belief_map,
    targets,
    rng,
    obstacles=None,
):
    if not uav.active:
        return []

    if obstacles is None:
        obstacles = []

    altitude = float(
        uav.position[2]
    )

    if altitude <= 0.0:
        return []

    (
        base_pd,
        base_pf,
        fov_radius,
    ) = sensing_profile(
        altitude
    )

    visible_cells = (
        cells_with_fov_coverage(
            uav
        )
    )
    target_cells = (
        get_target_cells(
            targets
        )
    )
    targets_by_id = {
        target.id: target
        for target in targets
    }

    cell_size = float(
        CONFIG[
            "grid_cell_m"
        ]
    )
    map_size = float(
        CONFIG[
            "map_size"
        ]
    )

    # A visible cell center may extend half a cell diagonal beyond the FOV.
    # Cylinders outside this disk cannot intersect any of the camera rays.
    ray_radius = fov_radius + np.sqrt(0.5) * cell_size
    los_obstacles = [
        obstacle for obstacle in obstacles
        if np.linalg.norm(obstacle.position - uav.position[:2])
        <= ray_radius + obstacle.radius
    ]

    sensing_log = []

    for (
        gx,
        gy,
        coverage_fraction,
    ) in visible_cells:
        cell_center = np.array(
            [
                min(
                    (
                        float(gx)
                        + 0.5
                    )
                    * cell_size,
                    map_size,
                ),
                min(
                    (
                        float(gy)
                        + 0.5
                    )
                    * cell_size,
                    map_size,
                ),
            ],
            dtype=np.float64,
        )

        if (
            bool(
                CONFIG[
                    "sensing_obstacle_occlusion"
                ]
            )
            and not camera_ground_point_has_los(
                uav,
                cell_center,
                obstacles=los_obstacles,
            )
        ):
            continue

        candidate_target_ids = (
            target_cells.get(
                (gx, gy),
                [],
            )
        )

        target_ids_in_fov = []

        for target_id in (
            candidate_target_ids
        ):
            target = targets_by_id[
                target_id
            ]

            in_footprint = (
                np.linalg.norm(
                    target.position
                    - uav.position[:2]
                )
                <= fov_radius
            )

            target_visible = (
                in_footprint
                and (
                    not bool(
                        CONFIG[
                            "sensing_obstacle_occlusion"
                        ]
                    )
                    or camera_target_has_los(
                        uav,
                        target,
                        obstacles=los_obstacles,
                    )
                )
            )

            if target_visible:
                target_ids_in_fov.append(
                    target_id
                )

        has_target = bool(
            target_ids_in_fov
        )

        effective_pf = float(
            1.0
            - (
                1.0 - base_pf
            )
            ** coverage_fraction
        )
        effective_pd = float(
            coverage_fraction
            * base_pd
            + (
                1.0
                - coverage_fraction
            )
            * effective_pf
        )

        if has_target:
            positive_probability = (
                base_pd
            )
        else:
            positive_probability = (
                effective_pf
            )

        observation = int(
            rng.random()
            < positive_probability
        )

        prior = float(
            belief_map[
                gy,
                gx,
            ]
        )
        posterior = bayes_update(
            prior,
            observation,
            effective_pd,
            effective_pf,
        )
        belief_map[
            gy,
            gx,
        ] = posterior

        sensing_log.append(
            {
                "gx": gx,
                "gy": gy,
                "target_ids": (
                    target_ids_in_fov
                ),
                "has_target": (
                    has_target
                ),
                "observation": (
                    observation
                ),
                "prior": prior,
                "posterior": (
                    posterior
                ),
                "pd": effective_pd,
                "pf": effective_pf,
                "base_pd": base_pd,
                "base_pf": base_pf,
                "coverage_fraction": (
                    float(
                        coverage_fraction
                    )
                ),
            }
        )

    return sensing_log


# --- frozen notebook cell 37 ---
def check_confirmation(
    uav,
    sensing_record,
):
    if not uav.active:
        return "none", []

    posterior = float(
        sensing_record[
            "posterior"
        ]
    )
    observation = int(
        sensing_record[
            "observation"
        ]
    )
    target_ids = list(
        sensing_record[
            "target_ids"
        ]
    )

    if (
        float(uav.position[2])
        > float(
            CONFIG[
                "fine_altitude"
            ]
        )
    ):
        return "none", []

    if (
        posterior
        < float(
            CONFIG[
                "confirmation_threshold"
            ]
        )
    ):
        return "none", []

    if observation != 1:
        return "none", []

    if target_ids:
        return (
            "true_confirmation",
            target_ids,
        )

    return (
        "false_confirmation",
        [],
    )


# --- frozen notebook cell 38 ---
@dataclass
class ConfirmationEvent:
    target_id: int | None
    uav_id: int
    step: int

    cell: tuple
    uav_position: np.ndarray

    altitude: float
    belief: float
    observation: int

    confirmation_type: str
    report_status: str | None = None


# --- frozen notebook cell 39 ---
def create_confirmation_events(
    uav,
    sensing_record,
    step
):
    confirmation_type, target_ids = check_confirmation(
        uav,
        sensing_record
    )

    if confirmation_type == "none":
        return []

    common = {
        "uav_id": int(uav.id),
        "step": int(step),
        "cell": (
            int(sensing_record["gx"]),
            int(sensing_record["gy"]),
        ),
        "uav_position": uav.position.copy(),
        "altitude": float(uav.position[2]),
        "belief": float(sensing_record["posterior"]),
        "observation": int(sensing_record["observation"]),
        "confirmation_type": confirmation_type,
    }

    if confirmation_type == "false_confirmation":
        return [ConfirmationEvent(target_id=None,**common,)]

    return [ConfirmationEvent(target_id=int(target_id),**common,) for target_id in target_ids]


# Export public and internal helpers required by downstream production layers.
__all__ = [name for name in globals() if not name.startswith("__")]
