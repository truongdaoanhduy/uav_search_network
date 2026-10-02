"""Production source migrated from frozen fix_test_gpu.ipynb.

Do not import the notebook at runtime. The notebook is retained only as a
reference/regression artifact. Production changes belong in Python modules.
Migrated from notebook cells 125..138.
"""

from .reward_energy import *  # noqa: F401,F403

# --- frozen notebook cell 125 ---
def _normalized_relative_vector(
    source_position,
    destination_position,
):
    source_position = np.asarray(
        source_position,
        dtype=np.float64,
    )
    destination_position = np.asarray(
        destination_position,
        dtype=np.float64,
    )

    if (
        source_position.shape != (3,)
        or destination_position.shape
        != (3,)
    ):
        raise ValueError(
            "positions must have shape (3,)"
        )

    altitude_span = max(
        1e-9,
        float(CONFIG["altitude_max"])
        - float(CONFIG["altitude_min"]),
    )

    scale = np.array(
        [
            float(CONFIG["map_size"]),
            float(CONFIG["map_size"]),
            altitude_span,
        ],
        dtype=np.float64,
    )

    return np.clip(
        (
            destination_position
            - source_position
        )
        / scale,
        -1.0,
        1.0,
    ).astype(
        np.float32
    )


# --- frozen notebook cell 126 ---
def extract_belief_patch(
    belief_map,
    position,
):
    patch_cells = int(
        CONFIG["belief_patch_cells"]
    )

    if (
        patch_cells < 1
        or patch_cells % 2 == 0
    ):
        raise ValueError(
            "belief_patch_cells must be "
            "a positive odd integer"
        )

    belief_map = np.asarray(
        belief_map,
        dtype=np.float64,
    )

    if belief_map.ndim != 2:
        raise ValueError(
            "belief_map must be 2-D"
        )

    gx, gy = world_to_grid(
        position
    )

    radius = (
        patch_cells // 2
    )

    patch = np.full(
        (
            patch_cells,
            patch_cells,
        ),
        float(
            CONFIG["belief_prior"]
        ),
        dtype=np.float32,
    )

    height, width = (
        belief_map.shape
    )

    for patch_y in range(
        patch_cells
    ):
        map_y = (
            gy
            + patch_y
            - radius
        )

        if not 0 <= map_y < height:
            continue

        for patch_x in range(
            patch_cells
        ):
            map_x = (
                gx
                + patch_x
                - radius
            )

            if not 0 <= map_x < width:
                continue

            patch[
                patch_y,
                patch_x,
            ] = float(
                belief_map[
                    map_y,
                    map_x,
                ]
            )

    return patch


# --- frozen notebook cell 127 ---
def build_destination_mask(
    sender_id,
    uavs,
    buffer=None,
):
    choices = communication_choices(
        sender_id,
        len(uavs),
    )

    mask = np.zeros(
        len(choices),
        dtype=np.int8,
    )

    # Silent is always valid.
    mask[0] = 1

    sender = uavs[
        sender_id
    ]

    if not sender.active:
        return mask

    if buffer is not None:
        has_transmittable_report = any(
            not report_is_complete(
                report
            )
            for report in buffer
        )

        if not has_transmittable_report:
            return mask

    gcs_position = np.asarray(
        CONFIG[
            "gcs_position"
        ],
        dtype=np.float64,
    )

    for index, destination in enumerate(
        choices[1:],
        start=1,
    ):
        if destination == GCS_NODE:
            distance = float(
                np.linalg.norm(
                    sender.position
                    - gcs_position
                )
            )
            mask[index] = int(
                distance
                <= float(
                    CONFIG[
                        "gcs_contact_range_m"
                    ]
                )
            )
            continue

        peer = uavs[
            destination
        ]

        if not peer.active:
            continue

        distance = float(
            np.linalg.norm(
                sender.position
                - peer.position
            )
        )

        mask[index] = int(
            distance
            <= float(
                CONFIG[
                    "peer_contact_range_m"
                ]
            )
        )

    return mask


# --- frozen notebook cell 128 ---
def build_obstacle_features(
    uav,
    obstacles,
):
    count = int(
        CONFIG[
            "observation_nearest_obstacles"
        ]
    )

    if count < 0:
        raise ValueError(
            "observation_nearest_obstacles "
            "must be >= 0"
        )

    features = np.zeros(
        (count, 5),
        dtype=np.float32,
    )

    if count == 0:
        return features

    map_size = max(
        1e-9,
        float(
            CONFIG[
                "map_size"
            ]
        ),
    )
    altitude_max = max(
        1e-9,
        float(
            CONFIG[
                "altitude_max"
            ]
        ),
    )
    diagonal = max(
        1e-9,
        np.sqrt(2.0)
        * map_size,
    )

    ranked = sorted(
        obstacles,
        key=lambda obstacle: (
            float(
                np.linalg.norm(
                    np.asarray(
                        obstacle.position,
                        dtype=np.float64,
                    )
                    - uav.position[:2]
                )
            )
        ),
    )

    for row_index, obstacle in enumerate(
        ranked[:count]
    ):
        relative_xy = (
            np.asarray(
                obstacle.position,
                dtype=np.float64,
            )
            - uav.position[:2]
        )
        horizontal_distance = float(
            np.linalg.norm(
                relative_xy
            )
        )

        features[
            row_index
        ] = np.array(
            [
                np.clip(
                    relative_xy[0]
                    / map_size,
                    -1.0,
                    1.0,
                ),
                np.clip(
                    relative_xy[1]
                    / map_size,
                    -1.0,
                    1.0,
                ),
                np.clip(
                    horizontal_distance
                    / diagonal,
                    0.0,
                    1.0,
                ),
                np.clip(
                    float(
                        obstacle.radius
                    )
                    / map_size,
                    0.0,
                    1.0,
                ),
                np.clip(
                    float(
                        obstacle.height
                    )
                    / altitude_max,
                    0.0,
                    1.0,
                ),
            ],
            dtype=np.float32,
        )

    return features


# --- frozen notebook cell 129 ---
def coarse_belief_uncertainty_map(
    belief_map,
    output_cells=None,
):
    values = np.asarray(
        belief_map,
        dtype=np.float64,
    )

    if values.ndim != 2:
        raise ValueError(
            "belief_map must be 2-D"
        )

    if output_cells is None:
        output_cells = int(
            CONFIG[
                "belief_coarse_cells"
            ]
        )

    output_cells = int(
        output_cells
    )

    if output_cells < 1:
        raise ValueError(
            "belief_coarse_cells "
            "must be >= 1"
        )

    if (
        output_cells
        > values.shape[0]
        or output_cells
        > values.shape[1]
    ):
        raise ValueError(
            "belief_coarse_cells cannot "
            "exceed belief-map size"
        )

    probabilities = np.clip(
        values,
        1e-12,
        1.0 - 1e-12,
    )
    entropy = (
        -probabilities
        * np.log2(
            probabilities
        )
        - (
            1.0
            - probabilities
        )
        * np.log2(
            1.0
            - probabilities
        )
    )

    if (
        entropy.shape[0] % output_cells == 0
        and entropy.shape[1] % output_cells == 0
    ):
        row_size = entropy.shape[0] // output_cells
        col_size = entropy.shape[1] // output_cells
        pooled = (
            entropy.reshape(
                output_cells,
                row_size,
                output_cells,
                col_size,
            )
            .mean(axis=(1, 3))
            .astype(np.float32)
        )
        return np.clip(
            pooled,
            0.0,
            1.0,
        )

    row_groups = np.array_split(
        np.arange(
            entropy.shape[0]
        ),
        output_cells,
    )
    col_groups = np.array_split(
        np.arange(
            entropy.shape[1]
        ),
        output_cells,
    )

    pooled = np.zeros(
        (
            output_cells,
            output_cells,
        ),
        dtype=np.float32,
    )

    for row_index, rows in enumerate(
        row_groups
    ):
        for col_index, cols in enumerate(
            col_groups
        ):
            pooled[
                row_index,
                col_index,
            ] = float(
                np.mean(
                    entropy[
                        np.ix_(
                            rows,
                            cols,
                        )
                    ]
                )
            )

    return np.clip(
        pooled,
        0.0,
        1.0,
    )


# --- frozen notebook cell 130 ---
def build_agent_observation(
    uav_id,
    uavs,
    belief_maps,
    obstacles,
    report_buffers,
    pending_reports,
    current_step,
):
    uav = uavs[uav_id]

    map_size = float(
        CONFIG["map_size"]
    )
    altitude_min = float(
        CONFIG["altitude_min"]
    )
    altitude_span = max(
        1e-9,
        float(
            CONFIG[
                "altitude_max"
            ]
        )
        - altitude_min,
    )
    max_speed = max(
        1e-9,
        float(
            CONFIG[
                "max_speed"
            ]
        ),
    )
    battery_capacity = max(
        1e-9,
        float(
            CONFIG[
                "battery_j"
            ]
        ),
    )
    time_fraction = float(
        np.clip(
            float(
                current_step
            )
            / max(
                1.0,
                float(
                    CONFIG[
                        "max_steps"
                    ]
                ),
            ),
            0.0,
            1.0,
        )
    )

    self_state = np.array(
        [
            np.clip(
                uav.position[0]
                / map_size,
                0.0,
                1.0,
            ),
            np.clip(
                uav.position[1]
                / map_size,
                0.0,
                1.0,
            ),
            np.clip(
                (
                    uav.position[2]
                    - altitude_min
                )
                / altitude_span,
                0.0,
                1.0,
            ),
            np.clip(
                uav.velocity[0]
                / max_speed,
                -1.0,
                1.0,
            ),
            np.clip(
                uav.velocity[1]
                / max_speed,
                -1.0,
                1.0,
            ),
            np.clip(
                uav.velocity[2]
                / max_speed,
                -1.0,
                1.0,
            ),
            np.clip(
                uav.battery_j
                / battery_capacity,
                0.0,
                1.0,
            ),
            float(
                uav.active
            ),
            time_fraction,
        ],
        dtype=np.float32,
    )

    gcs_relative = (
        _normalized_relative_vector(
            uav.position,
            np.asarray(
                CONFIG[
                    "gcs_position"
                ],
                dtype=np.float64,
            ),
        )
    )

    peer_rows = []

    for peer_id, peer in enumerate(
        uavs
    ):
        if peer_id == uav_id:
            continue

        row = np.zeros(
            5,
            dtype=np.float32,
        )

        distance = float(
            np.linalg.norm(
                peer.position
                - uav.position
            )
        )

        visible = bool(
            uav.active
            and peer.active
            and distance
            <= float(
                CONFIG[
                    "peer_contact_range_m"
                ]
            )
        )

        if visible:
            row[:3] = (
                _normalized_relative_vector(
                    uav.position,
                    peer.position,
                )
            )
            row[3] = float(
                np.clip(
                    distance
                    / max(
                        1e-9,
                        float(
                            CONFIG[
                                "peer_contact_range_m"
                            ]
                        ),
                    ),
                    0.0,
                    1.0,
                )
            )
            row[4] = 1.0

        peer_rows.append(
            row
        )

    neighbors = np.stack(
        peer_rows,
        axis=0,
    ).astype(
        np.float32
    )

    obstacle_features = (
        build_obstacle_features(
            uav,
            obstacles,
        )
    )

    buffer = report_buffers[
        uav_id
    ]

    max_buffer_reports = max(
        1.0,
        float(
            CONFIG[
                "buffer_bytes"
            ]
        )
        / max(
            1.0,
            float(
                CONFIG[
                    "report_bytes"
                ]
            ),
        ),
    )

    used_fraction = np.clip(
        buffer_used_bytes(
            buffer
        )
        / max(
            1.0,
            float(
                CONFIG[
                    "buffer_bytes"
                ]
            ),
        ),
        0.0,
        1.0,
    )

    report_fraction = np.clip(
        len(buffer)
        / max_buffer_reports,
        0.0,
        1.0,
    )

    oldest_age_fraction = 0.0

    for report in buffer:
        if report.ttl_s <= 0.0:
            continue

        oldest_age_fraction = max(
            oldest_age_fraction,
            float(
                np.clip(
                    report_age_s(
                        report,
                        current_step,
                    )
                    / report.ttl_s,
                    0.0,
                    1.0,
                )
            ),
        )

    pending_bytes = (
        pending_source_used_bytes(
            pending_reports,
            uav_id,
        )
    )

    pending_fraction = np.clip(
        pending_bytes
        / max(
            1.0,
            float(
                CONFIG[
                    "pending_buffer_bytes"
                ]
            ),
        ),
        0.0,
        1.0,
    )

    head_delivery_fraction = 0.0

    for report in buffer:
        if report.size_bytes <= 0:
            continue

        head_delivery_fraction = float(
            np.clip(
                float(
                    report.delivered_bytes
                )
                / float(
                    report.size_bytes
                ),
                0.0,
                1.0,
            )
        )
        break

    buffer_state = np.array(
        [
            used_fraction,
            report_fraction,
            oldest_age_fraction,
            pending_fraction,
            head_delivery_fraction,
        ],
        dtype=np.float32,
    )

    return {
        "self_state": (
            self_state
        ),
        "gcs_relative": (
            gcs_relative
        ),
        "neighbors": (
            neighbors
        ),
        "obstacles": (
            obstacle_features
        ),
        "belief_patch": (
            extract_belief_patch(
                belief_maps[
                    uav_id
                ],
                uav.position,
            )
        ),
        "belief_coarse": (
            coarse_belief_uncertainty_map(
                belief_maps[
                    uav_id
                ]
            )
        ),
        "buffer": (
            buffer_state
        ),
        "destination_mask": (
            build_destination_mask(
                uav_id,
                uavs,
                buffer=buffer,
            )
        ),
    }


# --- frozen notebook cell 131 ---
def belief_map_summary(
    belief_map,
):
    values = np.asarray(
        belief_map,
        dtype=np.float64,
    )

    if values.ndim != 2:
        raise ValueError(
            "belief_map must be 2-D"
        )

    probabilities = np.clip(
        values,
        1e-12,
        1.0 - 1e-12,
    )

    entropy = (
        -probabilities
        * np.log2(
            probabilities
        )
        - (
            1.0
            - probabilities
        )
        * np.log2(
            1.0
            - probabilities
        )
    )

    mean_entropy = float(
        np.mean(
            entropy
        )
    )

    known_fraction = float(
        np.mean(
            np.abs(
                probabilities
                - float(
                    CONFIG[
                        "belief_prior"
                    ]
                )
            )
            > 0.05
        )
    )

    return np.array(
        [
            np.clip(
                mean_entropy,
                0.0,
                1.0,
            ),
            np.clip(
                known_fraction,
                0.0,
                1.0,
            ),
        ],
        dtype=np.float32,
    )


# --- frozen notebook cell 132 ---
def pool_belief_map(
    belief_map,
    output_cells=None,
):
    values = np.asarray(
        belief_map,
        dtype=np.float64,
    )

    if values.ndim != 2:
        raise ValueError(
            "belief_map must be 2-D"
        )

    if output_cells is None:
        output_cells = int(
            CONFIG[
                "critic_belief_grid_cells"
            ]
        )

    output_cells = int(
        output_cells
    )

    if output_cells < 1:
        raise ValueError(
            "critic_belief_grid_cells "
            "must be >= 1"
        )

    if (
        output_cells
        > values.shape[0]
        or output_cells
        > values.shape[1]
    ):
        raise ValueError(
            "critic_belief_grid_cells "
            "cannot exceed belief-map size"
        )

    if (
        values.shape[0] % output_cells == 0
        and values.shape[1] % output_cells == 0
    ):
        row_size = values.shape[0] // output_cells
        col_size = values.shape[1] // output_cells
        pooled = (
            values.reshape(
                output_cells,
                row_size,
                output_cells,
                col_size,
            )
            .mean(axis=(1, 3))
            .astype(np.float32)
        )
        return np.clip(
            pooled,
            0.0,
            1.0,
        )

    row_groups = np.array_split(
        np.arange(
            values.shape[0]
        ),
        output_cells,
    )
    col_groups = np.array_split(
        np.arange(
            values.shape[1]
        ),
        output_cells,
    )

    pooled = np.zeros(
        (
            output_cells,
            output_cells,
        ),
        dtype=np.float32,
    )

    for row_index, rows in enumerate(
        row_groups
    ):
        for col_index, cols in enumerate(
            col_groups
        ):
            pooled[
                row_index,
                col_index,
            ] = float(
                np.mean(
                    values[
                        np.ix_(
                            rows,
                            cols,
                        )
                    ]
                )
            )

    return np.clip(
        pooled,
        0.0,
        1.0,
    )


# --- frozen notebook cell 133 ---
def report_state_features(
    report,
    current_step,
    num_targets,
):
    if not isinstance(
        report,
        Report,
    ):
        raise TypeError(
            "report must be Report"
        )

    target_scale = max(
        1.0,
        float(
            num_targets
            - 1
        ),
    )
    age_fraction = float(
        np.clip(
            report_age_s(
                report,
                current_step,
            )
            / float(
                report.ttl_s
            ),
            0.0,
            1.0,
        )
    )
    delivery_fraction = float(
        np.clip(
            float(
                report.delivered_bytes
            )
            / max(
                1.0,
                float(
                    report.size_bytes
                ),
            ),
            0.0,
            1.0,
        )
    )

    return np.asarray(
        [
            1.0,
            np.clip(
                float(
                    report.target_id
                )
                / target_scale,
                0.0,
                1.0,
            ),
            age_fraction,
            delivery_fraction,
        ],
        dtype=np.float32,
    )


# --- frozen notebook cell 134 ---
def fixed_report_slots(
    reports,
    slot_count,
    current_step,
    num_targets,
):
    slot_count = int(
        slot_count
    )

    if slot_count < 0:
        raise ValueError(
            "slot_count must be >= 0"
        )

    features = np.zeros(
        (
            slot_count,
            4,
        ),
        dtype=np.float32,
    )

    ordered = sorted(
        reports,
        key=lambda report: (
            int(
                report.created_step
            ),
            int(
                report.target_id
            ),
        ),
    )

    for index, report in enumerate(
        ordered[:slot_count]
    ):
        features[index] = (
            report_state_features(
                report,
                current_step,
                num_targets,
            )
        )

    return features


# --- frozen notebook cell 135 ---
def build_global_state(
    uavs,
    targets,
    belief_maps,
    obstacles,
    report_buffers,
    pending_reports,
    gcs_received_target_ids,
    current_step,
    observations=None,
):
    values = []

    observation_keys = (
        "self_state",
        "gcs_relative",
        "neighbors",
        "obstacles",
        "belief_patch",
        "belief_coarse",
        "buffer",
        "destination_mask",
    )

    for uav_id in range(
        len(uavs)
    ):
        if observations is None:
            observation = (
                build_agent_observation(
                    uav_id,
                    uavs,
                    belief_maps,
                    obstacles,
                    report_buffers,
                    pending_reports,
                    current_step,
                )
            )
        else:
            observation = observations[
                f"uav_{uav_id}"
            ]

        for key in (
            observation_keys
        ):
            values.extend(
                np.asarray(
                    observation[
                        key
                    ],
                    dtype=np.float32,
                )
                .reshape(-1)
                .tolist()
            )

        buffer_slot_count = int(
            CONFIG[
                "buffer_bytes"
            ]
            // CONFIG[
                "report_bytes"
            ]
        )
        pending_slot_count = int(
            CONFIG[
                "pending_buffer_bytes"
            ]
            // CONFIG[
                "report_bytes"
            ]
        )

        values.extend(
            fixed_report_slots(
                report_buffers[
                    uav_id
                ],
                buffer_slot_count,
                current_step,
                len(targets),
            )
            .reshape(-1)
            .tolist()
        )

        source_pending = [
            report
            for report
            in pending_reports
            if int(
                report.source_uav
            )
            == int(
                uav_id
            )
        ]

        values.extend(
            fixed_report_slots(
                source_pending,
                pending_slot_count,
                current_step,
                len(targets),
            )
            .reshape(-1)
            .tolist()
        )

        values.extend(
            belief_map_summary(
                belief_maps[
                    uav_id
                ]
            ).tolist()
        )

        values.extend(
            pool_belief_map(
                belief_maps[
                    uav_id
                ]
            )
            .reshape(-1)
            .tolist()
        )

    map_size = max(
        1e-9,
        float(
            CONFIG[
                "map_size"
            ]
        ),
    )
    altitude_max = max(
        1e-9,
        float(
            CONFIG[
                "altitude_max"
            ]
        ),
    )

    for target in targets:
        values.extend(
            [
                np.clip(
                    target.position[0]
                    / map_size,
                    0.0,
                    1.0,
                ),
                np.clip(
                    target.position[1]
                    / map_size,
                    0.0,
                    1.0,
                ),
                float(
                    target.confirmed
                ),
                float(
                    target.id
                    in gcs_received_target_ids
                ),
            ]
        )

    for obstacle in obstacles:
        values.extend(
            [
                np.clip(
                    obstacle.position[0]
                    / map_size,
                    0.0,
                    1.0,
                ),
                np.clip(
                    obstacle.position[1]
                    / map_size,
                    0.0,
                    1.0,
                ),
                np.clip(
                    float(
                        obstacle.radius
                    )
                    / map_size,
                    0.0,
                    1.0,
                ),
                np.clip(
                    float(
                        obstacle.height
                    )
                    / altitude_max,
                    0.0,
                    1.0,
                ),
            ]
        )

    values.append(
        np.clip(
            float(
                current_step
            )
            / max(
                1.0,
                float(
                    CONFIG[
                        "max_steps"
                    ]
                ),
            ),
            0.0,
            1.0,
        )
    )

    return np.asarray(
        values,
        dtype=np.float32,
    )


# --- frozen notebook cell 136 ---
_CHECKPOINT_TYPE_KEY = "__uav_checkpoint_type__"


# --- frozen notebook cell 137 ---
def _checkpoint_pack(value):
    """Convert environment state to weights-only-safe checkpoint containers."""
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(
            value
        )
        return {
            _CHECKPOINT_TYPE_KEY: "ndarray",
            "dtype": array.dtype.str,
            "shape": list(
                array.shape
            ),
            "data": array.tobytes(
                order="C"
            ),
        }

    if isinstance(value, np.generic):
        return value.item()

    if (
        is_dataclass(value)
        and not isinstance(value, type)
    ):
        return {
            _CHECKPOINT_TYPE_KEY: "dataclass",
            "name": type(value).__name__,
            "fields": _checkpoint_pack(
                asdict(value)
            ),
        }

    if isinstance(value, dict):
        return {
            _CHECKPOINT_TYPE_KEY: "dict",
            "items": [
                [
                    _checkpoint_pack(key),
                    _checkpoint_pack(item),
                ]
                for key, item in value.items()
            ],
        }

    if isinstance(value, list):
        return {
            _CHECKPOINT_TYPE_KEY: "list",
            "items": [
                _checkpoint_pack(item)
                for item in value
            ],
        }

    if isinstance(value, tuple):
        return {
            _CHECKPOINT_TYPE_KEY: "tuple",
            "items": [
                _checkpoint_pack(item)
                for item in value
            ],
        }

    if isinstance(value, set):
        return {
            _CHECKPOINT_TYPE_KEY: "set",
            "items": [
                _checkpoint_pack(item)
                for item in sorted(
                    value,
                    key=repr,
                )
            ],
        }

    if value is None or isinstance(
        value,
        (bool, int, float, str),
    ):
        return value

    raise TypeError(
        "unsupported checkpoint value "
        f"{type(value).__name__}"
    )


# --- frozen notebook cell 138 ---
def _checkpoint_unpack(value):
    """Restore values produced by _checkpoint_pack."""
    if not isinstance(value, dict):
        return value

    kind = value.get(
        _CHECKPOINT_TYPE_KEY
    )

    if kind == "ndarray":
        data = value["data"]
        if not isinstance(
            data,
            (bytes, bytearray),
        ):
            raise TypeError(
                "checkpoint ndarray payload "
                "must be bytes-like"
            )
        array = np.frombuffer(
            data,
            dtype=np.dtype(
                value["dtype"]
            ),
        ).copy()
        return array.reshape(
            tuple(
                int(size)
                for size
                in value["shape"]
            )
        )

    if kind == "dict":
        return {
            _checkpoint_unpack(key): (
                _checkpoint_unpack(item)
            )
            for key, item
            in value["items"]
        }

    if kind == "list":
        return [
            _checkpoint_unpack(item)
            for item in value["items"]
        ]

    if kind == "tuple":
        return tuple(
            _checkpoint_unpack(item)
            for item in value["items"]
        )

    if kind == "set":
        return {
            _checkpoint_unpack(item)
            for item in value["items"]
        }

    if kind == "dataclass":
        classes = {
            "UAV": UAV,
            "Target": Target,
            "Obstacle": Obstacle,
            "Report": Report,
            "PeerTransferState": (
                PeerTransferState
            ),
        }
        class_name = value["name"]
        if class_name not in classes:
            raise ValueError(
                "unsupported checkpoint "
                f"dataclass {class_name!r}"
            )
        fields = _checkpoint_unpack(
            value["fields"]
        )
        return classes[class_name](
            **fields
        )

    raise ValueError(
        "invalid packed checkpoint value"
    )


# Export public and internal helpers required by downstream production layers.
__all__ = [name for name in globals() if not name.startswith("__")]
