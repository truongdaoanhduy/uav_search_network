"""Production source migrated from frozen fix_test_gpu.ipynb.

Do not import the notebook at runtime. The notebook is retained only as a
reference/regression artifact. Production changes belong in Python modules.
Migrated from notebook cells 118..124.
"""

from ..network.backends import *  # noqa: F401,F403

# --- frozen notebook cell 118 ---
def binary_entropy_bits(probability):
    probability = float(
        np.clip(
            probability,
            1e-12,
            1.0 - 1e-12,
        )
    )

    return float(
        -probability
        * np.log2(probability)
        - (1.0 - probability)
        * np.log2(
            1.0 - probability
        )
    )


# --- frozen notebook cell 119 ---
def sensing_information_gain_bits(
    sensing_records,
):
    entropy_reduction = 0.0

    for record in (
        sensing_records
    ):
        prior_entropy = (
            binary_entropy_bits(
                record[
                    "prior"
                ]
            )
        )
        posterior_entropy = (
            binary_entropy_bits(
                record[
                    "posterior"
                ]
            )
        )

        # Signed realized entropy reduction.
        # A later increase in uncertainty cancels earlier shaping
        # instead of allowing repeated positive-only reward.
        entropy_reduction += (
            prior_entropy
            - posterior_entropy
        )

    return float(
        entropy_reduction
    )


# --- frozen notebook cell 120 ---
def belief_certainty_potential(
    belief_maps,
):
    values = np.asarray(
        belief_maps,
        dtype=np.float64,
    )

    if values.ndim != 3:
        raise ValueError(
            "belief_maps must have shape "
            "(num_uavs, height, width)"
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

    return float(
        np.clip(
            1.0 - mean_entropy,
            0.0,
            1.0,
        )
    )


# --- frozen notebook cell 121 ---
def propulsion_energy_j(
    uav,
    hybrid_action,
    dt=None,
    realized_acceleration_mps2=None,
):
    if not isinstance(uav, UAV):
        raise TypeError(
            "uav must be UAV"
        )

    if not isinstance(
        hybrid_action,
        HybridAction,
    ):
        raise TypeError(
            "hybrid_action must be HybridAction"
        )

    if dt is None:
        dt = CONFIG["dt"]

    dt = float(dt)

    if (
        not np.isfinite(dt)
        or dt <= 0.0
    ):
        raise ValueError(
            "dt must be finite and > 0"
        )

    if not uav.active:
        return 0.0

    speed = float(
        np.linalg.norm(
            uav.velocity
        )
    )

    commanded_accel = float(
        np.linalg.norm(
            hybrid_action.movement
        )
        * float(
            CONFIG["max_accel"]
        )
    )

    if realized_acceleration_mps2 is None:
        acceleration_for_energy = (
            commanded_accel
        )
    else:
        realized_acceleration_mps2 = (
            np.asarray(
                realized_acceleration_mps2,
                dtype=np.float64,
            )
        )
        if realized_acceleration_mps2.shape != (3,):
            raise ValueError(
                "realized_acceleration_mps2 must "
                "have shape (3,)"
            )
        if not np.all(
            np.isfinite(
                realized_acceleration_mps2
            )
        ):
            raise ValueError(
                "realized_acceleration_mps2 must "
                "contain only finite values"
            )
        acceleration_for_energy = min(
            float(
                np.linalg.norm(
                    realized_acceleration_mps2
                )
            ),
            float(CONFIG["max_accel"]),
        )

    if (
        uav.position[2] <= 0.0
        and speed <= 1e-12
        and commanded_accel <= 1e-12
    ):
        power_w = float(
            CONFIG[
                "energy_idle_power_w"
            ]
        )
    else:
        power_w = (
            float(
                CONFIG[
                    "energy_hover_power_w"
                ]
            )
            + float(
                CONFIG[
                    "energy_speed_sq_coeff"
                ]
            )
            * speed
            * speed
            + float(
                CONFIG[
                    "energy_accel_sq_coeff"
                ]
            )
            * acceleration_for_energy
            * acceleration_for_energy
        )

    if (
        not np.isfinite(power_w)
        or power_w < 0.0
    ):
        raise ValueError(
            "computed propulsion power "
            "must be finite and >= 0"
        )

    return float(
        power_w * dt
    )


# --- frozen notebook cell 122 ---
def apply_uav_energy_budget(
    uavs,
    hybrid_actions,
    communication_energy_j,
    dt=None,
    realized_accelerations_mps2=None,
    propulsion_energy_override_j=None,
):
    if dt is None:
        dt = CONFIG["dt"]

    if len(uavs) != len(hybrid_actions):
        raise ValueError(
            "uavs and hybrid_actions must have the same length"
        )

    communication_energy_j = np.asarray(
        communication_energy_j,
        dtype=np.float64,
    ).copy()

    if communication_energy_j.shape != (len(uavs),):
        raise ValueError(
            "communication_energy_j has the wrong shape"
        )

    if (
        not np.all(np.isfinite(communication_energy_j))
        or np.any(communication_energy_j < 0.0)
    ):
        raise ValueError(
            "communication energy must be finite and >= 0"
        )

    if realized_accelerations_mps2 is not None:
        realized_accelerations_mps2 = np.asarray(
            realized_accelerations_mps2,
            dtype=np.float64,
        )
        if realized_accelerations_mps2.shape != (
            len(uavs),
            3,
        ):
            raise ValueError(
                "realized_accelerations_mps2 has the wrong shape"
            )
        if not np.all(
            np.isfinite(realized_accelerations_mps2)
        ):
            raise ValueError(
                "realized_accelerations_mps2 must contain only finite values"
            )

    if propulsion_energy_override_j is not None:
        propulsion_energy_override_j = np.asarray(
            propulsion_energy_override_j,
            dtype=np.float64,
        )
        if propulsion_energy_override_j.shape != (
            len(uavs),
        ):
            raise ValueError(
                "propulsion_energy_override_j has the wrong shape"
            )
        if (
            not np.all(
                np.isfinite(propulsion_energy_override_j)
            )
            or np.any(
                propulsion_energy_override_j < 0.0
            )
        ):
            raise ValueError(
                "propulsion override must be finite and >= 0"
            )

    propulsion = np.zeros(
        len(uavs),
        dtype=np.float64,
    )
    total = np.zeros(
        len(uavs),
        dtype=np.float64,
    )
    depleted_ids = []

    for index, (uav, hybrid_action) in enumerate(
        zip(uavs, hybrid_actions)
    ):
        if propulsion_energy_override_j is None:
            propulsion[index] = propulsion_energy_j(
                uav,
                hybrid_action,
                dt=dt,
                realized_acceleration_mps2=(
                    None
                    if realized_accelerations_mps2 is None
                    else realized_accelerations_mps2[index]
                ),
            )
        else:
            propulsion[index] = float(
                propulsion_energy_override_j[index]
            )

        available = max(
            0.0,
            float(uav.battery_j),
        )

        # Propulsion has priority because it has already determined the
        # physical trajectory for this decision step. Communication can
        # consume only what remains after that propulsion allocation.
        propulsion[index] = min(
            propulsion[index],
            available,
        )
        communication_energy_j[index] = min(
            communication_energy_j[index],
            max(
                0.0,
                available - propulsion[index],
            ),
        )
        total[index] = (
            propulsion[index]
            + communication_energy_j[index]
        )

        uav.battery_j = max(
            0.0,
            available - total[index],
        )

        if (
            uav.active
            and uav.battery_j <= 1e-12
        ):
            uav.active = False
            uav.velocity = np.zeros(
                3,
                dtype=np.float64,
            )
            depleted_ids.append(
                int(uav.id)
            )

    return {
        "propulsion_j": propulsion,
        "communication_j": communication_energy_j.copy(),
        "total_j": total,
        "depleted_uav_ids": depleted_ids,
    }


# --- frozen notebook cell 123 ---
def apply_battery_limited_motion(
    uavs,
    hybrid_actions,
    motion_result,
    positions_before,
    velocities_before,
    battery_before_j,
    active_before,
    dt=None,
):
    if dt is None:
        dt = CONFIG["dt"]
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and > 0")

    count = len(uavs)
    positions_before = np.asarray(
        positions_before,
        dtype=np.float64,
    )
    velocities_before = np.asarray(
        velocities_before,
        dtype=np.float64,
    )
    battery_before_j = np.asarray(
        battery_before_j,
        dtype=np.float64,
    )
    active_before = np.asarray(
        active_before,
        dtype=bool,
    )

    if positions_before.shape != (count, 3):
        raise ValueError("positions_before has the wrong shape")
    if velocities_before.shape != (count, 3):
        raise ValueError("velocities_before has the wrong shape")
    if battery_before_j.shape != (count,):
        raise ValueError("battery_before_j has the wrong shape")
    if active_before.shape != (count,):
        raise ValueError("active_before has the wrong shape")

    original_realized = np.asarray(
        motion_result["realized_acceleration_mps2"],
        dtype=np.float64,
    )
    if original_realized.shape != (count, 3):
        raise ValueError(
            "motion_result realized acceleration has the wrong shape"
        )

    full_positions = np.stack(
        [
            uav.position.copy()
            for uav in uavs
        ],
        axis=0,
    )
    full_velocities = np.stack(
        [
            uav.velocity.copy()
            for uav in uavs
        ],
        axis=0,
    )

    full_required = np.zeros(
        count,
        dtype=np.float64,
    )
    battery_fraction = np.ones(
        count,
        dtype=np.float64,
    )
    battery_limited = np.zeros(
        count,
        dtype=bool,
    )

    for index, (uav, hybrid_action) in enumerate(
        zip(uavs, hybrid_actions)
    ):
        if not active_before[index]:
            continue
        required = propulsion_energy_j(
            uav,
            hybrid_action,
            dt=dt,
            realized_acceleration_mps2=(
                original_realized[index]
            ),
        )
        available = max(
            0.0,
            float(battery_before_j[index]),
        )
        full_required[index] = required
        if required > available + 1e-12:
            fraction = (
                0.0
                if required <= 0.0
                else float(
                    np.clip(
                        available / required,
                        0.0,
                        1.0,
                    )
                )
            )
            battery_fraction[index] = fraction
            battery_limited[index] = True
            # The discrete-time model treats this as a uniformly scaled
            # segment over the same dt, then the UAV is inactive at the end.
            uav.position = (
                positions_before[index]
                + fraction
                * (
                    full_positions[index]
                    - positions_before[index]
                )
            )
            uav.velocity = (
                full_velocities[index]
                * fraction
            )

    # APF is the only collision-avoidance mechanism. Battery limiting may
    # shorten the realized segment, but it must not reintroduce the old
    # hard-stop peer collision resolver.
    blocked = np.asarray(
        motion_result.get(
            "blocked",
            np.zeros(count, dtype=bool),
        ),
        dtype=bool,
    ).copy()
    blocked_by_peer = np.asarray(
        motion_result.get(
            "blocked_by_peer",
            np.zeros(count, dtype=bool),
        ),
        dtype=bool,
    ).copy()

    final_velocities = np.stack(
        [
            uav.velocity.copy()
            for uav in uavs
        ],
        axis=0,
    )
    final_realized = (
        final_velocities
        - velocities_before
    ) / dt
    motion_result[
        "realized_acceleration_mps2"
    ] = final_realized
    motion_result["blocked"] = blocked
    motion_result[
        "blocked_by_peer"
    ] = blocked_by_peer

    propulsion_required = np.zeros(
        count,
        dtype=np.float64,
    )
    propulsion_charged = np.zeros(
        count,
        dtype=np.float64,
    )

    for index, (uav, hybrid_action) in enumerate(
        zip(uavs, hybrid_actions)
    ):
        if not active_before[index]:
            continue
        available = max(
            0.0,
            float(battery_before_j[index]),
        )
        final_required = propulsion_energy_j(
            uav,
            hybrid_action,
            dt=dt,
            realized_acceleration_mps2=(
                final_realized[index]
            ),
        )
        propulsion_required[index] = (
            final_required
        )
        if battery_limited[index]:
            propulsion_charged[index] = (
                available
            )
        elif final_required > (
            available + 1e-12
        ):
            # A newly blocked path can require a larger braking impulse than
            # the originally feasible segment. It consumes the remaining
            # battery and becomes inactive without re-introducing motion.
            battery_limited[index] = True
            battery_fraction[index] = 0.0
            propulsion_charged[index] = (
                available
            )
            uav.position = (
                positions_before[index].copy()
            )
            uav.velocity = np.zeros(
                3,
                dtype=np.float64,
            )
            final_realized[index] = (
                -velocities_before[index]
                / dt
            )
        else:
            propulsion_charged[index] = (
                final_required
            )

    motion_result[
        "realized_acceleration_mps2"
    ] = final_realized
    motion_result[
        "battery_limited"
    ] = battery_limited
    motion_result[
        "battery_fraction"
    ] = battery_fraction

    for index, uav in enumerate(uavs):
        if battery_limited[index]:
            uav.active = False
            uav.velocity = np.zeros(
                3,
                dtype=np.float64,
            )

    communication_budget_j = np.maximum(
        0.0,
        battery_before_j - propulsion_charged,
    )
    communication_budget_j[
        battery_limited
    ] = 0.0

    return {
        "propulsion_required_j": propulsion_required,
        "propulsion_charged_j": propulsion_charged,
        "communication_budget_j": communication_budget_j,
        "battery_limited": battery_limited,
        "battery_fraction": battery_fraction,
    }


# --- frozen notebook cell 124 ---
def transmission_intent_with_energy_budget(
    intent,
    energy_budget_j,
    dt,
    backend_name,
):
    if intent is None:
        return None
    if not isinstance(intent, TransmissionIntent):
        raise TypeError("intent must be TransmissionIntent or None")

    energy_budget_j = float(energy_budget_j)
    dt = float(dt)
    if (
        not np.isfinite(energy_budget_j)
        or energy_budget_j < 0.0
    ):
        raise ValueError(
            "energy_budget_j must be finite and >= 0"
        )
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and > 0")
    if energy_budget_j <= 0.0:
        return None

    tx_power_w = float(intent.tx_power_w)
    if tx_power_w <= 0.0:
        return None

    full_step_tx_energy = tx_power_w * dt
    if energy_budget_j >= full_step_tx_energy - 1e-12:
        return intent

    backend = str(backend_name).strip().lower()
    if backend in {"uavnetsim", "uav_net_sim"}:
        bit_rate_bps = float(
            CONFIG["uavnetsim_bit_rate_bps"]
        )
    else:
        bit_rate_bps = float(
            CONFIG["comm_max_link_rate_bps"]
        )

    affordable_duration_s = min(
        dt,
        energy_budget_j / tx_power_w,
    )
    max_bytes = int(
        np.floor(
            bit_rate_bps
            * affordable_duration_s
            / 8.0
            + 1e-9
        )
    )
    if max_bytes <= 0:
        return None
    if max_bytes >= int(intent.requested_bytes):
        return intent

    return TransmissionIntent(
        sender=int(intent.sender),
        recipient=int(intent.recipient),
        target_id=int(intent.target_id),
        requested_bytes=max_bytes,
        tx_power_w=tx_power_w,
    )


# Export public and internal helpers required by downstream production layers.
__all__ = [name for name in globals() if not name.startswith("__")]
