from __future__ import annotations

import numpy as np

from config import CONFIG
from uav_marl.network.backends import SimpleNetworkBackend
from uav_marl.network.communication import (
    GCS_NODE,
    HybridAction,
    TransmissionIntent,
)
from uav_marl.world.entities import Report, UAV
from uav_marl.world.reward_energy import (
    apply_battery_limited_motion,
    apply_uav_energy_budget,
)


def test_failed_simple_tx_attempt_consumes_radio_energy():
    uavs = []
    for index in range(int(CONFIG["num_uavs"])):
        position = (
            np.array([5000.0, 5000.0, 100.0])
            if index == 0
            else np.array([10.0 * index, 0.0, 0.0])
        )
        uavs.append(
            UAV(
                index,
                position,
                np.zeros(3),
                float(CONFIG["battery_j"]),
                True,
            )
        )

    report_buffers = [[] for _ in uavs]
    report_buffers[0].append(
        Report(
            target_id=0,
            source_uav=0,
            created_step=0,
            size_bytes=int(CONFIG["report_bytes"]),
            ttl_s=float(CONFIG["report_ttl"]),
            delivered_bytes=0,
        )
    )

    backend = SimpleNetworkBackend()
    backend.reset(
        uavs=uavs,
        gcs_position=np.asarray(CONFIG["gcs_position"], dtype=float),
        obstacles=[],
        report_buffers=report_buffers,
        pending_reports=[],
        gcs_received_target_ids=set(),
    )
    request = TransmissionIntent(
        sender=0,
        recipient=GCS_NODE,
        target_id=0,
        requested_bytes=100_000,
        tx_power_w=float(CONFIG["tx_power_max_w"]),
    )

    result = backend.step(
        [request],
        dt=float(CONFIG["dt"]),
        current_step=1,
    )[0]
    energy_j = float(
        backend.last_step_communication_energy_by_uav()[0]
    )

    assert int(result.get("tx_bytes", 0)) == 0
    assert energy_j > 0.0


def test_battery_limited_motion_scales_segment_and_disables_tx_budget():
    uav = UAV(
        0,
        np.array([101.0, 100.0, 10.0]),
        np.array([1.0, 0.0, 0.0]),
        10.0,
        True,
    )
    action = HybridAction(
        movement=np.array([0.5, 0.0, 0.0]),
        destination=None,
        tx_power_w=0.0,
    )
    motion_result = {
        "realized_acceleration_mps2": np.array(
            [[1.0, 0.0, 0.0]]
        ),
        "blocked": np.array([False]),
        "blocked_by_peer": np.array([False]),
    }

    limited = apply_battery_limited_motion(
        [uav],
        [action],
        motion_result,
        positions_before=np.array([[100.0, 100.0, 10.0]]),
        velocities_before=np.array([[0.0, 0.0, 0.0]]),
        battery_before_j=np.array([10.0]),
        active_before=np.array([True]),
        dt=1.0,
    )

    fraction = float(limited["battery_fraction"][0])
    assert bool(limited["battery_limited"][0])
    assert 0.0 < fraction < 1.0
    assert float(limited["communication_budget_j"][0]) == 0.0
    assert 100.0 < float(uav.position[0]) < 101.0

    energy_result = apply_uav_energy_budget(
        [uav],
        [action],
        np.array([0.0]),
        dt=1.0,
        realized_accelerations_mps2=motion_result[
            "realized_acceleration_mps2"
        ],
        propulsion_energy_override_j=limited[
            "propulsion_charged_j"
        ],
    )

    assert float(energy_result["total_j"][0]) == 10.0
    assert abs(float(uav.battery_j)) <= 1e-12
    assert uav.active is False
