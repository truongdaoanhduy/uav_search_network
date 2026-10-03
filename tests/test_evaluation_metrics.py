from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from uav_marl.algorithms.masac import (
    finalize_episode_diagnostics,
    new_episode_diagnostics,
    summarize_evaluation_results,
    update_episode_diagnostics,
)


def _fake_info(*, step: int, newly_confirmed=(), newly_delivered=(), total_j=0.0):
    zeros = np.zeros(2, dtype=bool)
    return {
        "step": step,
        "information_gain_bits": 0.0,
        "sensing_record_count": 0,
        "positive_observation_count": 0,
        "target_positive_observation_count": 0,
        "targets_in_fov_ids": [],
        "false_confirmation_count": 0,
        "expired_target_ids": [],
        "dropped_report_count": 0,
        "newly_confirmed_target_ids": list(newly_confirmed),
        "newly_delivered_target_ids": list(newly_delivered),
        "reward_components": {},
        "motion": {
            "blocked": zeros.copy(),
            "blocked_by_obstacle": zeros.copy(),
            "blocked_by_peer": zeros.copy(),
            "boundary_clipped": zeros.copy(),
            "horizontal_boundary_clipped": zeros.copy(),
            "altitude_clipped": zeros.copy(),
        },
        "energy": {
            "communication_j": np.zeros(2, dtype=np.float64),
            "total_j": np.full(2, total_j / 2.0, dtype=np.float64),
        },
        "communication_edges": [],
        "network_metrics": {},
    }


def _fake_env():
    uavs = [
        SimpleNamespace(
            position=np.array([0.0, 0.0, 0.0]),
            active=True,
            battery_j=1000.0,
        ),
        SimpleNamespace(
            position=np.array([100.0, 0.0, 0.0]),
            active=True,
            battery_j=1000.0,
        ),
    ]
    target = SimpleNamespace(
        id=0,
        position=np.array([250.0, 250.0, 0.0]),
        confirmed=False,
    )
    return SimpleNamespace(
        uavs=uavs,
        targets=[target],
        obstacles=[],
        num_targets=1,
        num_uavs=2,
        gcs_received_target_ids=set(),
    )


def test_cpu_evaluation_tracks_mission_latency_distance_and_efficiency():
    env = _fake_env()
    diagnostics = new_episode_diagnostics(env=env)

    env.uavs[0].position = np.array([3.0, 4.0, 0.0])
    env.targets[0].confirmed = True
    update_episode_diagnostics(
        diagnostics,
        _fake_info(step=1, newly_confirmed=[0], total_j=60.0),
        env=env,
    )

    env.gcs_received_target_ids.add(0)
    update_episode_diagnostics(
        diagnostics,
        _fake_info(step=3, newly_delivered=[0], total_j=40.0),
        env=env,
    )

    metrics = finalize_episode_diagnostics(
        diagnostics,
        env,
        episode_return=10.0,
        episode_length=3,
        success=True,
    )

    assert metrics["distance_total_m"] == 5.0
    assert metrics["first_confirmation_observed"] == 1.0
    assert metrics["all_targets_confirmed"] == 1.0
    assert metrics["first_delivery_observed"] == 1.0
    assert metrics["time_to_first_confirm_s"] == 1.0
    assert metrics["time_to_all_confirm_s"] == 1.0
    assert metrics["time_to_first_delivery_s"] == 3.0
    assert metrics["target_confirmation_to_delivery_latency_s"] == 2.0
    assert metrics["target_confirmation_to_delivery_latency_max_s"] == 2.0
    assert metrics["target_confirmation_to_delivery_latency_count"] == 1
    assert metrics["report_delivery_given_confirmation_rate_percent"] == 100.0
    assert metrics["energy_per_confirmed_target_j"] == 100.0
    assert metrics["energy_per_delivered_target_j"] == 100.0


def test_conditional_time_metrics_ignore_unobserved_episodes_but_keep_observed_rate():
    observed = {
        "success": 1.0,
        "return": 5.0,
        "length": 10,
        "first_confirmation_observed": 1.0,
        "all_targets_confirmed": 1.0,
        "first_delivery_observed": 1.0,
        "time_to_first_confirm_s": 4.0,
        "time_to_all_confirm_s": 8.0,
        "time_to_first_delivery_s": 9.0,
        "distance_total_m": 100.0,
        "report_delivery_given_confirmation_rate_percent": 100.0,
        "target_confirmation_to_delivery_latency_count": 1,
        "action_destination_silent_rate": 0.8,
        "action_destination_peer_rate": 0.1,
        "action_destination_gcs_rate": 0.1,
        "action_motion_saturation_rate": 0.9,
    }
    unobserved = {
        "success": 0.0,
        "return": -5.0,
        "length": 10,
        "first_confirmation_observed": 0.0,
        "all_targets_confirmed": 0.0,
        "first_delivery_observed": 0.0,
        "distance_total_m": 80.0,
        "report_delivery_given_confirmation_rate_percent": 0.0,
        "target_confirmation_to_delivery_latency_count": 0,
        "action_destination_silent_rate": 1.0,
        "action_destination_peer_rate": 0.0,
        "action_destination_gcs_rate": 0.0,
        "action_motion_saturation_rate": 1.0,
    }

    summary = summarize_evaluation_results([observed, unobserved])

    assert summary["first_confirmation_observed"] == 0.5
    assert summary["all_targets_confirmed"] == 0.5
    assert summary["first_delivery_observed"] == 0.5
    assert summary["time_to_first_confirm_s"] == 4.0
    assert summary["time_to_all_confirm_s"] == 8.0
    assert summary["time_to_first_delivery_s"] == 9.0
    assert summary["distance_total_m"] == 90.0
    assert summary["action_destination_silent_rate"] == 0.9
    assert summary["action_destination_peer_rate"] == 0.05
    assert summary["action_destination_gcs_rate"] == 0.05
    assert summary["action_motion_saturation_rate"] == 0.95
