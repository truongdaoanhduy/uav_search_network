from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from config import CONFIG
from uav_marl.envs.uav_search import UAVSearchEnv
from uav_marl.world.entities import Report

ROOT = Path(__file__).resolve().parents[1]


def test_paper_v2_is_default_reward_profile():
    config_source = (ROOT / "configs" / "config.yaml").read_text()
    reward_source = (ROOT / "configs" / "reward" / "paper_v2.yaml").read_text()

    assert "- reward: paper_v2" in config_source
    assert "coverage_shaping: 5.0" in reward_source
    assert "communication_progress_shaping: 5.0" in reward_source
    assert "normalize_safety_by_uavs: true" in reward_source
    assert "confirmation: 30.0" in reward_source
    assert "blocked: 0.2" in reward_source
    assert "apf_intervention: 0.2" in reward_source
    assert "boundary: 0.1" in reward_source


def test_cpu_communication_progress_potential_tracks_partial_and_complete_delivery():
    env = UAVSearchEnv(backend_name="simple")
    try:
        env.reset(seed=44)
        report = Report(
            target_id=0,
            source_uav=0,
            created_step=0,
            size_bytes=int(CONFIG["report_bytes"]),
            ttl_s=float(CONFIG["report_ttl"]),
            delivered_bytes=int(CONFIG["report_bytes"]) // 2,
        )
        env.report_buffers[0].append(report)

        expected_partial = 0.5 / float(env.num_targets)
        assert np.isclose(
            env._communication_progress_potential(),
            expected_partial,
        )

        env.gcs_received_target_ids.add(0)
        expected_complete = 1.0 / float(env.num_targets)
        assert np.isclose(
            env._communication_progress_potential(),
            expected_complete,
        )
    finally:
        env.close()


def test_cpu_coverage_potential_and_reward_v2_components_are_active():
    env = UAVSearchEnv(backend_name="simple")
    try:
        observations, _ = env.reset(seed=44)
        for uav in env.uavs:
            uav.position[2] = 100.0

        observations = env._observations()
        actions = {}
        for agent_id, observation in observations.items():
            legal = np.flatnonzero(
                np.asarray(
                    observation["destination_mask"],
                    dtype=np.int8,
                )
            )
            actions[agent_id] = {
                "motion": np.zeros(3, dtype=np.float32),
                "destination": int(legal[0]),
                "power": np.array([-1.0], dtype=np.float32),
            }

        _, _, _, _, info = env.step(actions)
        components = info["reward_components"]

        assert env._coverage_potential() > 0.0
        assert components["coverage_shaping"] > 0.0
        assert "communication_progress" in components
        assert "information_gain" in components
        assert "blocked_motion" in components
        assert "boundary" in components
    finally:
        env.close()


def test_gpu_reward_v2_source_has_matching_potentials_and_normalized_safety():
    source = (ROOT / "uav_marl" / "envs" / "gpu.py").read_text()

    assert "def _coverage_potential(self):" in source
    assert "def _communication_progress_potential(self):" in source
    assert '"reward_coverage_shaping"' in source
    assert '"reward_communication_progress_shaping"' in source
    assert '"reward_normalize_safety_by_uavs"' in source
    assert "reward_coverage_shaping" in source
    assert "reward_communication_progress" in source
    assert '"reward_apf_intervention"' in source
    assert 'motion["apf_active"]' in source


def test_cpu_reward_penalizes_apf_intervention_when_legacy_blocked_is_false(monkeypatch):
    monkeypatch.setitem(CONFIG, "apf_enabled", True)
    monkeypatch.setitem(CONFIG, "reward_apf_intervention", 0.2)
    env = UAVSearchEnv(backend_name="simple")
    try:
        env.reset(seed=44)
        for obstacle in env.obstacles:
            obstacle.position[:] = np.asarray([2800.0, 2800.0], dtype=np.float64)
            obstacle.radius = 80.0
            obstacle.height = 30.0
        positions = np.asarray(
            [
                [100.0, 100.0, 50.0],
                [135.0, 100.0, 50.0],
                [800.0, 800.0, 50.0],
                [1200.0, 800.0, 50.0],
                [1600.0, 800.0, 50.0],
                [2000.0, 800.0, 50.0],
            ],
            dtype=np.float64,
        )
        for uav, position in zip(env.uavs, positions):
            uav.position[:] = position
            uav.velocity[:] = 0.0
            uav.active = True

        observations = env._observations()
        actions = {}
        for index, agent_id in enumerate(env.agent_ids):
            legal = np.flatnonzero(
                np.asarray(
                    observations[agent_id]["destination_mask"],
                    dtype=np.int8,
                )
            )
            motion = np.zeros(3, dtype=np.float32)
            if index == 0:
                motion[0] = 1.0
            elif index == 1:
                motion[0] = -1.0
            actions[agent_id] = {
                "motion": motion,
                "destination": int(legal[0]),
                "power": np.array([-1.0], dtype=np.float32),
            }

        _, _, _, _, info = env.step(actions)

        assert not np.any(info["motion"]["blocked"])
        assert np.all(info["motion"]["apf_peer_active"][:2])
        assert info["reward_components"]["apf_intervention"] < 0.0
    finally:
        env.close()


def test_cpu_investigation_potential_rewards_descending_near_high_belief(monkeypatch):
    monkeypatch.setitem(CONFIG, "reward_investigation_belief_threshold", 0.8)
    env = UAVSearchEnv(backend_name="simple")
    try:
        env.reset(seed=44)
        for belief in env.belief_maps:
            belief.fill(float(CONFIG["belief_prior"]))
        for uav in env.uavs:
            uav.active = True
            uav.position[:] = np.array([1500.0, 1500.0, 150.0])
            uav.velocity[:] = 0.0

        gx = int(env.uavs[0].position[0] // float(CONFIG["grid_cell_m"]))
        gy = int(env.uavs[0].position[1] // float(CONFIG["grid_cell_m"]))
        env.belief_maps[0][gy, gx] = 0.9

        high_altitude = env._investigation_potential()
        env.uavs[0].position[2] = 100.0
        mid_altitude = env._investigation_potential()
        env.uavs[0].position[2] = float(CONFIG["fine_altitude"])
        fine_altitude = env._investigation_potential()

        assert 0.0 < high_altitude < mid_altitude < fine_altitude
        assert fine_altitude <= 1.0
    finally:
        env.close()


def test_gpu_investigation_potential_matches_altitude_progress(monkeypatch):
    import torch

    from uav_marl.envs.gpu import FullGpuUAVBatchEnv

    monkeypatch.setitem(CONFIG, "reward_investigation_belief_threshold", 0.8)
    env = FullGpuUAVBatchEnv(1, device="cpu", seed=44, strict_cuda=False)
    try:
        env.active.zero_()
        env.active[0, 0] = True
        env.positions[0, 0] = torch.tensor([1500.0, 1500.0, 150.0])
        env.velocities.zero_()
        env.belief.fill_(float(CONFIG["belief_prior"]))
        gx = int(env.positions[0, 0, 0].item() // env.cell_size)
        gy = int(env.positions[0, 0, 1].item() // env.cell_size)
        env.belief[0, 0, gy, gx] = 0.9

        high_altitude = float(env._investigation_potential()[0])
        env.positions[0, 0, 2] = 100.0
        mid_altitude = float(env._investigation_potential()[0])
        env.positions[0, 0, 2] = float(CONFIG["fine_altitude"])
        fine_altitude = float(env._investigation_potential()[0])

        assert 0.0 < high_altitude < mid_altitude < fine_altitude
        assert fine_altitude <= 1.0
    finally:
        env.close()


def test_cpu_reward_adds_positive_investigation_shaping_when_descending_near_evidence(monkeypatch):
    monkeypatch.setitem(CONFIG, "reward_investigation_shaping", 10.0)
    monkeypatch.setitem(CONFIG, "reward_investigation_belief_threshold", 0.8)
    env = UAVSearchEnv(backend_name="simple")
    try:
        observations, _ = env.reset(seed=44)
        for belief in env.belief_maps:
            belief.fill(float(CONFIG["belief_prior"]))
        env.uavs[0].position[2] = 100.0
        gx = int(env.uavs[0].position[0] // float(CONFIG["grid_cell_m"]))
        gy = int(env.uavs[0].position[1] // float(CONFIG["grid_cell_m"]))
        high_gx = min(env.belief_maps.shape[-1] - 1, gx + 6)
        env.belief_maps[0][gy, high_gx] = 0.9

        observations = env._observations()
        actions = {}
        for index, (agent_id, observation) in enumerate(observations.items()):
            legal = np.flatnonzero(np.asarray(observation["destination_mask"], dtype=np.int8))
            motion = np.zeros(3, dtype=np.float32)
            if index == 0:
                motion[2] = -1.0
            actions[agent_id] = {
                "motion": motion,
                "destination": int(legal[0]),
                "power": np.array([-1.0], dtype=np.float32),
            }

        _, _, _, _, info = env.step(actions)

        assert "investigation_shaping" in info["reward_components"]
        assert info["reward_components"]["investigation_shaping"] > 0.0
    finally:
        env.close()


def test_gpu_reward_adds_positive_investigation_shaping_when_descending_near_evidence(monkeypatch):
    import torch

    from uav_marl.envs.gpu import FullGpuUAVBatchEnv

    monkeypatch.setitem(CONFIG, "reward_investigation_shaping", 10.0)
    monkeypatch.setitem(CONFIG, "reward_investigation_belief_threshold", 0.8)
    env = FullGpuUAVBatchEnv(1, device="cpu", seed=44, strict_cuda=False)
    try:
        env.belief.fill_(float(CONFIG["belief_prior"]))
        env.positions[0, 0, 2] = 100.0
        gx = int(env.positions[0, 0, 0].item() // env.cell_size)
        gy = int(env.positions[0, 0, 1].item() // env.cell_size)
        high_gx = min(env.grid_n - 1, gx + 6)
        env.belief[0, 0, gy, high_gx] = 0.9

        continuous = torch.zeros((1, env.num_uavs, env.continuous_dim))
        continuous[0, 0, 2] = -1.0
        destination = torch.zeros((1, env.num_uavs), dtype=torch.long)

        *_, metrics = env.step(continuous, destination)

        assert float(metrics["reward_investigation_shaping"][0]) > 0.0
    finally:
        env.close()


def test_evaluation_summary_keeps_new_reward_decomposition_components():
    from uav_marl.algorithms.masac import summarize_evaluation_results

    row = {
        "success": 0.0,
        "return": -1.0,
        "length": 10,
        "reward_component_coverage_shaping": 0.4,
        "reward_component_investigation_shaping": 0.3,
        "reward_component_communication_progress": 0.2,
        "reward_component_apf_intervention": -0.1,
    }
    summary = summarize_evaluation_results([row])

    assert summary["reward_component_coverage_shaping"] == pytest.approx(0.4)
    assert summary["reward_component_investigation_shaping"] == pytest.approx(0.3)
    assert summary["reward_component_communication_progress"] == pytest.approx(0.2)
    assert summary["reward_component_apf_intervention"] == pytest.approx(-0.1)


def test_cpu_investigation_shaping_does_not_reward_standing_still(monkeypatch):
    monkeypatch.setitem(CONFIG, "reward_investigation_shaping", 10.0)
    monkeypatch.setitem(CONFIG, "reward_investigation_belief_threshold", 0.8)
    env = UAVSearchEnv(backend_name="simple")
    try:
        observations, _ = env.reset(seed=44)
        for belief in env.belief_maps:
            belief.fill(float(CONFIG["belief_prior"]))
        env.uavs[0].position[2] = 100.0
        env.uavs[0].velocity[:] = 0.0
        gx = int(env.uavs[0].position[0] // float(CONFIG["grid_cell_m"]))
        gy = int(env.uavs[0].position[1] // float(CONFIG["grid_cell_m"]))
        high_gx = min(env.belief_maps.shape[-1] - 1, gx + 6)
        env.belief_maps[0][gy, high_gx] = 0.9

        observations = env._observations()
        actions = {}
        for agent_id, observation in observations.items():
            legal = np.flatnonzero(np.asarray(observation["destination_mask"], dtype=np.int8))
            actions[agent_id] = {
                "motion": np.zeros(3, dtype=np.float32),
                "destination": int(legal[0]),
                "power": np.array([-1.0], dtype=np.float32),
            }

        _, _, _, _, info = env.step(actions)

        assert info["reward_components"]["investigation_shaping"] <= 0.0
    finally:
        env.close()


def test_cpu_early_all_inactive_pays_remaining_horizon_step_cost(monkeypatch):
    monkeypatch.setitem(CONFIG, "reward_step_penalty", 0.01)
    env = UAVSearchEnv(backend_name="simple")
    try:
        observations, _ = env.reset(seed=44)
        for uav in env.uavs:
            uav.battery_j = 1.0
            uav.velocity[:] = 0.0

        observations = env._observations()
        actions = {}
        for agent_id, observation in observations.items():
            legal = np.flatnonzero(
                np.asarray(observation["destination_mask"], dtype=np.int8)
            )
            actions[agent_id] = {
                "motion": np.zeros(3, dtype=np.float32),
                "destination": int(legal[0]),
                "power": np.array([-1.0], dtype=np.float32),
            }

        _, _, terminated, _, info = env.step(actions)

        assert terminated
        assert not info["success"]
        assert not any(uav.active for uav in env.uavs)
        remaining_steps = max(0, int(CONFIG["max_steps"]) - int(info["step"]))
        assert info["reward_components"]["inactive_horizon_makeup"] == pytest.approx(
            -float(CONFIG["reward_step_penalty"]) * remaining_steps
        )
    finally:
        env.close()


def test_gpu_early_all_inactive_pays_remaining_horizon_step_cost(monkeypatch):
    import torch

    from uav_marl.envs.gpu import FullGpuUAVBatchEnv

    monkeypatch.setitem(CONFIG, "reward_step_penalty", 0.01)
    env = FullGpuUAVBatchEnv(1, device="cpu", seed=44, strict_cuda=False)
    try:
        env.battery.fill_(1.0)
        env.velocities.zero_()
        continuous = torch.zeros((1, env.num_uavs, env.continuous_dim))
        destination = torch.zeros((1, env.num_uavs), dtype=torch.long)

        _, _, _, terminated, _, metrics = env.step(continuous, destination)

        assert bool(terminated[0])
        assert not bool(metrics["success"][0])
        assert not bool(env.active[0].any())
        remaining_steps = max(0, int(CONFIG["max_steps"]) - int(env.step_count[0]))
        assert float(metrics["reward_inactive_horizon_makeup"][0]) == pytest.approx(
            -float(CONFIG["reward_step_penalty"]) * remaining_steps,
            abs=1e-5,
        )
    finally:
        env.close()


def test_cpu_investigation_potential_rewards_approaching_high_belief_without_oracle(monkeypatch):
    monkeypatch.setitem(CONFIG, "reward_investigation_belief_threshold", 0.8)
    env = UAVSearchEnv(backend_name="simple")
    try:
        env.reset(seed=44)
        for belief in env.belief_maps:
            belief.fill(float(CONFIG["belief_prior"]))
        for uav in env.uavs:
            uav.active = True
            uav.position[:] = np.array([1500.0, 1500.0, 150.0])
            uav.velocity[:] = 0.0

        gx = int(env.uavs[0].position[0] // float(CONFIG["grid_cell_m"]))
        gy = int(env.uavs[0].position[1] // float(CONFIG["grid_cell_m"]))
        evidence_gx = min(env.belief_maps.shape[-1] - 1, gx + 5)
        env.belief_maps[0][gy, evidence_gx] = 0.9

        far = env._investigation_potential()
        env.uavs[0].position[0] += 75.0
        closer = env._investigation_potential()

        assert closer > far > 0.0
    finally:
        env.close()


def test_gpu_investigation_potential_rewards_approaching_high_belief_without_oracle(monkeypatch):
    import torch

    from uav_marl.envs.gpu import FullGpuUAVBatchEnv

    monkeypatch.setitem(CONFIG, "reward_investigation_belief_threshold", 0.8)
    env = FullGpuUAVBatchEnv(1, device="cpu", seed=44, strict_cuda=False)
    try:
        env.active.zero_()
        env.active[0, 0] = True
        env.positions[0, 0] = torch.tensor([1500.0, 1500.0, 150.0])
        env.velocities.zero_()
        env.belief.fill_(float(CONFIG["belief_prior"]))
        gx = int(env.positions[0, 0, 0].item() // env.cell_size)
        gy = int(env.positions[0, 0, 1].item() // env.cell_size)
        evidence_gx = min(env.grid_n - 1, gx + 5)
        env.belief[0, 0, gy, evidence_gx] = 0.9

        far = float(env._investigation_potential()[0])
        env.positions[0, 0, 0] += 75.0
        closer = float(env._investigation_potential()[0])

        assert closer > far > 0.0
    finally:
        env.close()


def test_cpu_evaluation_reward_groups_include_new_shaping_and_failure_terms():
    source = (Path(__file__).resolve().parents[1] / "uav_marl" / "algorithms" / "masac.py").read_text()
    search_block = source[source.index("    reward_search = ("):source.index("    reward_communication = (")]
    safety_block = source[source.index("    reward_safety = ("):source.index("    reward_energy = ")]
    mission_block = source[source.index("    reward_mission = ("):source.index("\n\n    trace = {")]

    assert '"investigation_shaping"' in search_block
    assert '"apf_intervention"' in safety_block
    assert '"inactive_horizon_makeup"' in mission_block
