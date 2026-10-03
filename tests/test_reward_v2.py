from __future__ import annotations

from pathlib import Path

import numpy as np

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
