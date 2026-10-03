from __future__ import annotations

from pathlib import Path

import pytest

from config import CONFIG
from uav_marl.training.gpu import (
    build_episode_curve_payload,
    build_paper_curve_payload,
    resolve_vector_gradient_steps,
)

ROOT = Path(__file__).resolve().parents[1]


def _episode(value: float = 1.0):
    return {
        "return": 10.0 * value,
        "length": 100.0,
        "end_step": 100.0,
        "end_reason_code": 3.0,
        "end_by_success": 0.0,
        "end_by_all_uavs_inactive": 0.0,
        "end_by_horizon": 1.0,
        "confirmed_targets": 2.0,
        "delivered_targets": 1.0,
        "episode_energy_j": 5000.0,
        "target_search_rate_percent": 4.0 * value,
        "target_delivery_rate_percent": 2.0 * value,
        "success": 0.0,
        "coverage_percent": 20.0 * value,
        "target_encounter_rate_percent": 10.0,
        "information_gain_bits": 30.0,
        "blocked_motion_rate": 0.2,
        "peer_safety_block_rate": 0.1,
        "obstacle_block_rate": 0.0,
        "boundary_clip_rate": 0.05,
        "false_confirmations": 0.0,
        "reports_created": 2.0,
        "expired_reports": 0.0,
        "dropped_reports": 0.0,
        "report_expiry_rate_percent": 0.0,
        "report_drop_rate_percent": 0.0,
        "report_delivery_given_confirmation_rate_percent": 50.0,
        "communication_energy_j": 2.0,
        "battery_remaining_mean_percent": 90.0,
        "depleted_uav_count": 0.0,
        "distance_total_m": 1000.0,
        "network_tx_attempts": 10.0,
        "network_phy_success_percent": 80.0,
        "network_tx_payload_success_ratio": 0.8,
        "network_throughput_kbps": 12.0,
        "network_nlos_attempt_rate_percent": 10.0,
        "gcs_in_range_uav_fraction": 0.5,
        "report_delivery_latency_s": 4.0,
        "report_delivery_latency_max_s": 6.0,
        "report_delivery_latency_sample_count": 1.0,
        "energy_per_confirmed_target_j": 2500.0,
        "energy_per_delivered_target_j": 5000.0,
        "time_to_first_confirm_s": 20.0,
        "time_to_all_confirm_s": None,
        "time_to_first_delivery_s": 40.0,
        "first_confirmation_observed": 1.0,
        "all_targets_confirmed": 0.0,
        "first_delivery_observed": 1.0,
        "reward_search": 1.0,
        "reward_communication": 2.0,
        "reward_safety": -3.0,
        "reward_energy": -0.1,
        "reward_mission": -1.0,
    }


def test_single_episode_metrics_are_raw_not_fake_seed_statistics():
    payload = build_episode_curve_payload(
        [_episode()],
        episodes_completed=1,
        episode_component_keys=(
            "reward_search",
            "reward_communication",
            "reward_safety",
            "reward_energy",
            "reward_mission",
        ),
    )
    assert payload["episode/return"] == 10.0
    assert payload["episode/coverage_percent"] == 20.0
    assert payload["episode/success"] == 0.0
    assert payload["episode/total_energy_j"] == 5000.0
    assert "episode/return_mean" not in payload
    assert "episode/return_std" not in payload
    assert "episode/coverage_percent_mean" not in payload
    assert "episode/coverage_percent_std" not in payload


def test_multi_episode_bins_are_explicitly_named_as_bin_means():
    payload = build_episode_curve_payload(
        [_episode(1.0), _episode(2.0)],
        episodes_completed=2,
        episode_component_keys=(),
    )
    assert payload["episode/return_bin_mean"] == 15.0
    assert payload["episode/coverage_percent_bin_mean"] == 30.0
    assert "episode/return" not in payload
    assert not any(key.endswith("_std") for key in payload)


def test_paper_curve_has_smoothed_display_metric_without_fake_std_fields():
    state = {}
    payload = build_paper_curve_payload(
        [_episode(), _episode(2.0)],
        episodes_completed=2,
        ema_state=state,
        ema_beta=0.85,
    )
    assert "paper/return" in payload
    assert "paper/coverage_percent" in payload
    assert "paper/return_window_mean" not in payload
    assert "paper/return_window_std" not in payload
    assert "paper/window_size" not in payload


def test_vector_gradient_steps_scale_for_4096_envs(monkeypatch):
    monkeypatch.setitem(CONFIG, "training_auto_scale_gradient_steps", True)
    monkeypatch.setitem(CONFIG, "training_min_replay_sample_ratio", 1.0)
    assert resolve_vector_gradient_steps(
        1,
        num_envs=4096,
        batch_size=256,
        train_freq=1,
    ) == 16
    assert resolve_vector_gradient_steps(
        1,
        num_envs=2,
        batch_size=256,
        train_freq=1,
    ) == 1
    assert resolve_vector_gradient_steps(
        -1,
        num_envs=4096,
        batch_size=256,
        train_freq=1,
    ) == -1


def test_vector_gradient_auto_scaling_can_be_disabled(monkeypatch):
    monkeypatch.setitem(CONFIG, "training_auto_scale_gradient_steps", False)
    assert resolve_vector_gradient_steps(
        3,
        num_envs=4096,
        batch_size=256,
        train_freq=1,
    ) == 3


def test_workspace_formatter_uses_episode_axis_paper_style_and_kaggle_column():
    source = (ROOT / "scripts" / "format_wandb_paper_view.py").read_text()
    assert '"episodes_completed"' in source
    assert 'title_x="Episode"' in source
    assert "smoothing_show_original=True" in source
    assert "smoothing_factor=0.95" in source
    assert '"summary:global_step"' not in source
    assert '"config:kaggle_account"' in source
    assert '"config:kaggle"' not in source
    assert '"config:execution_platform"' not in source
    assert '"episode/return"' in source
    assert '"episode/coverage_percent"' in source
    assert "episode/coverage_percent_mean" not in source
    assert "02 — Safety + Errors" in source
    assert "03 — Energy" in source
    assert "04 — Network" in source
    assert "05 — Episode + Mission Timing" in source


def test_actual_runs_table_formatter_locks_training_columns():
    source = (ROOT / "scripts" / "format_wandb_runs_table.py").read_text()
    assert 'DEFAULT_WORKSPACE = "Truongdaoanhduy\'s workspace"' in source
    assert '"config:kaggle_account.value"' in source
    assert '"summary:episodes_completed"' in source
    assert 'run_feed.lock_columns = True' in source
    assert "Config('run_role') = 'training'" in source


def test_wandb_run_roles_are_explicit():
    gpu_source = (ROOT / "uav_marl" / "training" / "gpu.py").read_text()
    cpu_source = (ROOT / "uav_marl" / "evaluation" / "postprocess.py").read_text()
    assert '"run_role": "training"' in gpu_source
    assert '"run_role": "cpu_evaluation"' in cpu_source


def test_cuda_graph_policy_runtime_flag_is_wired():
    config_source = (ROOT / "uav_marl" / "configuration.py").read_text()
    gpu_source = (ROOT / "uav_marl" / "training" / "gpu.py").read_text()
    assert '"training_cuda_graph_policy_actions": "runtime.cuda_graph_policy_actions"' in config_source
    assert 'class _MasacPolicyActionCudaGraph' in gpu_source
    assert 'torch.cuda.CUDAGraph()' in gpu_source
    assert 'self.graph.replay()' in gpu_source
    assert '"cuda_graph_policy_actions": bool(cuda_graph_enabled)' in gpu_source


def test_wandb_training_and_cpu_runs_expose_kaggle_account():
    gpu_source = (ROOT / "uav_marl" / "training" / "gpu.py").read_text()
    cpu_source = (ROOT / "uav_marl" / "evaluation" / "postprocess.py").read_text()
    assert '"kaggle_account": str(kaggle_account)' in gpu_source
    assert 'run.summary["kaggle_account"] = str(kaggle_account)' in gpu_source
    assert '"kaggle_account": resolve_kaggle_account()' in cpu_source
