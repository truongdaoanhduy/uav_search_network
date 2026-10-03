"""Configuration composition and legacy compatibility.

New experiments use five independent config groups:
Task / Reward / Algorithm / Runtime / Experiment.

The notebook implementation still consumes the historical flat CONFIG mapping.
apply_to_legacy_config() is the compatibility bridge while the training engine
is gradually moved from the notebook into Python modules.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .algorithms import get_algorithm

_NETWORK_BACKENDS = {
    "simple",
    "packet_approx",
    "uavnetsim_gpu",
    "uavnetsim",
}
_PROVIDERS = {"kaggle", "vast", "local", "server"}


def to_plain_dict(cfg: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    if OmegaConf.is_config(cfg):
        value = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(value, dict):
            raise TypeError("resolved configuration must be a mapping")
        return value
    return deepcopy(dict(cfg))


def resolved_config_sha256(cfg: DictConfig | Mapping[str, Any]) -> str:
    plain = to_plain_dict(cfg)
    payload = json.dumps(
        plain,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derive_transition_budget(
    *,
    total_episodes: int,
    num_envs: int,
    max_steps: int,
) -> int:
    """Return a safe vectorized interaction ceiling for an episode budget.

    total_episodes remains the user-facing training budget. The returned
    transition count is only an internal hard ceiling used by the legacy engine.
    """
    total_episodes = int(total_episodes)
    num_envs = int(num_envs)
    max_steps = int(max_steps)
    if total_episodes < 1:
        raise ValueError("total_episodes must be >= 1")
    if num_envs < 1:
        raise ValueError("num_envs must be >= 1")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    vector_waves = math.ceil(total_episodes / num_envs)
    return int(vector_waves * num_envs * max_steps)


def _get(cfg: Mapping[str, Any], path: str) -> Any:
    current: Any = cfg
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(f"missing configuration field: {path}")
        current = current[part]
    return current


def validate_config(cfg: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    plain = to_plain_dict(cfg)
    get_algorithm(_get(plain, "algorithm.name"))

    seed = int(_get(plain, "seed"))
    if seed < 0:
        raise ValueError("seed must be >= 0")

    provider = str(_get(plain, "runtime.provider")).strip().lower()
    if provider not in _PROVIDERS:
        raise ValueError(
            f"runtime.provider must be one of {sorted(_PROVIDERS)}, got {provider!r}"
        )

    execution_mode = str(
        _get(plain, "runtime.execution_mode")
    ).strip().lower()
    if execution_mode not in {"cpu", "gpu", "auto"}:
        raise ValueError("runtime.execution_mode must be cpu, gpu, or auto")

    num_envs = int(_get(plain, "runtime.num_envs"))
    if num_envs < 1:
        raise ValueError("runtime.num_envs must be >= 1")

    max_gpus = int(_get(plain, "runtime.max_gpus"))
    if execution_mode == "gpu" and max_gpus < 1:
        raise ValueError("GPU runtime requires runtime.max_gpus >= 1")

    device = str(_get(plain, "runtime.device")).strip().lower()
    if execution_mode == "gpu":
        valid_device = (
            device in {"auto", "cuda"}
            or (
                device.startswith("cuda:")
                and device.split(":", 1)[1].isdigit()
            )
        )
        if not valid_device:
            raise ValueError(
                "GPU runtime.device must be auto, cuda, or cuda:<index>"
            )
        if (
            bool(_get(plain, "runtime.auto_multi_gpu"))
            and max_gpus > 1
            and device not in {"auto", "cuda", "cuda:0"}
        ):
            raise ValueError(
                "multi-GPU DDP requires runtime.device=cuda:0/auto; "
                "use CUDA_VISIBLE_DEVICES to remap physical GPUs"
            )

    multi_gpu = bool(_get(plain, "runtime.auto_multi_gpu")) and max_gpus > 1
    strategy = str(
        _get(plain, "runtime.multi_gpu_strategy")
    ).strip().lower()
    if multi_gpu and strategy != "ddp":
        raise ValueError(
            "episode-budget full-flow currently supports multi-GPU only "
            "with runtime.multi_gpu_strategy=ddp"
        )

    # Production GPU episode-budget runs use the same distributed worker for
    # world_size 1 and 2 so W&B episode/paper metrics stay identical. These
    # optimizer/compiler switches are not yet implemented in that worker.
    if execution_mode == "gpu" and _get(
        plain, "runtime.compile_mode"
    ) is not None:
        raise ValueError(
            "GPU episode-budget full-flow currently requires "
            "runtime.compile_mode=null"
        )
    if execution_mode == "gpu" and bool(
        _get(plain, "runtime.fused_adam")
    ):
        raise ValueError(
            "GPU episode-budget full-flow currently requires "
            "runtime.fused_adam=false"
        )
    train_freq = int(_get(plain, "algorithm.train_freq"))
    gradient_steps = int(_get(plain, "algorithm.gradient_steps"))
    if train_freq < 1:
        raise ValueError("algorithm.train_freq must be >= 1")
    if gradient_steps == 0 or gradient_steps < -1:
        raise ValueError(
            "algorithm.gradient_steps must be -1 or a positive integer"
        )

    backend = str(
        _get(plain, "runtime.network_backend")
    ).strip().lower()
    if backend not in _NETWORK_BACKENDS:
        raise ValueError(
            f"runtime.network_backend must be one of {sorted(_NETWORK_BACKENDS)}"
        )

    if backend == "uavnetsim" and num_envs != 1:
        raise ValueError(
            "packet-event UavNetSim training currently requires runtime.num_envs=1"
        )

    total_episodes = int(_get(plain, "experiment.total_episodes"))
    if total_episodes < 1:
        raise ValueError("experiment.total_episodes must be >= 1")

    max_steps = int(_get(plain, "task.scenario.max_steps"))
    if max_steps < 1:
        raise ValueError("task.scenario.max_steps must be >= 1")

    wandb_enabled = bool(_get(plain, "experiment.wandb.enabled"))
    wandb_mode = str(
        _get(plain, "experiment.wandb.mode")
    ).strip().lower()
    wandb_required = bool(_get(plain, "experiment.wandb.required"))
    if not wandb_enabled or not wandb_required or wandb_mode != "online":
        raise ValueError(
            "production full-flow requires experiment.wandb enabled=true, "
            "required=true, mode=online"
        )

    if not bool(_get(plain, "experiment.visualization.enabled")):
        raise ValueError(
            "production full-flow requires post-train visualization enabled"
        )
    if not bool(_get(plain, "experiment.visualization.required")):
        raise ValueError(
            "production full-flow requires visualization.required=true"
        )

    visualization_session = str(
        _get(plain, "experiment.visualization.session")
    ).strip().lower()
    if visualization_session not in {"separate_cpu", "same_session"}:
        raise ValueError(
            "experiment.visualization.session must be separate_cpu or same_session"
        )

    return plain


def _assign(
    target: dict[str, Any],
    source: Mapping[str, Any],
    mapping: Mapping[str, str],
) -> None:
    for legacy_key, path in mapping.items():
        target[legacy_key] = deepcopy(_get(source, path))


_TASK_MAP = {
    "map_size": "task.scenario.map_size",
    "num_uavs": "task.scenario.num_uavs",
    "num_targets": "task.scenario.num_targets",
    "num_obstacles": "task.scenario.num_obstacles",
    "altitude_min": "task.scenario.altitude_min",
    "altitude_max": "task.scenario.altitude_max",
    "max_speed": "task.scenario.max_speed",
    "max_accel": "task.scenario.max_accel",
    "dt": "task.scenario.dt",
    "max_steps": "task.scenario.max_steps",
    "gcs_position": "task.scenario.gcs_position",
    "gcs_exclusion_radius_m": "task.scenario.gcs_exclusion_radius_m",
    "target_exclusion_radius_m": "task.scenario.target_exclusion_radius_m",
    "battery_j": "task.scenario.battery_j",
    "grid_cell_m": "task.scenario.grid_cell_m",
    "safety_distance": "task.scenario.safety_distance",
    "obstacle_clearance_m": "task.scenario.obstacle_clearance_m",
    "launch_min_spacing_m": "task.scenario.launch_min_spacing_m",
    "launch_radius_m": "task.scenario.launch_radius_m",
    "obstacle_radius_min_m": "task.scenario.obstacle_radius_min_m",
    "obstacle_radius_max_m": "task.scenario.obstacle_radius_max_m",
    "obstacle_height_min_m": "task.scenario.obstacle_height_min_m",
    "obstacle_height_max_m": "task.scenario.obstacle_height_max_m",
    "pd": "task.sensing.pd",
    "pf": "task.sensing.pf",
    "altitude_anchors": "task.sensing.altitude_anchors",
    "belief_prior": "task.sensing.belief_prior",
    "confirmation_threshold": "task.sensing.confirmation_threshold",
    "fine_altitude": "task.sensing.fine_altitude",
    "verified_empty_belief": "task.sensing.verified_empty_belief",
    "camera_full_fov_deg": "task.sensing.camera_full_fov_deg",
    "sensing_obstacle_occlusion": "task.sensing.obstacle_occlusion",
    "report_bytes": "task.communication.report_bytes",
    "buffer_bytes": "task.communication.buffer_bytes",
    "pending_buffer_bytes": "task.communication.pending_buffer_bytes",
    "report_ttl": "task.communication.report_ttl",
    "peer_contact_range_m": "task.communication.peer_contact_range_m",
    "gcs_contact_range_m": "task.communication.gcs_contact_range_m",
    "tx_power_min_w": "task.communication.tx_power_min_w",
    "tx_power_max_w": "task.communication.tx_power_max_w",
    "comm_bandwidth_hz": "task.communication.bandwidth_hz",
    "comm_max_link_rate_bps": "task.communication.max_link_rate_bps",
    "comm_snr_threshold_db": "task.communication.snr_threshold_db",
    "comm_noise_power_dbm": "task.communication.noise_power_dbm",
    "comm_reference_gain_db": "task.communication.reference_gain_db",
    "comm_reference_distance_m": "task.communication.reference_distance_m",
    "comm_path_loss_exponent": "task.communication.path_loss_exponent",
    "comm_nlos_additional_loss_db": "task.communication.nlos_additional_loss_db",
    "uavnetsim_path": "task.communication.uavnetsim.path",
    "uavnetsim_mac_protocol": "task.communication.uavnetsim.mac_protocol",
    "uavnetsim_channel_mode": "task.communication.uavnetsim.channel_mode",
    "uavnetsim_los_model": "task.communication.uavnetsim.los_model",
    "uavnetsim_nlos_model": "task.communication.uavnetsim.nlos_model",
    "uavnetsim_payload_bytes": "task.communication.uavnetsim.payload_bytes",
    "uavnetsim_packet_lifetime_s": "task.communication.uavnetsim.packet_lifetime_s",
    "uavnetsim_max_queue_size": "task.communication.uavnetsim.max_queue_size",
    "uavnetsim_obstacle_polygon_sides": "task.communication.uavnetsim.obstacle_polygon_sides",
    "uavnetsim_carrier_frequency_hz": "task.communication.uavnetsim.carrier_frequency_hz",
    "uavnetsim_bandwidth_hz": "task.communication.uavnetsim.bandwidth_hz",
    "uavnetsim_bit_rate_bps": "task.communication.uavnetsim.bit_rate_bps",
    "uavnetsim_sinr_threshold_db": "task.communication.uavnetsim.sinr_threshold_db",
    "uavnetsim_cca_threshold_dbm": "task.communication.uavnetsim.cca_threshold_dbm",
    "uavnetsim_thermal_noise_density_dbm_hz": "task.communication.uavnetsim.thermal_noise_density_dbm_hz",
    "uavnetsim_receiver_noise_figure_db": "task.communication.uavnetsim.receiver_noise_figure_db",
    "uavnetsim_max_retransmission_attempt": "task.communication.uavnetsim.max_retransmission_attempt",
    "full_gpu_packet_ack_bits": "task.communication.gpu_surrogate.packet_ack_bits",
    "full_gpu_packet_sifs_us": "task.communication.gpu_surrogate.packet_sifs_us",
    "full_gpu_packet_slot_us": "task.communication.gpu_surrogate.packet_slot_us",
    "full_gpu_packet_cw_min": "task.communication.gpu_surrogate.packet_cw_min",
    "full_gpu_packet_difs_us": "task.communication.gpu_surrogate.packet_difs_us",
    "full_gpu_packet_contention_decay": "task.communication.gpu_surrogate.packet_contention_decay",
    "full_gpu_network_interference": "task.communication.gpu_surrogate.network_interference",
    "full_gpu_network_expected_backoff": "task.communication.gpu_surrogate.expected_backoff",
    "belief_patch_cells": "task.observation.belief_patch_cells",
    "belief_coarse_cells": "task.observation.belief_coarse_cells",
    "observation_nearest_obstacles": "task.observation.nearest_obstacles",
    "critic_belief_grid_cells": "task.observation.critic_belief_grid_cells",
    "energy_idle_power_w": "task.energy.idle_power_w",
    "energy_hover_power_w": "task.energy.hover_power_w",
    "energy_speed_sq_coeff": "task.energy.speed_sq_coeff",
    "energy_accel_sq_coeff": "task.energy.accel_sq_coeff",
}


_RUNTIME_MAP = {
    "full_gpu_execution_mode": "runtime.execution_mode",
    "full_gpu_device": "runtime.device",
    "full_gpu_auto_multi_gpu": "runtime.auto_multi_gpu",
    "full_gpu_max_gpus": "runtime.max_gpus",
    "full_gpu_multi_gpu_strategy": "runtime.multi_gpu_strategy",
    "full_gpu_num_envs": "runtime.num_envs",
    "training_auto_scale_gradient_steps": (
        "runtime.offpolicy.auto_scale_gradient_steps"
    ),
    "training_min_replay_sample_ratio": (
        "runtime.offpolicy.min_replay_sample_ratio"
    ),
    "training_num_envs": "runtime.num_envs",
    "training_vector_context": "runtime.vector.context",
    "training_vector_shared_memory": "runtime.vector.shared_memory",
    "training_overlap_env_and_updates": "runtime.vector.overlap_env_and_updates",
    "training_vector_backend": "runtime.vector.backend",
    "training_cuda_num_envs": "runtime.vector.cuda_num_envs",
    "training_torch_sensing_enabled": "runtime.torch_sensing.enabled",
    "training_torch_sensing_device": "runtime.torch_sensing.device",
    "training_torch_sensing_dtype": "runtime.torch_sensing.dtype",
    "training_torch_sensing_batch_timeout_s": "runtime.torch_sensing.batch_timeout_s",
    "training_torch_overlap_env_and_updates": "runtime.torch_sensing.overlap_env_and_updates",
    "training_amp_enabled": "runtime.amp.enabled",
    "training_amp_dtype": "runtime.amp.dtype",
    "full_gpu_benchmark_env_counts": "runtime.gpu.benchmark_env_counts",
    "full_gpu_benchmark_vector_steps": "runtime.gpu.benchmark_vector_steps",
    "full_gpu_world_candidates": "runtime.gpu.world_candidates",
    "full_gpu_world_sampling_rounds": "runtime.gpu.world_sampling_rounds",
}


_EXPERIMENT_MAP = {
    "training_log_interval_steps": "experiment.logging.log_interval_steps",
    "training_reward_curve_bin_episodes": "experiment.logging.reward_curve_bin_episodes",
    "training_paper_curve_bin_episodes": "experiment.logging.paper_curve_bin_episodes",
    "training_paper_curve_ema_beta": "experiment.logging.paper_curve_ema_beta",
    "training_eval_interval_steps": "experiment.evaluation.interval_steps",
    "training_eval_interval_episodes": "experiment.evaluation.interval_episodes",
    "training_eval_episodes": "experiment.evaluation.episodes",
    "training_eval_seed_offset": "experiment.evaluation.seed_offset",
    "training_inline_evaluation": "experiment.evaluation.inline",
    "training_inline_visualization": "experiment.visualization.inline",
    "training_auto_postprocess_visualization": "experiment.visualization.enabled",
    "training_auto_postprocess_required": "experiment.visualization.required",
    "training_postprocess_mode": "experiment.visualization.session",
    "training_auto_postprocess_render_video": "experiment.visualization.render_video",
    "training_auto_postprocess_log_wandb_media": "experiment.visualization.upload_media_to_wandb",
    "training_auto_postprocess_output_dir": "experiment.visualization.output_dir",
    "training_visualize_2d": "experiment.visualization.render_2d",
    "training_visualization_interval_episodes": "experiment.visualization.interval_episodes",
    "training_visualization_trajectory_stride": "experiment.visualization.trajectory_stride",
    "training_visualization_max_link_edges": "experiment.visualization.max_link_edges",
    "training_checkpoint_interval_steps": "experiment.checkpoint.interval_steps",
    "training_checkpoint_interval_episodes": "experiment.checkpoint.interval_episodes",
    "training_checkpoint_include_replay": "experiment.checkpoint.include_replay",
    "training_upload_periodic_checkpoints_wandb": "experiment.checkpoint.upload_periodic_to_wandb",
    "training_publish_final_checkpoint_wandb_artifact": "experiment.checkpoint.publish_final_wandb_artifact",
    "training_final_checkpoint_artifact_alias": "experiment.checkpoint.final_artifact_alias",
    "training_enable_wandb": "experiment.wandb.enabled",
    "training_wandb_entity": "experiment.wandb.entity",
    "training_wandb_project": "experiment.wandb.project",
    "training_wandb_mode": "experiment.wandb.mode",
    "training_wandb_required": "experiment.wandb.required",
    "training_wandb_require_online_gpu": "experiment.wandb.require_online_gpu",
    "training_wandb_disable_system_stats": "experiment.wandb.disable_system_stats",
    "training_require_kaggle_account_name": "experiment.wandb.require_kaggle_account_name",
    "training_wandb_name_format": "experiment.wandb.name_format",
    "postprocess_cpu_eval_episodes": "experiment.postprocess.eval_episodes",
    "postprocess_cpu_video_fps": "experiment.postprocess.video_fps",
    "postprocess_cpu_upload_huggingface": "experiment.postprocess.upload_huggingface",
    "training_auto_postprocess_upload_huggingface": "experiment.postprocess.upload_huggingface",
    "postprocess_hf_private_repo": "experiment.postprocess.hf_private_repo",
    "paper_plot_ema_beta": "experiment.paper_plot.ema_beta",
    "paper_plot_raw_alpha": "experiment.paper_plot.raw_alpha",
    "paper_plot_raw_linewidth": "experiment.paper_plot.raw_linewidth",
    "paper_plot_main_linewidth": "experiment.paper_plot.main_linewidth",
    "paper_plot_band_alpha": "experiment.paper_plot.band_alpha",
    "training_enable_csv": "experiment.outputs.csv",
    "training_enable_tensorboard": "experiment.outputs.tensorboard",
    "kaggle_wandb_secret_names": "experiment.secrets.kaggle_wandb_names",
    "kaggle_hf_token_secret_names": "experiment.secrets.kaggle_hf_token_names",
    "kaggle_hf_repo_secret_names": "experiment.secrets.kaggle_hf_repo_names",
}


def apply_to_legacy_config(
    legacy_config: dict[str, Any],
    cfg: DictConfig | Mapping[str, Any],
) -> dict[str, Any]:
    """Mutate and return the historical flat CONFIG used by the notebook."""
    plain = validate_config(cfg)
    algorithm = get_algorithm(_get(plain, "algorithm.name"))

    legacy_config["seed"] = int(_get(plain, "seed"))
    _assign(legacy_config, plain, _TASK_MAP)
    _assign(legacy_config, plain, _RUNTIME_MAP)
    _assign(legacy_config, plain, _EXPERIMENT_MAP)

    backend = str(_get(plain, "runtime.network_backend")).strip().lower()
    legacy_config["network_backend"] = backend
    if backend in {"simple", "packet_approx", "uavnetsim_gpu"}:
        legacy_config["full_gpu_network_model"] = backend

    reward = _get(plain, "reward")
    for key, value in reward.items():
        if key == "name":
            continue
        legacy_config[f"reward_{key}"] = deepcopy(value)

    algorithm_cfg = _get(plain, "algorithm")
    prefix = algorithm.legacy_prefix
    for key, value in algorithm_cfg.items():
        if key == "name":
            continue
        if key == "hidden_dims":
            value = tuple(value)
        legacy_config[f"{prefix}_{key}"] = deepcopy(value)

    # Preserve the historical key for older notebook helpers while making the
    # modern resolved config explicit and library-like.
    legacy_config[f"{prefix}_updates_per_step"] = int(
        algorithm_cfg["gradient_steps"]
    )

    legacy_config["training_output_dir"] = f"outputs/{algorithm.name}"
    if algorithm.name == "matd3":
        legacy_config["matd3_training_output_dir"] = "outputs/matd3"
        legacy_config["matd3_wandb_project"] = str(
            _get(plain, "experiment.wandb.project")
        )

    legacy_config["_runtime_provider"] = str(
        _get(plain, "runtime.provider")
    ).strip().lower()
    legacy_config["_experiment_name"] = str(
        _get(plain, "experiment.name")
    )
    legacy_config["_selected_algorithm"] = algorithm.name
    legacy_config["_target_episodes"] = int(
        _get(plain, "experiment.total_episodes")
    )
    legacy_config["_resolved_run_config"] = deepcopy(plain)
    legacy_config["_resolved_config_sha256"] = resolved_config_sha256(plain)

    return legacy_config


def describe_run(cfg: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    plain = validate_config(cfg)
    total_episodes = int(_get(plain, "experiment.total_episodes"))
    num_envs = int(_get(plain, "runtime.num_envs"))
    max_steps = int(_get(plain, "task.scenario.max_steps"))
    return {
        "algorithm": str(_get(plain, "algorithm.name")),
        "task": str(_get(plain, "task.name")),
        "reward": str(_get(plain, "reward.name")),
        "runtime_provider": str(_get(plain, "runtime.provider")),
        "execution_mode": str(_get(plain, "runtime.execution_mode")),
        "network_backend": str(_get(plain, "runtime.network_backend")),
        "num_envs": num_envs,
        "max_gpus": int(_get(plain, "runtime.max_gpus")),
        "seed": int(_get(plain, "seed")),
        "total_episodes": total_episodes,
        "max_steps": max_steps,
        "derived_transition_ceiling": derive_transition_budget(
            total_episodes=total_episodes,
            num_envs=num_envs,
            max_steps=max_steps,
        ),
        "config_sha256": resolved_config_sha256(plain),
    }
