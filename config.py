# LEGACY COMPATIBILITY CONFIG. New experiments should use train.py + configs/.
# The composable runner resolves Task/Reward/Algorithm/Runtime/Experiment and
# applies resolved values to this dictionary before constructing production Python modules.
CONFIG = {
    "seed": 44,

    "map_size": 3000,
    "num_uavs": 6,
    "num_targets": 50,
    "num_obstacles": 30,

    "altitude_min": 0,
    "altitude_max": 150,
    "max_speed": 20,
    "max_accel": 2.0,

    "dt": 1,
    "max_steps": 1000,

    "gcs_position": [1500, 0, 0.0],
    "gcs_exclusion_radius_m": 150,
    "target_exclusion_radius_m": 250,

    "battery_j": 277200,

    "grid_cell_m": 25,

    "pd": [0.99, 0.90, 0.80, 0.70],
    "pf": [0.01, 0.10, 0.20, 0.30],
    "altitude_anchors": [0.0, 50.0, 100.0, 150.0],

    "belief_prior": 0.5,
    "confirmation_threshold": 0.99,
    "fine_altitude": 50.0,
    "verified_empty_belief": 0.01,

    "safety_distance": 30,
    "obstacle_clearance_m": 30.0,
    # Risk-aware APF safety layer. Influence distances are derived each step
    # from relative closing speed and braking capability.
    "apf_enabled": True,
    "apf_soft_gain": 1.5,
    "apf_lookahead_s": 3.0,
    "apf_braking_margin": 1.5,
    "launch_min_spacing_m": 60.0,

    "report_bytes": 1_000_000,
    "buffer_bytes": 3_000_000,
    "pending_buffer_bytes": 3_000_000,
    "report_ttl": 300.0,

    "camera_full_fov_deg": 90,
    "sensing_obstacle_occlusion": True,

    "launch_radius_m": 300,

    "obstacle_radius_min_m": 50.0,
    "obstacle_radius_max_m": 130.0,
    "obstacle_height_min_m": 30.0,
    "obstacle_height_max_m": 120.0,

    "peer_contact_range_m": 500,
    "gcs_contact_range_m": 500,

    "tx_power_min_w": 0.1,
    "tx_power_max_w": 0.4,

    "comm_bandwidth_hz": 10_000_000.0,
    "comm_max_link_rate_bps": 2_000_000.0,
    "comm_snr_threshold_db": 6.0,
    "comm_noise_power_dbm": -110.0,
    "comm_reference_gain_db": -70.0,
    "comm_reference_distance_m": 1.0,
    "comm_path_loss_exponent": 2.0,
    "comm_nlos_additional_loss_db": 19.0,

    # Network backend used by the future environment.
    "network_backend": "simple",

    # UavNetSim backend (optional external simulator).
    # Leave path as None and pass repo_path=... to UavNetSimBackend,
    # or set the UAVNETSIM_PATH environment variable.
    "uavnetsim_path": None,
    "uavnetsim_mac_protocol": "CSMA_CA",
    "uavnetsim_channel_mode": "a2a",
    "uavnetsim_los_model": "free_space",
    "uavnetsim_nlos_model": "urban",
    "uavnetsim_payload_bytes": 1024,
    "uavnetsim_packet_lifetime_s": 10.0,
    "uavnetsim_max_queue_size": 200,
    "uavnetsim_obstacle_polygon_sides": 16,
    # Pin the radio/MAC constants used by the installed UavNetSim version so
    # the CPU validator and CUDA training surrogate stay on the same contract.
    "uavnetsim_carrier_frequency_hz": 2_400_000_000.0,
    "uavnetsim_bandwidth_hz": 22_000_000.0,
    "uavnetsim_bit_rate_bps": 2_000_000.0,
    "uavnetsim_sinr_threshold_db": 6.0,
    "uavnetsim_cca_threshold_dbm": -82.0,
    "uavnetsim_thermal_noise_density_dbm_hz": -174.0,
    "uavnetsim_receiver_noise_figure_db": 7.0,
    "uavnetsim_max_retransmission_attempt": 5,

    # GPU-native training approximation of UavNetSim's packet/MAC costs.
    # "simple" preserves the original tensor link model. "packet_approx"
    # adds packet queue limits and expected CSMA/CA airtime. "uavnetsim_gpu"
    # additionally uses UavNetSim's A2A path-gain law, simultaneous-link SINR
    # interference, ACK/backoff service cost, queue limits, relay custody and
    # GCS delivery, all as batched CUDA tensor math. It deliberately remains a
    # deterministic time-slotted surrogate rather than pretending SimPy's
    # event scheduler itself has been ported to CUDA. Use real UavNetSim for
    # packet-event validation and protocol-library experiments.
    "full_gpu_network_model": "uavnetsim_gpu",
    "full_gpu_packet_ack_bits": 240,
    "full_gpu_packet_sifs_us": 10.0,
    "full_gpu_packet_slot_us": 20.0,
    "full_gpu_packet_cw_min": 31,
    "full_gpu_packet_difs_us": 50.0,
    # Empirical shared-channel efficiency per additional simultaneous sender.
    # 0.8 matches the current UavNetSim CSMA/CA calibration reasonably well
    # for 1-4 concurrent GCS uplinks in the fixed seed-44 validation scene.
    "full_gpu_packet_contention_decay": 0.8,
    # Use exact simultaneous-transmitter SINR for the richer GPU surrogate.
    "full_gpu_network_interference": True,
    # Keep the deterministic expected-backoff model reproducible across runs.
    "full_gpu_network_expected_backoff": True,

    # Environment observation.
    "belief_patch_cells": 13,
    "belief_coarse_cells": 10,
    "observation_nearest_obstacles": 3,
    "critic_belief_grid_cells": 10,

    # Mission-level energy model. These are project baseline values,
    # not UavNetSim propulsion parameters.
    "energy_idle_power_w": 20.0,
    "energy_hover_power_w": 120.0,
    "energy_speed_sq_coeff": 0.5,
    "energy_accel_sq_coeff": 2.0,

    # Cooperative team reward (paper_v2).
    # Potential-based shaping uses gamma*Phi(s') - Phi(s).
    "reward_info_gain": 10.0,
    "reward_coverage_shaping": 5.0,
    "reward_communication_progress_shaping": 5.0,
    "reward_shaping_gamma": 0.99,
    "reward_confirmation": 30.0,
    "reward_delivery": 50.0,
    "reward_false_confirmation": 5.0,
    "reward_blocked": 0.2,
    "reward_boundary": 0.1,
    "reward_normalize_safety_by_uavs": True,
    "reward_expired_report": 10.0,
    "reward_dropped_report": 10.0,
    "reward_energy_per_kj": 0.01,
    "reward_step_penalty": 0.01,
    "reward_all_delivered_bonus": 500.0,

    # Hybrid multi-agent SAC (CTDE).
    "masac_hidden_dims": (256, 256),
    "masac_replay_capacity": 50_000,
    "masac_batch_size": 256,
    "masac_gamma": 0.99,
    "masac_tau": 0.005,
    "masac_actor_lr": 3e-4,
    "masac_critic_lr": 3e-4,
    "masac_alpha_lr": 3e-4,
    "masac_log_std_min": -5.0,
    "masac_log_std_max": 2.0,
    "masac_gumbel_temperature": 1.0,
    # Multiplier for SAC continuous target entropy -|A|.
    "masac_continuous_target_entropy_scale": 1.0,
    "masac_discrete_target_entropy_ratio": 0.98,
    "masac_initial_alpha_continuous": 0.2,
    "masac_initial_alpha_discrete": 0.2,
    "masac_learning_starts": 2_000,
    # Library-style off-policy schedule: collect one vector step, then one
    # gradient step. Set gradient_steps=-1 only when intentionally matching
    # gradient updates to the number of newly collected transitions.
    "masac_train_freq": 1,
    "masac_gradient_steps": 1,
    "masac_updates_per_step": 1,  # deprecated compatibility alias
    "masac_gradient_clip_norm": 10.0,

    # Training experiment infrastructure.
    "training_output_dir": "outputs/masac",
    # Laptop profile: 2 CPU workers slightly beat 4 with lower memory use.
    "training_num_envs": 2,
    "training_vector_context": "fork",
    "training_vector_shared_memory": True,
    "training_overlap_env_and_updates": True,

    # Default: parallel CPU simulation plus batched CUDA actor/critic work.
    # GPU sensing remains opt-in: measured T4 runs were slower with it.
    # Explicit "torch_threaded" also enables that experimental collector.
    "training_vector_backend": "auto",
    "training_cuda_num_envs": 4,
    "training_torch_sensing_enabled": False,
    "training_torch_sensing_device": "auto",
    # float64 preserves the current CPU sensing semantics/trajectory.
    # float32 is faster on T4 but is an explicitly non-reference mode.
    "training_torch_sensing_dtype": "float64",
    "training_torch_sensing_batch_timeout_s": 0.002,
    # Avoid nondeterministic scheduling/competition between env CUDA work
    # and learner CUDA work until a measured deterministic overlap path wins.
    "training_torch_overlap_env_and_updates": False,

    # Optional CUDA learner acceleration. Disabled by default until benchmarked
    # for speed, stability, and same-seed reproducibility on the active GPU.
    "training_amp_enabled": False,
    "training_amp_dtype": "float16",

    # Full-GPU path in test_gpu.ipynb. The mission environment, replay,
    # inference, and learner tensors stay on one CUDA device. On the current
    # Kaggle Tesla T4, an earlier extended env-only sweep reached about
    # 617/1008/1481 transitions/s at 256/512/1024 envs and OOMed at 2048.
    # Later full-training runs did succeed at 2048 envs, so treat that OOM as
    # historical rather than a current hard limit. Re-sweep for active code/hardware.
    # Use 512 as a lower-memory fallback for larger replay/checkpoints.
    # Re-sweep when the GPU, UAV count, map, observation contract, or replay
    # capacity changes.
    # Device mode for test_gpu.ipynb: auto, cpu, or gpu. Auto uses CUDA
    # when available and otherwise runs the same tensor path on CPU for audit.
    "full_gpu_execution_mode": "auto",
    "full_gpu_device": "cuda:0",
    # Automatic single-node GPU selection. With >=2 visible CUDA devices the
    # production wrapper launches two NCCL/DDP ranks (one process/GPU), each
    # with its own environment shard and learner replica; with one visible GPU
    # it falls back to the existing single-GPU path.
    "full_gpu_auto_multi_gpu": True,
    "full_gpu_max_gpus": 2,
    "full_gpu_multi_gpu_strategy": "ddp",
    "full_gpu_num_envs": 1024,
    # For off-policy training with vectorized envs, ensure each optimizer burst
    # samples at least this fraction of the newly collected transitions.
    # 1.0 means total minibatch samples per burst >= new transitions.
    "training_auto_scale_gradient_steps": True,
    "training_min_replay_sample_ratio": 1.0,
    "full_gpu_benchmark_env_counts": (
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
    ),
    "full_gpu_benchmark_vector_steps": 200,
    # Fixed-size rejection pools keep world generation tensorized. Multiple
    # rounds make placement robust without moving the geometry to NumPy/CPU.
    "full_gpu_world_candidates": 256,
    "full_gpu_world_sampling_rounds": 4,

    "training_log_interval_steps": 100,
    "training_eval_interval_steps": 5_000,
    # In episode-target GPU runs, evaluation follows completed episodes rather
    # than raw transitions so W&B produces meaningful learning curves.
    "training_eval_interval_episodes": 5_000,
    "training_eval_episodes": 1,
    "training_eval_seed_offset": 10_000,
    # Keep expensive reference evaluation/rendering out of the optimizer loop.
    # Production GPU jobs stop after publishing/copying the final checkpoint.
    # The launcher starts a separate CPU-only session for deterministic
    # reference evaluation + PNG/MP4 so visualization does not consume GPU quota.
    "training_inline_evaluation": False,
    "training_inline_visualization": False,
    "training_auto_postprocess_visualization": True,
    "training_auto_postprocess_required": True,
    "training_postprocess_mode": "separate_cpu",
    "training_auto_postprocess_render_video": True,
    # Visualization files stay in the job/local output; W&B remains metrics-only.
    "training_auto_postprocess_log_wandb_media": False,
    "training_auto_postprocess_upload_huggingface": False,
    "training_auto_postprocess_output_dir": "post_train_visualization",
    # Keep episode/* scientifically raw: one W&B history point per completed
    # episode. Paper-style curves are a separate aggregated/smoothed view so
    # visualization never destroys or replaces the underlying observations.
    "training_reward_curve_bin_episodes": 1,
    # paper/* is a presentation-only series: average 256 completed episodes
    # (~195 points over 50k episodes), then apply a moderate EMA. This closely
    # matches paper-style learning curves while keeping episode/* fully raw.
    # beta=0.85 smooths visible spikes without hiding long-term degradation.
    "training_paper_curve_bin_episodes": 256,
    "training_paper_curve_ema_beta": 0.85,
    # Publication-style rendering only. episode metrics remain unsmoothed raw data.
    # A faint raw trace plus a bold EMA keeps every episode visible and auditable.
    "paper_plot_ema_beta": 0.95,
    "paper_plot_raw_alpha": 0.12,
    "paper_plot_raw_linewidth": 0.9,
    "paper_plot_main_linewidth": 2.6,
    "paper_plot_band_alpha": 0.16,
    # 2D is the default evaluation visualization: it is much cheaper than 3D
    # rendering and still shows search trajectories, coverage, targets,
    # obstacles, GCS, and successful communication links.
    "training_visualize_2d": True,
    "training_visualization_interval_episodes": 10_000,
    "training_visualization_trajectory_stride": 10,
    "training_visualization_max_link_edges": 64,
    # Exact rolling latest.pt checkpoints include replay + worker state;
    # use a wider interval to limit multi-GB checkpoint I/O.
    "training_checkpoint_interval_steps": 50_000,
    # DDP episode-target runs save model/optimizer checkpoints periodically.
    # They intentionally omit the sharded replay/environment state and are
    # therefore model-resume checkpoints, not bit-exact simulator resumes.
    "training_checkpoint_interval_episodes": 5_000,
    "training_checkpoint_include_replay": True,
    # Keep periodic checkpoints in Kaggle output, but avoid repeatedly uploading
    # ~tens-of-MB files while GPUs are reserved. Publish only the final model as
    # a W&B model artifact; CPU post-processing can mirror it to Hugging Face.
    "training_upload_periodic_checkpoints_wandb": False,
    "training_publish_final_checkpoint_wandb_artifact": True,
    "training_final_checkpoint_artifact_alias": "final",
    "training_enable_csv": True,
    "training_enable_tensorboard": True,
    "training_enable_wandb": True,
    # Keep W&B focused on experiment metrics. Disable automatic CPU/RAM/GPU
    # telemetry panels (power, clocks, memory, ECC errors, etc.).
    "training_wandb_disable_system_stats": True,
    "training_wandb_entity": "uav_search_paper",
    "training_wandb_project": "uav_search_target",
    "training_wandb_mode": "online",
    # Full-flow training is always online in W&B. Debug/unit helpers may be used
    # directly without W&B, but train_full_gpu_auto rejects offline/disabled runs.
    "training_wandb_require_online_gpu": True,
    # When a production run executes on Kaggle, resolve the authenticated Kaggle
    # username and expose it as the W&B config column kaggle_account.
    "training_require_kaggle_account_name": True,
    # GPU training logs a single W&B run from rank 0. The display name is
    # exactly: <ALGORITHM>-seed<SEED>-<WANDB_RUN_ID>.
    "training_wandb_required": True,
    "training_wandb_name_format": "{algorithm}-seed{seed}-{id}",

    # Hybrid MATD3 (CTDE + discrete destination adaptation).
    "matd3_hidden_dims": (256, 256),
    "matd3_replay_capacity": 50_000,
    "matd3_batch_size": 256,
    "matd3_gamma": 0.99,
    "matd3_tau": 0.005,
    "matd3_actor_lr": 3e-4,
    "matd3_critic_lr": 3e-4,
    "matd3_policy_delay": 2,
    "matd3_target_policy_noise": 0.2,
    "matd3_target_noise_clip": 0.5,
    "matd3_exploration_noise": 0.1,
    "matd3_discrete_epsilon_start": 1.0,
    "matd3_discrete_epsilon_end": 0.05,
    "matd3_discrete_epsilon_decay_steps": 100_000,
    "matd3_gumbel_temperature": 1.0,
    "matd3_learning_starts": 2_000,
    "matd3_train_freq": 1,
    "matd3_gradient_steps": 1,
    "matd3_updates_per_step": 1,  # deprecated compatibility alias
    "matd3_gradient_clip_norm": 10.0,

    # MATD3 experiment output/logging.
    "matd3_training_output_dir": "outputs/matd3",
    "matd3_wandb_project": "uav_search_target",

    # Kaggle runtime integration.
    "kaggle_wandb_secret_names": (
        "WANDB_API_KEY",
        "wandb_key",
        "WANDB_KEY",
    ),
    # CPU post-process / checkpoint mirroring. Set these as Kaggle Secrets on
    # the CPU account; never hard-code tokens in the notebook.
    "kaggle_hf_token_secret_names": (
        "HF_TOKEN",
        "HUGGINGFACE_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    ),
    "kaggle_hf_repo_secret_names": (
        "HF_REPO_ID",
        "HUGGINGFACE_REPO_ID",
    ),
    "postprocess_cpu_eval_episodes": 1,
    "postprocess_cpu_video_fps": 8,
    "postprocess_cpu_upload_huggingface": True,
    "postprocess_hf_private_repo": True,
}
