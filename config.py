CONFIG = {
    "seed": 44,

    "map_size": 5000,
    "num_uavs": 6,
    "num_targets": 10,
    "num_obstacles": 20,

    "altitude_min": 0,
    "altitude_max": 150,
    "max_speed": 20,
    "max_accel": 2.0,

    "dt": 1,
    "max_steps": 800,

    "gcs_position": [2500, 0, 0.0],
    "gcs_exclusion_radius_m": 200,
    "target_exclusion_radius_m": 400,

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
    "launch_min_spacing_m": 45.0,

    "report_bytes": 1_000_000,
    "buffer_bytes": 3_000_000,
    "pending_buffer_bytes": 3_000_000,
    "report_ttl": 300.0,

    "camera_full_fov_deg": 90,
    "sensing_obstacle_occlusion": True,

    "launch_radius_m": 300,

    "obstacle_radius_min_m": 80.0,
    "obstacle_radius_max_m": 220.0,
    "obstacle_height_min_m": 30.0,
    "obstacle_height_max_m": 120.0,

    "peer_contact_range_m": 1000,
    "gcs_contact_range_m": 1000,

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

    # Cooperative team reward.
    # Potential-based belief-certainty shaping scale.
    "reward_info_gain": 10.0,
    "reward_shaping_gamma": 0.99,
    "reward_confirmation": 20.0,
    "reward_delivery": 50.0,
    "reward_false_confirmation": 5.0,
    "reward_blocked": 0.2,
    "reward_boundary": 0.05,
    "reward_expired_report": 10.0,
    "reward_dropped_report": 10.0,
    "reward_energy_per_kj": 0.01,
    "reward_step_penalty": 0.01,
    "reward_all_delivered_bonus": 100.0,

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
    "masac_updates_per_step": 1,
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

    "training_log_interval_steps": 100,
    "training_eval_interval_steps": 5_000,
    "training_eval_episodes": 1,
    "training_eval_seed_offset": 10_000,
    # Exact rolling latest.pt checkpoints include replay + worker state;
    # use a wider interval to limit multi-GB checkpoint I/O.
    "training_checkpoint_interval_steps": 50_000,
    "training_checkpoint_include_replay": True,
    "training_enable_csv": True,
    "training_enable_tensorboard": True,
    "training_enable_wandb": False,
    "training_wandb_entity": "uav_search_paper",
    "training_wandb_project": "uav_search_target",
    "training_wandb_mode": "offline",

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
    "matd3_updates_per_step": 1,
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
}
