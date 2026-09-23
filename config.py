CONFIG = {
    "map_size": 5000,
    "num_uavs": 6,
    "num_targets": 10,
    "num_obstacles": 6,

    "altitude_min": 0,
    "altitude_max": 150,
    "max_speed": 20,
    "max_accel": 2.0,

    "dt": 1,
    "max_steps": 600,

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
    # "fine_altitude": 50.0,

    "safety_distance": 30,
    "launch_min_spacing_m": 45.0,

    "report_bytes": 1_000_000,
    "buffer_bytes": 3_000_000,
    "report_ttl": 300.0,

    "camera_full_fov_deg": 90,

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
}
