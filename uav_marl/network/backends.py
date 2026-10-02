"""Production source migrated from frozen fix_test_gpu.ipynb.

Do not import the notebook at runtime. The notebook is retained only as a
reference/regression artifact. Production changes belong in Python modules.
Migrated from notebook cells 108..117.
"""

from .communication import *  # noqa: F401,F403

# --- frozen notebook cell 108 ---
class NetworkBackend(ABC):
    """Backend-neutral interface for mission-level network simulation."""

    @abstractmethod
    def reset(
        self,
        *,
        uavs,
        gcs_position,
        obstacles,
        report_buffers,
        pending_reports,
        gcs_received_target_ids,
    ):
        """Reset backend state and bind the current mission state."""
        raise NotImplementedError

    @abstractmethod
    def sync_positions(self, uavs):
        """Synchronize backend node positions with mission state."""
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        requests,
        dt,
        current_step,
    ):
        """Execute one mission-step worth of network requests."""
        raise NotImplementedError

    @abstractmethod
    def metrics(self):
        """Return backend metrics without mutating backend state."""
        raise NotImplementedError

    @abstractmethod
    def last_step_communication_energy_by_uav(self):
        """Return communication energy used by each mission UAV last step."""
        raise NotImplementedError


# --- frozen notebook cell 109 ---
class SimpleNetworkBackend(NetworkBackend):
    """Thin backend wrapper around the existing simple-network primitives."""

    def __init__(self):
        self._initialized = False
        self._uavs = None
        self._gcs_position = None
        self._obstacles = None
        self._report_buffers = None
        self._pending_reports = None
        self._gcs_received_target_ids = None
        self._peer_transfer_states = create_peer_transfer_states()
        self._metric_state = self._new_metric_state()
        self._last_step_comm_energy_by_uav = np.zeros(
            int(CONFIG["num_uavs"]),
            dtype=np.float64,
        )

    @staticmethod
    def _new_metric_state():
        return {
            "steps": 0,
            "requests": 0,
            "tx_bytes": 0,
            "communication_energy_j": 0.0,
            "expired_peer_states": 0,
            "custody_retry_events": 0,
            "status_counts": {},
        }

    @staticmethod
    def _validate_uavs(uavs):
        if not isinstance(uavs, list):
            raise TypeError("uavs must be a list")

        num_uavs = int(CONFIG["num_uavs"])

        if len(uavs) != num_uavs:
            raise ValueError(
                "uavs length does not match CONFIG['num_uavs']"
            )

        for expected_id, uav in enumerate(uavs):
            if not isinstance(uav, UAV):
                raise TypeError(
                    "uavs must contain only UAV objects"
                )

            if int(uav.id) != expected_id:
                raise ValueError(
                    "uavs must be ordered by contiguous UAV id"
                )

    @staticmethod
    def _validate_obstacles(obstacles):
        if not isinstance(obstacles, list):
            raise TypeError("obstacles must be a list")

        for obstacle in obstacles:
            if not isinstance(obstacle, Obstacle):
                raise TypeError(
                    "obstacles must contain only Obstacle objects"
                )

    @staticmethod
    def _validate_report_state(
        report_buffers,
        pending_reports,
        gcs_received_target_ids,
    ):
        if not isinstance(report_buffers, list):
            raise TypeError(
                "report_buffers must be a list"
            )

        if len(report_buffers) != int(
            CONFIG["num_uavs"]
        ):
            raise ValueError(
                "report_buffers length does not match "
                "CONFIG['num_uavs']"
            )

        for buffer in report_buffers:
            if not isinstance(buffer, list):
                raise TypeError(
                    "each report buffer must be a list"
                )

            for report in buffer:
                if not isinstance(report, Report):
                    raise TypeError(
                        "report buffers must contain "
                        "only Report objects"
                    )

        if not isinstance(pending_reports, list):
            raise TypeError(
                "pending_reports must be a list"
            )

        for report in pending_reports:
            if not isinstance(report, Report):
                raise TypeError(
                    "pending_reports must contain "
                    "only Report objects"
                )

        if not isinstance(
            gcs_received_target_ids,
            set,
        ):
            raise TypeError(
                "gcs_received_target_ids must be a set"
            )

    @staticmethod
    def _validated_gcs_position(gcs_position):
        position = np.asarray(
            gcs_position,
            dtype=np.float64,
        )

        if position.shape != (3,):
            raise ValueError(
                "gcs_position must have shape (3,)"
            )

        if not np.all(np.isfinite(position)):
            raise ValueError(
                "gcs_position must contain only finite values"
            )

        configured_position = np.asarray(
            CONFIG["gcs_position"],
            dtype=np.float64,
        )

        # Existing simple-network primitives read GCS position from
        # CONFIG directly. Reject a mismatch instead of silently using
        # two different GCS locations.
        if not np.allclose(
            position,
            configured_position,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "SimpleNetworkBackend gcs_position must match "
                "CONFIG['gcs_position']"
            )

        return position.copy()

    def reset(
        self,
        *,
        uavs,
        gcs_position,
        obstacles,
        report_buffers,
        pending_reports,
        gcs_received_target_ids,
    ):
        self._validate_uavs(uavs)
        self._validate_obstacles(obstacles)
        self._validate_report_state(
            report_buffers,
            pending_reports,
            gcs_received_target_ids,
        )

        self._uavs = uavs
        self._gcs_position = (
            self._validated_gcs_position(
                gcs_position
            )
        )
        self._obstacles = obstacles
        self._report_buffers = report_buffers
        self._pending_reports = pending_reports
        self._gcs_received_target_ids = (
            gcs_received_target_ids
        )
        self._peer_transfer_states = (
            create_peer_transfer_states()
        )
        self._metric_state = (
            self._new_metric_state()
        )
        self._last_step_comm_energy_by_uav = np.zeros(
            int(CONFIG["num_uavs"]),
            dtype=np.float64,
        )
        self._initialized = True

    def sync_positions(self, uavs):
        if not self._initialized:
            raise RuntimeError(
                "backend must be reset before sync_positions"
            )

        self._validate_uavs(uavs)
        self._uavs = uavs

    def step(
        self,
        requests,
        dt,
        current_step,
    ):
        if not self._initialized:
            raise RuntimeError(
                "backend must be reset before step"
            )

        if (
            isinstance(current_step, (bool, np.bool_))
            or not isinstance(
                current_step,
                (int, np.integer),
            )
        ):
            raise TypeError(
                "current_step must be an integer"
            )

        current_step = int(current_step)

        if current_step < 0:
            raise ValueError(
                "current_step must be >= 0"
            )

        if isinstance(dt, (bool, np.bool_)):
            raise TypeError(
                "dt must be numeric and not boolean"
            )

        dt = float(dt)

        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(
                "dt must be finite and > 0"
            )

        # Report/peer TTL helpers currently use CONFIG['dt'].
        # Enforce one mission clock until those helpers are made
        # backend-neutral.
        if not np.isclose(
            dt,
            float(CONFIG["dt"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "SimpleNetworkBackend dt must match CONFIG['dt']"
            )

        if requests is None:
            request_list = []
        elif isinstance(
            requests,
            TransmissionIntent,
        ):
            request_list = [requests]
        elif isinstance(requests, (list, tuple)):
            request_list = list(requests)
        else:
            raise TypeError(
                "requests must be a TransmissionIntent, "
                "a list/tuple of intents, or None"
            )

        for request in request_list:
            if not isinstance(
                request,
                TransmissionIntent,
            ):
                raise TypeError(
                    "requests must contain only "
                    "TransmissionIntent objects"
                )

        sender_ids = [
            int(request.sender)
            for request in request_list
        ]

        if len(sender_ids) != len(
            set(sender_ids)
        ):
            raise ValueError(
                "each sender may submit at most one "
                "transmission request per network step"
            )

        expired_peer_states = (
            cleanup_peer_transfer_states(
                self._peer_transfer_states,
                current_step,
            )
        )
        custody_retry_results = (
            retry_completed_peer_transfers(
                self._peer_transfer_states,
                self._report_buffers,
                current_step,
                self._gcs_received_target_ids,
            )
        )

        self._metric_state[
            "expired_peer_states"
        ] += len(expired_peer_states)
        self._metric_state[
            "custody_retry_events"
        ] += len(custody_retry_results)

        self._last_step_comm_energy_by_uav.fill(
            0.0
        )

        results = []

        for request in request_list:
            result = execute_transmission_intent(
                request,
                self._uavs,
                self._report_buffers,
                self._pending_reports,
                self._gcs_received_target_ids,
                self._peer_transfer_states,
                current_step,
                obstacles=self._obstacles,
                dt=dt,
            )

            results.append(result)

            status = str(
                result.get(
                    "status",
                    "unknown",
                )
            )
            status_counts = self._metric_state[
                "status_counts"
            ]
            status_counts[status] = (
                status_counts.get(status, 0)
                + 1
            )
            tx_bytes = int(
                result.get(
                    "tx_bytes",
                    0,
                )
            )
            self._metric_state["tx_bytes"] += tx_bytes

            # TX attempts consume radio energy even if SNR is too low
            # and no payload bytes are delivered.
            if (
                status != "already_delivered"
                and int(request.requested_bytes) > 0
                and request.tx_power_w > 0.0
            ):
                phy_rate_bps = float(
                    CONFIG[
                        "comm_max_link_rate_bps"
                    ]
                )
                attempted_bytes = min(
                    int(request.requested_bytes),
                    int(
                        np.floor(
                            phy_rate_bps
                            * dt
                            / 8.0
                        )
                    ),
                )
                duration_s = min(
                    dt,
                    attempted_bytes
                    * 8.0
                    / phy_rate_bps,
                )
                energy_j = (
                    float(request.tx_power_w)
                    * duration_s
                )

                self._last_step_comm_energy_by_uav[
                    request.sender
                ] += energy_j
                self._metric_state[
                    "communication_energy_j"
                ] += energy_j

        self._metric_state["steps"] += 1
        self._metric_state["requests"] += len(
            request_list
        )

        return results

    @property
    def peer_transfer_states(self):
        if not self._initialized:
            raise RuntimeError(
                "backend must be reset before accessing "
                "peer_transfer_states"
            )

        return self._peer_transfer_states

    def metrics(self):
        status_counts = dict(
            self._metric_state[
                "status_counts"
            ]
        )

        return {
            "steps": int(
                self._metric_state["steps"]
            ),
            "requests": int(
                self._metric_state["requests"]
            ),
            "tx_bytes": int(
                self._metric_state["tx_bytes"]
            ),
            "communication_energy_j": float(
                self._metric_state[
                    "communication_energy_j"
                ]
            ),
            "expired_peer_states": int(
                self._metric_state[
                    "expired_peer_states"
                ]
            ),
            "custody_retry_events": int(
                self._metric_state[
                    "custody_retry_events"
                ]
            ),
            "status_counts": status_counts,
            "active_peer_transfers": len(
                self._peer_transfer_states
            ),
        }

    def last_step_communication_energy_by_uav(self):
        if not self._initialized:
            raise RuntimeError(
                "backend must be reset before "
                "reading communication energy"
            )

        return (
            self._last_step_comm_energy_by_uav
            .copy()
        )


# --- frozen notebook cell 110 ---
def resolve_uavnetsim_repo_path(repo_path=None):
    candidates = [
        repo_path,
        CONFIG.get("uavnetsim_path"),
        os.environ.get("UAVNETSIM_PATH"),
    ]

    # Also support a normal pip-installed UavNetSim distribution. The
    # package exposes simulator/, entities/, and utils/ as top-level packages,
    # so there is intentionally no importable top-level "uavnetsim" module.
    try:
        from importlib.metadata import distribution
        installed_root = Path(
            distribution("uavnetsim").locate_file("")
        ).resolve()
        candidates.append(installed_root)
    except Exception:
        pass

    checked = []

    for candidate in candidates:
        if candidate in (None, ""):
            continue

        path = Path(candidate).expanduser().resolve()
        checked.append(str(path))

        required = (
            path / "simulator" / "simulator.py",
            path / "entities" / "packet.py",
            path / "utils" / "config.py",
        )

        if all(item.is_file() for item in required):
            return path

    suffix = (
        f" Checked: {checked}."
        if checked
        else ""
    )

    raise FileNotFoundError(
        "UavNetSim repository was not found. "
        "Pass repo_path=... to UavNetSimBackend, set "
        "CONFIG['uavnetsim_path'], or set UAVNETSIM_PATH."
        + suffix
    )


# --- frozen notebook cell 111 ---
def uavnetsim_available(repo_path=None):
    try:
        resolve_uavnetsim_repo_path(
            repo_path
        )
    except FileNotFoundError:
        return False

    return True


# --- frozen notebook cell 112 ---
def _uavnetsim_scene_feature(
    obstacle,
    obstacle_id,
):
    if not isinstance(obstacle, Obstacle):
        raise TypeError(
            "obstacle must be an Obstacle"
        )

    sides = int(
        CONFIG[
            "uavnetsim_obstacle_polygon_sides"
        ]
    )

    if sides < 8:
        raise ValueError(
            "uavnetsim_obstacle_polygon_sides "
            "must be >= 8"
        )

    cx = float(obstacle.position[0])
    cy = float(obstacle.position[1])
    radius = float(obstacle.radius)

    footprint = []

    for index in range(sides):
        angle = (
            2.0
            * np.pi
            * index
            / sides
        )

        footprint.append(
            {
                "x": float(
                    cx
                    + radius
                    * np.cos(angle)
                ),
                "y": float(
                    cy
                    + radius
                    * np.sin(angle)
                ),
                "z": 0.0,
            }
        )

    return {
        "id": f"mission-obstacle-{obstacle_id}",
        "category": "building",
        "height": float(obstacle.height),
        "material": "itu_concrete",
        "source": "uav_search_target",
        "footprint": footprint,
    }


# --- frozen notebook cell 113 ---
def build_uavnetsim_scene_payload(
    obstacles,
):
    if not isinstance(obstacles, list):
        raise TypeError(
            "obstacles must be a list"
        )

    features = [
        _uavnetsim_scene_feature(
            obstacle,
            obstacle_id,
        )
        for obstacle_id, obstacle
        in enumerate(obstacles)
    ]

    map_size = float(
        CONFIG["map_size"]
    )

    if (
        not np.isfinite(map_size)
        or map_size <= 0.0
    ):
        raise ValueError(
            "CONFIG['map_size'] must be "
            "finite and > 0"
        )

    return {
        "schema_version": 1,
        "name": "uav-search-target",
        "anchor": {
            "latitude": 0.0,
            "longitude": 0.0,
        },
        "size_x": map_size,
        "size_y": map_size,
        "features": features,
    }


# --- frozen notebook cell 114 ---
def write_uavnetsim_scene(
    obstacles,
    directory,
):
    directory = Path(directory)
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    scene_path = (
        directory / "scene.json"
    )

    payload = (
        build_uavnetsim_scene_payload(
            obstacles
        )
    )

    scene_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    return scene_path


# --- frozen notebook cell 115 ---
class MarlSelectedNextHopRouting:
    """Routing shim that preserves the MARL-selected mission next hop."""

    def __init__(
        self,
        simulator,
        my_drone,
    ):
        self.simulator = simulator
        self.my_drone = my_drone

    def next_hop_selection(
        self,
        packet,
    ):
        enquire = False

        next_hop_id = getattr(
            packet,
            "forced_next_hop_id",
            packet.dst_drone.identifier,
        )

        try:
            next_hop_id = int(
                next_hop_id
            )
        except (TypeError, ValueError):
            return (
                False,
                packet,
                enquire,
            )

        if not (
            0
            <= next_hop_id
            < len(self.simulator.drones)
        ):
            return (
                False,
                packet,
                enquire,
            )

        if (
            next_hop_id
            == self.my_drone.identifier
        ):
            return (
                False,
                packet,
                enquire,
            )

        packet.next_hop_id = (
            next_hop_id
        )

        if (
            self.my_drone.identifier
            not in packet.intermediate_drones
        ):
            packet.intermediate_drones.append(
                self.my_drone.identifier
            )

        return (
            True,
            packet,
            enquire,
        )

    def packet_reception(
        self,
        packet,
        src_drone_id,
    ):
        import copy

        from entities.packet import (
            AckPacket,
            DataPacket,
        )
        from utils import config as uav_config

        if isinstance(
            packet,
            DataPacket,
        ):
            packet_copy = copy.copy(
                packet
            )

            destination_id = (
                packet_copy
                .dst_drone
                .identifier
            )

            if (
                destination_id
                == self.my_drone.identifier
            ):
                if (
                    packet_copy.packet_id
                    not in (
                        self.simulator
                        .metrics
                        .datapacket_arrived
                    )
                ):
                    self.simulator.metrics.calculate_metrics(
                        packet_copy
                    )

                uav_config.GL_ID_ACK_PACKET += 1

                ack_packet = AckPacket(
                    src_drone=self.my_drone,
                    dst_drone=(
                        self.simulator
                        .drones[src_drone_id]
                    ),
                    ack_packet_id=(
                        uav_config
                        .GL_ID_ACK_PACKET
                    ),
                    ack_packet_length=(
                        uav_config
                        .ACK_PACKET_LENGTH
                    ),
                    ack_packet=packet_copy,
                    simulator=self.simulator,
                    channel_id=(
                        packet_copy.channel_id
                    ),
                )

                yield self.simulator.env.timeout(
                    uav_config.SIFS_DURATION
                )

                ack_packet.increase_ttl()

                self.my_drone.mac_protocol.phy.unicast(
                    ack_packet,
                    src_drone_id,
                )

                yield self.simulator.env.timeout(
                    ack_packet.packet_length
                    / uav_config.BIT_RATE
                    * 1e6
                )

                return

            if (
                self.my_drone
                .transmitting_queue
                .qsize()
                < self.my_drone.max_queue_size
            ):
                self.my_drone.transmitting_queue.put(
                    packet_copy
                )

            return

        if isinstance(
            packet,
            AckPacket,
        ):
            data_packet = (
                packet.ack_packet
            )

            if (
                getattr(
                    data_packet,
                    "first_attempt_time",
                    None,
                )
                is not None
            ):
                self.simulator.metrics.mac_delay.append(
                    (
                        self.simulator.env.now
                        - data_packet.first_attempt_time
                    )
                    / 1e3
                )

            self.my_drone.remove_from_queue(
                data_packet
            )

            key = (
                f"wait_ack"
                f"{self.my_drone.identifier}"
                f"_{data_packet.packet_id}"
            )

            finish_state = (
                self.my_drone
                .mac_protocol
                .wait_ack_process_finish
                .get(
                    key,
                    1,
                )
            )

            if finish_state == 0:
                process = (
                    self.my_drone
                    .mac_protocol
                    .wait_ack_process_dict
                    .get(key)
                )

                if (
                    process is not None
                    and not process.triggered
                ):
                    self.my_drone.mac_protocol.wait_ack_process_finish[
                        key
                    ] = 1

                    process.interrupt()

            return

    def penalize(
        self,
        packet,
    ):
        return None


# --- frozen notebook cell 116 ---
class UavNetSimBackend(NetworkBackend):
    """Adapter from mission reports to UavNetSim packet-level networking."""

    def __init__(
        self,
        repo_path=None,
        seed=None,
    ):
        self.repo_path = repo_path
        self.seed = (
            int(CONFIG.get("seed", 44))
            if seed is None
            else int(seed)
        )

        self._initialized = False
        self._uavs = None
        self._gcs_position = None
        self._obstacles = None
        self._report_buffers = None
        self._pending_reports = None
        self._gcs_received_target_ids = None
        self._peer_transfer_states = (
            create_peer_transfer_states()
        )

        self._repo = None
        self._simpy = None
        self._simulator_class = None
        self._data_packet_class = None
        self._uav_config = None

        self._scene_directory = None
        self._environment = None
        self._simulator = None
        self._gcs_node_id = int(
            CONFIG["num_uavs"]
        )

        self._packet_records = {}
        self._metric_state = (
            self._new_metric_state()
        )
        self._last_step_comm_energy_by_uav = np.zeros(
            int(CONFIG["num_uavs"]),
            dtype=np.float64,
        )
        self._step_energy_budget_j = None

    @staticmethod
    def _new_metric_state():
        return {
            "steps": 0,
            "requests": 0,
            "packets_injected": 0,
            "packets_delivered": 0,
            "packets_failed": 0,
            "queue_limit_events": 0,
            "payload_bytes_deferred": 0,
            "payload_bytes_injected": 0,
            "payload_bytes_delivered": 0,
            "communication_energy_j": 0.0,
            "status_counts": {},
        }

    @staticmethod
    def _validate_backend_config():
        payload_bytes = int(
            CONFIG[
                "uavnetsim_payload_bytes"
            ]
        )

        if payload_bytes <= 0:
            raise ValueError(
                "uavnetsim_payload_bytes "
                "must be > 0"
            )

        packet_lifetime_s = float(
            CONFIG[
                "uavnetsim_packet_lifetime_s"
            ]
        )

        if (
            not np.isfinite(
                packet_lifetime_s
            )
            or packet_lifetime_s <= 0.0
        ):
            raise ValueError(
                "uavnetsim_packet_lifetime_s "
                "must be finite and > 0"
            )

        max_queue_size = int(
            CONFIG[
                "uavnetsim_max_queue_size"
            ]
        )

        if max_queue_size <= 0:
            raise ValueError(
                "uavnetsim_max_queue_size "
                "must be > 0"
            )

    def _load_uavnetsim(self):
        self._repo = (
            resolve_uavnetsim_repo_path(
                self.repo_path
            )
        )

        repo_string = str(
            self._repo
        )

        if repo_string not in sys.path:
            sys.path.insert(
                0,
                repo_string,
            )

        # UavNetSim's simulator.log configures a relative
        # running_log.log file at import time when the root logger has
        # no handlers. Add a temporary NullHandler so importing the
        # external simulator never writes files into this project.
        import logging

        root_logger = logging.getLogger()
        temporary_handler = None

        if not root_logger.handlers:
            temporary_handler = (
                logging.NullHandler()
            )
            root_logger.addHandler(
                temporary_handler
            )

        try:
            self._simpy = (
                importlib.import_module(
                    "simpy"
                )
            )
            simulator_module = (
                importlib.import_module(
                    "simulator.simulator"
                )
            )
            packet_module = (
                importlib.import_module(
                    "entities.packet"
                )
            )
            self._uav_config = (
                importlib.import_module(
                    "utils.config"
                )
            )
        finally:
            if (
                temporary_handler
                is not None
            ):
                root_logger.removeHandler(
                    temporary_handler
                )

        simulator_file = Path(
            simulator_module.__file__
        ).resolve()

        if (
            self._repo
            not in simulator_file.parents
        ):
            raise RuntimeError(
                "A different simulator package "
                "named 'simulator' is already "
                "loaded in this Python process"
            )

        self._simulator_class = (
            simulator_module.Simulator
        )
        self._data_packet_class = (
            packet_module.DataPacket
        )

    def _configure_uavnetsim(
        self,
        scene_directory,
    ):
        uav_config = self._uav_config

        num_network_nodes = (
            int(CONFIG["num_uavs"])
            + 1
        )

        uav_config.NUMBER_OF_DRONES = (
            num_network_nodes
        )
        uav_config.MAX_TTL = (
            num_network_nodes + 1
        )

        uav_config.ROUTING_PROTOCOL = (
            "DRL"
        )
        uav_config.DRL_ROUTING_PROTOCOL_CLASS = (
            MarlSelectedNextHopRouting
        )
        uav_config.DRL_HELLO_PACKET_CLASS = (
            None
        )
        uav_config.DRL_NEIGHBOR_TABLE_CLASS = (
            None
        )

        uav_config.MAC_PROTOCOL = str(
            CONFIG[
                "uavnetsim_mac_protocol"
            ]
        )
        uav_config.MOBILITY_MODEL = (
            "GaussMarkov3D"
        )

        uav_config.CHANNEL_MODE = str(
            CONFIG[
                "uavnetsim_channel_mode"
            ]
        )
        uav_config.LOS_A2A_MODEL = str(
            CONFIG[
                "uavnetsim_los_model"
            ]
        )
        uav_config.NLOS_A2A_MODEL = str(
            CONFIG[
                "uavnetsim_nlos_model"
            ]
        )

        uav_config.STATIC_CASE = 1
        uav_config.HETEROGENEOUS = 0

        uav_config.TRAFFIC_PATTERN = (
            "UNIFORM"
        )
        uav_config.PACKET_ARRIVAL_RATE = (
            1e-12
        )

        uav_config.MAP_LENGTH = float(
            CONFIG["map_size"]
        )
        uav_config.MAP_WIDTH = float(
            CONFIG["map_size"]
        )
        uav_config.MAP_HEIGHT = float(
            CONFIG["altitude_max"]
        )
        uav_config.UAV_MIN_ALTITUDE = (
            float(
                CONFIG[
                    "altitude_min"
                ]
            )
        )
        uav_config.UAV_MAX_ALTITUDE = (
            float(
                CONFIG[
                    "altitude_max"
                ]
            )
        )
        uav_config.UAV_BOUNDARY_CLEARANCE = (
            0.0
        )
        uav_config.UAV_BUILDING_CLEARANCE = (
            0.0
        )

        uav_config.MAX_QUEUE_SIZE = int(
            CONFIG[
                "uavnetsim_max_queue_size"
            ]
        )

        uav_config.AVERAGE_PAYLOAD_LENGTH = (
            int(
                CONFIG[
                    "uavnetsim_payload_bytes"
                ]
            )
            * 8
        )
        uav_config.VARIABLE_PAYLOAD_LENGTH = (
            0
        )
        uav_config.PACKET_LIFETIME = (
            float(
                CONFIG[
                    "uavnetsim_packet_lifetime_s"
                ]
            )
            * 1e6
        )

        uav_config.INITIAL_ENERGY = max(
            float(CONFIG["battery_j"]),
            10_000.0,
        )
        uav_config.ENERGY_THRESHOLD = (
            0.0
        )

        uav_config.TRANSMITTING_POWER = (
            float(
                CONFIG[
                    "tx_power_min_w"
                ]
            )
        )

        # Pin UavNetSim radio/MAC values to the project config. This avoids
        # silent behavior changes when the external simulator is upgraded and
        # keeps the CUDA surrogate on exactly the same physical/MAC contract.
        uav_config.CARRIER_FREQUENCY = float(
            CONFIG["uavnetsim_carrier_frequency_hz"]
        )
        uav_config.BANDWIDTH = float(
            CONFIG["uavnetsim_bandwidth_hz"]
        )
        uav_config.BIT_RATE = float(
            CONFIG["uavnetsim_bit_rate_bps"]
        )
        uav_config.SINR_THRESHOLD_DB = float(
            CONFIG["uavnetsim_sinr_threshold_db"]
        )
        uav_config.CCA_THRESHOLD_DBM = float(
            CONFIG["uavnetsim_cca_threshold_dbm"]
        )
        uav_config.THERMAL_NOISE_DENSITY_DBM_HZ = float(
            CONFIG["uavnetsim_thermal_noise_density_dbm_hz"]
        )
        uav_config.RECEIVER_NOISE_FIGURE_DB = float(
            CONFIG["uavnetsim_receiver_noise_figure_db"]
        )
        uav_config.MAX_RETRANSMISSION_ATTEMPT = int(
            CONFIG["uavnetsim_max_retransmission_attempt"]
        )
        uav_config.CW_MIN = int(
            CONFIG["full_gpu_packet_cw_min"]
        )
        uav_config.SLOT_DURATION = float(
            CONFIG["full_gpu_packet_slot_us"]
        )
        uav_config.DIFS_DURATION = float(
            CONFIG["full_gpu_packet_difs_us"]
        )
        uav_config.SIFS_DURATION = float(
            CONFIG["full_gpu_packet_sifs_us"]
        )
        uav_config.ACK_PACKET_LENGTH = int(
            CONFIG["full_gpu_packet_ack_bits"]
        )

        uav_config.SIONNA_SCENE_PATH = str(
            Path(scene_directory)
            / "scene.xml"
        )

    def _patch_dynamic_tx_power(self):
        original_transmit = (
            self._simulator
            .channel
            .transmit
        )
        uav_config = self._uav_config

        def transmit_with_packet_power(
            packet,
            transmitter_id,
            receiver_ids,
        ):
            previous_power = (
                uav_config
                .TRANSMITTING_POWER
            )

            packet_power = float(
                getattr(
                    packet,
                    "mission_tx_power_w",
                    previous_power,
                )
            )

            uav_config.TRANSMITTING_POWER = (
                packet_power
            )

            try:
                return original_transmit(
                    packet,
                    transmitter_id,
                    receiver_ids,
                )
            finally:
                uav_config.TRANSMITTING_POWER = (
                    previous_power
                )

        self._simulator.channel.transmit = (
            transmit_with_packet_power
        )

        for drone in self._simulator.drones:
            phy = (
                drone
                .mac_protocol
                .phy
            )

            def consume_energy(
                packet,
                phy=phy,
            ):
                power_w = float(
                    getattr(
                        packet,
                        "mission_tx_power_w",
                        uav_config
                        .TRANSMITTING_POWER,
                    )
                )

                duration_s = (
                    packet.packet_length
                    / uav_config.BIT_RATE
                )

                phy.my_drone.residual_energy = max(
                    0.0,
                    (
                        phy.my_drone
                        .residual_energy
                        - duration_s
                        * power_w
                    ),
                )

            phy._consume_transmit_energy = (
                consume_energy
            )

            def unicast_with_energy_budget(
                packet,
                next_hop_id,
                phy=phy,
                consume_energy=consume_energy,
            ):
                power_w = float(
                    getattr(
                        packet,
                        "mission_tx_power_w",
                        uav_config.TRANSMITTING_POWER,
                    )
                )
                duration_s = (
                    float(packet.packet_length)
                    / float(uav_config.BIT_RATE)
                )
                required_j = duration_s * power_w
                if (
                    float(phy.my_drone.residual_energy)
                    + 1e-12
                    < required_j
                ):
                    packet.mission_energy_blocked = True
                    return False

                consume_energy(packet)
                phy.my_drone.simulator.channel.transmit(
                    packet,
                    phy.my_drone.identifier,
                    [next_hop_id],
                )
                return True

            phy.unicast = unicast_with_energy_budget

    def _freeze_external_mobility(self):
        total_us = (
            (
                int(CONFIG["max_steps"])
                + 2
            )
            * float(CONFIG["dt"])
            * 1e6
        )

        freeze_interval = (
            total_us + 1e6
        )

        for drone in self._simulator.drones:
            drone.speed = 0.0
            drone.velocity = [
                0.0,
                0.0,
                0.0,
            ]
            drone.velocity_mean = 0.0

            model = getattr(
                drone,
                "mobility_model",
                None,
            )

            if model is not None:
                if hasattr(
                    model,
                    "position_update_interval",
                ):
                    model.position_update_interval = (
                        freeze_interval
                    )

                if hasattr(
                    model,
                    "direction_update_interval",
                ):
                    model.direction_update_interval = (
                        freeze_interval
                    )

    def reset(
        self,
        *,
        uavs,
        gcs_position,
        obstacles,
        report_buffers,
        pending_reports,
        gcs_received_target_ids,
    ):
        self._validate_backend_config()

        SimpleNetworkBackend._validate_uavs(
            uavs
        )
        SimpleNetworkBackend._validate_obstacles(
            obstacles
        )
        SimpleNetworkBackend._validate_report_state(
            report_buffers,
            pending_reports,
            gcs_received_target_ids,
        )

        validated_gcs = (
            SimpleNetworkBackend
            ._validated_gcs_position(
                gcs_position
            )
        )

        if self._initialized:
            self.close()

        self._load_uavnetsim()

        self._scene_directory = (
            tempfile.TemporaryDirectory(
                prefix="uav_search_uavnetsim_"
            )
        )

        write_uavnetsim_scene(
            obstacles,
            self._scene_directory.name,
        )

        self._configure_uavnetsim(
            self._scene_directory.name
        )

        self._environment = (
            self._simpy.Environment()
        )

        total_us = (
            (
                int(CONFIG["max_steps"])
                + 2
            )
            * float(CONFIG["dt"])
            * 1e6
        )

        self._simulator = (
            self._simulator_class(
                seed=self.seed,
                env=self._environment,
                n_drones=(
                    int(
                        CONFIG[
                            "num_uavs"
                        ]
                    )
                    + 1
                ),
                total_simulation_time=(
                    total_us
                ),
                drone_speed=0.0,
            )
        )

        self._freeze_external_mobility()
        self._patch_dynamic_tx_power()

        self._uavs = uavs
        self._gcs_position = (
            validated_gcs
        )
        self._obstacles = obstacles
        self._report_buffers = (
            report_buffers
        )
        self._pending_reports = (
            pending_reports
        )
        self._gcs_received_target_ids = (
            gcs_received_target_ids
        )

        self._peer_transfer_states = (
            create_peer_transfer_states()
        )
        self._packet_records = {}
        self._metric_state = (
            self._new_metric_state()
        )
        self._last_step_comm_energy_by_uav = np.zeros(
            int(CONFIG["num_uavs"]),
            dtype=np.float64,
        )
        self._step_energy_budget_j = None

        self._initialized = True

        self.sync_positions(
            uavs
        )

        # Execute all processes scheduled at t=0 once.
        self._environment.run(
            until=1.0
        )

        # UavNetSim mobility is deliberately frozen; re-apply the
        # mission positions after its t=0 initialization.
        self.sync_positions(
            uavs
        )

    def sync_positions(
        self,
        uavs,
    ):
        if not self._initialized:
            raise RuntimeError(
                "backend must be reset before "
                "sync_positions"
            )

        SimpleNetworkBackend._validate_uavs(
            uavs
        )

        map_size = float(
            CONFIG["map_size"]
        )
        altitude_min = float(
            CONFIG["altitude_min"]
        )
        altitude_max = float(
            CONFIG["altitude_max"]
        )

        for mission_uav in uavs:
            position = np.asarray(
                mission_uav.position,
                dtype=np.float64,
            )
            velocity = np.asarray(
                mission_uav.velocity,
                dtype=np.float64,
            )

            if (
                position.shape != (3,)
                or velocity.shape != (3,)
            ):
                raise ValueError(
                    "UAV position and velocity "
                    "must have shape (3,)"
                )

            if (
                not np.all(
                    np.isfinite(position)
                )
                or not np.all(
                    np.isfinite(velocity)
                )
            ):
                raise ValueError(
                    "UAV position and velocity "
                    "must be finite"
                )

            if not (
                0.0
                <= position[0]
                <= map_size
                and 0.0
                <= position[1]
                <= map_size
                and altitude_min
                <= position[2]
                <= altitude_max
            ):
                raise ValueError(
                    "UAV position is outside "
                    "mission bounds"
                )

            external_drone = (
                self._simulator
                .drones[mission_uav.id]
            )

            # Mission motion remains the source of truth. Directly
            # synchronize coordinates instead of invoking UavNetSim
            # mobility/collision resolution.
            external_drone._coords = [
                float(value)
                for value in position
            ]
            external_drone.velocity = [
                float(value)
                for value in velocity
            ]
            external_drone.sleep = bool(
                not mission_uav.active
            )

        gcs_drone = (
            self._simulator
            .drones[
                self._gcs_node_id
            ]
        )
        gcs_drone._coords = [
            float(value)
            for value
            in self._gcs_position
        ]
        gcs_drone.velocity = [
            0.0,
            0.0,
            0.0,
        ]

        self._uavs = uavs

    @staticmethod
    def _validate_step_clock(
        dt,
        current_step,
    ):
        if (
            isinstance(
                current_step,
                (bool, np.bool_),
            )
            or not isinstance(
                current_step,
                (int, np.integer),
            )
        ):
            raise TypeError(
                "current_step must be an integer"
            )

        current_step = int(
            current_step
        )

        if current_step < 0:
            raise ValueError(
                "current_step must be >= 0"
            )

        if isinstance(
            dt,
            (bool, np.bool_),
        ):
            raise TypeError(
                "dt must be numeric and not boolean"
            )

        dt = float(dt)

        if (
            not np.isfinite(dt)
            or dt <= 0.0
        ):
            raise ValueError(
                "dt must be finite and > 0"
            )

        if not np.isclose(
            dt,
            float(CONFIG["dt"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "UavNetSimBackend dt must match "
                "CONFIG['dt']"
            )

        return dt, current_step

    @staticmethod
    def _normalize_requests(
        requests,
    ):
        if requests is None:
            request_list = []
        elif isinstance(
            requests,
            TransmissionIntent,
        ):
            request_list = [
                requests
            ]
        elif isinstance(
            requests,
            (list, tuple),
        ):
            request_list = list(
                requests
            )
        else:
            raise TypeError(
                "requests must be a "
                "TransmissionIntent, a list/"
                "tuple of intents, or None"
            )

        for request in request_list:
            if not isinstance(
                request,
                TransmissionIntent,
            ):
                raise TypeError(
                    "requests must contain only "
                    "TransmissionIntent objects"
                )

        sender_ids = [
            int(request.sender)
            for request in request_list
        ]

        if len(sender_ids) != len(
            set(sender_ids)
        ):
            raise ValueError(
                "each sender may submit at most "
                "one transmission request per "
                "network step"
            )

        return request_list

    def _report_generation_key(
        self,
        intent,
        report,
    ):
        return (
            int(intent.sender),
            int(intent.recipient),
            int(report.target_id),
            int(report.source_uav),
            int(report.created_step),
            int(report.size_bytes),
            float(report.ttl_s),
        )

    def _report_for_intent(
        self,
        intent,
        current_step,
    ):
        if (
            intent.target_id
            in self._gcs_received_target_ids
        ):
            return None

        sender_buffer = (
            self._report_buffers[
                intent.sender
            ]
        )

        cleanup_report_buffer(
            sender_buffer,
            current_step,
        )

        report = find_report_in_buffer(
            sender_buffer,
            intent.target_id,
        )

        if report is None:
            return None

        if report_is_expired(
            report,
            current_step,
        ):
            return None

        return report

    def _inflight_bytes(
        self,
        transfer_key,
    ):
        return sum(
            int(record["payload_bytes"])
            for record
            in self._packet_records.values()
            if (
                record["status"]
                == "in_flight"
                and record["transfer_key"]
                == transfer_key
            )
        )

    def _recipient_node_id(
        self,
        recipient,
    ):
        if recipient == GCS_NODE:
            return self._gcs_node_id

        return int(recipient)

    def _request_is_active(
        self,
        intent,
    ):
        if not self._uavs[
            intent.sender
        ].active:
            return False

        if intent.recipient == GCS_NODE:
            return True

        return bool(
            self._uavs[
                intent.recipient
            ].active
        )

    def _request_is_within_contact_range(
        self,
        intent,
    ):
        sender_position = np.asarray(
            self._uavs[
                intent.sender
            ].position,
            dtype=np.float64,
        )

        if intent.recipient == GCS_NODE:
            recipient_position = np.asarray(
                self._gcs_position,
                dtype=np.float64,
            )
            max_range = float(
                CONFIG[
                    "gcs_contact_range_m"
                ]
            )
        else:
            recipient_position = np.asarray(
                self._uavs[
                    intent.recipient
                ].position,
                dtype=np.float64,
            )
            max_range = float(
                CONFIG[
                    "peer_contact_range_m"
                ]
            )

        distance = float(
            np.linalg.norm(
                sender_position
                - recipient_position
            )
        )

        return bool(
            distance <= max_range
        )

    def _inject_request(
        self,
        intent,
        current_step,
    ):
        if not self._request_is_active(
            intent
        ):
            return {
                "transfer_key": None,
                "injected_bytes": 0,
                "injected_packets": 0,
                "reason": "inactive_node",
            }

        if not self._request_is_within_contact_range(
            intent
        ):
            return {
                "transfer_key": None,
                "injected_bytes": 0,
                "injected_packets": 0,
                "reason": "out_of_contact_range",
            }

        report = self._report_for_intent(
            intent,
            current_step,
        )

        if report is None:
            return {
                "transfer_key": None,
                "injected_bytes": 0,
                "injected_packets": 0,
                "reason": "no_report",
            }

        transfer_key = (
            self._report_generation_key(
                intent,
                report,
            )
        )

        inflight_bytes = (
            self._inflight_bytes(
                transfer_key
            )
        )

        bytes_to_inject = max(
            0,
            int(intent.requested_bytes)
            - inflight_bytes,
        )

        if bytes_to_inject <= 0:
            return {
                "transfer_key": transfer_key,
                "injected_bytes": 0,
                "injected_packets": 0,
                "reason": "already_in_flight",
            }

        recipient_node_id = (
            self._recipient_node_id(
                intent.recipient
            )
        )

        source_drone = (
            self._simulator
            .drones[intent.sender]
        )
        destination_drone = (
            self._simulator
            .drones[recipient_node_id]
        )

        payload_limit = int(
            CONFIG[
                "uavnetsim_payload_bytes"
            ]
        )

        injected_bytes = 0
        injected_packets = 0

        remaining = (
            bytes_to_inject
        )

        while remaining > 0:
            payload_bytes = min(
                payload_limit,
                remaining,
            )

            (
                self._uav_config
                .GL_ID_DATA_PACKET
            ) += 1

            packet_id = int(
                self._uav_config
                .GL_ID_DATA_PACKET
            )

            packet_length_bits = int(
                self._uav_config
                .IP_HEADER_LENGTH
                + self._uav_config
                .MAC_HEADER_LENGTH
                + self._uav_config
                .PHY_HEADER_LENGTH
                + payload_bytes
                * 8
            )

            channel_id = (
                source_drone
                .channel_assigner
                .channel_assign()
            )

            packet = (
                self._data_packet_class(
                    src_drone=source_drone,
                    dst_drone=destination_drone,
                    creation_time=(
                        self._environment.now
                    ),
                    data_packet_id=(
                        packet_id
                    ),
                    data_packet_length=(
                        packet_length_bits
                    ),
                    simulator=(
                        self._simulator
                    ),
                    channel_id=(
                        channel_id
                    ),
                )
            )

            packet.transmission_mode = 0
            packet.forced_next_hop_id = (
                recipient_node_id
            )
            packet.mission_tx_power_w = (
                float(
                    intent.tx_power_w
                )
            )

            packet.waiting_start_time = (
                self._environment.now
            )

            if (
                source_drone
                .transmitting_queue
                .qsize()
                >= source_drone.max_queue_size
            ):
                self._metric_state[
                    "queue_limit_events"
                ] += 1
                self._metric_state[
                    "payload_bytes_deferred"
                ] += int(remaining)
                break

            self._simulator.metrics.record_generated(
                packet
            )

            source_drone.transmitting_queue.put(
                packet
            )

            self._packet_records[
                packet_id
            ] = {
                "packet": packet,
                "transfer_key": transfer_key,
                "sender": int(
                    intent.sender
                ),
                "recipient": int(
                    intent.recipient
                ),
                "target_id": int(
                    intent.target_id
                ),
                "source_uav": int(
                    report.source_uav
                ),
                "created_step": int(
                    report.created_step
                ),
                "size_bytes": int(
                    report.size_bytes
                ),
                "ttl_s": float(
                    report.ttl_s
                ),
                "payload_bytes": int(
                    payload_bytes
                ),
                "tx_power_w": float(
                    intent.tx_power_w
                ),
                "status": "in_flight",
            }

            injected_bytes += int(
                payload_bytes
            )
            injected_packets += 1
            remaining -= int(
                payload_bytes
            )

        self._metric_state[
            "packets_injected"
        ] += injected_packets
        self._metric_state[
            "payload_bytes_injected"
        ] += injected_bytes

        deferred_bytes = int(
            remaining
        )

        return {
            "transfer_key": transfer_key,
            "injected_bytes": (
                injected_bytes
            ),
            "injected_packets": (
                injected_packets
            ),
            "deferred_bytes": (
                deferred_bytes
            ),
            "reason": (
                "queue_limited"
                if deferred_bytes > 0
                else "injected"
            ),
        }

    def _collect_packet_outcomes(self):
        delivered_ids = (
            self._simulator
            .metrics
            .datapacket_arrived
        )

        delivered_by_key = {}
        failed_by_key = {}

        max_attempts = int(
            self._uav_config
            .MAX_RETRANSMISSION_ATTEMPT
        )

        for (
            packet_id,
            record,
        ) in self._packet_records.items():
            if (
                record["status"]
                != "in_flight"
            ):
                continue

            packet = record["packet"]
            transfer_key = (
                record["transfer_key"]
            )

            if packet_id in delivered_ids:
                recipient = int(
                    record["recipient"]
                )

                if (
                    recipient != GCS_NODE
                    and not self._uavs[
                        recipient
                    ].active
                ):
                    record["status"] = (
                        "failed"
                    )
                    failed_by_key[
                        transfer_key
                    ] = (
                        failed_by_key.get(
                            transfer_key,
                            0,
                        )
                        + int(
                            record[
                                "payload_bytes"
                            ]
                        )
                    )
                    self._metric_state[
                        "packets_failed"
                    ] += 1
                    continue

                record["status"] = (
                    "delivered"
                )

                delivered_by_key[
                    transfer_key
                ] = (
                    delivered_by_key.get(
                        transfer_key,
                        0,
                    )
                    + int(
                        record[
                            "payload_bytes"
                        ]
                    )
                )

                self._metric_state[
                    "packets_delivered"
                ] += 1
                self._metric_state[
                    "payload_bytes_delivered"
                ] += int(
                    record[
                        "payload_bytes"
                    ]
                )

                continue

            attempts = int(
                packet
                .number_retransmission_attempt
                .get(
                    record["sender"],
                    0,
                )
            )

            expired = bool(
                self._environment.now
                >= (
                    packet.creation_time
                    + packet.deadline
                )
            )

            if (
                attempts
                >= max_attempts
                or expired
            ):
                record["status"] = (
                    "failed"
                )
                failed_by_key[
                    transfer_key
                ] = (
                    failed_by_key.get(
                        transfer_key,
                        0,
                    )
                    + int(
                        record[
                            "payload_bytes"
                        ]
                    )
                )
                self._metric_state[
                    "packets_failed"
                ] += 1

        return (
            delivered_by_key,
            failed_by_key,
        )

    def _generation_matches_record(
        self,
        report,
        record,
    ):
        return bool(
            report.target_id
            == record["target_id"]
            and report.source_uav
            == record["source_uav"]
            and report.created_step
            == record["created_step"]
            and report.size_bytes
            == record["size_bytes"]
            and np.isclose(
                report.ttl_s,
                record["ttl_s"],
            )
        )

    def _commit_delivered_bytes(
        self,
        transfer_key,
        delivered_bytes,
        current_step,
    ):
        delivered_bytes = int(
            delivered_bytes
        )

        if delivered_bytes <= 0:
            return None

        matching_records = [
            record
            for record
            in self._packet_records.values()
            if (
                record["transfer_key"]
                == transfer_key
            )
        ]

        if not matching_records:
            return None

        record = (
            matching_records[0]
        )

        target_id = int(
            record["target_id"]
        )

        if (
            target_id
            in self._gcs_received_target_ids
        ):
            return {
                "status": "already_delivered",
                "committed_bytes": 0,
            }

        sender = int(
            record["sender"]
        )

        sender_report = find_report_in_buffer(
            self._report_buffers[
                sender
            ],
            target_id,
        )

        if (
            sender_report is None
            or not self._generation_matches_record(
                sender_report,
                record,
            )
        ):
            return {
                "status": "stale_generation",
                "committed_bytes": 0,
            }

        intent = TransmissionIntent(
            sender=sender,
            recipient=int(
                record["recipient"]
            ),
            target_id=target_id,
            requested_bytes=(
                delivered_bytes
            ),
            tx_power_w=float(
                record["tx_power_w"]
            ),
        )

        if (
            intent.recipient
            == GCS_NODE
        ):
            result = (
                commit_gcs_transfer_bytes(
                    intent,
                    delivered_bytes,
                    self._report_buffers,
                    self._pending_reports,
                    self._gcs_received_target_ids,
                    self._peer_transfer_states,
                    current_step,
                )
            )

            return {
                "status": (
                    "gcs_delivered"
                    if (
                        result["status"]
                        == "delivered"
                    )
                    else (
                        "gcs_partial"
                        if (
                            result["status"]
                            == "partial"
                        )
                        else result[
                            "status"
                        ]
                    )
                ),
                "committed_bytes": int(
                    result.get(
                        "committed_bytes",
                        0,
                    )
                ),
                "commit_result": result,
            }

        peer_commit = commit_peer_transfer_bytes(
            intent,
            delivered_bytes,
            self._report_buffers[
                sender
            ],
            self._peer_transfer_states,
            current_step,
        )

        state = peer_commit["state"]
        committed_bytes = int(
            peer_commit[
                "committed_bytes"
            ]
        )

        if not peer_commit["complete"]:
            return {
                "status": "peer_partial",
                "committed_bytes": (
                    committed_bytes
                ),
                "peer_state": state,
                "peer_commit_result": (
                    peer_commit
                ),
            }

        key = peer_transfer_key(
            sender,
            intent.recipient,
            target_id,
        )

        custody_result = (
            commit_completed_peer_transfer_to_receiver(
                key,
                self._peer_transfer_states,
                self._report_buffers,
                current_step,
                self._gcs_received_target_ids,
            )
        )

        status_map = {
            "enqueued": "peer_enqueued",
            "buffer_full": (
                "peer_buffer_full"
            ),
            "already_present": (
                "peer_already_present"
            ),
            "expired": "expired",
            "already_delivered": (
                "already_delivered"
            ),
        }

        return {
            "status": status_map.get(
                custody_result["status"],
                custody_result["status"],
            ),
            "committed_bytes": (
                committed_bytes
            ),
            "peer_state": state,
            "peer_commit_result": (
                peer_commit
            ),
            "custody_result": (
                custody_result
            ),
        }

    def _has_inflight_for_key(
        self,
        transfer_key,
    ):
        return any(
            (
                record["status"]
                == "in_flight"
                and record[
                    "transfer_key"
                ]
                == transfer_key
            )
            for record
            in self._packet_records.values()
        )

    def _cleanup_terminal_packet_records(
        self,
    ):
        for (
            packet_id,
            record,
        ) in list(
            self._packet_records.items()
        ):
            if (
                record["status"]
                == "in_flight"
            ):
                continue

            del self._packet_records[
                packet_id
            ]

    def set_step_energy_budget_j(
        self,
        energy_budget_j,
    ):
        if not self._initialized:
            raise RuntimeError(
                "backend must be reset before setting energy budget"
            )
        values = np.asarray(
            energy_budget_j,
            dtype=np.float64,
        )
        expected = (
            int(CONFIG["num_uavs"]),
        )
        if values.shape != expected:
            raise ValueError(
                "energy_budget_j has wrong shape"
            )
        if (
            not np.all(np.isfinite(values))
            or np.any(values < 0.0)
        ):
            raise ValueError(
                "energy_budget_j must be finite and >= 0"
            )
        self._step_energy_budget_j = (
            values.copy()
        )

    def step(
        self,
        requests,
        dt,
        current_step,
    ):
        if not self._initialized:
            raise RuntimeError(
                "backend must be reset before step"
            )

        dt, current_step = (
            self._validate_step_clock(
                dt,
                current_step,
            )
        )

        request_list = (
            self._normalize_requests(
                requests
            )
        )

        self.sync_positions(
            self._uavs
        )

        cleanup_peer_transfer_states(
            self._peer_transfer_states,
            current_step,
        )

        retry_completed_peer_transfers(
            self._peer_transfer_states,
            self._report_buffers,
            current_step,
            self._gcs_received_target_ids,
        )

        if self._step_energy_budget_j is not None:
            for drone, budget_j in zip(
                self._simulator.drones[
                    : int(CONFIG["num_uavs"])
                ],
                self._step_energy_budget_j,
            ):
                drone.residual_energy = float(
                    budget_j
                )
            self._step_energy_budget_j = None

        request_contexts = [
            (
                request,
                self._inject_request(
                    request,
                    current_step,
                ),
            )
            for request
            in request_list
        ]

        energy_before = [
            float(
                drone.residual_energy
            )
            for drone
            in self._simulator.drones[
                : int(
                    CONFIG["num_uavs"]
                )
            ]
        ]

        target_time = (
            self._environment.now
            + dt
            * 1e6
        )

        self._environment.run(
            until=target_time
        )

        energy_after = [
            float(
                drone.residual_energy
            )
            for drone
            in self._simulator.drones[
                : int(
                    CONFIG["num_uavs"]
                )
            ]
        ]

        self._last_step_comm_energy_by_uav = np.asarray(
            [
                max(
                    0.0,
                    before - after,
                )
                for before, after
                in zip(
                    energy_before,
                    energy_after,
                )
            ],
            dtype=np.float64,
        )

        communication_energy = float(
            np.sum(
                self._last_step_comm_energy_by_uav
            )
        )

        self._metric_state[
            "communication_energy_j"
        ] += communication_energy

        (
            delivered_by_key,
            failed_by_key,
        ) = self._collect_packet_outcomes()

        commit_results = {}

        for (
            transfer_key,
            delivered_bytes,
        ) in delivered_by_key.items():
            commit_results[
                transfer_key
            ] = (
                self._commit_delivered_bytes(
                    transfer_key,
                    delivered_bytes,
                    current_step,
                )
            )

        results = []

        for (
            request,
            context,
        ) in request_contexts:
            transfer_key = (
                context["transfer_key"]
            )

            if transfer_key is None:
                result = {
                    "status": "no_transfer",
                    "tx_bytes": 0,
                    "injected_bytes": 0,
                    "injected_packets": 0,
                    "deferred_bytes": int(
                        context.get(
                            "deferred_bytes",
                            0,
                        )
                    ),
                    "reason": (
                        context["reason"]
                    ),
                }
            else:
                delivered_bytes = int(
                    delivered_by_key.get(
                        transfer_key,
                        0,
                    )
                )
                failed_bytes = int(
                    failed_by_key.get(
                        transfer_key,
                        0,
                    )
                )
                commit_result = (
                    commit_results.get(
                        transfer_key
                    )
                )

                if commit_result is not None:
                    status = (
                        commit_result[
                            "status"
                        ]
                    )
                    committed_bytes = int(
                        commit_result.get(
                            "committed_bytes",
                            0,
                        )
                    )
                elif self._has_inflight_for_key(
                    transfer_key
                ):
                    status = "in_flight"
                    committed_bytes = 0
                elif failed_bytes > 0:
                    status = "packet_failed"
                    committed_bytes = 0
                else:
                    status = "no_transfer"
                    committed_bytes = 0

                result = {
                    "status": status,
                    "tx_bytes": (
                        committed_bytes
                    ),
                    "delivered_payload_bytes": (
                        delivered_bytes
                    ),
                    "failed_payload_bytes": (
                        failed_bytes
                    ),
                    "injected_bytes": int(
                        context[
                            "injected_bytes"
                        ]
                    ),
                    "injected_packets": int(
                        context[
                            "injected_packets"
                        ]
                    ),
                    "deferred_bytes": int(
                        context.get(
                            "deferred_bytes",
                            0,
                        )
                    ),
                    "reason": (
                        context["reason"]
                    ),
                    "commit_result": (
                        commit_result
                    ),
                }

            results.append(
                result
            )

            status = str(
                result["status"]
            )

            status_counts = (
                self._metric_state[
                    "status_counts"
                ]
            )

            status_counts[status] = (
                status_counts.get(
                    status,
                    0,
                )
                + 1
            )

        self._metric_state[
            "steps"
        ] += 1
        self._metric_state[
            "requests"
        ] += len(
            request_list
        )

        self._cleanup_terminal_packet_records()

        return results

    @property
    def peer_transfer_states(self):
        if not self._initialized:
            raise RuntimeError(
                "backend must be reset before "
                "accessing peer_transfer_states"
            )

        return self._peer_transfer_states

    def metrics(self):
        network_metrics = {}

        if self._simulator is not None:
            network_metrics = (
                self._simulator
                .metrics
                .snapshot()
            )

        return {
            "backend": "uavnetsim",
            "steps": int(
                self._metric_state[
                    "steps"
                ]
            ),
            "requests": int(
                self._metric_state[
                    "requests"
                ]
            ),
            "packets_injected": int(
                self._metric_state[
                    "packets_injected"
                ]
            ),
            "packets_delivered": int(
                self._metric_state[
                    "packets_delivered"
                ]
            ),
            "packets_failed": int(
                self._metric_state[
                    "packets_failed"
                ]
            ),
            "queue_limit_events": int(
                self._metric_state[
                    "queue_limit_events"
                ]
            ),
            "payload_bytes_deferred": int(
                self._metric_state[
                    "payload_bytes_deferred"
                ]
            ),
            "payload_bytes_injected": int(
                self._metric_state[
                    "payload_bytes_injected"
                ]
            ),
            "payload_bytes_delivered": int(
                self._metric_state[
                    "payload_bytes_delivered"
                ]
            ),
            "communication_energy_j": float(
                self._metric_state[
                    "communication_energy_j"
                ]
            ),
            "in_flight_packets": sum(
                record["status"]
                == "in_flight"
                for record
                in self._packet_records.values()
            ),
            "active_peer_transfers": len(
                self._peer_transfer_states
            ),
            "status_counts": dict(
                self._metric_state[
                    "status_counts"
                ]
            ),
            "network_metrics": dict(
                network_metrics
            ),
        }

    def last_step_communication_energy_by_uav(self):
        if not self._initialized:
            raise RuntimeError(
                "backend must be reset before "
                "reading communication energy"
            )

        return (
            self._last_step_comm_energy_by_uav
            .copy()
        )

    def close(self):
        if self._simulator is not None:
            self._simulator.close()

        if (
            self._scene_directory
            is not None
        ):
            self._scene_directory.cleanup()

        self._scene_directory = None
        self._environment = None
        self._simulator = None
        self._packet_records = {}
        self._initialized = False


# --- frozen notebook cell 117 ---
def create_network_backend(
    backend_name=None,
    **kwargs,
):
    if backend_name is None:
        backend_name = CONFIG.get(
            "network_backend",
            "simple",
        )

    normalized = str(
        backend_name
    ).strip().lower()

    if normalized == "simple":
        if kwargs:
            unexpected = ", ".join(
                sorted(kwargs)
            )
            raise TypeError(
                "SimpleNetworkBackend does not "
                f"accept: {unexpected}"
            )

        return SimpleNetworkBackend()

    if normalized in {
        "uavnetsim",
        "uav_net_sim",
    }:
        return UavNetSimBackend(
            **kwargs
        )

    raise ValueError(
        "network backend must be "
        "'simple' or 'uavnetsim'"
    )


# Export public and internal helpers required by downstream production layers.
__all__ = [name for name in globals() if not name.startswith("__")]
