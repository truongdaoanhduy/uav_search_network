"""Production source migrated from frozen fix_test_gpu.ipynb.

Do not import the notebook at runtime. The notebook is retained only as a
reference/regression artifact. Production changes belong in Python modules.
Migrated from notebook cells 60..107.
"""

from ..world.reports import *  # noqa: F401,F403

# --- frozen notebook cell 60 ---
def distance_3d(position_a, position_b):
    position_a = np.asarray(position_a,dtype=np.float64)
    position_b = np.asarray(position_b, dtype=np.float64)

    if position_a.shape != (3,):
        raise ValueError("position_a must have shape (3,)")

    if position_b.shape != (3,):
        raise ValueError("position_b must have shape (3,)")

    if not np.all(np.isfinite(position_a)):
        raise ValueError("position_a must contain only finite values")

    if not np.all(np.isfinite(position_b)):
        raise ValueError("position_b must contain only finite values")

    return np.linalg.norm(position_a - position_b)


# --- frozen notebook cell 61 ---
def positions_can_communicate(position_a,position_b,max_range_m):
    max_range_m = float(max_range_m)

    if (
        not np.isfinite(max_range_m)
        or max_range_m <= 0.0
    ):
        raise ValueError(
            "max_range_m must be finite and > 0"
        )

    return (distance_3d(position_a,position_b)<= max_range_m)


# --- frozen notebook cell 62 ---
def uavs_can_communicate(uav_a,uav_b):
    if uav_a.id == uav_b.id:
        return False

    if not uav_a.active:
        return False

    if not uav_b.active:
        return False

    return positions_can_communicate(
        uav_a.position,
        uav_b.position,
        CONFIG["peer_contact_range_m"]
    )


# --- frozen notebook cell 63 ---
def uav_can_reach_gcs(uav):
    if not uav.active:
        return False

    return positions_can_communicate(
        uav.position,
        CONFIG["gcs_position"],
        CONFIG["gcs_contact_range_m"],
    )


# --- frozen notebook cell 64 ---
def get_uav_neighbors(uav,uavs):
    neighbors = []

    for other in uavs:
        if uavs_can_communicate(uav,other):
            neighbors.append(other.id)

    return neighbors


# --- frozen notebook cell 65 ---
GCS_NODE = -1


# --- frozen notebook cell 66 ---
SILENT_DESTINATION = None


# --- frozen notebook cell 67 ---
def communication_choices(sender_id,num_uavs=None):
    if num_uavs is None:
        num_uavs = CONFIG["num_uavs"]

    if (isinstance(sender_id, bool) or not isinstance(sender_id,(int, np.integer))):
        raise TypeError("sender_id must be an integer")

    if (isinstance(num_uavs, bool) or not isinstance(num_uavs,(int, np.integer))):
        raise TypeError("num_uavs must be an integer")

    sender_id = int(sender_id)
    num_uavs = int(num_uavs)

    if num_uavs < 1:
        raise ValueError("num_uavs must be >= 1")

    if not 0 <= sender_id < num_uavs:
        raise ValueError("sender_id out of range")

    peers = tuple(
        uav_id
        for uav_id in range(num_uavs)
        if uav_id != sender_id
    )

    return (SILENT_DESTINATION,*peers,GCS_NODE)


# --- frozen notebook cell 68 ---
def decode_destination(
    sender_id,
    destination_index,
    num_uavs=None,
):
    if num_uavs is None:
        num_uavs = CONFIG["num_uavs"]

    if (
        isinstance(
            destination_index,
            (bool, np.bool_),
        )
        or not isinstance(
            destination_index,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "destination_index must be an integer"
        )

    destination_index = int(destination_index)

    choices = communication_choices(
        sender_id,
        num_uavs,
    )

    if not 0<= destination_index< len(choices):
        raise ValueError(
            "destination_index out of range"
        )

    return choices[destination_index]


# --- frozen notebook cell 69 ---
def decode_tx_power_w(power_action):
    power_array = np.asarray(power_action)

    if power_array.shape not in [(), (1,)]:
        raise ValueError(
            "power_action must be a scalar or have shape (1,)"
        )

    if (
        np.issubdtype(power_array.dtype, np.bool_)
        or not np.issubdtype(power_array.dtype, np.number)
    ):
        raise TypeError(
            "power_action must be numeric and not boolean"
        )

    power_value = float(power_array.item())

    if not np.isfinite(power_value):
        raise ValueError(
            "power_action must be finite"
        )

    if not -1.0 <= power_value <= 1.0:
        raise ValueError(
            "power_action must be within [-1, 1]"
        )

    power_min = float(CONFIG["tx_power_min_w"])
    power_max = float(CONFIG["tx_power_max_w"])

    if (
        not np.isfinite(power_min)
        or not np.isfinite(power_max)
        or power_min <= 0.0
        or power_max < power_min
    ):
        raise ValueError(
            "invalid TX power range"
        )

    normalized = (power_value + 1.0) / 2.0

    return float(
        power_min
        + normalized
        * (power_max - power_min)
    )


# --- frozen notebook cell 70 ---
@dataclass
class HybridAction:
    movement: np.ndarray
    destination: int | None
    tx_power_w: float


# --- frozen notebook cell 71 ---
def decode_hybrid_action(
    sender_id,
    action,
    num_uavs=None,
):
    if num_uavs is None:
        num_uavs = CONFIG["num_uavs"]

    if not isinstance(action, dict):
        raise TypeError(
            "action must be a dict"
        )

    required_keys = {
        "motion",
        "destination",
        "power",
    }

    if set(action.keys()) != required_keys:
        raise ValueError(
            "action must contain exactly "
            "motion, destination, power"
        )

    motion_object = np.asarray(
        action["motion"],
        dtype=object,
    )

    if motion_object.shape != (3,):
        raise ValueError(
            "motion must have shape (3,)"
        )

    for value in motion_object:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(
                "motion must be numeric and not boolean"
            )

        if not isinstance(
            value,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError(
                "motion must be numeric and not boolean"
            )

    motion = motion_object.astype(
        np.float64,
    )

    if not np.all(np.isfinite(motion)):
        raise ValueError(
            "motion must contain only finite values"
        )

    if (
        np.any(motion < -1.0)
        or np.any(motion > 1.0)
    ):
        raise ValueError(
            "motion must be within [-1, 1]"
        )

    destination = decode_destination(
        sender_id,
        action["destination"],
        num_uavs,
    )

    decoded_power_w = decode_tx_power_w(action["power"])

    if destination is SILENT_DESTINATION:
        tx_power_w = 0.0
    else:
        tx_power_w = decoded_power_w

    return HybridAction(
        movement=motion.copy(),
        destination=destination,
        tx_power_w=tx_power_w,
    )


# --- frozen notebook cell 72 ---
def select_report_for_transmission(
    buffer,
    current_step,
):
    if (
        isinstance(current_step, (bool, np.bool_))
        or not isinstance(current_step, (int, np.integer))
    ):
        raise TypeError("current_step must be an integer")

    current_step = int(current_step)

    if not isinstance(buffer, list):
        raise TypeError("buffer must be a list")

    for report in buffer:
        if not isinstance(report, Report):
            raise TypeError("buffer must contain only Report objects")

    cleanup_report_buffer(buffer, current_step)

    for report in buffer:
        if not report_is_complete(report):
            return report

    return None


# --- frozen notebook cell 73 ---
@dataclass(frozen=True)
class TransmissionIntent:
    sender: int
    recipient: int
    target_id: int
    requested_bytes: int
    tx_power_w: float

    def __post_init__(self):
        for name, value in (
            ("sender", self.sender),
            ("recipient", self.recipient),
            ("target_id", self.target_id),
            ("requested_bytes", self.requested_bytes),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(
                    value,
                    (int, np.integer),
                )
            ):
                raise TypeError(
                    f"{name} must be an integer"
                )

        sender = int(self.sender)
        recipient = int(self.recipient)
        target_id = int(self.target_id)
        requested_bytes = int(
            self.requested_bytes
        )
        tx_power_w = float(
            self.tx_power_w
        )

        num_uavs = int(
            CONFIG["num_uavs"]
        )

        if not 0 <= sender < num_uavs:
            raise ValueError(
                "sender out of range"
            )

        if (
            recipient != GCS_NODE
            and not 0 <= recipient < num_uavs
        ):
            raise ValueError(
                "recipient must be GCS_NODE "
                "or a valid UAV id"
            )

        if recipient == sender:
            raise ValueError(
                "sender and recipient "
                "must be different"
            )

        if target_id < 0:
            raise ValueError(
                "target_id must be >= 0"
            )

        if requested_bytes <= 0:
            raise ValueError(
                "requested_bytes must be > 0"
            )

        power_min = float(
            CONFIG["tx_power_min_w"]
        )
        power_max = float(
            CONFIG["tx_power_max_w"]
        )

        if (
            not np.isfinite(tx_power_w)
            or not power_min
            <= tx_power_w
            <= power_max
        ):
            raise ValueError(
                "tx_power_w outside "
                "configured range"
            )

        object.__setattr__(
            self,
            "sender",
            sender,
        )
        object.__setattr__(
            self,
            "recipient",
            recipient,
        )
        object.__setattr__(
            self,
            "target_id",
            target_id,
        )
        object.__setattr__(
            self,
            "requested_bytes",
            requested_bytes,
        )
        object.__setattr__(
            self,
            "tx_power_w",
            tx_power_w,
        )


# --- frozen notebook cell 74 ---
def build_transmission_intent(
    sender_id,
    hybrid_action,
    buffer,
    current_step,
    peer_transfer_states=None,
):
    if (
        isinstance(
            sender_id,
            (bool, np.bool_),
        )
        or not isinstance(
            sender_id,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "sender_id must be an integer"
        )

    sender_id = int(sender_id)

    num_uavs = int(
        CONFIG["num_uavs"]
    )

    if not 0 <= sender_id < num_uavs:
        raise ValueError(
            "sender_id out of range"
        )

    if not isinstance(
        hybrid_action,
        HybridAction,
    ):
        raise TypeError(
            "hybrid_action must be HybridAction"
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

    if not isinstance(
        buffer,
        list,
    ):
        raise TypeError(
            "buffer must be a list"
        )

    for report in buffer:
        if not isinstance(
            report,
            Report,
        ):
            raise TypeError(
                "buffer must contain only "
                "Report objects"
            )

    if (
        peer_transfer_states is not None
        and not isinstance(
            peer_transfer_states,
            dict,
        )
    ):
        raise TypeError(
            "peer_transfer_states must be "
            "a dict or None"
        )

    if (
        hybrid_action.destination
        is SILENT_DESTINATION
    ):
        return None

    cleanup_report_buffer(
        buffer,
        current_step,
    )

    destination = (
        hybrid_action.destination
    )

    if destination == GCS_NODE:
        for report in buffer:
            if report_is_complete(
                report
            ):
                continue

            requested_bytes = (
                report_remaining_bytes(
                    report
                )
            )

            if requested_bytes <= 0:
                continue

            return TransmissionIntent(
                sender=sender_id,
                recipient=destination,
                target_id=report.target_id,
                requested_bytes=requested_bytes,
                tx_power_w=(
                    hybrid_action.tx_power_w
                ),
            )

        return None

    # Peer destination: preserve FIFO order, but skip any report
    # that this exact peer already received completely.
    for report in buffer:
        if report_is_complete(
            report
        ):
            continue

        requested_bytes = int(
            report.size_bytes
        )

        if peer_transfer_states is not None:
            key = peer_transfer_key(
                sender_id,
                destination,
                report.target_id,
            )

            state = (
                peer_transfer_states.get(
                    key
                )
            )

            if state is not None:
                if not isinstance(
                    state,
                    PeerTransferState,
                ):
                    raise TypeError(
                        "peer_transfer_states "
                        "values must be "
                        "PeerTransferState"
                    )

                if peer_transfer_is_expired(
                    state,
                    current_step,
                ):
                    del peer_transfer_states[
                        key
                    ]
                    state = None

            if state is not None:
                if (
                    state.source_uav
                    != report.source_uav
                    or state.created_step
                    != report.created_step
                    or state.size_bytes
                    != report.size_bytes
                    or not np.isclose(
                        state.ttl_s,
                        report.ttl_s,
                    )
                ):
                    raise ValueError(
                        "peer transfer state "
                        "does not match "
                        "sender report"
                    )

                requested_bytes = (
                    peer_transfer_remaining_bytes(
                        state
                    )
                )

                if requested_bytes <= 0:
                    # This peer already has the full report.
                    # Continue FIFO search so later reports are
                    # not head-of-line blocked.
                    continue

        return TransmissionIntent(
            sender=sender_id,
            recipient=destination,
            target_id=report.target_id,
            requested_bytes=(
                requested_bytes
            ),
            tx_power_w=(
                hybrid_action.tx_power_w
            ),
        )

    return None


# --- frozen notebook cell 75 ---
def transmission_is_feasible(
    intent,
    uavs,
):
    if not isinstance(
        intent,
        TransmissionIntent,
    ):
        raise TypeError(
            "intent must be TransmissionIntent"
        )

    if not isinstance(uavs, list):
        raise TypeError(
            "uavs must be a list"
        )

    num_uavs = int(
        CONFIG["num_uavs"]
    )

    if len(uavs) != num_uavs:
        raise ValueError(
            "uavs length does not match "
            "CONFIG['num_uavs']"
        )

    for uav in uavs:
        if not isinstance(uav, UAV):
            raise TypeError(
                "uavs must contain only UAV objects"
            )

        if (
            isinstance(uav.id, (bool, np.bool_))
            or not isinstance(
                uav.id,
                (int, np.integer),
            )
        ):
            raise TypeError(
                "each UAV id must be an integer"
            )

    ids = [
        int(uav.id)
        for uav in uavs
    ]

    if len(set(ids)) != num_uavs:
        raise ValueError(
            "UAV ids must be unique"
        )

    if set(ids) != set(range(num_uavs)):
        raise ValueError(
            "UAV ids must be exactly "
            "0..num_uavs-1"
        )

    uavs_by_id = {
        int(uav.id): uav
        for uav in uavs
    }

    sender = uavs_by_id[
        intent.sender
    ]

    if not sender.active:
        return False

    if intent.recipient == GCS_NODE:
        return bool(
            uav_can_reach_gcs(
                sender
            )
        )

    recipient = uavs_by_id[
        intent.recipient
    ]

    return bool(
        uavs_can_communicate(
            sender,
            recipient,
        )
    )


# --- frozen notebook cell 76 ---
LOS_LINK = "los"


# --- frozen notebook cell 77 ---
NLOS_LINK = "nlos"


# --- frozen notebook cell 78 ---
def positions_have_los(
    position_a,
    position_b,
    obstacles,
):
    if obstacles is None:
        obstacles = []

    if not isinstance(obstacles, list):
        raise TypeError("obstacles must be a list")

    for obstacle in obstacles:
        if not isinstance(obstacle, Obstacle):
            raise TypeError(
                "obstacles must contain only Obstacle objects"
            )

    return not segment_intersects_obstacle(
        position_a,
        position_b,
        obstacles,
    )


# --- frozen notebook cell 79 ---
def classify_link_state(
    position_a,
    position_b,
    obstacles,
):
    if positions_have_los(
        position_a,
        position_b,
        obstacles,
    ):
        return LOS_LINK

    return NLOS_LINK


# --- frozen notebook cell 80 ---
def db_to_linear(db_value):
    db_value = float(db_value)

    if not np.isfinite(db_value):
        raise ValueError(
            "db_value must be finite"
        )

    return float(
        10.0 ** (db_value / 10.0)
    )


# --- frozen notebook cell 81 ---
def dbm_to_w(dbm_value):
    dbm_value = float(dbm_value)

    if not np.isfinite(dbm_value):
        raise ValueError(
            "dbm_value must be finite"
        )

    return float(
        10.0 ** (
            (dbm_value - 30.0) / 10.0
        )
    )


# --- frozen notebook cell 82 ---
def calculate_link_snr(
    tx_power_w,
    distance_m,
    additional_loss_db=0.0,
):
    tx_power_w = float(tx_power_w)
    distance_m = float(distance_m)
    additional_loss_db = float(additional_loss_db)

    if (
        not np.isfinite(tx_power_w)
        or tx_power_w <= 0.0
    ):
        raise ValueError(
            "tx_power_w must be finite and > 0"
        )

    if (
        not np.isfinite(distance_m)
        or distance_m < 0.0
    ):
        raise ValueError(
            "distance_m must be finite and >= 0"
        )

    if (
        not np.isfinite(additional_loss_db)
        or additional_loss_db < 0.0
    ):
        raise ValueError(
            "additional_loss_db must be finite and >= 0"
        )

    power_min = float(
        CONFIG["tx_power_min_w"]
    )
    power_max = float(
        CONFIG["tx_power_max_w"]
    )

    if not (
        power_min
        <= tx_power_w
        <= power_max
    ):
        raise ValueError(
            "tx_power_w outside configured range"
        )

    reference_distance_m = float(
        CONFIG["comm_reference_distance_m"]
    )
    path_loss_exponent = float(
        CONFIG["comm_path_loss_exponent"]
    )

    if (
        not np.isfinite(reference_distance_m)
        or reference_distance_m <= 0.0
    ):
        raise ValueError(
            "comm_reference_distance_m "
            "must be finite and > 0"
        )

    if (
        not np.isfinite(path_loss_exponent)
        or path_loss_exponent <= 0.0
    ):
        raise ValueError(
            "comm_path_loss_exponent "
            "must be finite and > 0"
        )

    reference_gain_linear = db_to_linear(
        CONFIG["comm_reference_gain_db"]
    )

    noise_power_w = dbm_to_w(
        CONFIG["comm_noise_power_dbm"]
    )

    if (
        reference_gain_linear <= 0.0
        or noise_power_w <= 0.0
    ):
        raise ValueError(
            "invalid communication model"
        )

    effective_distance_m = max(
        distance_m,
        reference_distance_m,
    )

    channel_power_gain = (
        reference_gain_linear
        * (
            reference_distance_m
            / effective_distance_m
        )
        ** path_loss_exponent
    )

    additional_gain = db_to_linear(
        -additional_loss_db
    )

    received_power_w = (
        tx_power_w
        * channel_power_gain
        * additional_gain
    )

    snr_linear = (
        received_power_w
        / noise_power_w
    )

    return float(snr_linear)


# --- frozen notebook cell 83 ---
def snr_linear_to_db(snr_linear):
    snr_linear = float(snr_linear)

    if not np.isfinite(snr_linear) or snr_linear < 0.0:
        raise ValueError(
            "snr_linear must be finite and >= 0"
        )

    if snr_linear == 0.0:
        return float("-inf")

    return float(
        10.0 * np.log10(snr_linear)
    )


# --- frozen notebook cell 84 ---
def transmission_distance_m(
    intent,
    uavs,
):
    if not isinstance(
        intent,
        TransmissionIntent,
    ):
        raise TypeError(
            "intent must be TransmissionIntent"
        )

    if not isinstance(uavs, list):
        raise TypeError(
            "uavs must be a list"
        )

    uavs_by_id = {
        int(uav.id): uav
        for uav in uavs
        if isinstance(uav, UAV)
    }

    if intent.sender not in uavs_by_id:
        raise ValueError(
            "sender UAV not found"
        )

    sender = uavs_by_id[
        intent.sender
    ]

    if intent.recipient == GCS_NODE:
        return distance_3d(
            sender.position,
            CONFIG["gcs_position"],
        )

    if intent.recipient not in uavs_by_id:
        raise ValueError(
            "recipient UAV not found"
        )

    recipient = uavs_by_id[
        intent.recipient
    ]

    return distance_3d(
        sender.position,
        recipient.position,
    )


# --- frozen notebook cell 85 ---
def calculate_intent_snr(
    intent,
    uavs,
    obstacles=None,
):
    if obstacles is None:
        obstacles = []

    if not transmission_is_feasible(
        intent,
        uavs,
    ):
        return 0.0

    distance_m = transmission_distance_m(
        intent,
        uavs,
    )

    uavs_by_id = {
        int(uav.id): uav
        for uav in uavs
        if isinstance(uav, UAV)
    }

    sender_position = uavs_by_id[
        intent.sender
    ].position

    if intent.recipient == GCS_NODE:
        recipient_position = np.asarray(
            CONFIG["gcs_position"],
            dtype=np.float64,
        )
    else:
        recipient_position = uavs_by_id[
            intent.recipient
        ].position

    link_state = classify_link_state(
        sender_position,
        recipient_position,
        obstacles,
    )

    if link_state == LOS_LINK:
        additional_loss_db = 0.0
    else:
        additional_loss_db = float(
            CONFIG["comm_nlos_additional_loss_db"]
        )

    return calculate_link_snr(
        intent.tx_power_w,
        distance_m,
        additional_loss_db=additional_loss_db,
    )


# --- frozen notebook cell 86 ---
def calculate_link_rate_bps(
    snr_linear,
):
    snr_linear = float(
        snr_linear
    )

    if (
        not np.isfinite(
            snr_linear
        )
        or snr_linear < 0.0
    ):
        raise ValueError(
            "snr_linear must be "
            "finite and >= 0"
        )

    bandwidth_hz = float(
        CONFIG[
            "comm_bandwidth_hz"
        ]
    )
    max_rate_bps = float(
        CONFIG[
            "comm_max_link_rate_bps"
        ]
    )

    if (
        not np.isfinite(
            bandwidth_hz
        )
        or bandwidth_hz <= 0.0
    ):
        raise ValueError(
            "comm_bandwidth_hz "
            "must be finite and > 0"
        )

    if (
        not np.isfinite(
            max_rate_bps
        )
        or max_rate_bps <= 0.0
    ):
        raise ValueError(
            "comm_max_link_rate_bps "
            "must be finite and > 0"
        )

    if snr_linear == 0.0:
        return 0.0

    shannon_rate_bps = float(
        bandwidth_hz
        * np.log1p(
            snr_linear
        )
        / np.log(2.0)
    )

    return min(
        shannon_rate_bps,
        max_rate_bps,
    )


# --- frozen notebook cell 87 ---
def calculate_intent_rate_bps(
    intent,
    uavs,
    obstacles=None,
):
    snr_linear = (
        calculate_intent_snr(
            intent,
            uavs,
            obstacles=obstacles,
        )
    )

    threshold_db = float(
        CONFIG[
            "comm_snr_threshold_db"
        ]
    )

    if (
        not np.isfinite(
            threshold_db
        )
    ):
        raise ValueError(
            "comm_snr_threshold_db "
            "must be finite"
        )

    if (
        snr_linear_to_db(
            snr_linear
        )
        < threshold_db
    ):
        return 0.0

    return calculate_link_rate_bps(
        snr_linear
    )


# --- frozen notebook cell 88 ---
def calculate_intent_tx_bytes(
    intent,
    uavs,
    obstacles=None,
    dt=None,
):
    if dt is None:
        dt = CONFIG["dt"]

    dt = float(dt)

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(
            "dt must be finite and > 0"
        )

    rate_bps = calculate_intent_rate_bps(
        intent,
        uavs,
        obstacles=obstacles,
    )

    capacity_bytes = (
        rate_bps
        * dt
        / 8.0
    )

    transferable_bytes = int(
        np.floor(capacity_bytes)
    )

    return min(
        int(intent.requested_bytes),
        transferable_bytes,
    )


# --- frozen notebook cell 89 ---
@dataclass
class PeerTransferState:
    sender: int
    recipient: int
    target_id: int
    source_uav: int
    created_step: int
    size_bytes: int
    ttl_s: float
    received_bytes: int = 0

    def __post_init__(self):
        for name, value in (
            ("sender", self.sender),
            ("recipient", self.recipient),
            ("target_id", self.target_id),
            ("source_uav", self.source_uav),
            ("created_step", self.created_step),
            ("size_bytes", self.size_bytes),
            ("received_bytes", self.received_bytes),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
            ):
                raise TypeError(
                    f"{name} must be an integer"
                )

        self.sender = int(self.sender)
        self.recipient = int(self.recipient)
        self.target_id = int(self.target_id)
        self.source_uav = int(self.source_uav)
        self.created_step = int(self.created_step)
        self.size_bytes = int(self.size_bytes)
        self.received_bytes = int(self.received_bytes)
        self.ttl_s = float(self.ttl_s)

        num_uavs = int(CONFIG["num_uavs"])

        if not 0 <= self.sender < num_uavs:
            raise ValueError("sender out of range")

        if not 0 <= self.recipient < num_uavs:
            raise ValueError("recipient out of range")

        if self.sender == self.recipient:
            raise ValueError(
                "sender and recipient must be different"
            )

        if self.target_id < 0:
            raise ValueError(
                "target_id must be >= 0"
            )

        if not 0 <= self.source_uav < num_uavs:
            raise ValueError(
                "source_uav out of range"
            )

        if self.created_step < 0:
            raise ValueError(
                "created_step must be >= 0"
            )

        if self.size_bytes <= 0:
            raise ValueError(
                "size_bytes must be > 0"
            )

        if (
            not np.isfinite(self.ttl_s)
            or self.ttl_s <= 0.0
        ):
            raise ValueError(
                "ttl_s must be finite and > 0"
            )

        if not (
            0
            <= self.received_bytes
            <= self.size_bytes
        ):
            raise ValueError(
                "received_bytes must be in "
                "[0, size_bytes]"
            )


# --- frozen notebook cell 90 ---
def make_info_pickle_safe(value):
    """Convert dataclasses in public info payloads to plain serializable data."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            key: make_info_pickle_safe(item)
            for key, item in asdict(value).items()
        }

    if isinstance(value, dict):
        return {
            key: make_info_pickle_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            make_info_pickle_safe(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            make_info_pickle_safe(item)
            for item in value
        )

    if isinstance(value, set):
        return [
            make_info_pickle_safe(item)
            for item in sorted(
                value,
                key=repr,
            )
        ]

    if isinstance(value, np.generic):
        return value.item()

    return value


# --- frozen notebook cell 91 ---
def create_peer_transfer_states():
    return {}


# --- frozen notebook cell 92 ---
def peer_transfer_key(
    sender,
    recipient,
    target_id,
):
    for name, value in (
        ("sender", sender),
        ("recipient", recipient),
        ("target_id", target_id),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
        ):
            raise TypeError(
                f"{name} must be an integer"
            )

    sender = int(sender)
    recipient = int(recipient)
    target_id = int(target_id)

    num_uavs = int(CONFIG["num_uavs"])

    if not 0 <= sender < num_uavs:
        raise ValueError(
            "sender out of range"
        )

    if not 0 <= recipient < num_uavs:
        raise ValueError(
            "recipient out of range"
        )

    if sender == recipient:
        raise ValueError(
            "sender and recipient must be different"
        )

    if target_id < 0:
        raise ValueError(
            "target_id must be >= 0"
        )

    return (
        sender,
        recipient,
        target_id,
    )


# --- frozen notebook cell 93 ---
def peer_transfer_remaining_bytes(state):
    if not isinstance(
        state,
        PeerTransferState,
    ):
        raise TypeError(
            "state must be PeerTransferState"
        )

    return max(
        0,
        int(
            state.size_bytes
            - state.received_bytes
        ),
    )


# --- frozen notebook cell 94 ---
def peer_transfer_is_complete(state):
    return (peer_transfer_remaining_bytes(state)== 0)


# --- frozen notebook cell 95 ---
def peer_transfer_age_s(
    state,
    current_step,
):
    if not isinstance(
        state,
        PeerTransferState,
    ):
        raise TypeError(
            "state must be PeerTransferState"
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

    if current_step < state.created_step:
        raise ValueError(
            "current_step must be >= "
            "state.created_step"
        )

    dt = float(CONFIG["dt"])

    if (
        not np.isfinite(dt)
        or dt <= 0.0
    ):
        raise ValueError(
            "CONFIG['dt'] must be finite and > 0"
        )

    return float((current_step - state.created_step)* dt)


# --- frozen notebook cell 96 ---
def peer_transfer_is_expired(
    state,
    current_step,
):
    return peer_transfer_age_s(state,current_step)>= state.ttl_s


# --- frozen notebook cell 97 ---
def find_report_in_buffer(
    buffer,
    target_id,
):
    if not isinstance(buffer, list):
        raise TypeError("buffer must be a list")

    if (
        isinstance(target_id, (bool, np.bool_))
        or not isinstance(
            target_id,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "target_id must be an integer"
        )

    target_id = int(target_id)

    for report in buffer:
        if not isinstance(
            report,
            Report,
        ):
            raise TypeError(
                "buffer must contain only "
                "Report objects"
            )

        if report.target_id == target_id:
            return report

    return None


# --- frozen notebook cell 98 ---
def get_or_create_peer_transfer_state(
    intent,
    sender_buffer,
    peer_transfer_states,
    current_step,
):
    if not isinstance(
        intent,
        TransmissionIntent,
    ):
        raise TypeError(
            "intent must be TransmissionIntent"
        )

    if intent.recipient == GCS_NODE:
        raise ValueError(
            "peer transfer state is only "
            "for UAV-to-UAV transmissions"
        )

    if not isinstance(
        peer_transfer_states,
        dict,
    ):
        raise TypeError(
            "peer_transfer_states must be a dict"
        )

    report = find_report_in_buffer(
        sender_buffer,
        intent.target_id,
    )

    if report is None:
        raise ValueError(
            "intent target report not found "
            "in sender buffer"
        )

    if report_is_expired(
        report,
        current_step,
    ):
        raise ValueError(
            "cannot create transfer state "
            "for an expired report"
        )

    key = peer_transfer_key(
        intent.sender,
        intent.recipient,
        intent.target_id,
    )

    state = peer_transfer_states.get(
        key
    )

    if state is not None:
        if not isinstance(
            state,
            PeerTransferState,
        ):
            raise TypeError(
                "peer_transfer_states values "
                "must be PeerTransferState"
            )

        # A stale partial copy must not block a fresh report with the
        # same target_id after the original report lifetime expires.
        if peer_transfer_is_expired(
            state,
            current_step,
        ):
            del peer_transfer_states[key]
            state = None

    if state is not None:
        if (
            state.source_uav
            != report.source_uav
            or state.created_step
            != report.created_step
            or state.size_bytes
            != report.size_bytes
            or not np.isclose(
                state.ttl_s,
                report.ttl_s,
            )
        ):
            raise ValueError(
                "existing peer transfer state "
                "does not match sender report"
            )

        return state

    state = PeerTransferState(
        sender=int(intent.sender),
        recipient=int(intent.recipient),
        target_id=int(intent.target_id),
        source_uav=int(report.source_uav),
        created_step=int(report.created_step),
        size_bytes=int(report.size_bytes),
        ttl_s=float(report.ttl_s),
        received_bytes=0,
    )

    peer_transfer_states[key] = state

    return state


# --- frozen notebook cell 99 ---
def commit_peer_transfer_bytes(
    intent,
    tx_bytes,
    sender_buffer,
    peer_transfer_states,
    current_step,
):
    if not isinstance(
        intent,
        TransmissionIntent,
    ):
        raise TypeError(
            "intent must be TransmissionIntent"
        )

    if (
        isinstance(tx_bytes, (bool, np.bool_))
        or not isinstance(
            tx_bytes,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "tx_bytes must be an integer"
        )

    tx_bytes = int(tx_bytes)

    if tx_bytes < 0:
        raise ValueError(
            "tx_bytes must be >= 0"
        )

    if tx_bytes > intent.requested_bytes:
        raise ValueError(
            "tx_bytes cannot exceed "
            "intent.requested_bytes"
        )

    state = get_or_create_peer_transfer_state(
        intent,
        sender_buffer,
        peer_transfer_states,
        current_step,
    )

    if peer_transfer_is_expired(
        state,
        current_step,
    ):
        raise ValueError(
            "cannot commit bytes to an "
            "expired peer transfer"
        )

    remaining = (
        peer_transfer_remaining_bytes(
            state
        )
    )

    committed_bytes = min(
        tx_bytes,
        remaining,
    )

    state.received_bytes += (
        committed_bytes
    )

    return {
        "state": state,
        "committed_bytes": (
            committed_bytes
        ),
        "remaining_bytes": (
            peer_transfer_remaining_bytes(
                state
            )
        ),
        "complete": (
            peer_transfer_is_complete(
                state
            )
        ),
    }


# --- frozen notebook cell 100 ---
def cleanup_peer_transfer_states(
    peer_transfer_states,
    current_step,
):
    if not isinstance(
        peer_transfer_states,
        dict,
    ):
        raise TypeError(
            "peer_transfer_states must be a dict"
        )

    expired_keys = []

    for key, state in list(
        peer_transfer_states.items()
    ):
        if not isinstance(
            state,
            PeerTransferState,
        ):
            raise TypeError(
                "peer_transfer_states values "
                "must be PeerTransferState"
            )

        if peer_transfer_is_expired(
            state,
            current_step,
        ):
            expired_keys.append(key)
            del peer_transfer_states[key]

    return expired_keys


# --- frozen notebook cell 101 ---
def known_gcs_delivered_bytes_for_transfer(
    state,
    report_buffers,
):
    if not isinstance(
        state,
        PeerTransferState,
    ):
        raise TypeError(
            "state must be PeerTransferState"
        )

    if not isinstance(
        report_buffers,
        list,
    ):
        raise TypeError(
            "report_buffers must be a list"
        )

    known_bytes = 0

    for buffer in report_buffers:
        if not isinstance(
            buffer,
            list,
        ):
            raise TypeError(
                "each report buffer must be a list"
            )

        for report in buffer:
            if not isinstance(
                report,
                Report,
            ):
                raise TypeError(
                    "report buffers must contain "
                    "only Report objects"
                )

            if report.target_id != state.target_id:
                continue

            # Only merge progress from the same report generation.
            if (
                report.source_uav
                != state.source_uav
                or report.created_step
                != state.created_step
                or report.size_bytes
                != state.size_bytes
                or not np.isclose(
                    report.ttl_s,
                    state.ttl_s,
                )
            ):
                continue

            known_bytes = max(
                known_bytes,
                int(
                    report.delivered_bytes
                ),
            )

    return min(
        known_bytes,
        int(state.size_bytes),
    )


# --- frozen notebook cell 102 ---
def commit_completed_peer_transfer_to_receiver(
    state_key,
    peer_transfer_states,
    report_buffers,
    current_step,
    gcs_received_target_ids=None,
):
    if not isinstance(
        peer_transfer_states,
        dict,
    ):
        raise TypeError(
            "peer_transfer_states must be a dict"
        )

    if not isinstance(
        report_buffers,
        list,
    ):
        raise TypeError(
            "report_buffers must be a list"
        )

    num_uavs = int(
        CONFIG["num_uavs"]
    )

    if len(report_buffers) != num_uavs:
        raise ValueError(
            "report_buffers length does not match "
            "CONFIG['num_uavs']"
        )

    for buffer in report_buffers:
        if not isinstance(buffer, list):
            raise TypeError(
                "each report buffer must be a list"
            )

    if gcs_received_target_ids is None:
        gcs_received_target_ids = set()

    if not isinstance(
        gcs_received_target_ids,
        set,
    ):
        raise TypeError(
            "gcs_received_target_ids must be a set"
        )

    if (
        not isinstance(state_key, tuple)
        or len(state_key) != 3
    ):
        raise TypeError(
            "state_key must be a "
            "(sender, recipient, target_id) tuple"
        )

    key = peer_transfer_key(
        state_key[0],
        state_key[1],
        state_key[2],
    )

    state = peer_transfer_states.get(
        key
    )

    if state is None:
        raise ValueError(
            "peer transfer state not found"
        )

    if not isinstance(
        state,
        PeerTransferState,
    ):
        raise TypeError(
            "peer_transfer_states values must "
            "be PeerTransferState"
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

    if not peer_transfer_is_complete(
        state
    ):
        return {
            "status": "incomplete",
            "report": None,
            "remaining_bytes": (
                peer_transfer_remaining_bytes(
                    state
                )
            ),
        }

    if peer_transfer_is_expired(
        state,
        current_step,
    ):
        del peer_transfer_states[key]

        return {
            "status": "expired",
            "report": None,
            "remaining_bytes": 0,
        }

    if (
        state.target_id
        in gcs_received_target_ids
    ):
        del peer_transfer_states[key]

        return {
            "status": "already_delivered",
            "report": None,
            "remaining_bytes": 0,
        }

    receiver_buffer = report_buffers[
        state.recipient
    ]

    cleanup_report_buffer(
        receiver_buffer,
        current_step,
    )

    known_gcs_delivered_bytes = (
        known_gcs_delivered_bytes_for_transfer(
            state,
            report_buffers,
        )
    )

    existing_report = find_report_in_buffer(
        receiver_buffer,
        state.target_id,
    )

    if existing_report is not None:
        existing_report.delivered_bytes = max(
            int(existing_report.delivered_bytes),
            int(known_gcs_delivered_bytes),
        )
        # The receiver already has a full copy. Keep the completed
        # transfer state so this sender does not retransmit it.
        return {
            "status": "already_present",
            "report": existing_report,
            "remaining_bytes": 0,
        }

    received_report = Report(
        target_id=state.target_id,
        source_uav=state.source_uav,
        created_step=state.created_step,
        size_bytes=state.size_bytes,
        ttl_s=state.ttl_s,
        delivered_bytes=(
            known_gcs_delivered_bytes
        ),
    )

    enqueued, reason = enqueue_report(
        receiver_buffer,
        received_report,
    )

    if enqueued:
        # Keep the completed state as a lightweight receipt that this
        # sender already transferred the full report to this recipient.
        # build_transmission_intent() will therefore return None for the
        # same hop instead of sending the same report again.
        return {
            "status": "enqueued",
            "report": received_report,
            "remaining_bytes": 0,
        }

    if reason == "buffer_full":
        return {
            "status": "buffer_full",
            "report": None,
            "remaining_bytes": 0,
        }

    if reason == "duplicate":
        existing_report = find_report_in_buffer(
            receiver_buffer,
            state.target_id,
        )

        return {
            "status": "already_present",
            "report": existing_report,
            "remaining_bytes": 0,
        }

    raise RuntimeError(
        f"unexpected enqueue result: {reason}"
    )


# --- frozen notebook cell 103 ---
def reports_are_same_generation(
    report_a,
    report_b,
):
    if not isinstance(
        report_a,
        Report,
    ):
        raise TypeError(
            "report_a must be Report"
        )

    if not isinstance(
        report_b,
        Report,
    ):
        raise TypeError(
            "report_b must be Report"
        )

    return bool(
        report_a.target_id
        == report_b.target_id
        and report_a.source_uav
        == report_b.source_uav
        and report_a.created_step
        == report_b.created_step
        and report_a.size_bytes
        == report_b.size_bytes
        and np.isclose(
            report_a.ttl_s,
            report_b.ttl_s,
        )
    )


# --- frozen notebook cell 104 ---
def remove_peer_transfer_states_for_target(
    peer_transfer_states,
    target_id,
):
    if not isinstance(
        peer_transfer_states,
        dict,
    ):
        raise TypeError(
            "peer_transfer_states must be a dict"
        )

    if (
        isinstance(target_id, (bool, np.bool_))
        or not isinstance(
            target_id,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "target_id must be an integer"
        )

    target_id = int(target_id)

    if target_id < 0:
        raise ValueError(
            "target_id must be >= 0"
        )

    removed_keys = []

    for key, state in list(
        peer_transfer_states.items()
    ):
        if not isinstance(
            state,
            PeerTransferState,
        ):
            raise TypeError(
                "peer_transfer_states values "
                "must be PeerTransferState"
            )

        if state.target_id == target_id:
            removed_keys.append(key)
            del peer_transfer_states[key]

    return removed_keys


# --- frozen notebook cell 105 ---
def commit_gcs_transfer_bytes(
    intent,
    tx_bytes,
    report_buffers,
    pending_reports,
    gcs_received_target_ids,
    peer_transfer_states,
    current_step,
):
    if not isinstance(
        intent,
        TransmissionIntent,
    ):
        raise TypeError(
            "intent must be TransmissionIntent"
        )

    if intent.recipient != GCS_NODE:
        raise ValueError(
            "commit_gcs_transfer_bytes "
            "requires a GCS intent"
        )

    if (
        isinstance(tx_bytes, (bool, np.bool_))
        or not isinstance(
            tx_bytes,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "tx_bytes must be an integer"
        )

    tx_bytes = int(tx_bytes)

    if tx_bytes < 0:
        raise ValueError(
            "tx_bytes must be >= 0"
        )

    if tx_bytes > intent.requested_bytes:
        raise ValueError(
            "tx_bytes cannot exceed "
            "intent.requested_bytes"
        )

    if not isinstance(
        report_buffers,
        list,
    ):
        raise TypeError(
            "report_buffers must be a list"
        )

    if len(report_buffers) != int(
        CONFIG["num_uavs"]
    ):
        raise ValueError(
            "report_buffers length does not "
            "match CONFIG['num_uavs']"
        )

    for buffer in report_buffers:
        if not isinstance(
            buffer,
            list,
        ):
            raise TypeError(
                "each report buffer "
                "must be a list"
            )

        for report in buffer:
            if not isinstance(
                report,
                Report,
            ):
                raise TypeError(
                    "report buffers must contain "
                    "only Report objects"
                )

    if not isinstance(
        pending_reports,
        list,
    ):
        raise TypeError(
            "pending_reports must be a list"
        )

    for report in pending_reports:
        if not isinstance(
            report,
            Report,
        ):
            raise TypeError(
                "pending_reports must contain "
                "only Report objects"
            )

    if not isinstance(
        gcs_received_target_ids,
        set,
    ):
        raise TypeError(
            "gcs_received_target_ids "
            "must be a set"
        )

    if not isinstance(
        peer_transfer_states,
        dict,
    ):
        raise TypeError(
            "peer_transfer_states must be a dict"
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

    current_step = int(
        current_step
    )

    if current_step < 0:
        raise ValueError(
            "current_step must be >= 0"
        )

    if (
        intent.target_id
        in gcs_received_target_ids
    ):
        # Make this path idempotent: if stale network copies remain for
        # any reason, calling the commit again restores the invariant
        # that a GCS-delivered target has no buffered/pending copies.
        cleanup_result = (
            mark_target_delivered_to_gcs(
                intent.target_id,
                gcs_received_target_ids,
                report_buffers,
                pending_reports,
            )
        )

        removed_peer_states = (
            remove_peer_transfer_states_for_target(
                peer_transfer_states,
                intent.target_id,
            )
        )

        return {
            "status": "already_delivered",
            "committed_bytes": 0,
            "delivered_bytes": 0,
            "remaining_bytes": 0,
            "removed_peer_states": (
                removed_peer_states
            ),
            "cleanup_result": (
                cleanup_result
            ),
        }

    sender_buffer = report_buffers[
        intent.sender
    ]

    sender_report = (
        find_report_in_buffer(
            sender_buffer,
            intent.target_id,
        )
    )

    if sender_report is None:
        raise ValueError(
            "intent target report "
            "not found in sender buffer"
        )

    if report_is_expired(
        sender_report,
        current_step,
    ):
        raise ValueError(
            "cannot commit an "
            "expired report to GCS"
        )

    report_copies = []

    for buffer in report_buffers:
        for report in buffer:
            if reports_are_same_generation(
                report,
                sender_report,
            ):
                report_copies.append(
                    report
                )

    if not report_copies:
        raise RuntimeError(
            "no matching report copies found"
        )

    known_delivered_bytes = max(
        int(report.delivered_bytes)
        for report in report_copies
    )

    remaining_bytes = max(
        0,
        int(sender_report.size_bytes)
        - known_delivered_bytes,
    )

    committed_bytes = min(
        tx_bytes,
        remaining_bytes,
    )

    new_delivered_bytes = (
        known_delivered_bytes
        + committed_bytes
    )

    for report in report_copies:
        report.delivered_bytes = (
            new_delivered_bytes
        )

    complete = (
        new_delivered_bytes
        >= sender_report.size_bytes
    )

    removed_peer_states = []

    if complete:
        mark_target_delivered_to_gcs(
            intent.target_id,
            gcs_received_target_ids,
            report_buffers,
            pending_reports,
        )

        removed_peer_states = (
            remove_peer_transfer_states_for_target(
                peer_transfer_states,
                intent.target_id,
            )
        )

    return {
        "status": (
            "delivered"
            if complete
            else "partial"
        ),
        "committed_bytes": (
            committed_bytes
        ),
        "delivered_bytes": (
            new_delivered_bytes
        ),
        "remaining_bytes": max(
            0,
            int(sender_report.size_bytes)
            - new_delivered_bytes,
        ),
        "removed_peer_states": (
            removed_peer_states
        ),
    }


# --- frozen notebook cell 106 ---
def retry_completed_peer_transfers(
    peer_transfer_states,
    report_buffers,
    current_step,
    gcs_received_target_ids,
):
    if not isinstance(
        peer_transfer_states,
        dict,
    ):
        raise TypeError(
            "peer_transfer_states must be a dict"
        )

    if not isinstance(
        report_buffers,
        list,
    ):
        raise TypeError(
            "report_buffers must be a list"
        )

    if len(report_buffers) != int(
        CONFIG["num_uavs"]
    ):
        raise ValueError(
            "report_buffers length does not "
            "match CONFIG['num_uavs']"
        )

    for buffer in report_buffers:
        if not isinstance(buffer, list):
            raise TypeError(
                "each report buffer must be a list"
            )

    if not isinstance(
        gcs_received_target_ids,
        set,
    ):
        raise TypeError(
            "gcs_received_target_ids "
            "must be a set"
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

    results = []

    for key, state in list(
        peer_transfer_states.items()
    ):
        if not isinstance(
            state,
            PeerTransferState,
        ):
            raise TypeError(
                "peer_transfer_states values "
                "must be PeerTransferState"
            )

        if not peer_transfer_is_complete(
            state
        ):
            continue

        receiver_buffer = report_buffers[
            state.recipient
        ]

        # Remove stale receiver copies before deciding whether the
        # completed state can act as a receipt.
        cleanup_report_buffer(
            receiver_buffer,
            current_step,
        )

        # Expiration and global GCS delivery take precedence over receipt
        # retention. commit_completed_peer_transfer_to_receiver() removes
        # the completed state in both cases.
        if (
            peer_transfer_is_expired(
                state,
                current_step,
            )
            or state.target_id
            in gcs_received_target_ids
        ):
            result = (
                commit_completed_peer_transfer_to_receiver(
                    key,
                    peer_transfer_states,
                    report_buffers,
                    current_step,
                    gcs_received_target_ids,
                )
            )

            results.append(
                {
                    "key": key,
                    "status": result["status"],
                    "result": result,
                }
            )
            continue

        # A non-expired completed state is also used as a receipt after
        # the receiver has a full copy. Do not repeatedly re-commit it.
        existing_report = find_report_in_buffer(
            receiver_buffer,
            state.target_id,
        )

        if (
            existing_report is not None
            and reports_are_same_generation(
                existing_report,
                Report(
                    target_id=state.target_id,
                    source_uav=state.source_uav,
                    created_step=state.created_step,
                    size_bytes=state.size_bytes,
                    ttl_s=state.ttl_s,
                    delivered_bytes=0,
                ),
            )
        ):
            results.append(
                {
                    "key": key,
                    "status": "receipt_present",
                }
            )
            continue

        result = (
            commit_completed_peer_transfer_to_receiver(
                key,
                peer_transfer_states,
                report_buffers,
                current_step,
                gcs_received_target_ids,
            )
        )

        results.append(
            {
                "key": key,
                "status": result["status"],
                "result": result,
            }
        )

    return results


# --- frozen notebook cell 107 ---
def execute_transmission_intent(
    intent,
    uavs,
    report_buffers,
    pending_reports,
    gcs_received_target_ids,
    peer_transfer_states,
    current_step,
    obstacles=None,
    dt=None,
):
    if not isinstance(
        intent,
        TransmissionIntent,
    ):
        raise TypeError(
            "intent must be TransmissionIntent"
        )

    if not isinstance(uavs, list):
        raise TypeError(
            "uavs must be a list"
        )

    if not isinstance(
        report_buffers,
        list,
    ):
        raise TypeError(
            "report_buffers must be a list"
        )

    if len(report_buffers) != int(
        CONFIG["num_uavs"]
    ):
        raise ValueError(
            "report_buffers length does not "
            "match CONFIG['num_uavs']"
        )

    if not isinstance(
        pending_reports,
        list,
    ):
        raise TypeError(
            "pending_reports must be a list"
        )

    if not isinstance(
        gcs_received_target_ids,
        set,
    ):
        raise TypeError(
            "gcs_received_target_ids "
            "must be a set"
        )

    if not isinstance(
        peer_transfer_states,
        dict,
    ):
        raise TypeError(
            "peer_transfer_states must be a dict"
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

    if obstacles is None:
        obstacles = []

    if not isinstance(obstacles, list):
        raise TypeError(
            "obstacles must be a list"
        )

    retry_results = (
        retry_completed_peer_transfers(
            peer_transfer_states,
            report_buffers,
            current_step,
            gcs_received_target_ids,
        )
    )

    # A stale intent can survive after another copy reaches GCS.
    # Restore the global-delivery invariant before doing any radio work.
    if (
        intent.target_id
        in gcs_received_target_ids
    ):
        cleanup_result = (
            mark_target_delivered_to_gcs(
                intent.target_id,
                gcs_received_target_ids,
                report_buffers,
                pending_reports,
            )
        )

        removed_peer_states = (
            remove_peer_transfer_states_for_target(
                peer_transfer_states,
                intent.target_id,
            )
        )

        return {
            "status": "already_delivered",
            "tx_bytes": 0,
            "retry_results": retry_results,
            "cleanup_result": cleanup_result,
            "removed_peer_states": (
                removed_peer_states
            ),
        }

    tx_bytes = calculate_intent_tx_bytes(
        intent,
        uavs,
        obstacles=obstacles,
        dt=dt,
    )

    if tx_bytes <= 0:
        return {
            "status": "no_transfer",
            "tx_bytes": 0,
            "retry_results": retry_results,
        }

    if intent.recipient == GCS_NODE:
        commit_result = (
            commit_gcs_transfer_bytes(
                intent,
                tx_bytes,
                report_buffers,
                pending_reports,
                gcs_received_target_ids,
                peer_transfer_states,
                current_step,
            )
        )

        return {
            "status": (
                "gcs_"
                + commit_result["status"]
            ),
            "tx_bytes": tx_bytes,
            "retry_results": retry_results,
            "commit_result": commit_result,
        }

    sender_buffer = report_buffers[
        intent.sender
    ]

    peer_result = (
        commit_peer_transfer_bytes(
            intent,
            tx_bytes,
            sender_buffer,
            peer_transfer_states,
            current_step,
        )
    )

    custody_result = None

    if peer_result["complete"]:
        key = peer_transfer_key(
            intent.sender,
            intent.recipient,
            intent.target_id,
        )

        custody_result = (
            commit_completed_peer_transfer_to_receiver(
                key,
                peer_transfer_states,
                report_buffers,
                current_step,
                gcs_received_target_ids,
            )
        )

    if custody_result is None:
        status = "peer_partial"
    else:
        status = (
            "peer_"
            + custody_result["status"]
        )

    return {
        "status": status,
        "tx_bytes": tx_bytes,
        "retry_results": retry_results,
        "peer_result": peer_result,
        "custody_result": custody_result,
    }


# Export public and internal helpers required by downstream production layers.
__all__ = [name for name in globals() if not name.startswith("__")]
