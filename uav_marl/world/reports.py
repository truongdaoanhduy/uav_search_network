"""Production source migrated from frozen fix_test_gpu.ipynb.

Do not import the notebook at runtime. The notebook is retained only as a
reference/regression artifact. Production changes belong in Python modules.
Migrated from notebook cells 40..59.
"""

from .sensing import *  # noqa: F401,F403

# --- frozen notebook cell 40 ---
def pending_source_used_bytes(
    pending_reports,
    source_uav,
):
    return sum(
        int(report.size_bytes)
        for report
        in pending_reports
        if int(report.source_uav)
        == int(source_uav)
    )


# --- frozen notebook cell 41 ---
def enqueue_pending_report(
    pending_reports,
    report,
):
    if not isinstance(
        pending_reports,
        list,
    ):
        raise TypeError(
            "pending_reports must be a list"
        )

    if not isinstance(
        report,
        Report,
    ):
        raise TypeError(
            "report must be a Report"
        )

    duplicate = any(
        (
            item.target_id
            == report.target_id
            and item.source_uav
            == report.source_uav
        )
        for item
        in pending_reports
    )

    if duplicate:
        return False, "duplicate"

    used_bytes = (
        pending_source_used_bytes(
            pending_reports,
            report.source_uav,
        )
    )

    capacity = int(
        CONFIG[
            "pending_buffer_bytes"
        ]
    )

    if (
        used_bytes
        + int(
            report.size_bytes
        )
        > capacity
    ):
        return False, "pending_full"

    pending_reports.append(
        report
    )

    return True, "enqueued"


# --- frozen notebook cell 42 ---
def process_confirmation_events(
    events,
    targets,
    report_buffers,
    pending_reports,
    gcs_received_target_ids,
):
    reports = []
    event_log = []

    if not isinstance(
        report_buffers,
        list,
    ):
        raise TypeError(
            "report_buffers must be a list"
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

    targets_by_id = {
        target.id: target
        for target in targets
    }

    generated_keys = set()

    for event in events:
        event_log.append(
            event
        )

        if (
            event.confirmation_type
            != "true_confirmation"
        ):
            continue

        if (
            event.target_id
            not in targets_by_id
        ):
            raise ValueError(
                "confirmation event "
                "target_id not found"
            )

        target = targets_by_id[
            event.target_id
        ]
        target.confirmed = True

        current_step = int(
            event.step
        )

        for buffer in (
            report_buffers
        ):
            remove_expired_reports(
                buffer,
                current_step,
            )

        remove_expired_reports(
            pending_reports,
            current_step,
        )

        if (
            target.id
            in gcs_received_target_ids
        ):
            event.report_status = (
                "already_delivered"
            )
            continue

        source_uav = int(
            event.uav_id
        )

        if not (
            0
            <= source_uav
            < len(
                report_buffers
            )
        ):
            raise ValueError(
                "confirmation event "
                "uav_id out of range"
            )

        generation_key = (
            source_uav,
            int(target.id),
        )

        report_already_buffered = (
            report_exists(
                report_buffers[
                    source_uav
                ],
                target.id,
            )
        )

        report_already_pending = any(
            (
                report.target_id
                == target.id
                and report.source_uav
                == source_uav
            )
            for report
            in pending_reports
        )

        if (
            report_already_buffered
            or report_already_pending
            or generation_key
            in generated_keys
        ):
            event.report_status = (
                "duplicate"
            )
            continue

        report = Report(
            target_id=target.id,
            source_uav=source_uav,
            created_step=(
                event.step
            ),
            size_bytes=(
                CONFIG[
                    "report_bytes"
                ]
            ),
            ttl_s=(
                CONFIG[
                    "report_ttl"
                ]
            ),
        )

        enqueued, reason = (
            enqueue_report(
                report_buffers[
                    source_uav
                ],
                report,
            )
        )

        stored = bool(
            enqueued
        )

        if enqueued:
            event.report_status = (
                "buffered"
            )

        if (
            not enqueued
            and reason
            == "buffer_full"
        ):
            (
                pending_enqueued,
                pending_reason,
            ) = enqueue_pending_report(
                pending_reports,
                report,
            )

            stored = bool(
                pending_enqueued
            )

            if pending_enqueued:
                event.report_status = (
                    "pending"
                )
            elif (
                pending_reason
                == "pending_full"
            ):
                event.report_status = (
                    "dropped"
                )
            elif (
                pending_reason
                == "duplicate"
            ):
                event.report_status = (
                    "duplicate"
                )

            if (
                not pending_enqueued
                and pending_reason
                not in {
                    "duplicate",
                    "pending_full",
                }
            ):
                raise RuntimeError(
                    "unexpected pending "
                    "enqueue result: "
                    f"{pending_reason}"
                )

        elif (
            not enqueued
            and reason
            == "duplicate"
        ):
            event.report_status = (
                "duplicate"
            )

        elif (
            not enqueued
            and reason
            != "duplicate"
        ):
            raise RuntimeError(
                "unexpected enqueue "
                f"result: {reason}"
            )

        if stored:
            reports.append(
                report
            )

        generated_keys.add(
            generation_key
        )

    return reports, event_log


# --- frozen notebook cell 43 ---
def create_report_buffers():
    return [[] for _ in range(CONFIG["num_uavs"])]


# --- frozen notebook cell 44 ---
def create_pending_reports():
    return []


# --- frozen notebook cell 45 ---
def create_gcs_received_target_ids():
    return set()


# --- frozen notebook cell 46 ---
def report_remaining_bytes(report):
    remaining = report.size_bytes - report.delivered_bytes
    return max(0, int(remaining))


# --- frozen notebook cell 47 ---
def report_is_complete(report):
    return report_remaining_bytes(report) == 0


# --- frozen notebook cell 48 ---
def buffer_used_bytes(buffer):
    return sum(report.size_bytes for report in buffer)


# --- frozen notebook cell 49 ---
def report_exists(buffer, target_id):
    return any(
        report.target_id == target_id for report in buffer
    )


# --- frozen notebook cell 50 ---
def report_exists_in_buffers(report_buffers, target_id, source_uav):
    return any(
        report.target_id == target_id
        and int(report.source_uav) == int(source_uav)
        for buffer in report_buffers
        for report in buffer
    )


# --- frozen notebook cell 51 ---
def pending_report_exists(pending_reports, target_id):
    return any(
        report.target_id == target_id
        for report in pending_reports
    )


# --- frozen notebook cell 52 ---
def enqueue_report(buffer, report):
    if not isinstance(buffer, list):
        raise TypeError("buffer must be a list")
    if not isinstance(report, Report):
        raise TypeError("report must be a Report")

    # A report whose bytes have all reached the GCS is retained until
    # mark_target_delivered_to_gcs() records mission-level delivery and
    # removes the report. This prevents losing delivery state between steps.
    if report_exists(buffer, report.target_id):
        return False, "duplicate"

    used_bytes = buffer_used_bytes(buffer)
    new_used_bytes = used_bytes + report.size_bytes

    if new_used_bytes > CONFIG["buffer_bytes"]:
        return False, "buffer_full"

    buffer.append(report)
    return True, "enqueued"


# --- frozen notebook cell 53 ---
def report_age_s(report, current_step):
    if (
        isinstance(current_step, (bool, np.bool_))
        or not isinstance(current_step, (int, np.integer))
    ):
        raise TypeError("current_step must be an integer")

    created_step = report.created_step

    if (
        isinstance(created_step, (bool, np.bool_))
        or not isinstance(created_step, (int, np.integer))
    ):
        raise TypeError("report.created_step must be an integer")

    current_step = int(current_step)
    created_step = int(created_step)

    if created_step < 0:
        raise ValueError("report.created_step must be >= 0")

    if current_step < created_step:
        raise ValueError(
            "current_step must be >= report.created_step"
        )

    dt = float(CONFIG["dt"])

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("CONFIG['dt'] must be finite and > 0")

    return float(
        (current_step - created_step) * dt
    )


# --- frozen notebook cell 54 ---
def report_is_expired(report, current_step):
    age_s = report_age_s(report,current_step)

    return age_s >= report.ttl_s


# --- frozen notebook cell 55 ---
def remove_expired_reports(
    buffer,
    current_step,
):
    kept_reports = []
    expired_reports = []

    for report in buffer:
        if report_is_expired(report, current_step):
            expired_reports.append(report)
        else:
            kept_reports.append(report)

    buffer[:] = kept_reports
    return expired_reports


# --- frozen notebook cell 56 ---
def remove_completed_reports(buffer):
    completed_reports = [
        report
        for report in buffer
        if report_is_complete(report)
    ]
    buffer[:] = [
        report
        for report in buffer
        if not report_is_complete(report)
    ]
    return completed_reports


# --- frozen notebook cell 57 ---
def cleanup_report_buffer(buffer, current_step):
    expired = remove_expired_reports(buffer, current_step)
    completed = [
        report
        for report in buffer
        if report_is_complete(report)
    ]
    # Completed reports are intentionally retained until the GCS-delivery
    # state is marked. FIFO selection skips them without deleting them.
    return {
        "expired": expired,
        "completed": completed,
    }


# --- frozen notebook cell 58 ---
def flush_pending_reports(
    pending_reports,
    report_buffers,
    current_step,
    gcs_received_target_ids,
    uavs,
):
    if not isinstance(pending_reports, list):
        raise TypeError("pending_reports must be a list")
    if not isinstance(report_buffers, list):
        raise TypeError("report_buffers must be a list")
    if not isinstance(gcs_received_target_ids, set):
        raise TypeError("gcs_received_target_ids must be a set")
    if not isinstance(uavs, list):
        raise TypeError("uavs must be a list")

    uavs_by_id = {}
    for uav in uavs:
        if not isinstance(uav, UAV):
            raise TypeError("uavs must contain only UAV objects")
        uav_id = int(uav.id)
        if uav_id in uavs_by_id:
            raise ValueError("UAV ids must be unique")
        uavs_by_id[uav_id] = uav

    for buffer in report_buffers:
        cleanup_report_buffer(buffer, current_step)

    kept_pending = []
    enqueued_target_ids = []
    expired_target_ids = []
    discarded_target_ids = []
    deferred_inactive_target_ids = []

    for report in pending_reports:
        if report.target_id in gcs_received_target_ids:
            discarded_target_ids.append(report.target_id)
            continue

        if report_is_expired(report, current_step):
            expired_target_ids.append(report.target_id)
            continue

        source_uav = int(report.source_uav)
        if not 0 <= source_uav < len(report_buffers):
            raise ValueError("pending report source_uav out of range")
        if source_uav not in uavs_by_id:
            raise ValueError("pending report source_uav not found")

        # Duplicate suppression uses the logical report key
        # (target_id, source_uav) across all replicated swarm buffers.
        if report_exists_in_buffers(
            report_buffers,
            report.target_id,
            source_uav,
        ):
            discarded_target_ids.append(report.target_id)
            continue

        if not uavs_by_id[source_uav].active:
            kept_pending.append(report)
            deferred_inactive_target_ids.append(report.target_id)
            continue

        enqueued, reason = enqueue_report(
            report_buffers[source_uav],
            report,
        )

        if enqueued:
            enqueued_target_ids.append(report.target_id)
        elif reason == "buffer_full":
            kept_pending.append(report)
        elif reason == "duplicate":
            discarded_target_ids.append(report.target_id)
        else:
            raise RuntimeError(f"unexpected enqueue result: {reason}")

    pending_reports[:] = kept_pending

    return {
        "enqueued_target_ids": enqueued_target_ids,
        "expired_target_ids": expired_target_ids,
        "discarded_target_ids": discarded_target_ids,
        "deferred_inactive_target_ids": deferred_inactive_target_ids,
    }


# --- frozen notebook cell 59 ---
def mark_target_delivered_to_gcs(
    target_id,
    gcs_received_target_ids,
    report_buffers,
    pending_reports,
):
    if isinstance(target_id, (bool, np.bool_)) or not isinstance(
        target_id,
        (int, np.integer),
    ):
        raise TypeError("target_id must be an integer")

    target_id = int(target_id)
    if target_id < 0:
        raise ValueError("target_id must be >= 0")
    if not isinstance(gcs_received_target_ids, set):
        raise TypeError("gcs_received_target_ids must be a set")

    gcs_received_target_ids.add(target_id)

    removed_from_buffers = 0
    for buffer in report_buffers:
        before = len(buffer)
        buffer[:] = [
            report
            for report in buffer
            if report.target_id != target_id
        ]
        removed_from_buffers += before - len(buffer)

    before_pending = len(pending_reports)
    pending_reports[:] = [
        report
        for report in pending_reports
        if report.target_id != target_id
    ]

    return {
        "removed_from_buffers": removed_from_buffers,
        "removed_from_pending": before_pending - len(pending_reports),
    }


# Export public and internal helpers required by downstream production layers.
__all__ = [name for name in globals() if not name.startswith("__")]
