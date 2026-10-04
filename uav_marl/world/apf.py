"""Risk-aware artificial potential field safety layer for multi-UAV motion."""

from __future__ import annotations

import math

import numpy as np
import torch


def _validate_parameters(
    *,
    max_accel,
    safety_distance,
    obstacle_clearance,
    soft_gain,
    lookahead_s,
    braking_margin,
    enabled=True,
):
    values = {
        "max_accel": float(max_accel),
        "safety_distance": float(safety_distance),
        "obstacle_clearance": float(obstacle_clearance),
        "soft_gain": float(soft_gain),
        "lookahead_s": float(lookahead_s),
        "braking_margin": float(braking_margin),
        "enabled": bool(enabled),
    }
    for name in (
        "max_accel",
        "safety_distance",
        "obstacle_clearance",
        "soft_gain",
        "lookahead_s",
        "braking_margin",
    ):
        if not math.isfinite(values[name]):
            raise ValueError(f"{name} must be finite")
    if values["max_accel"] <= 0.0:
        raise ValueError("max_accel must be > 0")
    if values["safety_distance"] < 0.0:
        raise ValueError("safety_distance must be >= 0")
    if values["obstacle_clearance"] < 0.0:
        raise ValueError("obstacle_clearance must be >= 0")
    if values["soft_gain"] < 0.0:
        raise ValueError("soft_gain must be >= 0")
    if values["lookahead_s"] < 0.0:
        raise ValueError("lookahead_s must be >= 0")
    if values["braking_margin"] < 1.0:
        raise ValueError("braking_margin must be >= 1")
    return values


def _risk_strength_numpy(
    risk_clearance,
    stopping_distance,
    warning_distance,
    gain,
):
    risk_clearance = np.asarray(risk_clearance, dtype=np.float64)
    stopping_distance = np.asarray(stopping_distance, dtype=np.float64)
    warning_distance = np.asarray(warning_distance, dtype=np.float64)
    width = warning_distance - stopping_distance
    q = np.where(
        width > 1e-12,
        (warning_distance - risk_clearance)
        / np.maximum(width, 1e-12),
        0.0,
    )
    q = np.clip(q, 0.0, 1.0)
    return float(gain) * q * q


def _risk_strength_torch(
    risk_clearance,
    stopping_distance,
    warning_distance,
    gain,
):
    width = warning_distance - stopping_distance
    q = torch.where(
        width > 1e-12,
        (warning_distance - risk_clearance)
        / width.clamp_min(1e-12),
        torch.zeros_like(width),
    ).clamp(0.0, 1.0)
    return float(gain) * q.square()


def _normalize_numpy(vectors, *, max_norm=1.0):
    vectors = np.asarray(vectors, dtype=np.float64)
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    scale = np.minimum(
        1.0,
        float(max_norm) / np.maximum(norm, 1e-12),
    )
    return vectors * scale


def _normalize_torch(vectors, *, max_norm=1.0):
    norm = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    scale = torch.minimum(
        torch.ones_like(norm),
        torch.as_tensor(
            float(max_norm),
            dtype=vectors.dtype,
            device=vectors.device,
        )
        / norm.clamp_min(1e-12),
    )
    return vectors * scale


def _cylinder_clearance_numpy(
    points,
    obstacle_xy,
    expanded_radius,
    expanded_height,
):
    points = np.asarray(points, dtype=np.float64)
    obstacle_xy = np.asarray(obstacle_xy, dtype=np.float64)
    expanded_radius = np.asarray(expanded_radius, dtype=np.float64)
    expanded_height = np.asarray(expanded_height, dtype=np.float64)

    count_uav = int(points.shape[0])
    count_obs = int(obstacle_xy.shape[0])
    clearance = np.zeros((count_uav, count_obs), dtype=np.float64)
    direction = np.zeros((count_uav, count_obs, 3), dtype=np.float64)

    for i in range(count_uav):
        for j in range(count_obs):
            rel_xy = points[i, :2] - obstacle_xy[j]
            radial = float(np.linalg.norm(rel_xy))
            if radial > 1e-12:
                horizontal_dir = rel_xy / radial
            else:
                horizontal_dir = np.array([1.0, 0.0], dtype=np.float64)

            radial_out = radial - float(expanded_radius[j])
            z_above = float(points[i, 2] - expanded_height[j])
            inside = radial_out <= 0.0 and z_above <= 0.0

            if inside:
                radial_penetration = max(0.0, -radial_out)
                top_penetration = max(0.0, -z_above)
                if radial_penetration <= top_penetration:
                    clearance[i, j] = -radial_penetration
                    direction[i, j, :2] = horizontal_dir
                else:
                    clearance[i, j] = -top_penetration
                    direction[i, j, 2] = 1.0
                continue

            horizontal_out = max(0.0, radial_out)
            vertical_out = max(0.0, z_above)
            vector = np.array(
                [
                    horizontal_dir[0] * horizontal_out,
                    horizontal_dir[1] * horizontal_out,
                    vertical_out,
                ],
                dtype=np.float64,
            )
            distance = float(np.linalg.norm(vector))
            clearance[i, j] = distance
            if distance > 1e-12:
                direction[i, j] = vector / distance
            elif radial_out >= z_above:
                direction[i, j, :2] = horizontal_dir
            else:
                direction[i, j, 2] = 1.0

    return clearance, direction


def _cylinder_clearance_torch(
    points,
    obstacle_xy,
    expanded_radius,
    expanded_height,
):
    rel_xy = points[..., :2].unsqueeze(2) - obstacle_xy.unsqueeze(1)
    radial = torch.linalg.vector_norm(rel_xy, dim=-1)
    horizontal_dir = rel_xy / radial.clamp_min(1e-12).unsqueeze(-1)

    fallback = torch.zeros_like(horizontal_dir)
    fallback[..., 0] = 1.0
    horizontal_dir = torch.where(
        (radial > 1e-12).unsqueeze(-1),
        horizontal_dir,
        fallback,
    )

    radial_out = radial - expanded_radius.unsqueeze(1)
    z_above = points[..., 2].unsqueeze(2) - expanded_height.unsqueeze(1)
    inside = (radial_out <= 0.0) & (z_above <= 0.0)

    radial_penetration = (-radial_out).clamp_min(0.0)
    top_penetration = (-z_above).clamp_min(0.0)
    side_exit = radial_penetration <= top_penetration

    inside_clearance = -torch.minimum(
        radial_penetration,
        top_penetration,
    )
    inside_direction = torch.zeros(
        (*radial.shape, 3),
        dtype=points.dtype,
        device=points.device,
    )
    inside_direction[..., :2] = torch.where(
        side_exit.unsqueeze(-1),
        horizontal_dir,
        torch.zeros_like(horizontal_dir),
    )
    inside_direction[..., 2] = (~side_exit).to(points.dtype)

    horizontal_out = radial_out.clamp_min(0.0)
    vertical_out = z_above.clamp_min(0.0)
    outside_vector = torch.cat(
        (
            horizontal_dir * horizontal_out.unsqueeze(-1),
            vertical_out.unsqueeze(-1),
        ),
        dim=-1,
    )
    outside_distance = torch.linalg.vector_norm(outside_vector, dim=-1)
    outside_direction = (
        outside_vector / outside_distance.clamp_min(1e-12).unsqueeze(-1)
    )

    side_fallback = radial_out >= z_above
    boundary_fallback = torch.zeros_like(outside_direction)
    boundary_fallback[..., :2] = torch.where(
        side_fallback.unsqueeze(-1),
        horizontal_dir,
        torch.zeros_like(horizontal_dir),
    )
    boundary_fallback[..., 2] = (~side_fallback).to(points.dtype)
    outside_direction = torch.where(
        (outside_distance > 1e-12).unsqueeze(-1),
        outside_direction,
        boundary_fallback,
    )

    clearance = torch.where(inside, inside_clearance, outside_distance)
    direction = torch.where(
        inside.unsqueeze(-1),
        inside_direction,
        outside_direction,
    )
    return clearance, direction


def _obstacle_soft_direction_numpy(
    outward_direction,
    nominal_velocities,
):
    horizontal = outward_direction[..., :2]
    horizontal_norm = np.linalg.norm(horizontal, axis=-1, keepdims=True)
    radial_xy = horizontal / np.maximum(horizontal_norm, 1e-12)

    tangent_a = np.concatenate(
        (
            -radial_xy[..., 1:2],
            radial_xy[..., 0:1],
            np.zeros((*radial_xy.shape[:-1], 1), dtype=np.float64),
        ),
        axis=-1,
    )
    tangent_b = -tangent_a
    nominal = nominal_velocities[:, None, :]

    choose_a = (
        np.sum(tangent_a * nominal, axis=-1)
        >= np.sum(tangent_b * nominal, axis=-1)
    )
    tangent = np.where(
        choose_a[..., None],
        tangent_a,
        tangent_b,
    )

    mixed = _normalize_numpy(outward_direction + tangent)
    return np.where(
        (horizontal_norm[..., 0] > 1e-9)[..., None],
        mixed,
        outward_direction,
    )


def _obstacle_soft_direction_torch(
    outward_direction,
    nominal_velocities,
):
    horizontal = outward_direction[..., :2]
    horizontal_norm = torch.linalg.vector_norm(
        horizontal,
        dim=-1,
        keepdim=True,
    )
    radial_xy = horizontal / horizontal_norm.clamp_min(1e-12)

    tangent_a = torch.cat(
        (
            -radial_xy[..., 1:2],
            radial_xy[..., 0:1],
            torch.zeros_like(radial_xy[..., :1]),
        ),
        dim=-1,
    )
    tangent_b = -tangent_a
    nominal = nominal_velocities.unsqueeze(2)

    choose_a = (
        (tangent_a * nominal).sum(dim=-1)
        >= (tangent_b * nominal).sum(dim=-1)
    )
    tangent = torch.where(
        choose_a.unsqueeze(-1),
        tangent_a,
        tangent_b,
    )

    mixed = _normalize_torch(outward_direction + tangent)
    return torch.where(
        (horizontal_norm[..., 0] > 1e-9).unsqueeze(-1),
        mixed,
        outward_direction,
    )


def apf_repulsion_numpy(
    positions,
    velocities,
    active,
    obstacle_xy,
    obstacle_radius,
    obstacle_height,
    *,
    nominal_velocities=None,
    max_accel,
    safety_distance,
    obstacle_clearance,
    soft_gain,
    lookahead_s,
    braking_margin,
    enabled=True,
):
    """Risk-aware predictive APF using the nominal MARL motion."""
    params = _validate_parameters(
        max_accel=max_accel,
        safety_distance=safety_distance,
        obstacle_clearance=obstacle_clearance,
        soft_gain=soft_gain,
        lookahead_s=lookahead_s,
        braking_margin=braking_margin,
        enabled=enabled,
    )

    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    if nominal_velocities is None:
        nominal_velocities = velocities
    nominal_velocities = np.asarray(nominal_velocities, dtype=np.float64)
    active = np.asarray(active, dtype=bool)

    obstacle_xy = np.asarray(obstacle_xy, dtype=np.float64).reshape(-1, 2)
    obstacle_radius = np.asarray(
        obstacle_radius,
        dtype=np.float64,
    ).reshape(-1)
    obstacle_height = np.asarray(
        obstacle_height,
        dtype=np.float64,
    ).reshape(-1)

    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [U, 3]")
    if velocities.shape != positions.shape:
        raise ValueError("velocities must match positions")
    if nominal_velocities.shape != positions.shape:
        raise ValueError("nominal_velocities must match positions")
    if active.shape != positions.shape[:1]:
        raise ValueError("active must have shape [U]")
    if not (
        obstacle_xy.shape[0]
        == obstacle_radius.shape[0]
        == obstacle_height.shape[0]
    ):
        raise ValueError("obstacle arrays must have matching length")

    count = positions.shape[0]
    peer_accel = np.zeros_like(positions)
    peer_emergency_accel = np.zeros_like(positions)
    peer_active = np.zeros(count, dtype=bool)
    peer_emergency = np.zeros(count, dtype=bool)
    peer_min_clearance = np.full(count, np.inf, dtype=np.float64)

    if params["enabled"]:
        lookahead = params["lookahead_s"]

        for left in range(count):
            if not active[left]:
                continue
            for right in range(left + 1, count):
                if not active[right]:
                    continue

                relative = positions[left] - positions[right]
                relative_velocity = (
                    nominal_velocities[left] - nominal_velocities[right]
                )

                now_distance = float(np.linalg.norm(relative))
                if now_distance > 1e-12:
                    now_direction = relative / now_distance
                else:
                    now_direction = np.array([-1.0, 0.0, 0.0])

                denom = float(relative_velocity @ relative_velocity)
                if denom > 1e-12:
                    t_closest = float(
                        np.clip(
                            -(relative @ relative_velocity) / denom,
                            0.0,
                            lookahead,
                        )
                    )
                else:
                    t_closest = 0.0

                closest = relative + relative_velocity * t_closest
                closest_distance = float(np.linalg.norm(closest))
                if closest_distance > 1e-12:
                    soft_direction = closest / closest_distance
                else:
                    soft_direction = now_direction

                current_clearance = (
                    now_distance - params["safety_distance"]
                )
                predicted_clearance = (
                    closest_distance - params["safety_distance"]
                )
                risk_clearance = min(
                    current_clearance,
                    predicted_clearance,
                )

                peer_min_clearance[left] = min(
                    peer_min_clearance[left],
                    risk_clearance,
                )
                peer_min_clearance[right] = min(
                    peer_min_clearance[right],
                    risk_clearance,
                )

                closing_speed = max(
                    0.0,
                    -float(relative_velocity @ now_direction),
                )
                stopping_distance = (
                    closing_speed * closing_speed
                    / max(4.0 * params["max_accel"], 1e-12)
                )
                warning_distance = (
                    params["braking_margin"] * stopping_distance
                )

                emergency = (
                    current_clearance <= stopping_distance
                    or predicted_clearance <= 0.0
                )
                caution = (
                    closing_speed > 1e-9
                    and not emergency
                    and risk_clearance <= warning_distance
                )

                if emergency:
                    emergency_vector = (
                        params["max_accel"] * now_direction
                    )
                    peer_emergency_accel[left] += emergency_vector
                    peer_emergency_accel[right] -= emergency_vector
                    peer_emergency[left] = True
                    peer_emergency[right] = True
                    peer_active[left] = True
                    peer_active[right] = True
                elif caution:
                    strength = float(
                        _risk_strength_numpy(
                            risk_clearance,
                            stopping_distance,
                            warning_distance,
                            params["soft_gain"],
                        )
                    )
                    correction = (
                        params["max_accel"]
                        * strength
                        * soft_direction
                    )
                    peer_accel[left] += correction
                    peer_accel[right] -= correction
                    peer_active[left] = True
                    peer_active[right] = True

    obstacle_accel = np.zeros_like(positions)
    obstacle_emergency_accel = np.zeros_like(positions)
    obstacle_active = np.zeros(count, dtype=bool)
    obstacle_emergency = np.zeros(count, dtype=bool)
    obstacle_min_clearance = np.full(count, np.inf, dtype=np.float64)

    if params["enabled"] and obstacle_xy.shape[0] > 0:
        expanded_radius = (
            obstacle_radius + params["obstacle_clearance"]
        )
        expanded_height = (
            obstacle_height + params["obstacle_clearance"]
        )

        current_clearance, current_direction = _cylinder_clearance_numpy(
            positions,
            obstacle_xy,
            expanded_radius,
            expanded_height,
        )
        predicted_positions = (
            positions + nominal_velocities * params["lookahead_s"]
        )
        predicted_clearance, predicted_direction = _cylinder_clearance_numpy(
            predicted_positions,
            obstacle_xy,
            expanded_radius,
            expanded_height,
        )

        use_predicted = predicted_clearance < current_clearance
        risk_clearance = np.minimum(
            current_clearance,
            predicted_clearance,
        )
        obstacle_min_clearance = np.min(risk_clearance, axis=1)

        risk_direction = np.where(
            use_predicted[..., None],
            predicted_direction,
            current_direction,
        )
        soft_direction = _obstacle_soft_direction_numpy(
            risk_direction,
            nominal_velocities,
        )

        closing_speed = np.maximum(
            0.0,
            -np.sum(
                nominal_velocities[:, None, :] * current_direction,
                axis=-1,
            ),
        )
        stopping_distance = (
            closing_speed * closing_speed
            / max(2.0 * params["max_accel"], 1e-12)
        )
        warning_distance = (
            params["braking_margin"] * stopping_distance
        )

        emergency_pair = (
            (current_clearance <= stopping_distance)
            | (predicted_clearance <= 0.0)
        ) & active[:, None]
        caution_pair = (
            (closing_speed > 1e-9)
            & ~emergency_pair
            & (risk_clearance <= warning_distance)
            & active[:, None]
        )

        soft_strength = _risk_strength_numpy(
            risk_clearance,
            stopping_distance,
            warning_distance,
            params["soft_gain"],
        ) * caution_pair.astype(np.float64)

        obstacle_accel = (
            params["max_accel"]
            * np.sum(
                soft_strength[..., None] * soft_direction,
                axis=1,
            )
        )
        obstacle_emergency_accel = (
            params["max_accel"]
            * np.sum(
                emergency_pair[..., None].astype(np.float64)
                * current_direction,
                axis=1,
            )
        )
        obstacle_active = np.any(
            caution_pair | emergency_pair,
            axis=1,
        )
        obstacle_emergency = np.any(
            emergency_pair,
            axis=1,
        )

    soft_accel = peer_accel + obstacle_accel
    emergency_accel = _normalize_numpy(
        peer_emergency_accel + obstacle_emergency_accel,
        max_norm=params["max_accel"],
    )
    emergency = peer_emergency | obstacle_emergency

    soft_accel[~active] = 0.0
    emergency_accel[~active] = 0.0

    return {
        "acceleration_mps2": soft_accel,
        "soft_acceleration_mps2": soft_accel,
        "emergency_acceleration_mps2": emergency_accel,
        "peer_acceleration_mps2": peer_accel,
        "obstacle_acceleration_mps2": obstacle_accel,
        "active": peer_active | obstacle_active,
        "peer_active": peer_active,
        "obstacle_active": obstacle_active,
        "emergency": emergency,
        "peer_emergency": peer_emergency,
        "obstacle_emergency": obstacle_emergency,
        "min_peer_clearance_m": peer_min_clearance,
        "min_obstacle_clearance_m": obstacle_min_clearance,
    }


def apf_repulsion_torch(
    positions,
    velocities,
    active,
    obstacle_xy,
    obstacle_radius,
    obstacle_height,
    *,
    nominal_velocities=None,
    max_accel,
    safety_distance,
    obstacle_clearance,
    soft_gain,
    lookahead_s,
    braking_margin,
    enabled=True,
):
    """Batched torch equivalent of apf_repulsion_numpy."""
    params = _validate_parameters(
        max_accel=max_accel,
        safety_distance=safety_distance,
        obstacle_clearance=obstacle_clearance,
        soft_gain=soft_gain,
        lookahead_s=lookahead_s,
        braking_margin=braking_margin,
        enabled=enabled,
    )

    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [E, U, 3]")
    if velocities.shape != positions.shape:
        raise ValueError("velocities must match positions")
    if nominal_velocities is None:
        nominal_velocities = velocities
    if nominal_velocities.shape != positions.shape:
        raise ValueError("nominal_velocities must match positions")
    if active.shape != positions.shape[:2]:
        raise ValueError("active must have shape [E, U]")

    active = active.to(dtype=torch.bool)
    envs, uavs, _ = positions.shape

    peer_accel = torch.zeros_like(positions)
    peer_emergency_accel = torch.zeros_like(positions)
    peer_active = torch.zeros_like(active)
    peer_emergency = torch.zeros_like(active)
    peer_min_clearance = torch.full(
        active.shape,
        float("inf"),
        dtype=positions.dtype,
        device=positions.device,
    )

    if params["enabled"]:
        relative = positions.unsqueeze(2) - positions.unsqueeze(1)
        relative_velocity = (
            nominal_velocities.unsqueeze(2)
            - nominal_velocities.unsqueeze(1)
        )

        denom = relative_velocity.square().sum(dim=-1)
        t_closest = torch.where(
            denom > 1e-12,
            -(
                relative * relative_velocity
            ).sum(dim=-1) / denom.clamp_min(1e-12),
            torch.zeros_like(denom),
        ).clamp(0.0, params["lookahead_s"])

        closest = (
            relative
            + relative_velocity * t_closest.unsqueeze(-1)
        )
        closest_distance = torch.linalg.vector_norm(
            closest,
            dim=-1,
        )
        now_distance = torch.linalg.vector_norm(
            relative,
            dim=-1,
        )
        now_direction = (
            relative
            / now_distance.clamp_min(1e-12).unsqueeze(-1)
        )

        indices = torch.arange(
            uavs,
            device=positions.device,
            dtype=positions.dtype,
        )
        fallback_sign = torch.sign(
            indices.view(1, uavs, 1)
            - indices.view(1, 1, uavs)
        )
        fallback = torch.zeros_like(relative)
        fallback[..., 0] = fallback_sign
        now_direction = torch.where(
            (now_distance > 1e-12).unsqueeze(-1),
            now_direction,
            fallback,
        )
        soft_direction = (
            closest
            / closest_distance.clamp_min(1e-12).unsqueeze(-1)
        )
        soft_direction = torch.where(
            (closest_distance > 1e-12).unsqueeze(-1),
            soft_direction,
            now_direction,
        )

        eye = torch.eye(
            uavs,
            device=positions.device,
            dtype=torch.bool,
        ).unsqueeze(0)
        pair_mask = (
            active.unsqueeze(2)
            & active.unsqueeze(1)
            & ~eye
        )

        current_clearance = (
            now_distance - params["safety_distance"]
        )
        predicted_clearance = (
            closest_distance - params["safety_distance"]
        )
        risk_clearance = torch.minimum(
            current_clearance,
            predicted_clearance,
        )

        masked_clearance = torch.where(
            pair_mask,
            risk_clearance,
            torch.full_like(risk_clearance, float("inf")),
        )
        peer_min_clearance = masked_clearance.amin(dim=2)

        closing_speed = (
            -(
                relative_velocity * now_direction
            ).sum(dim=-1)
        ).clamp_min(0.0)
        stopping_distance = (
            closing_speed.square()
            / max(4.0 * params["max_accel"], 1e-12)
        )
        warning_distance = (
            params["braking_margin"] * stopping_distance
        )

        emergency_pair = (
            (current_clearance <= stopping_distance)
            | (predicted_clearance <= 0.0)
        ) & pair_mask
        caution_pair = (
            (closing_speed > 1e-9)
            & ~emergency_pair
            & (risk_clearance <= warning_distance)
            & pair_mask
        )

        peer_strength = _risk_strength_torch(
            risk_clearance,
            stopping_distance,
            warning_distance,
            params["soft_gain"],
        ) * caution_pair.to(positions.dtype)

        peer_accel = (
            params["max_accel"]
            * (
                peer_strength.unsqueeze(-1)
                * soft_direction
            ).sum(dim=2)
        )
        peer_emergency_accel = (
            params["max_accel"]
            * (
                emergency_pair.to(positions.dtype).unsqueeze(-1)
                * now_direction
            ).sum(dim=2)
        )
        peer_active = (
            caution_pair | emergency_pair
        ).any(dim=2)
        peer_emergency = emergency_pair.any(dim=2)

    if obstacle_xy.ndim != 3 or obstacle_xy.shape[-1] != 2:
        raise ValueError("obstacle_xy must have shape [E, O, 2]")
    if obstacle_radius.shape != obstacle_xy.shape[:2]:
        raise ValueError("obstacle_radius must have shape [E, O]")
    if obstacle_height.shape != obstacle_xy.shape[:2]:
        raise ValueError("obstacle_height must have shape [E, O]")
    if obstacle_xy.shape[0] != envs:
        raise ValueError("obstacle batch dimension must match positions")

    obstacle_accel = torch.zeros_like(positions)
    obstacle_emergency_accel = torch.zeros_like(positions)
    obstacle_active = torch.zeros_like(active)
    obstacle_emergency = torch.zeros_like(active)
    obstacle_min_clearance = torch.full(
        active.shape,
        float("inf"),
        dtype=positions.dtype,
        device=positions.device,
    )

    if params["enabled"] and obstacle_xy.shape[1] > 0:
        expanded_radius = (
            obstacle_radius + params["obstacle_clearance"]
        )
        expanded_height = (
            obstacle_height + params["obstacle_clearance"]
        )

        current_clearance, current_direction = _cylinder_clearance_torch(
            positions,
            obstacle_xy,
            expanded_radius,
            expanded_height,
        )
        predicted_positions = (
            positions + nominal_velocities * params["lookahead_s"]
        )
        predicted_clearance, predicted_direction = _cylinder_clearance_torch(
            predicted_positions,
            obstacle_xy,
            expanded_radius,
            expanded_height,
        )

        use_predicted = predicted_clearance < current_clearance
        risk_clearance = torch.minimum(
            current_clearance,
            predicted_clearance,
        )

        obstacle_min_clearance = torch.where(
            active.unsqueeze(-1),
            risk_clearance,
            torch.full_like(risk_clearance, float("inf")),
        ).amin(dim=2)

        risk_direction = torch.where(
            use_predicted.unsqueeze(-1),
            predicted_direction,
            current_direction,
        )
        soft_direction = _obstacle_soft_direction_torch(
            risk_direction,
            nominal_velocities,
        )

        closing_speed = (
            -(
                nominal_velocities.unsqueeze(2)
                * current_direction
            ).sum(dim=-1)
        ).clamp_min(0.0)
        stopping_distance = (
            closing_speed.square()
            / max(2.0 * params["max_accel"], 1e-12)
        )
        warning_distance = (
            params["braking_margin"] * stopping_distance
        )

        emergency_pair = (
            (current_clearance <= stopping_distance)
            | (predicted_clearance <= 0.0)
        ) & active.unsqueeze(-1)
        caution_pair = (
            (closing_speed > 1e-9)
            & ~emergency_pair
            & (risk_clearance <= warning_distance)
            & active.unsqueeze(-1)
        )

        obstacle_strength = _risk_strength_torch(
            risk_clearance,
            stopping_distance,
            warning_distance,
            params["soft_gain"],
        ) * caution_pair.to(positions.dtype)

        obstacle_accel = (
            params["max_accel"]
            * (
                obstacle_strength.unsqueeze(-1)
                * soft_direction
            ).sum(dim=2)
        )
        obstacle_emergency_accel = (
            params["max_accel"]
            * (
                emergency_pair.to(positions.dtype).unsqueeze(-1)
                * current_direction
            ).sum(dim=2)
        )
        obstacle_active = (
            caution_pair | emergency_pair
        ).any(dim=2)
        obstacle_emergency = emergency_pair.any(dim=2)

    soft_accel = peer_accel + obstacle_accel
    emergency_accel = _normalize_torch(
        peer_emergency_accel + obstacle_emergency_accel,
        max_norm=params["max_accel"],
    )
    emergency = peer_emergency | obstacle_emergency

    soft_accel = torch.where(
        active.unsqueeze(-1),
        soft_accel,
        torch.zeros_like(soft_accel),
    )
    emergency_accel = torch.where(
        active.unsqueeze(-1),
        emergency_accel,
        torch.zeros_like(emergency_accel),
    )

    return {
        "acceleration_mps2": soft_accel,
        "soft_acceleration_mps2": soft_accel,
        "emergency_acceleration_mps2": emergency_accel,
        "peer_acceleration_mps2": peer_accel,
        "obstacle_acceleration_mps2": obstacle_accel,
        "active": peer_active | obstacle_active,
        "peer_active": peer_active,
        "obstacle_active": obstacle_active,
        "emergency": emergency,
        "peer_emergency": peer_emergency,
        "obstacle_emergency": obstacle_emergency,
        "min_peer_clearance_m": peer_min_clearance,
        "min_obstacle_clearance_m": obstacle_min_clearance,
    }
