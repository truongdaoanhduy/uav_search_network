"""Production source migrated from frozen fix_test_gpu.ipynb.

Do not import the notebook at runtime. The notebook is retained only as a
reference/regression artifact. Production changes belong in Python modules.
Migrated from notebook cells 3..6.
"""

# --- frozen notebook cell 3 ---
from __future__ import annotations

import ast
import random
import sys
import types
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Hydra configs/ is the user-facing source of truth. config.py is a
# compatibility mapping imported through the Python module path; production
# code must not depend on the process current working directory.
from config import CONFIG


def _require_cuda(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("full-GPU mode requires an available CUDA device")


def contact_graph_reachability_fraction_numpy(
    positions,
    active,
    gcs_position,
    *,
    peer_range_m,
    gcs_range_m,
):
    """Fraction of active UAVs with a range-graph path to the GCS."""
    positions = np.asarray(positions, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    gcs_position = np.asarray(gcs_position, dtype=np.float64).reshape(3)
    active_count = int(np.count_nonzero(active))
    if active_count == 0:
        return 0.0

    peer_distance = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :],
        axis=-1,
    )
    peer_edges = (
        peer_distance <= float(peer_range_m)
    ) & active[:, None] & active[None, :]
    gcs_distance = np.linalg.norm(
        positions - gcs_position[None, :],
        axis=-1,
    )
    reachable = active & (gcs_distance <= float(gcs_range_m))
    for _ in range(max(0, int(active.shape[0]) - 1)):
        expanded = reachable | np.any(
            peer_edges & reachable[None, :],
            axis=1,
        )
        if np.array_equal(expanded, reachable):
            break
        reachable = expanded
    return float(np.count_nonzero(reachable) / active_count)


def contact_graph_reachability_fraction_torch(
    positions,
    active,
    gcs_position,
    *,
    peer_range_m,
    gcs_range_m,
):
    """Batched torch equivalent of contact_graph_reachability_fraction_numpy."""
    active = active.to(dtype=torch.bool)
    gcs_position = torch.as_tensor(
        gcs_position,
        device=positions.device,
        dtype=positions.dtype,
    ).reshape(1, 1, 3)
    peer_distance = torch.linalg.vector_norm(
        positions.unsqueeze(2) - positions.unsqueeze(1),
        dim=-1,
    )
    peer_edges = (
        peer_distance <= float(peer_range_m)
    ) & active.unsqueeze(2) & active.unsqueeze(1)
    gcs_distance = torch.linalg.vector_norm(
        positions - gcs_position,
        dim=-1,
    )
    reachable = active & (gcs_distance <= float(gcs_range_m))
    for _ in range(max(0, int(positions.shape[1]) - 1)):
        reachable = reachable | (
            peer_edges & reachable.unsqueeze(1)
        ).any(dim=-1)
    return (
        reachable.float().sum(dim=-1)
        / active.float().sum(dim=-1).clamp_min(1.0)
    )


# --- frozen notebook cell 4 ---
import os as _bootstrap_os
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
import copy
import csv
import importlib
import json
import os
import random
import secrets
import subprocess
import tempfile
import sys
import time
from abc import ABC, abstractmethod
from itertools import pairwise
from pathlib import Path
from typing import ClassVar
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from config import CONFIG


def _validate_apf_parameters(
    *,
    max_accel,
    safety_distance,
    obstacle_clearance,
    peer_influence,
    obstacle_influence,
    peer_gain,
    obstacle_gain,
    lookahead_s,
    emergency_gain=4.0,
    braking_margin=1.5,
):
    values = {
        "max_accel": float(max_accel),
        "safety_distance": float(safety_distance),
        "obstacle_clearance": float(obstacle_clearance),
        "peer_influence": float(peer_influence),
        "obstacle_influence": float(obstacle_influence),
        "peer_gain": float(peer_gain),
        "obstacle_gain": float(obstacle_gain),
        "lookahead_s": float(lookahead_s),
        "emergency_gain": float(emergency_gain),
        "braking_margin": float(braking_margin),
    }
    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if values["max_accel"] <= 0.0:
        raise ValueError("max_accel must be > 0")
    if values["safety_distance"] < 0.0:
        raise ValueError("safety_distance must be >= 0")
    if values["obstacle_clearance"] < 0.0:
        raise ValueError("obstacle_clearance must be >= 0")
    if values["peer_influence"] <= 0.0:
        raise ValueError("peer_influence must be > 0")
    if values["obstacle_influence"] <= 0.0:
        raise ValueError("obstacle_influence must be > 0")
    if values["peer_gain"] < 0.0:
        raise ValueError("peer_gain must be >= 0")
    if values["obstacle_gain"] < 0.0:
        raise ValueError("obstacle_gain must be >= 0")
    if values["lookahead_s"] < 0.0:
        raise ValueError("lookahead_s must be >= 0")
    if values["emergency_gain"] < 2.0:
        raise ValueError(
            "emergency_gain must be >= 2 so APF can override "
            "a full-scale opposing policy acceleration"
        )
    if values["braking_margin"] < 1.0:
        raise ValueError("braking_margin must be >= 1")
    return values


def _apf_strength_numpy(clearance, influence, gain):
    clearance = np.asarray(clearance, dtype=np.float64)
    normalized = np.clip(
        1.0 - np.maximum(clearance, 0.0) / float(influence),
        0.0,
        1.0,
    )
    return float(gain) * normalized * normalized


def _apf_strength_torch(clearance, influence, gain):
    normalized = (
        1.0
        - torch.clamp_min(clearance, 0.0) / float(influence)
    ).clamp(0.0, 1.0)
    return float(gain) * normalized.square()


def _cylinder_clearance_numpy(
    points,
    obstacle_xy,
    expanded_radius,
    expanded_height,
):
    """Signed distance/direction to the exterior of finite vertical cylinders.

    Negative clearance means the point is inside the expanded safety cylinder.
    Direction always points toward the nearest safe exterior.
    """
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
    z_above = (
        points[..., 2].unsqueeze(2)
        - expanded_height.unsqueeze(1)
    )
    inside = (radial_out <= 0.0) & (z_above <= 0.0)

    radial_penetration = (-radial_out).clamp_min(0.0)
    top_penetration = (-z_above).clamp_min(0.0)
    choose_side = radial_penetration <= top_penetration
    inside_clearance = -torch.minimum(
        radial_penetration,
        top_penetration,
    )
    inside_direction = torch.zeros(
        (*radial.shape, 3),
        device=points.device,
        dtype=points.dtype,
    )
    inside_direction[..., :2] = torch.where(
        choose_side.unsqueeze(-1),
        horizontal_dir,
        torch.zeros_like(horizontal_dir),
    )
    inside_direction[..., 2] = (~choose_side).to(points.dtype)

    horizontal_out = radial_out.clamp_min(0.0)
    vertical_out = z_above.clamp_min(0.0)
    outside_vector = torch.cat(
        (
            horizontal_dir * horizontal_out.unsqueeze(-1),
            vertical_out.unsqueeze(-1),
        ),
        dim=-1,
    )
    outside_distance = torch.linalg.vector_norm(
        outside_vector,
        dim=-1,
    )
    outside_direction = (
        outside_vector
        / outside_distance.clamp_min(1e-12).unsqueeze(-1)
    )
    boundary_fallback = torch.zeros_like(outside_direction)
    side_fallback = radial_out >= z_above
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

    clearance = torch.where(
        inside,
        inside_clearance,
        outside_distance,
    )
    direction = torch.where(
        inside.unsqueeze(-1),
        inside_direction,
        outside_direction,
    )
    return clearance, direction


def apf_repulsion_numpy(
    positions,
    velocities,
    active,
    obstacle_xy,
    obstacle_radius,
    obstacle_height,
    *,
    max_accel,
    safety_distance,
    obstacle_clearance,
    peer_influence,
    obstacle_influence,
    peer_gain,
    obstacle_gain,
    lookahead_s,
    emergency_gain=4.0,
    braking_margin=1.5,
):
    """Velocity-aware predictive APF for peer and obstacle avoidance."""
    params = _validate_apf_parameters(
        max_accel=max_accel,
        safety_distance=safety_distance,
        obstacle_clearance=obstacle_clearance,
        peer_influence=peer_influence,
        obstacle_influence=obstacle_influence,
        peer_gain=peer_gain,
        obstacle_gain=obstacle_gain,
        lookahead_s=lookahead_s,
        emergency_gain=emergency_gain,
        braking_margin=braking_margin,
    )
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
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
    peer_active = np.zeros(count, dtype=bool)
    lookahead = params["lookahead_s"]

    for left in range(count):
        if not active[left]:
            continue
        for right in range(left + 1, count):
            if not active[right]:
                continue
            relative = positions[left] - positions[right]
            relative_velocity = velocities[left] - velocities[right]
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
            distance = float(np.linalg.norm(closest))
            now_distance = float(np.linalg.norm(relative))
            if now_distance > 1e-12:
                now_direction = relative / now_distance
            else:
                now_direction = np.array([-1.0, 0.0, 0.0])
            if distance > 1e-12:
                direction = closest / distance
            else:
                direction = now_direction

            clearance = distance - params["safety_distance"]
            strength = float(
                _apf_strength_numpy(
                    clearance,
                    params["peer_influence"],
                    params["peer_gain"],
                )
            )

            closing_speed = max(
                0.0,
                -float(relative_velocity @ now_direction),
            )
            # Both active peers can contribute max_accel in opposite
            # directions, so the relative braking acceleration is 2*a_max.
            braking_distance = (
                closing_speed * closing_speed
                / max(4.0 * params["max_accel"], 1e-12)
                * params["braking_margin"]
            )
            current_clearance = (
                now_distance - params["safety_distance"]
            )
            emergency = (
                current_clearance <= braking_distance
                or clearance <= 0.0
            )
            if emergency:
                strength = max(
                    strength,
                    params["emergency_gain"],
                )
                # Emergency braking must act opposite the present closing
                # direction, not a future closest-point direction that may
                # have already crossed to the other side.
                direction = now_direction
            if strength <= 0.0:
                continue
            acceleration = (
                params["max_accel"] * strength * direction
            )
            peer_accel[left] += acceleration
            peer_accel[right] -= acceleration
            peer_active[left] = True
            peer_active[right] = True

    obstacle_accel = np.zeros_like(positions)
    obstacle_active = np.zeros(count, dtype=bool)
    if obstacle_xy.shape[0] > 0:
        expanded_radius = (
            obstacle_radius + params["obstacle_clearance"]
        )
        expanded_height = (
            obstacle_height + params["obstacle_clearance"]
        )
        current_clearance, current_direction = (
            _cylinder_clearance_numpy(
                positions,
                obstacle_xy,
                expanded_radius,
                expanded_height,
            )
        )
        predicted_positions = positions + velocities * lookahead
        predicted_clearance, predicted_direction = (
            _cylinder_clearance_numpy(
                predicted_positions,
                obstacle_xy,
                expanded_radius,
                expanded_height,
            )
        )
        use_predicted = predicted_clearance < current_clearance
        clearance = np.where(
            use_predicted,
            predicted_clearance,
            current_clearance,
        )
        direction = np.where(
            use_predicted[..., None],
            predicted_direction,
            current_direction,
        )
        strength = _apf_strength_numpy(
            clearance,
            params["obstacle_influence"],
            params["obstacle_gain"],
        )
        closing_speed = np.maximum(
            0.0,
            -np.sum(
                velocities[:, None, :] * current_direction,
                axis=-1,
            ),
        )
        braking_distance = (
            closing_speed * closing_speed
            / max(2.0 * params["max_accel"], 1e-12)
            * params["braking_margin"]
        )
        emergency = (
            (current_clearance <= braking_distance)
            | (clearance <= 0.0)
        )
        strength = np.maximum(
            strength,
            emergency.astype(np.float64)
            * params["emergency_gain"],
        )
        direction = np.where(
            emergency[..., None],
            current_direction,
            direction,
        )
        strength *= active[:, None]
        obstacle_accel = (
            params["max_accel"]
            * np.sum(strength[..., None] * direction, axis=1)
        )
        obstacle_active = np.any(strength > 0.0, axis=1)

    acceleration = peer_accel + obstacle_accel
    acceleration[~active] = 0.0
    return {
        "acceleration_mps2": acceleration,
        "peer_acceleration_mps2": peer_accel,
        "obstacle_acceleration_mps2": obstacle_accel,
        "active": peer_active | obstacle_active,
        "peer_active": peer_active,
        "obstacle_active": obstacle_active,
    }


def apf_repulsion_torch(
    positions,
    velocities,
    active,
    obstacle_xy,
    obstacle_radius,
    obstacle_height,
    *,
    max_accel,
    safety_distance,
    obstacle_clearance,
    peer_influence,
    obstacle_influence,
    peer_gain,
    obstacle_gain,
    lookahead_s,
    emergency_gain=4.0,
    braking_margin=1.5,
):
    """Batched torch equivalent of :func:`apf_repulsion_numpy`."""
    params = _validate_apf_parameters(
        max_accel=max_accel,
        safety_distance=safety_distance,
        obstacle_clearance=obstacle_clearance,
        peer_influence=peer_influence,
        obstacle_influence=obstacle_influence,
        peer_gain=peer_gain,
        obstacle_gain=obstacle_gain,
        lookahead_s=lookahead_s,
        emergency_gain=emergency_gain,
        braking_margin=braking_margin,
    )
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [E, U, 3]")
    if velocities.shape != positions.shape:
        raise ValueError("velocities must match positions")
    if active.shape != positions.shape[:2]:
        raise ValueError("active must have shape [E, U]")
    active = active.to(dtype=torch.bool)
    envs, uavs, _ = positions.shape

    relative = positions.unsqueeze(2) - positions.unsqueeze(1)
    relative_velocity = (
        velocities.unsqueeze(2) - velocities.unsqueeze(1)
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
    distance = torch.linalg.vector_norm(closest, dim=-1)
    direction = closest / distance.clamp_min(1e-12).unsqueeze(-1)
    now_distance = torch.linalg.vector_norm(relative, dim=-1)
    now_direction = (
        relative / now_distance.clamp_min(1e-12).unsqueeze(-1)
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
    fallback = torch.zeros_like(direction)
    fallback[..., 0] = fallback_sign
    direction = torch.where(
        (distance > 1e-12).unsqueeze(-1),
        direction,
        torch.where(
            (now_distance > 1e-12).unsqueeze(-1),
            now_direction,
            fallback,
        ),
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
    peer_clearance = distance - params["safety_distance"]
    peer_strength = _apf_strength_torch(
        peer_clearance,
        params["peer_influence"],
        params["peer_gain"],
    )
    closing_speed = (
        -(
            relative_velocity * now_direction
        ).sum(dim=-1)
    ).clamp_min(0.0)
    braking_distance = (
        closing_speed.square()
        / max(4.0 * params["max_accel"], 1e-12)
        * params["braking_margin"]
    )
    current_clearance = (
        now_distance - params["safety_distance"]
    )
    emergency = (
        (current_clearance <= braking_distance)
        | (peer_clearance <= 0.0)
    ) & pair_mask
    peer_strength = torch.maximum(
        peer_strength,
        emergency.to(positions.dtype)
        * params["emergency_gain"],
    )
    direction = torch.where(
        emergency.unsqueeze(-1),
        now_direction,
        direction,
    )
    peer_strength = peer_strength * pair_mask.to(positions.dtype)
    peer_accel = (
        params["max_accel"]
        * (
            peer_strength.unsqueeze(-1) * direction
        ).sum(dim=2)
    )
    peer_active = (peer_strength > 0.0).any(dim=2)

    if obstacle_xy.ndim != 3 or obstacle_xy.shape[-1] != 2:
        raise ValueError("obstacle_xy must have shape [E, O, 2]")
    if obstacle_radius.shape != obstacle_xy.shape[:2]:
        raise ValueError("obstacle_radius must have shape [E, O]")
    if obstacle_height.shape != obstacle_xy.shape[:2]:
        raise ValueError("obstacle_height must have shape [E, O]")
    if obstacle_xy.shape[0] != envs:
        raise ValueError("obstacle batch dimension must match positions")

    if obstacle_xy.shape[1] == 0:
        obstacle_accel = torch.zeros_like(positions)
        obstacle_active = torch.zeros_like(active)
    else:
        expanded_radius = (
            obstacle_radius + params["obstacle_clearance"]
        )
        expanded_height = (
            obstacle_height + params["obstacle_clearance"]
        )
        current_clearance, current_direction = (
            _cylinder_clearance_torch(
                positions,
                obstacle_xy,
                expanded_radius,
                expanded_height,
            )
        )
        predicted_positions = (
            positions + velocities * params["lookahead_s"]
        )
        predicted_clearance, predicted_direction = (
            _cylinder_clearance_torch(
                predicted_positions,
                obstacle_xy,
                expanded_radius,
                expanded_height,
            )
        )
        use_predicted = predicted_clearance < current_clearance
        clearance = torch.where(
            use_predicted,
            predicted_clearance,
            current_clearance,
        )
        direction_obstacle = torch.where(
            use_predicted.unsqueeze(-1),
            predicted_direction,
            current_direction,
        )
        obstacle_strength = _apf_strength_torch(
            clearance,
            params["obstacle_influence"],
            params["obstacle_gain"],
        )
        closing_speed = (
            -(
                velocities.unsqueeze(2)
                * current_direction
            ).sum(dim=-1)
        ).clamp_min(0.0)
        braking_distance = (
            closing_speed.square()
            / max(2.0 * params["max_accel"], 1e-12)
            * params["braking_margin"]
        )
        emergency = (
            (current_clearance <= braking_distance)
            | (clearance <= 0.0)
        ) & active.unsqueeze(-1)
        obstacle_strength = torch.maximum(
            obstacle_strength,
            emergency.to(positions.dtype)
            * params["emergency_gain"],
        )
        direction_obstacle = torch.where(
            emergency.unsqueeze(-1),
            current_direction,
            direction_obstacle,
        )
        obstacle_strength = (
            obstacle_strength
            * active.unsqueeze(-1).to(positions.dtype)
        )
        obstacle_accel = (
            params["max_accel"]
            * (
                obstacle_strength.unsqueeze(-1)
                * direction_obstacle
            ).sum(dim=2)
        )
        obstacle_active = (
            obstacle_strength > 0.0
        ).any(dim=2)

    acceleration = peer_accel + obstacle_accel
    acceleration = torch.where(
        active.unsqueeze(-1),
        acceleration,
        torch.zeros_like(acceleration),
    )
    return {
        "acceleration_mps2": acceleration,
        "peer_acceleration_mps2": peer_accel,
        "obstacle_acceleration_mps2": obstacle_accel,
        "active": peer_active | obstacle_active,
        "peer_active": peer_active,
        "obstacle_active": obstacle_active,
    }


# --- frozen notebook cell 5 ---
def set_seed(seed=44):
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    if hasattr(
        torch.backends,
        "cudnn",
    ):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        if hasattr(
            torch.backends.cudnn,
            "allow_tf32",
        ):
            torch.backends.cudnn.allow_tf32 = False

    if (
        hasattr(
            torch.backends,
            "cuda",
        )
        and hasattr(
            torch.backends.cuda,
            "matmul",
        )
    ):
        torch.backends.cuda.matmul.allow_tf32 = False


# --- frozen notebook cell 6 ---
from dataclasses import asdict, dataclass, is_dataclass


# Export public and internal helpers required by downstream production layers.
__all__ = [name for name in globals() if not name.startswith("__")]
