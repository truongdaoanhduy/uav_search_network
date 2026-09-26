"""GPU-native batched Simple UAV search environment and MARL trainer.

All hot-path environment state lives on one torch device.  The CPU only
orchestrates kernel launches, logging, and checkpoints; it never steps a
per-environment Python/Gym simulator.

This is intentionally a separate implementation from test.ipynb so the CPU
reference remains available for parity/regression checks.
"""
# ruff: noqa: I001

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

import argparse
import copy
import json
import math
import time
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from config import CONFIG


def configure_determinism(seed: int) -> None:
    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False


def _binary_entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-12, 1.0 - 1e-12)
    return -p * torch.log2(p) - (1.0 - p) * torch.log2(1.0 - p)


def project_motion_action(action: torch.Tensor) -> torch.Tensor:
    action = action.clamp(-1.0, 1.0)
    norm = torch.linalg.vector_norm(action, dim=-1, keepdim=True)
    return action / torch.maximum(norm, torch.ones_like(norm))


class GpuBatchedUAVEnv:
    """Tensorized multi-UAV Simple backend.

    Shapes use E=environments, U=UAVs, T=targets, O=obstacles.
    No NumPy/Gym environment is used in reset(), step(), observations(), or
    global_state().
    """

    def __init__(
        self,
        num_envs: int,
        device: str = "cuda:0",
        seed: int = 44,
        dtype: torch.dtype = torch.float32,
    ):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        self.dtype = dtype
        self.num_envs = int(num_envs)
        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.seed = int(seed)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.seed)

        self.U = int(CONFIG["num_uavs"])
        self.T = int(CONFIG["num_targets"])
        self.O = int(CONFIG["num_obstacles"])
        self.map_size = float(CONFIG["map_size"])
        self.alt_min = float(CONFIG["altitude_min"])
        self.alt_max = float(CONFIG["altitude_max"])
        self.alt_span = max(1e-9, self.alt_max - self.alt_min)
        self.max_speed = float(CONFIG["max_speed"])
        self.max_accel = float(CONFIG["max_accel"])
        self.dt = float(CONFIG["dt"])
        self.max_steps = int(CONFIG["max_steps"])
        self.cell = float(CONFIG["grid_cell_m"])
        self.G = math.ceil(self.map_size / self.cell)
        self.D = self.U + 1
        self.buffer_slots = int(CONFIG["buffer_bytes"] // CONFIG["report_bytes"])
        self.pending_slots = int(
            CONFIG["pending_buffer_bytes"] // CONFIG["report_bytes"]
        )
        if self.buffer_slots < 1 or self.pending_slots < 1:
            raise ValueError("GPU environment requires at least one report slot")

        self.gcs = torch.tensor(
            CONFIG["gcs_position"], device=self.device, dtype=self.dtype
        )
        self.altitude_anchors = torch.tensor(
            CONFIG["altitude_anchors"], device=self.device, dtype=self.dtype
        )
        self.pd_anchors = torch.tensor(
            CONFIG["pd"], device=self.device, dtype=self.dtype
        )
        self.pf_anchors = torch.tensor(
            CONFIG["pf"], device=self.device, dtype=self.dtype
        )

        # sender-specific categorical index -> recipient.
        # -2=silent, -1=GCS, 0..U-1=peer id.
        table = []
        for sender in range(self.U):
            peers = [p for p in range(self.U) if p != sender]
            table.append([-2, *peers, -1])
        self.destination_table = torch.tensor(
            table, device=self.device, dtype=torch.long
        )

        # Candidate sensing cells. At 150 m / 25 m, +/-7 safely covers every
        # intersecting belief cell including boundary-touching cells.
        max_fov = self.alt_max * math.tan(
            math.radians(float(CONFIG["camera_full_fov_deg"])) / 2.0
        )
        radius_cells = math.ceil(max_fov / self.cell) + 1
        oy, ox = torch.meshgrid(
            torch.arange(
                -radius_cells, radius_cells + 1, device=self.device
            ),
            torch.arange(
                -radius_cells, radius_cells + 1, device=self.device
            ),
            indexing="ij",
        )
        self.cell_offsets = torch.stack([ox.reshape(-1), oy.reshape(-1)], dim=-1)
        self.C = int(self.cell_offsets.shape[0])

        self.patch_cells = int(CONFIG["belief_patch_cells"])
        patch_r = self.patch_cells // 2
        py, px = torch.meshgrid(
            torch.arange(-patch_r, patch_r + 1, device=self.device),
            torch.arange(-patch_r, patch_r + 1, device=self.device),
            indexing="ij",
        )
        self.patch_offsets = torch.stack([px.reshape(-1), py.reshape(-1)], dim=-1)

        self.coarse_cells = int(CONFIG["belief_coarse_cells"])
        self.critic_cells = int(CONFIG["critic_belief_grid_cells"])
        if self.G % self.coarse_cells or self.G % self.critic_cells:
            raise ValueError("GPU pooling currently requires evenly divisible grid")
        self.nearest_obstacles = int(CONFIG["observation_nearest_obstacles"])

        self.local_obs_dim = (
            9
            + 3
            + (self.U - 1) * 5
            + self.nearest_obstacles * 5
            + self.patch_cells * self.patch_cells
            + self.coarse_cells * self.coarse_cells
            + 5
            + self.D
        )
        self.state_dim = (
            self.U
            * (
                self.local_obs_dim
                + (self.buffer_slots + self.pending_slots) * 4
                + 2
                + self.critic_cells * self.critic_cells
            )
            + self.T * 4
            + self.O * 4
            + 1
        )

        self._allocate()
        self.reset()

    def _allocate(self) -> None:
        E, U, T, O, G = self.num_envs, self.U, self.T, self.O, self.G
        dev, dt = self.device, self.dtype
        self.pos = torch.zeros((E, U, 3), device=dev, dtype=dt)
        self.vel = torch.zeros_like(self.pos)
        self.battery = torch.zeros((E, U), device=dev, dtype=dt)
        self.active = torch.ones((E, U), device=dev, dtype=torch.bool)

        self.target_xy = torch.zeros((E, T, 2), device=dev, dtype=dt)
        self.target_confirmed = torch.zeros((E, T), device=dev, dtype=torch.bool)
        self.target_delivered = torch.zeros((E, T), device=dev, dtype=torch.bool)

        self.obs_xy = torch.zeros((E, O, 2), device=dev, dtype=dt)
        self.obs_radius = torch.zeros((E, O), device=dev, dtype=dt)
        self.obs_height = torch.zeros((E, O), device=dev, dtype=dt)

        prior = float(CONFIG["belief_prior"])
        self.belief = torch.full((E, U, G, G), prior, device=dev, dtype=dt)
        self.belief_entropy = _binary_entropy(self.belief)
        self.entropy_sum = self.belief_entropy.sum(dim=(1, 2, 3))
        self.known_count = torch.zeros((E, U), device=dev, dtype=torch.long)

        # holder/source/target representation of report copies.
        self.report_valid = torch.zeros(
            (E, U, U, T), device=dev, dtype=torch.bool
        )
        self.report_created = torch.full(
            (E, U, U, T), -1, device=dev, dtype=torch.long
        )
        self.report_order = torch.full(
            (E, U, U, T), 2**60, device=dev, dtype=torch.long
        )
        self.next_report_order = torch.zeros((E, U), device=dev, dtype=torch.long)

        self.pending_valid = torch.zeros((E, U, T), device=dev, dtype=torch.bool)
        self.pending_created = torch.full(
            (E, U, T), -1, device=dev, dtype=torch.long
        )
        self.pending_order = torch.full(
            (E, U, T), 2**60, device=dev, dtype=torch.long
        )
        self.next_pending_order = torch.zeros((E, U), device=dev, dtype=torch.long)

        self.gcs_progress = torch.zeros((E, U, T), device=dev, dtype=dt)
        self.peer_progress = torch.zeros(
            (E, U, U, U, T), device=dev, dtype=dt
        )
        self.peer_valid = torch.zeros(
            (E, U, U, U, T), device=dev, dtype=torch.bool
        )
        self.peer_created = torch.full(
            (E, U, U, U, T), -1, device=dev, dtype=torch.long
        )

        self.current_step = torch.zeros(E, device=dev, dtype=torch.long)
        self.episode_index = torch.zeros(E, device=dev, dtype=torch.long)
        self.done = torch.zeros(E, device=dev, dtype=torch.bool)

        # Diagnostics kept on device.
        self.diag_boundary = torch.zeros(E, device=dev, dtype=torch.long)
        self.diag_horizontal_boundary = torch.zeros(E, device=dev, dtype=torch.long)
        self.diag_altitude_boundary = torch.zeros(E, device=dev, dtype=torch.long)
        self.diag_blocked = torch.zeros(E, device=dev, dtype=torch.long)
        self.diag_false_confirm = torch.zeros(E, device=dev, dtype=torch.long)
        self.diag_total_energy = torch.zeros(E, device=dev, dtype=dt)

    def _rand(self, shape) -> torch.Tensor:
        return torch.rand(
            shape, device=self.device, dtype=self.dtype, generator=self.generator
        )

    def _generate_world(self, n: int):
        """Generate n constrained worlds using CUDA/torch tensors only."""
        U, T, O = self.U, self.T, self.O
        pos = torch.zeros((n, U, 3), device=self.device, dtype=self.dtype)
        vel = torch.zeros_like(pos)

        launch_r = float(CONFIG["launch_radius_m"])
        launch_min = float(CONFIG["launch_min_spacing_m"])
        for u in range(U):
            unresolved = torch.ones(n, device=self.device, dtype=torch.bool)
            attempts = 0
            while bool(unresolved.any()):
                attempts += 1
                if attempts > 10000:
                    raise RuntimeError("GPU world generator could not place UAVs")
                cand = (self._rand((n, 2)) * 2.0 - 1.0) * launch_r
                cand = cand + self.gcs[:2]
                radial = torch.sum((cand - self.gcs[:2]) ** 2, dim=-1) <= launch_r**2
                bounds = (
                    (cand[:, 0] >= 0.0)
                    & (cand[:, 0] <= self.map_size)
                    & (cand[:, 1] >= 0.0)
                    & (cand[:, 1] <= self.map_size)
                )
                valid = radial & bounds
                if u:
                    dist = torch.linalg.vector_norm(
                        cand[:, None, :] - pos[:, :u, :2], dim=-1
                    )
                    valid &= (dist >= launch_min).all(dim=-1)
                take = unresolved & valid
                pos[take, u, :2] = cand[take]
                pos[take, u, 2] = self.alt_min
                unresolved &= ~take

        obs_xy = torch.zeros((n, O, 2), device=self.device, dtype=self.dtype)
        obs_r = torch.zeros((n, O), device=self.device, dtype=self.dtype)
        obs_h = torch.zeros((n, O), device=self.device, dtype=self.dtype)
        rmin, rmax = (
            float(CONFIG["obstacle_radius_min_m"]),
            float(CONFIG["obstacle_radius_max_m"]),
        )
        hmin, hmax = (
            float(CONFIG["obstacle_height_min_m"]),
            float(CONFIG["obstacle_height_max_m"]),
        )
        safety = float(CONFIG["safety_distance"])
        gcs_ex = float(CONFIG["gcs_exclusion_radius_m"])
        for o in range(O):
            unresolved = torch.ones(n, device=self.device, dtype=torch.bool)
            attempts = 0
            while bool(unresolved.any()):
                attempts += 1
                if attempts > 10000:
                    raise RuntimeError("GPU world generator could not place obstacles")
                r = rmin + self._rand((n,)) * (rmax - rmin)
                h = hmin + self._rand((n,)) * (hmax - hmin)
                xy = torch.stack(
                    [
                        r + self._rand((n,)) * (self.map_size - 2.0 * r),
                        r + self._rand((n,)) * (self.map_size - 2.0 * r),
                    ],
                    dim=-1,
                )
                valid = (
                    torch.linalg.vector_norm(xy - self.gcs[:2], dim=-1)
                    > gcs_ex + r
                )
                launch_dist = torch.linalg.vector_norm(
                    xy[:, None, :] - pos[:, :, :2], dim=-1
                )
                valid &= (launch_dist > r[:, None] + safety).all(dim=-1)
                if o:
                    dd = torch.linalg.vector_norm(
                        xy[:, None, :] - obs_xy[:, :o, :], dim=-1
                    )
                    valid &= (
                        dd > r[:, None] + obs_r[:, :o]
                    ).all(dim=-1)
                take = unresolved & valid
                obs_xy[take, o] = xy[take]
                obs_r[take, o] = r[take]
                obs_h[take, o] = h[take]
                unresolved &= ~take

        target_xy = torch.zeros((n, T, 2), device=self.device, dtype=self.dtype)
        target_ex = float(CONFIG["target_exclusion_radius_m"])
        target_cells = torch.full(
            (n, T, 2), -1, device=self.device, dtype=torch.long
        )
        for t in range(T):
            unresolved = torch.ones(n, device=self.device, dtype=torch.bool)
            attempts = 0
            while bool(unresolved.any()):
                attempts += 1
                if attempts > 10000:
                    raise RuntimeError("GPU world generator could not place targets")
                xy = self._rand((n, 2)) * self.map_size
                d_obs = torch.linalg.vector_norm(
                    xy[:, None, :] - obs_xy, dim=-1
                )
                outside_obs = (d_obs > obs_r).all(dim=-1)
                valid = outside_obs & (
                    torch.linalg.vector_norm(xy - self.gcs[:2], dim=-1)
                    > target_ex
                )
                cells = torch.floor(xy / self.cell).long().clamp(0, self.G - 1)
                if t:
                    duplicate = (
                        cells[:, None, :] == target_cells[:, :t, :]
                    ).all(dim=-1).any(dim=-1)
                    valid &= ~duplicate
                take = unresolved & valid
                target_xy[take, t] = xy[take]
                target_cells[take, t] = cells[take]
                unresolved &= ~take

        return pos, vel, target_xy, obs_xy, obs_r, obs_h

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        n = int(env_ids.numel())
        if n == 0:
            return self.observations(), self.global_state(), self.destination_mask()

        pos, vel, targets, oxy, orad, oh = self._generate_world(n)
        self.pos[env_ids] = pos
        self.vel[env_ids] = vel
        self.battery[env_ids] = float(CONFIG["battery_j"])
        self.active[env_ids] = True
        self.target_xy[env_ids] = targets
        self.target_confirmed[env_ids] = False
        self.target_delivered[env_ids] = False
        self.obs_xy[env_ids] = oxy
        self.obs_radius[env_ids] = orad
        self.obs_height[env_ids] = oh

        prior = float(CONFIG["belief_prior"])
        self.belief[env_ids] = prior
        self.belief_entropy[env_ids] = _binary_entropy(
            torch.full((1,), prior, device=self.device, dtype=self.dtype)
        )
        self.entropy_sum[env_ids] = float(self.U * self.G * self.G)
        self.known_count[env_ids] = 0

        self.report_valid[env_ids] = False
        self.report_created[env_ids] = -1
        self.report_order[env_ids] = 2**60
        self.next_report_order[env_ids] = 0
        self.pending_valid[env_ids] = False
        self.pending_created[env_ids] = -1
        self.pending_order[env_ids] = 2**60
        self.next_pending_order[env_ids] = 0
        self.gcs_progress[env_ids] = 0.0
        self.peer_progress[env_ids] = 0.0
        self.peer_valid[env_ids] = False
        self.peer_created[env_ids] = -1

        self.current_step[env_ids] = 0
        self.done[env_ids] = False
        self.diag_boundary[env_ids] = 0
        self.diag_horizontal_boundary[env_ids] = 0
        self.diag_altitude_boundary[env_ids] = 0
        self.diag_blocked[env_ids] = 0
        self.diag_false_confirm[env_ids] = 0
        self.diag_total_energy[env_ids] = 0.0
        self.episode_index[env_ids] += 1

        return self.observations(), self.global_state(), self.destination_mask()

    @staticmethod
    def _circle_sqrt_integral_torch(
        x: torch.Tensor, radius: torch.Tensor
    ) -> torch.Tensor:
        eps = torch.finfo(x.dtype).eps
        rs = radius.clamp_min(eps)
        x = torch.maximum(torch.minimum(x, radius), -radius)
        root = torch.sqrt((radius * radius - x * x).clamp_min(0.0))
        ratio = (x / rs).clamp(-1.0, 1.0)
        return 0.5 * (x * root + radius * radius * torch.asin(ratio))

    def _coverage_fraction(
        self,
        circle_x: torch.Tensor,
        circle_y: torch.Tensor,
        radius: torch.Tensor,
        gx: torch.Tensor,
        gy: torch.Tensor,
    ) -> torch.Tensor:
        """Exact circle/axis-aligned-cell overlap fraction, vectorized."""
        x_min = gx.to(self.dtype) * self.cell - circle_x
        x_max = (
            ((gx + 1).to(self.dtype) * self.cell).clamp_max(self.map_size)
            - circle_x
        )
        y_min = gy.to(self.dtype) * self.cell - circle_y
        y_max = (
            ((gy + 1).to(self.dtype) * self.cell).clamp_max(self.map_size)
            - circle_y
        )

        left = torch.maximum(x_min, -radius)
        right = torch.minimum(x_max, radius)
        valid = (
            (radius > 0.0)
            & (right > left)
            & (y_max > -radius)
            & (y_min < radius)
        )

        def crossings(y_edge):
            has = torch.abs(y_edge) < radius
            xc = torch.sqrt((radius * radius - y_edge * y_edge).clamp_min(0.0))
            neg = torch.where(has, -xc, left)
            pos = torch.where(has, xc, left)
            neg = torch.maximum(left, torch.minimum(right, neg))
            pos = torch.maximum(left, torch.minimum(right, pos))
            return neg, pos

        ymin_neg, ymin_pos = crossings(y_min)
        ymax_neg, ymax_pos = crossings(y_max)
        cuts = torch.stack(
            [left, right, ymin_neg, ymin_pos, ymax_neg, ymax_pos], dim=-1
        )
        cuts = torch.sort(cuts, dim=-1).values
        a, b = cuts[..., :-1], cuts[..., 1:]
        width = b - a
        mid = 0.5 * (a + b)
        rr = radius.unsqueeze(-1)
        ymin2 = y_min.unsqueeze(-1)
        ymax2 = y_max.unsqueeze(-1)
        half = torch.sqrt((rr * rr - mid * mid).clamp_min(0.0))
        upper = torch.minimum(ymax2, half)
        lower = torch.maximum(ymin2, -half)
        arc = self._circle_sqrt_integral_torch(b, rr) - self._circle_sqrt_integral_torch(a, rr)
        upper_i = torch.where(ymax2 < half, ymax2 * width, arc)
        lower_i = torch.where(ymin2 > -half, ymin2 * width, -arc)
        terms = torch.where(
            (width > 0.0) & (upper > lower),
            upper_i - lower_i,
            torch.zeros_like(width),
        )
        area = terms.sum(dim=-1)
        return torch.where(
            valid,
            (area / (self.cell * self.cell)).clamp(0.0, 1.0),
            torch.zeros_like(area),
        )

    def _segment_hits_obstacles(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        margin: float = 0.0,
    ) -> torch.Tensor:
        """Return whether each segment intersects any vertical cylinder.

        start/end: [E, ..., 3].  Output: [E, ...].
        """
        if start.shape != end.shape or start.shape[0] != self.num_envs:
            raise ValueError("segment tensors have incompatible shapes")
        rest_ndim = start.ndim - 2
        center = self.obs_xy.view(
            self.num_envs, *([1] * rest_ndim), self.O, 2
        )
        radius = (self.obs_radius + float(margin)).view(
            self.num_envs, *([1] * rest_ndim), self.O
        )
        height = (self.obs_height + float(margin)).view(
            self.num_envs, *([1] * rest_ndim), self.O
        )
        s = start.unsqueeze(-2)
        d = (end - start).unsqueeze(-2)
        sz = s[..., 2]
        dz = d[..., 2]
        parallel = dz.abs() <= 1e-12

        t_ground = (0.0 - sz) / torch.where(
            parallel, torch.ones_like(dz), dz
        )
        t_top = (height - sz) / torch.where(
            parallel, torch.ones_like(dz), dz
        )
        z_enter = torch.maximum(
            torch.zeros_like(t_ground), torch.minimum(t_ground, t_top)
        )
        z_exit = torch.minimum(
            torch.ones_like(t_ground), torch.maximum(t_ground, t_top)
        )
        parallel_valid = parallel & (sz >= 0.0) & (sz <= height)
        nonparallel_valid = (~parallel) & (z_enter <= z_exit)
        z_valid = parallel_valid | nonparallel_valid
        z_enter = torch.where(parallel, torch.zeros_like(z_enter), z_enter)
        z_exit = torch.where(parallel, torch.ones_like(z_exit), z_exit)

        rel = s[..., :2] - center
        dxy = d[..., :2]
        denom = (dxy * dxy).sum(dim=-1)
        numer = -(rel * dxy).sum(dim=-1)
        t_star = numer / denom.clamp_min(1e-12)
        t_star = torch.maximum(z_enter, torch.minimum(z_exit, t_star))
        t_star = torch.where(denom <= 1e-12, z_enter, t_star)
        closest = rel + t_star.unsqueeze(-1) * dxy
        dist2 = (closest * closest).sum(dim=-1)
        hit = z_valid & (dist2 <= radius * radius)
        return hit.any(dim=-1)

    def _apply_motion(self, motion: torch.Tensor):
        U = self.U
        motion = project_motion_action(motion)
        old_pos = self.pos.clone()
        old_vel = self.vel.clone()

        accel = motion * self.max_accel
        cand_vel = self.vel + accel * self.dt
        speed = torch.linalg.vector_norm(cand_vel, dim=-1, keepdim=True)
        cand_vel = cand_vel * torch.minimum(
            torch.ones_like(speed),
            torch.full_like(speed, self.max_speed) / speed.clamp_min(1e-12),
        )
        cand_vel = torch.where(
            self.active.unsqueeze(-1), cand_vel, torch.zeros_like(cand_vel)
        )
        raw_pos = old_pos + cand_vel * self.dt
        cand_pos = raw_pos.clone()
        cand_pos[..., 0] = cand_pos[..., 0].clamp(0.0, self.map_size)
        cand_pos[..., 1] = cand_pos[..., 1].clamp(0.0, self.map_size)
        cand_pos[..., 2] = cand_pos[..., 2].clamp(self.alt_min, self.alt_max)

        hclip = (raw_pos[..., :2] - cand_pos[..., :2]).abs().amax(dim=-1) > 1e-6
        zclip = (raw_pos[..., 2] - cand_pos[..., 2]).abs() > 1e-6
        boundary = (hclip | zclip) & self.active
        cand_vel = (cand_pos - old_pos) / self.dt

        blocked_obs = self._segment_hits_obstacles(
            old_pos, cand_pos, margin=float(CONFIG["obstacle_clearance_m"])
        ) & self.active
        blocked = blocked_obs.clone()
        blocked_peer = torch.zeros_like(blocked)

        # Monotonic fixed-point iteration. U passes are sufficient because each
        # pass can only add blocked UAVs.
        eye = torch.eye(U, device=self.device, dtype=torch.bool).unsqueeze(0)
        upper = torch.triu(
            torch.ones((U, U), device=self.device, dtype=torch.bool), diagonal=1
        ).unsqueeze(0)
        for _ in range(U):
            effective = torch.where(
                blocked.unsqueeze(-1), old_pos, cand_pos
            )
            ra = old_pos[:, :, None, :] - old_pos[:, None, :, :]
            ma = (
                effective[:, :, None, :] - old_pos[:, :, None, :]
                - (effective[:, None, :, :] - old_pos[:, None, :, :])
            )
            denom = (ma * ma).sum(dim=-1)
            t = -(ra * ma).sum(dim=-1) / denom.clamp_min(1e-12)
            t = t.clamp(0.0, 1.0)
            closest = ra + t.unsqueeze(-1) * ma
            dist = torch.linalg.vector_norm(closest, dim=-1)
            active_pair = (
                self.active[:, :, None] & self.active[:, None, :] & ~eye
            )
            violation = (
                (dist < float(CONFIG["safety_distance"]))
                & active_pair
                & upper
            )
            peer_hit = violation | violation.transpose(1, 2)
            newly = peer_hit.any(dim=-1)
            blocked_peer |= newly
            blocked |= newly

        final_pos = torch.where(blocked.unsqueeze(-1), old_pos, cand_pos)
        final_vel = torch.where(
            blocked.unsqueeze(-1), torch.zeros_like(cand_vel), cand_vel
        )
        final_vel = torch.where(
            self.active.unsqueeze(-1), final_vel, torch.zeros_like(final_vel)
        )
        self.pos = final_pos
        self.vel = final_vel
        realized_accel = (final_vel - old_vel) / self.dt

        self.diag_boundary += boundary.sum(dim=-1)
        self.diag_horizontal_boundary += hclip.sum(dim=-1)
        self.diag_altitude_boundary += zclip.sum(dim=-1)
        self.diag_blocked += blocked.sum(dim=-1)
        return {
            "blocked": blocked,
            "blocked_by_obstacle": blocked_obs,
            "blocked_by_peer": blocked_peer,
            "boundary_clipped": boundary,
            "horizontal_boundary_clipped": hclip,
            "altitude_clipped": zclip,
            "realized_acceleration": realized_accel,
            "projected_motion": motion,
        }

    def _interp_profile(self, altitude: torch.Tensor):
        # Four anchors; bucketize + linear interpolation.
        idx = torch.bucketize(
            altitude.contiguous(), self.altitude_anchors[1:-1]
        )
        lo = self.altitude_anchors[idx]
        hi = self.altitude_anchors[idx + 1]
        frac = (altitude - lo) / (hi - lo).clamp_min(1e-12)
        pd = self.pd_anchors[idx] + frac * (
            self.pd_anchors[idx + 1] - self.pd_anchors[idx]
        )
        pf = self.pf_anchors[idx] + frac * (
            self.pf_anchors[idx + 1] - self.pf_anchors[idx]
        )
        fov = altitude * math.tan(
            math.radians(float(CONFIG["camera_full_fov_deg"])) / 2.0
        )
        return pd, pf, fov

    def _update_belief_cells(
        self,
        env_idx: torch.Tensor,
        uav_idx: torch.Tensor,
        gx: torch.Tensor,
        gy: torch.Tensor,
        posterior: torch.Tensor,
    ) -> None:
        old = self.belief[env_idx, uav_idx, gy, gx]
        old_entropy = self.belief_entropy[env_idx, uav_idx, gy, gx]
        new = posterior.clamp(1e-8, 1.0 - 1e-8)
        new_entropy = _binary_entropy(new)
        self.belief[env_idx, uav_idx, gy, gx] = new
        self.belief_entropy[env_idx, uav_idx, gy, gx] = new_entropy

        delta_e = new_entropy - old_entropy
        self.entropy_sum.scatter_add_(0, env_idx, delta_e)

        old_known = (old - float(CONFIG["belief_prior"])).abs() > 0.05
        new_known = (new - float(CONFIG["belief_prior"])).abs() > 0.05
        delta_known = new_known.long() - old_known.long()
        flat_u = env_idx * self.U + uav_idx
        flat_counts = self.known_count.view(-1)
        flat_counts.scatter_add_(0, flat_u, delta_known)

    def _sense(self, next_step: torch.Tensor):
        E, U, T, C = self.num_envs, self.U, self.T, self.C
        altitude = self.pos[..., 2]
        base_pd, base_pf, fov = self._interp_profile(altitude)

        center_cell = torch.floor(self.pos[..., :2] / self.cell).long()
        candidate = center_cell[:, :, None, :] + self.cell_offsets[None, None, :, :]
        gx = candidate[..., 0]
        gy = candidate[..., 1]
        in_grid = (gx >= 0) & (gx < self.G) & (gy >= 0) & (gy < self.G)

        cx = self.pos[..., 0, None].expand(E, U, C)
        cy = self.pos[..., 1, None].expand(E, U, C)
        rr = fov[..., None].expand(E, U, C)
        coverage = self._coverage_fraction(cx, cy, rr, gx, gy)
        visible = (
            in_grid
            & (coverage > 0.0)
            & self.active[..., None]
            & (altitude[..., None] > 0.0)
        )

        # Cell-center camera LOS.
        cell_center = torch.stack(
            [
                ((gx.to(self.dtype) + 0.5) * self.cell).clamp_max(self.map_size),
                ((gy.to(self.dtype) + 0.5) * self.cell).clamp_max(self.map_size),
                torch.zeros_like(gx, dtype=self.dtype),
            ],
            dim=-1,
        )
        start = self.pos[:, :, None, :].expand(E, U, C, 3)
        if bool(CONFIG["sensing_obstacle_occlusion"]):
            cell_blocked = self._segment_hits_obstacles(start, cell_center)
            visible &= ~cell_blocked

        # Target visibility (footprint + target LOS), then map visible targets to
        # their candidate belief cells.
        target_end = torch.cat(
            [
                self.target_xy,
                torch.zeros((E, T, 1), device=self.device, dtype=self.dtype),
            ],
            dim=-1,
        )
        tstart = self.pos[:, :, None, :].expand(E, U, T, 3)
        tend = target_end[:, None, :, :].expand(E, U, T, 3)
        target_dist = torch.linalg.vector_norm(
            self.target_xy[:, None, :, :] - self.pos[:, :, None, :2], dim=-1
        )
        target_visible = target_dist <= fov[..., None]
        if bool(CONFIG["sensing_obstacle_occlusion"]):
            target_visible &= ~self._segment_hits_obstacles(tstart, tend)
        target_visible &= self.active[..., None] & (altitude[..., None] > 0.0)

        tcell = torch.floor(self.target_xy / self.cell).long().clamp(0, self.G - 1)
        same_cell = (
            (gx[..., None] == tcell[:, None, None, :, 0])
            & (gy[..., None] == tcell[:, None, None, :, 1])
        )
        candidate_targets = same_cell & target_visible[:, :, None, :]
        has_target = candidate_targets.any(dim=-1)

        effective_pf = 1.0 - torch.pow(
            (1.0 - base_pf[..., None]).clamp_min(0.0),
            coverage,
        )
        effective_pd = (
            coverage * base_pd[..., None]
            + (1.0 - coverage) * effective_pf
        )
        positive_probability = torch.where(
            has_target, base_pd[..., None], effective_pf
        )

        # One device RNG stream: deterministic for a fixed GPU/runtime.
        random_draw = self._rand((E, U, C))
        observation = random_draw < positive_probability

        safe_gx = gx.clamp(0, self.G - 1)
        safe_gy = gy.clamp(0, self.G - 1)
        e_idx = torch.arange(E, device=self.device)[:, None, None].expand(E, U, C)
        u_idx = torch.arange(U, device=self.device)[None, :, None].expand(E, U, C)
        prior = self.belief[e_idx, u_idx, safe_gy, safe_gx]

        numerator_pos = effective_pd * prior
        denom_pos = numerator_pos + effective_pf * (1.0 - prior)
        numerator_neg = (1.0 - effective_pd) * prior
        denom_neg = numerator_neg + (1.0 - effective_pf) * (1.0 - prior)
        posterior = torch.where(
            observation,
            numerator_pos / denom_pos.clamp_min(1e-8),
            numerator_neg / denom_neg.clamp_min(1e-8),
        ).clamp(1e-8, 1.0 - 1e-8)

        update_mask = visible
        flat = update_mask.reshape(-1)
        ef = e_idx.reshape(-1)[flat]
        uf = u_idx.reshape(-1)[flat]
        gxf = safe_gx.reshape(-1)[flat]
        gyf = safe_gy.reshape(-1)[flat]
        postf = posterior.reshape(-1)[flat]
        self._update_belief_cells(ef, uf, gxf, gyf, postf)

        qualifies = (
            visible
            & observation
            & (posterior >= float(CONFIG["confirmation_threshold"]))
            & (altitude[..., None] <= float(CONFIG["fine_altitude"]))
        )
        true_events = qualifies[..., None] & candidate_targets
        false_events = qualifies & ~has_target

        # False confirmation overwrites the posterior with verified-empty belief.
        if bool(false_events.any()):
            ff = false_events.reshape(-1)
            fe = e_idx.reshape(-1)[ff]
            fu = u_idx.reshape(-1)[ff]
            fgx = safe_gx.reshape(-1)[ff]
            fgy = safe_gy.reshape(-1)[ff]
            empty = torch.full(
                (int(ff.sum()),),
                float(CONFIG["verified_empty_belief"]),
                device=self.device,
                dtype=self.dtype,
            )
            self._update_belief_cells(fe, fu, fgx, fgy, empty)

        false_count = false_events.sum(dim=(1, 2))
        self.diag_false_confirm += false_count

        event_by_source_target = true_events.any(dim=2)
        confirmed_before = self.target_confirmed.clone()
        any_confirm = event_by_source_target.any(dim=1)
        self.target_confirmed |= any_confirm
        newly_confirmed = (~confirmed_before & self.target_confirmed).sum(dim=-1)

        dropped = self._create_reports_from_confirmations(
            event_by_source_target, next_step
        )
        return {
            "newly_confirmed": newly_confirmed,
            "false_confirmations": false_count,
            "dropped_reports": dropped,
            "sensing_records": visible.sum(dim=(1, 2)),
        }

    def _create_reports_from_confirmations(
        self, events: torch.Tensor, next_step: torch.Tensor
    ) -> torch.Tensor:
        """Create source-owned reports with the same 3-report buffer limits."""
        existing = self.report_valid.any(dim=1)
        new = (
            events
            & ~self.target_delivered[:, None, :]
            & ~existing
            & ~self.pending_valid
        )
        dropped = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        for source in range(self.U):
            source_new = new[:, source, :]
            buffer_used = self.report_valid[:, source].sum(dim=(1, 2))
            free = (self.buffer_slots - buffer_used).clamp_min(0)
            rank = torch.cumsum(source_new.long(), dim=-1)
            buffered = source_new & (rank <= free[:, None])

            old_next = self.next_report_order[:, source].clone()
            order = old_next[:, None] + rank - 1
            self.report_valid[:, source, source, :] |= buffered
            self.report_created[:, source, source, :] = torch.where(
                buffered,
                next_step[:, None],
                self.report_created[:, source, source, :],
            )
            self.report_order[:, source, source, :] = torch.where(
                buffered,
                order,
                self.report_order[:, source, source, :],
            )
            self.next_report_order[:, source] += buffered.sum(dim=-1)

            remaining = source_new & ~buffered
            pending_used = self.pending_valid[:, source].sum(dim=-1)
            pfree = (self.pending_slots - pending_used).clamp_min(0)
            prank = torch.cumsum(remaining.long(), dim=-1)
            pend = remaining & (prank <= pfree[:, None])
            pold = self.next_pending_order[:, source].clone()
            porder = pold[:, None] + prank - 1
            self.pending_valid[:, source, :] |= pend
            self.pending_created[:, source, :] = torch.where(
                pend,
                next_step[:, None],
                self.pending_created[:, source, :],
            )
            self.pending_order[:, source, :] = torch.where(
                pend,
                porder,
                self.pending_order[:, source, :],
            )
            self.next_pending_order[:, source] += pend.sum(dim=-1)
            dropped += (remaining & ~pend).sum(dim=-1)
        return dropped

    def _expire_reports(self, step: torch.Tensor) -> torch.Tensor:
        ttl_steps = float(CONFIG["report_ttl"]) / self.dt
        report_exp = self.report_valid & (
            step[:, None, None, None] - self.report_created
        ).to(self.dtype).ge(ttl_steps)
        pending_exp = self.pending_valid & (
            step[:, None, None] - self.pending_created
        ).to(self.dtype).ge(ttl_steps)

        expired_targets = report_exp.any(dim=(1, 2)) | pending_exp.any(dim=1)
        self.report_valid &= ~report_exp
        self.report_created.masked_fill_(report_exp, -1)
        self.report_order.masked_fill_(report_exp, 2**60)
        self.pending_valid &= ~pending_exp
        self.pending_created.masked_fill_(pending_exp, -1)
        self.pending_order.masked_fill_(pending_exp, 2**60)

        peer_exp = self.peer_valid & (
            step[:, None, None, None, None] - self.peer_created
        ).to(self.dtype).ge(ttl_steps)
        self.peer_valid &= ~peer_exp
        self.peer_progress.masked_fill_(peer_exp, 0.0)
        self.peer_created.masked_fill_(peer_exp, -1)
        return expired_targets.sum(dim=-1)

    def _flush_pending(self, step: torch.Tensor) -> None:
        del step
        big = torch.full(
            (self.num_envs, self.T), 2**60, device=self.device, dtype=torch.long
        )
        for source in range(self.U):
            used = self.report_valid[:, source].sum(dim=(1, 2))
            free = (self.buffer_slots - used).clamp_min(0)
            scores = torch.where(
                self.pending_valid[:, source],
                self.pending_order[:, source],
                big,
            )
            sorted_score, sorted_target = torch.sort(scores, dim=-1)
            valid_sorted = sorted_score < 2**60
            take_sorted = valid_sorted & (
                torch.arange(self.T, device=self.device)[None, :]
                < free[:, None]
            )
            take = torch.zeros_like(self.pending_valid[:, source])
            take.scatter_(1, sorted_target, take_sorted)

            rank = torch.cumsum(take.long(), dim=-1)
            old_next = self.next_report_order[:, source].clone()
            new_order = old_next[:, None] + rank - 1
            self.report_valid[:, source, source, :] |= take
            self.report_created[:, source, source, :] = torch.where(
                take,
                self.pending_created[:, source],
                self.report_created[:, source, source, :],
            )
            self.report_order[:, source, source, :] = torch.where(
                take,
                new_order,
                self.report_order[:, source, source, :],
            )
            self.next_report_order[:, source] += take.sum(dim=-1)
            self.pending_valid[:, source] &= ~take
            self.pending_created[:, source].masked_fill_(take, -1)
            self.pending_order[:, source].masked_fill_(take, 2**60)

    def _retry_completed_peer_transfers(self) -> None:
        size = float(CONFIG["report_bytes"])
        e = torch.arange(self.num_envs, device=self.device)
        # At most buffer_slots receipts can enter any receiver per step.
        for recipient in range(self.U):
            used = self.report_valid[:, recipient].sum(dim=(1, 2))
            free = (self.buffer_slots - used).clamp_min(0)
            # candidates [E,sender,source,target]
            cand = (
                self.peer_valid[:, :, recipient]
                & (self.peer_progress[:, :, recipient] >= size)
            )
            # A copy already present at receiver needs no new slot.
            already = self.report_valid[:, recipient][
                :, None, :, :
            ].expand_as(cand)
            cand &= ~already
            flat = cand.reshape(self.num_envs, -1)
            created = self.peer_created[:, :, recipient].reshape(
                self.num_envs, -1
            )
            score = torch.where(
                flat,
                created * (self.U * self.T + 1)
                + torch.arange(
                    flat.shape[1], device=self.device, dtype=torch.long
                )[None, :],
                torch.full_like(created, 2**60),
            )
            values, indices = torch.sort(score, dim=-1)
            for k in range(self.buffer_slots):
                take = (k < free) & (values[:, k] < 2**60)
                idx = indices[:, k]
                sender = idx // (self.U * self.T)
                rem = idx % (self.U * self.T)
                source = rem // self.T
                target = rem % self.T

                existing = self.report_valid[e, recipient, source, target]
                take &= ~existing
                self.report_valid[e, recipient, source, target] |= take
                created_value = self.peer_created[
                    e, sender, recipient, source, target
                ]
                self.report_created[e, recipient, source, target] = torch.where(
                    take,
                    created_value,
                    self.report_created[e, recipient, source, target],
                )
                order = self.next_report_order[:, recipient]
                self.report_order[e, recipient, source, target] = torch.where(
                    take,
                    order,
                    self.report_order[e, recipient, source, target],
                )
                self.next_report_order[:, recipient] += take.long()

    def _network_step(
        self,
        destination_index: torch.Tensor,
        power_action: torch.Tensor,
        step: torch.Tensor,
    ):
        E, U, T = self.num_envs, self.U, self.T
        dev = self.device
        self._retry_completed_peer_transfers()

        comm_energy = torch.zeros((E, U), device=dev, dtype=self.dtype)
        tx_bytes_total = torch.zeros(E, device=dev, dtype=self.dtype)
        delivered_before = self.target_delivered.clone()
        size = float(CONFIG["report_bytes"])
        max_rate = float(CONFIG["comm_max_link_rate_bps"])
        bandwidth = float(CONFIG["comm_bandwidth_hz"])
        threshold = float(CONFIG["comm_snr_threshold_db"])
        ref_gain = 10.0 ** (float(CONFIG["comm_reference_gain_db"]) / 10.0)
        noise_w = 10.0 ** (
            (float(CONFIG["comm_noise_power_dbm"]) - 30.0) / 10.0
        )
        ref_d = float(CONFIG["comm_reference_distance_m"])
        path_exp = float(CONFIG["comm_path_loss_exponent"])
        nlos_gain = 10.0 ** (
            -float(CONFIG["comm_nlos_additional_loss_db"]) / 10.0
        )
        pmin, pmax = (
            float(CONFIG["tx_power_min_w"]),
            float(CONFIG["tx_power_max_w"]),
        )
        tx_power = pmin + ((power_action.squeeze(-1).clamp(-1, 1) + 1.0) / 2.0) * (
            pmax - pmin
        )
        e = torch.arange(E, device=dev)

        for sender in range(U):
            dindex = destination_index[:, sender].long().clamp(0, self.D - 1)
            recipient = self.destination_table[sender, dindex]
            non_silent = recipient != -2
            peer = recipient >= 0
            gcs = recipient == -1
            safe_recipient = recipient.clamp(0, U - 1)

            # FIFO candidates held by sender.
            valid = self.report_valid[:, sender].clone()  # [E,source,T]
            # GCS-delivered targets never transmit.
            valid &= ~self.target_delivered[:, None, :]

            # For peer destinations skip logical reports already present or fully
            # transferred to this receiver.
            receiver_reports = self.report_valid[
                e, safe_recipient
            ]  # [E,source,T]
            peer_complete = self.peer_valid[
                e, sender, safe_recipient
            ] & (
                self.peer_progress[e, sender, safe_recipient] >= size
            )
            exclude_peer = receiver_reports | peer_complete
            valid &= ~(peer[:, None, None] & exclude_peer)
            valid &= non_silent[:, None, None]

            orders = self.report_order[:, sender]
            score = torch.where(
                valid,
                orders,
                torch.full_like(orders, 2**60),
            ).reshape(E, -1)
            chosen = torch.argmin(score, dim=-1)
            has_report = score[e, chosen] < 2**60
            source = chosen // T
            target = chosen % T

            peer_progress = self.peer_progress[
                e, sender, safe_recipient, source, target
            ]
            gcs_progress = self.gcs_progress[e, source, target]
            requested = torch.where(
                peer,
                size - peer_progress,
                size - gcs_progress,
            ).clamp_min(0.0)
            attempt = has_report & non_silent & self.active[:, sender] & (requested > 0)

            start = self.pos[:, sender]
            peer_end = self.pos[e, safe_recipient]
            end = torch.where(gcs[:, None], self.gcs[None, :], peer_end)
            distance = torch.linalg.vector_norm(end - start, dim=-1)
            contact = torch.where(
                gcs,
                distance <= float(CONFIG["gcs_contact_range_m"]),
                distance <= float(CONFIG["peer_contact_range_m"]),
            )
            contact &= torch.where(
                peer, self.active[e, safe_recipient], torch.ones_like(peer)
            )
            blocked = self._segment_hits_obstacles(start, end)
            add_gain = torch.where(
                blocked,
                torch.full_like(distance, nlos_gain),
                torch.ones_like(distance),
            )
            eff_d = torch.maximum(
                distance, torch.full_like(distance, ref_d)
            )
            channel_gain = ref_gain * torch.pow(ref_d / eff_d, path_exp)
            snr = tx_power[:, sender] * channel_gain * add_gain / noise_w
            snr_db = 10.0 * torch.log10(snr.clamp_min(1e-30))
            rate = bandwidth * torch.log1p(snr) / math.log(2.0)
            rate = rate.clamp_max(max_rate)
            rate = torch.where(snr_db >= threshold, rate, torch.zeros_like(rate))
            rate = torch.where(contact & attempt, rate, torch.zeros_like(rate))
            tx_bytes = torch.minimum(
                requested,
                torch.floor(rate * self.dt / 8.0),
            )
            tx_bytes_total += tx_bytes

            # Radio energy is charged for a real attempt even when SNR yields 0.
            attempted_bytes = torch.minimum(
                requested,
                torch.full_like(requested, math.floor(max_rate * self.dt / 8.0)),
            )
            duration = torch.minimum(
                torch.full_like(requested, self.dt),
                attempted_bytes * 8.0 / max_rate,
            )
            comm_energy[:, sender] += torch.where(
                attempt,
                tx_power[:, sender] * duration,
                torch.zeros_like(duration),
            )

            # Commit peer bytes.
            peer_take = attempt & peer
            p_old = self.peer_progress[
                e, sender, safe_recipient, source, target
            ]
            p_new = torch.minimum(
                torch.full_like(p_old, size), p_old + tx_bytes
            )
            self.peer_progress[
                e, sender, safe_recipient, source, target
            ] = torch.where(peer_take, p_new, p_old)
            self.peer_valid[
                e, sender, safe_recipient, source, target
            ] |= peer_take
            selected_created = self.report_created[e, sender, source, target]
            old_pc = self.peer_created[
                e, sender, safe_recipient, source, target
            ]
            self.peer_created[
                e, sender, safe_recipient, source, target
            ] = torch.where(
                peer_take & (old_pc < 0), selected_created, old_pc
            )

            # Commit GCS bytes.
            gcs_take = attempt & gcs
            gp_old = self.gcs_progress[e, source, target]
            gp_new = torch.minimum(
                torch.full_like(gp_old, size), gp_old + tx_bytes
            )
            self.gcs_progress[e, source, target] = torch.where(
                gcs_take, gp_new, gp_old
            )
            complete = gcs_take & (gp_new >= size)
            current_del = self.target_delivered[e, target]
            self.target_delivered[e, target] = current_del | complete

        newly_delivered_mask = ~delivered_before & self.target_delivered
        # Mission delivery removes every copy/pending/peer state for that target.
        keep_t = ~newly_delivered_mask
        self.report_valid &= keep_t[:, None, None, :]
        self.pending_valid &= keep_t[:, None, :]
        self.peer_valid &= keep_t[:, None, None, None, :]
        self._retry_completed_peer_transfers()
        return {
            "communication_energy": comm_energy,
            "tx_bytes": tx_bytes_total,
            "newly_delivered": newly_delivered_mask.sum(dim=-1),
        }

    def _apply_energy(
        self,
        motion: torch.Tensor,
        realized_accel: torch.Tensor,
        communication_energy: torch.Tensor,
    ) -> torch.Tensor:
        was_active = self.active.clone()
        speed = torch.linalg.vector_norm(self.vel, dim=-1)
        commanded_accel = (
            torch.linalg.vector_norm(motion, dim=-1) * self.max_accel
        )
        accel = torch.linalg.vector_norm(realized_accel, dim=-1).clamp_max(
            self.max_accel
        )
        idle = (
            (self.pos[..., 2] <= 0.0)
            & (speed <= 1e-12)
            & (commanded_accel <= 1e-12)
        )
        hover_power = (
            float(CONFIG["energy_hover_power_w"])
            + float(CONFIG["energy_speed_sq_coeff"]) * speed.square()
            + float(CONFIG["energy_accel_sq_coeff"]) * accel.square()
        )
        power = torch.where(
            idle,
            torch.full_like(speed, float(CONFIG["energy_idle_power_w"])),
            hover_power,
        )
        propulsion = torch.where(
            was_active, power * self.dt, torch.zeros_like(power)
        )
        total = propulsion + communication_energy
        self.battery = (self.battery - total).clamp_min(0.0)
        depleted = was_active & (self.battery <= 0.0)
        self.active &= ~depleted
        self.vel = torch.where(
            self.active.unsqueeze(-1), self.vel, torch.zeros_like(self.vel)
        )
        self.diag_total_energy += total.sum(dim=-1)
        return total

    def destination_mask(self) -> torch.Tensor:
        E, U = self.num_envs, self.U
        mask = torch.zeros((E, U, self.D), device=self.device, dtype=self.dtype)
        mask[..., 0] = 1.0
        has_report = self.report_valid.any(dim=(2, 3))
        for sender in range(U):
            enabled = self.active[:, sender] & has_report[:, sender]
            choices = self.destination_table[sender]
            for index in range(1, self.D):
                recipient = int(choices[index])
                if recipient == -1:
                    dist = torch.linalg.vector_norm(
                        self.pos[:, sender] - self.gcs, dim=-1
                    )
                    valid = dist <= float(CONFIG["gcs_contact_range_m"])
                else:
                    dist = torch.linalg.vector_norm(
                        self.pos[:, sender] - self.pos[:, recipient], dim=-1
                    )
                    valid = (
                        self.active[:, recipient]
                        & (dist <= float(CONFIG["peer_contact_range_m"]))
                    )
                mask[:, sender, index] = (enabled & valid).to(self.dtype)
        return mask

    def _belief_patch(self) -> torch.Tensor:
        E, U, P = self.num_envs, self.U, self.patch_cells
        center = torch.floor(self.pos[..., :2] / self.cell).long().clamp(0, self.G - 1)
        coords = center[:, :, None, :] + self.patch_offsets[None, None, :, :]
        gx, gy = coords[..., 0], coords[..., 1]
        valid = (gx >= 0) & (gx < self.G) & (gy >= 0) & (gy < self.G)
        sx, sy = gx.clamp(0, self.G - 1), gy.clamp(0, self.G - 1)
        e = torch.arange(E, device=self.device)[:, None, None].expand(E, U, P * P)
        u = torch.arange(U, device=self.device)[None, :, None].expand(E, U, P * P)
        values = self.belief[e, u, sy, sx]
        values = torch.where(
            valid,
            values,
            torch.full_like(values, float(CONFIG["belief_prior"])),
        )
        return values.reshape(E, U, P * P)

    def _coarse_uncertainty(self) -> torch.Tensor:
        c = self.coarse_cells
        block = self.G // c
        return (
            self.belief_entropy.view(
                self.num_envs, self.U, c, block, c, block
            )
            .mean(dim=(3, 5))
            .reshape(self.num_envs, self.U, c * c)
            .clamp(0.0, 1.0)
        )

    def _pooled_belief(self) -> torch.Tensor:
        c = self.critic_cells
        block = self.G // c
        return (
            self.belief.view(
                self.num_envs, self.U, c, block, c, block
            )
            .mean(dim=(3, 5))
            .reshape(self.num_envs, self.U, c * c)
            .clamp(0.0, 1.0)
        )

    def _obstacle_features(self) -> torch.Tensor:
        rel = (
            self.obs_xy[:, None, :, :] - self.pos[:, :, None, :2]
        )
        dist = torch.linalg.vector_norm(rel, dim=-1)
        count = self.nearest_obstacles
        _, idx = torch.topk(
            dist, k=count, dim=-1, largest=False, sorted=True
        )
        gather_xy = idx.unsqueeze(-1).expand(-1, -1, -1, 2)
        nearest_rel = torch.gather(rel, 2, gather_xy)
        nearest_dist = torch.gather(dist, 2, idx)
        radius = torch.gather(
            self.obs_radius[:, None, :].expand(-1, self.U, -1), 2, idx
        )
        height = torch.gather(
            self.obs_height[:, None, :].expand(-1, self.U, -1), 2, idx
        )
        diagonal = math.sqrt(2.0) * self.map_size
        return torch.stack(
            [
                (nearest_rel[..., 0] / self.map_size).clamp(-1.0, 1.0),
                (nearest_rel[..., 1] / self.map_size).clamp(-1.0, 1.0),
                (nearest_dist / diagonal).clamp(0.0, 1.0),
                (radius / self.map_size).clamp(0.0, 1.0),
                (height / self.alt_max).clamp(0.0, 1.0),
            ],
            dim=-1,
        ).reshape(self.num_envs, self.U, count * 5)

    def _neighbor_features(self) -> torch.Tensor:
        rows = []
        scale = torch.tensor(
            [self.map_size, self.map_size, self.alt_span],
            device=self.device,
            dtype=self.dtype,
        )
        for sender in range(self.U):
            peers = [p for p in range(self.U) if p != sender]
            rel = (
                self.pos[:, peers] - self.pos[:, sender, None]
            )
            dist = torch.linalg.vector_norm(rel, dim=-1)
            visible = (
                self.active[:, sender, None]
                & self.active[:, peers]
                & (dist <= float(CONFIG["peer_contact_range_m"]))
            )
            normalized = (rel / scale).clamp(-1.0, 1.0)
            feature = torch.cat(
                [
                    normalized,
                    (dist / float(CONFIG["peer_contact_range_m"]))
                    .clamp(0.0, 1.0)
                    .unsqueeze(-1),
                    torch.ones_like(dist).unsqueeze(-1),
                ],
                dim=-1,
            )
            feature = torch.where(
                visible.unsqueeze(-1), feature, torch.zeros_like(feature)
            )
            rows.append(feature.reshape(self.num_envs, -1))
        return torch.stack(rows, dim=1)

    def _main_report_slots(self) -> torch.Tensor:
        E, U, T = self.num_envs, self.U, self.T
        out = torch.zeros(
            (E, U, self.buffer_slots, 4),
            device=self.device,
            dtype=self.dtype,
        )
        target_scale = max(1.0, float(T - 1))
        for holder in range(U):
            valid = self.report_valid[:, holder].reshape(E, U * T)
            order = self.report_order[:, holder].reshape(E, U * T)
            score = torch.where(valid, order, torch.full_like(order, 2**60))
            values, idx = torch.sort(score, dim=-1)
            for slot in range(self.buffer_slots):
                flat = idx[:, slot]
                source = flat // T
                target = flat % T
                is_valid = values[:, slot] < 2**60
                created = self.report_created[
                    torch.arange(E, device=self.device), holder, source, target
                ]
                age = (
                    (self.current_step - created).to(self.dtype)
                    * self.dt
                    / float(CONFIG["report_ttl"])
                ).clamp(0.0, 1.0)
                progress = (
                    self.gcs_progress[
                        torch.arange(E, device=self.device), source, target
                    ]
                    / float(CONFIG["report_bytes"])
                ).clamp(0.0, 1.0)
                out[:, holder, slot, 0] = is_valid.to(self.dtype)
                out[:, holder, slot, 1] = torch.where(
                    is_valid,
                    target.to(self.dtype) / target_scale,
                    torch.zeros_like(age),
                )
                out[:, holder, slot, 2] = torch.where(
                    is_valid, age, torch.zeros_like(age)
                )
                out[:, holder, slot, 3] = torch.where(
                    is_valid, progress, torch.zeros_like(progress)
                )
        return out

    def _pending_report_slots(self) -> torch.Tensor:
        E, U, T = self.num_envs, self.U, self.T
        out = torch.zeros(
            (E, U, self.pending_slots, 4),
            device=self.device,
            dtype=self.dtype,
        )
        target_scale = max(1.0, float(T - 1))
        e = torch.arange(E, device=self.device)
        for source in range(U):
            score = torch.where(
                self.pending_valid[:, source],
                self.pending_order[:, source],
                torch.full_like(self.pending_order[:, source], 2**60),
            )
            values, target = torch.sort(score, dim=-1)
            for slot in range(self.pending_slots):
                tid = target[:, slot]
                is_valid = values[:, slot] < 2**60
                created = self.pending_created[e, source, tid]
                age = (
                    (self.current_step - created).to(self.dtype)
                    * self.dt
                    / float(CONFIG["report_ttl"])
                ).clamp(0.0, 1.0)
                progress = (
                    self.gcs_progress[e, source, tid]
                    / float(CONFIG["report_bytes"])
                ).clamp(0.0, 1.0)
                out[:, source, slot, 0] = is_valid.to(self.dtype)
                out[:, source, slot, 1] = torch.where(
                    is_valid,
                    tid.to(self.dtype) / target_scale,
                    torch.zeros_like(age),
                )
                out[:, source, slot, 2] = torch.where(
                    is_valid, age, torch.zeros_like(age)
                )
                out[:, source, slot, 3] = torch.where(
                    is_valid, progress, torch.zeros_like(progress)
                )
        return out

    def _buffer_features(self) -> torch.Tensor:
        E, U, T = self.num_envs, self.U, self.T
        valid = self.report_valid
        count = valid.sum(dim=(2, 3)).to(self.dtype)
        used_fraction = (
            count * float(CONFIG["report_bytes"]) / float(CONFIG["buffer_bytes"])
        ).clamp(0.0, 1.0)
        report_fraction = (count / max(1.0, float(self.buffer_slots))).clamp(0.0, 1.0)

        created = torch.where(
            valid,
            self.report_created,
            self.current_step[:, None, None, None],
        )
        age = (
            (self.current_step[:, None, None, None] - created).to(self.dtype)
            * self.dt
            / float(CONFIG["report_ttl"])
        ).clamp(0.0, 1.0)
        oldest = torch.where(valid, age, torch.zeros_like(age)).amax(dim=(2, 3))

        pending_count = self.pending_valid.sum(dim=-1).to(self.dtype)
        pending_fraction = (
            pending_count
            * float(CONFIG["report_bytes"])
            / float(CONFIG["pending_buffer_bytes"])
        ).clamp(0.0, 1.0)

        # FIFO head progress.
        order = self.report_order.reshape(E, U, U * T)
        flat_valid = valid.reshape(E, U, U * T)
        score = torch.where(
            flat_valid, order, torch.full_like(order, 2**60)
        )
        idx = torch.argmin(score, dim=-1)
        source = idx // T
        target = idx % T
        e = torch.arange(E, device=self.device)[:, None].expand(E, U)
        progress = self.gcs_progress[e, source, target] / float(
            CONFIG["report_bytes"]
        )
        has = score.gather(-1, idx.unsqueeze(-1)).squeeze(-1) < 2**60
        progress = torch.where(has, progress, torch.zeros_like(progress))
        return torch.stack(
            [used_fraction, report_fraction, oldest, pending_fraction, progress],
            dim=-1,
        )

    def observations(self) -> torch.Tensor:
        E, U = self.num_envs, self.U
        time_fraction = (
            self.current_step.to(self.dtype) / max(1.0, float(self.max_steps))
        ).clamp(0.0, 1.0)
        self_state = torch.cat(
            [
                (self.pos[..., 0:1] / self.map_size).clamp(0.0, 1.0),
                (self.pos[..., 1:2] / self.map_size).clamp(0.0, 1.0),
                ((self.pos[..., 2:3] - self.alt_min) / self.alt_span).clamp(0.0, 1.0),
                (self.vel / self.max_speed).clamp(-1.0, 1.0),
                (
                    self.battery.unsqueeze(-1) / float(CONFIG["battery_j"])
                ).clamp(0.0, 1.0),
                self.active.to(self.dtype).unsqueeze(-1),
                time_fraction[:, None, None].expand(E, U, 1),
            ],
            dim=-1,
        )
        scale = torch.tensor(
            [self.map_size, self.map_size, self.alt_span],
            device=self.device,
            dtype=self.dtype,
        )
        gcs_rel = (
            (self.gcs[None, None, :] - self.pos) / scale
        ).clamp(-1.0, 1.0)
        mask = self.destination_mask()
        obs = torch.cat(
            [
                self_state,
                gcs_rel,
                self._neighbor_features(),
                self._obstacle_features(),
                self._belief_patch(),
                self._coarse_uncertainty(),
                self._buffer_features(),
                mask,
            ],
            dim=-1,
        )
        if obs.shape != (E, U, self.local_obs_dim):
            raise RuntimeError(
                f"GPU observation shape {tuple(obs.shape)} != "
                f"{(E, U, self.local_obs_dim)}"
            )
        return obs

    def global_state(self, local_obs: torch.Tensor | None = None) -> torch.Tensor:
        if local_obs is None:
            local_obs = self.observations()
        E, U = self.num_envs, self.U
        main_slots = self._main_report_slots()
        pending_slots = self._pending_report_slots()
        # Per-UAV mean entropy is needed, not the environment aggregate.
        per_uav_entropy = self.belief_entropy.mean(dim=(2, 3))
        known_fraction = self.known_count.to(self.dtype) / float(self.G * self.G)
        summary = torch.stack(
            [per_uav_entropy.clamp(0.0, 1.0), known_fraction.clamp(0.0, 1.0)],
            dim=-1,
        )
        pooled = self._pooled_belief()
        per_uav = torch.cat(
            [
                local_obs,
                main_slots.reshape(E, U, -1),
                pending_slots.reshape(E, U, -1),
                summary,
                pooled,
            ],
            dim=-1,
        ).reshape(E, -1)

        target_features = torch.cat(
            [
                (self.target_xy / self.map_size).clamp(0.0, 1.0),
                self.target_confirmed.to(self.dtype).unsqueeze(-1),
                self.target_delivered.to(self.dtype).unsqueeze(-1),
            ],
            dim=-1,
        ).reshape(E, -1)
        obstacle_features = torch.stack(
            [
                (self.obs_xy[..., 0] / self.map_size).clamp(0.0, 1.0),
                (self.obs_xy[..., 1] / self.map_size).clamp(0.0, 1.0),
                (self.obs_radius / self.map_size).clamp(0.0, 1.0),
                (self.obs_height / self.alt_max).clamp(0.0, 1.0),
            ],
            dim=-1,
        ).reshape(E, -1)
        time_feature = (
            self.current_step.to(self.dtype) / max(1.0, float(self.max_steps))
        ).clamp(0.0, 1.0).unsqueeze(-1)
        state = torch.cat(
            [per_uav, target_features, obstacle_features, time_feature], dim=-1
        )
        if state.shape != (E, self.state_dim):
            raise RuntimeError(
                f"GPU state shape {tuple(state.shape)} != {(E, self.state_dim)}"
            )
        return state

    def belief_potential(self) -> torch.Tensor:
        return (
            1.0
            - self.entropy_sum
            / float(self.U * self.G * self.G)
        ).clamp(0.0, 1.0)

    def step(
        self,
        continuous_action: torch.Tensor,
        destination_index: torch.Tensor,
    ):
        if continuous_action.shape != (
            self.num_envs,
            self.U,
            4,
        ):
            raise ValueError("continuous_action must have shape [E,U,4]")
        if destination_index.shape != (self.num_envs, self.U):
            raise ValueError("destination_index must have shape [E,U]")
        if bool(self.done.any()):
            raise RuntimeError("reset done GPU environments before calling step again")

        potential_before = self.belief_potential()
        next_step = self.current_step + 1
        expired = self._expire_reports(next_step)
        self._flush_pending(next_step)

        motion_result = self._apply_motion(continuous_action[..., :3])
        sense_result = self._sense(next_step)
        self._flush_pending(next_step)
        network = self._network_step(
            destination_index,
            continuous_action[..., 3:4],
            next_step,
        )
        self._flush_pending(next_step)
        energy = self._apply_energy(
            motion_result["projected_motion"],
            motion_result["realized_acceleration"],
            network["communication_energy"],
        )

        potential_after = self.belief_potential()
        success = self.target_delivered.all(dim=-1)
        all_inactive = ~self.active.any(dim=-1)
        horizon = next_step >= self.max_steps
        terminated = success | all_inactive | horizon
        shaping_next = torch.where(
            terminated, torch.zeros_like(potential_after), potential_after
        )
        information_shaping = float(CONFIG["reward_info_gain"]) * (
            float(CONFIG["reward_shaping_gamma"]) * shaping_next
            - potential_before
        )

        blocked_count = motion_result["blocked"].sum(dim=-1)
        boundary_count = motion_result["boundary_clipped"].sum(dim=-1)
        reward = (
            information_shaping
            + float(CONFIG["reward_confirmation"])
            * sense_result["newly_confirmed"].to(self.dtype)
            + float(CONFIG["reward_delivery"])
            * network["newly_delivered"].to(self.dtype)
            - float(CONFIG["reward_false_confirmation"])
            * sense_result["false_confirmations"].to(self.dtype)
            - float(CONFIG["reward_blocked"]) * blocked_count.to(self.dtype)
            - float(CONFIG["reward_boundary"]) * boundary_count.to(self.dtype)
            - float(CONFIG["reward_expired_report"]) * expired.to(self.dtype)
            - float(CONFIG["reward_dropped_report"])
            * sense_result["dropped_reports"].to(self.dtype)
            - float(CONFIG["reward_energy_per_kj"])
            * energy.sum(dim=-1)
            / 1000.0
            - float(CONFIG["reward_step_penalty"])
            + torch.where(
                success,
                torch.full_like(information_shaping, float(CONFIG["reward_all_delivered_bonus"])),
                torch.zeros_like(information_shaping),
            )
        )

        self.current_step = next_step
        self.done = terminated
        obs = self.observations()
        state = self.global_state(obs)
        info = {
            "success": success,
            "newly_confirmed": sense_result["newly_confirmed"],
            "newly_delivered": network["newly_delivered"],
            "false_confirmations": sense_result["false_confirmations"],
            "dropped_reports": sense_result["dropped_reports"],
            "expired_reports": expired,
            "blocked": blocked_count,
            "boundary": boundary_count,
            "tx_bytes": network["tx_bytes"],
            "energy_j": energy.sum(dim=-1),
        }
        return obs, state, reward, terminated, info


class GpuReplayBuffer:
    """Replay storage that never leaves the training device in the hot path."""

    def __init__(
        self,
        capacity: int,
        num_agents: int,
        observation_dim: int,
        state_dim: int,
        continuous_dim: int,
        discrete_dim: int,
        device: torch.device,
        seed: int,
    ):
        self.capacity = int(capacity)
        self.num_agents = int(num_agents)
        self.observation_dim = int(observation_dim)
        self.state_dim = int(state_dim)
        self.continuous_dim = int(continuous_dim)
        self.discrete_dim = int(discrete_dim)
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))
        C, U = self.capacity, self.num_agents
        kw = {"device": self.device, "dtype": torch.float32}
        self.observations = torch.empty((C, U, observation_dim), **kw)
        self.next_observations = torch.empty_like(self.observations)
        self.states = torch.empty((C, state_dim), **kw)
        self.next_states = torch.empty_like(self.states)
        self.continuous_actions = torch.empty((C, U, continuous_dim), **kw)
        self.destination_indices = torch.empty(
            (C, U), device=self.device, dtype=torch.long
        )
        self.destination_masks = torch.empty((C, U, discrete_dim), **kw)
        self.next_destination_masks = torch.empty_like(self.destination_masks)
        self.rewards = torch.empty((C, 1), **kw)
        self.terminated = torch.empty((C, 1), **kw)
        self.position = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add_batch(
        self,
        observations,
        states,
        continuous_actions,
        destination_indices,
        destination_masks,
        rewards,
        next_observations,
        next_states,
        next_destination_masks,
        terminated,
    ):
        n = int(observations.shape[0])
        if n > self.capacity:
            raise ValueError("batch larger than replay capacity")
        indices = (
            torch.arange(n, device=self.device, dtype=torch.long)
            + self.position
        ) % self.capacity
        self.observations[indices] = observations
        self.states[indices] = states
        self.continuous_actions[indices] = continuous_actions
        self.destination_indices[indices] = destination_indices
        self.destination_masks[indices] = destination_masks
        self.rewards[indices, 0] = rewards
        self.next_observations[indices] = next_observations
        self.next_states[indices] = next_states
        self.next_destination_masks[indices] = next_destination_masks
        self.terminated[indices, 0] = terminated.to(torch.float32)
        self.position = (self.position + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, batch_size: int):
        if self.size < int(batch_size):
            raise ValueError("not enough replay samples")
        idx = torch.randint(
            0,
            self.size,
            (int(batch_size),),
            device=self.device,
            generator=self.generator,
        )
        return {
            "observations": self.observations[idx],
            "states": self.states[idx],
            "continuous_actions": self.continuous_actions[idx],
            "destination_indices": self.destination_indices[idx],
            "destination_masks": self.destination_masks[idx],
            "rewards": self.rewards[idx],
            "next_observations": self.next_observations[idx],
            "next_states": self.next_states[idx],
            "next_destination_masks": self.next_destination_masks[idx],
            "terminated": self.terminated[idx],
        }


def build_mlp(input_dim, hidden_dims, output_dim):
    dims = (int(input_dim), *tuple(int(x) for x in hidden_dims), int(output_dim))
    layers = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


def masked_logits(logits, mask):
    return logits.masked_fill(~(mask > 0.5), torch.finfo(logits.dtype).min)


def radial_squash(pre_squash, eps=1e-6):
    radius = torch.linalg.vector_norm(pre_squash, dim=-1, keepdim=True)
    sr = torch.tanh(radius)
    scale = torch.where(
        radius <= eps,
        1.0 - radius.square() / 3.0,
        sr / radius.clamp_min(eps),
    )
    action = pre_squash * scale
    radial_derivative = (1.0 - sr.square()).clamp_min(eps)
    log_det = torch.log(radial_derivative) + 2.0 * torch.log(scale.clamp_min(eps))
    return action, log_det


class MasacActor(nn.Module):
    def __init__(self, obs_dim, continuous_dim, discrete_dim, hidden):
        super().__init__()
        hidden = tuple(hidden)
        self.encoder = build_mlp(obs_dim, hidden[:-1], hidden[-1])
        self.mean = nn.Linear(hidden[-1], continuous_dim)
        self.log_std = nn.Linear(hidden[-1], continuous_dim)
        self.discrete = nn.Linear(hidden[-1], discrete_dim)

    def sample(self, obs, mask, deterministic=False):
        h = self.encoder(obs)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(
            float(CONFIG["masac_log_std_min"]),
            float(CONFIG["masac_log_std_max"]),
        )
        dist = Normal(mean, torch.exp(log_std))
        pre = mean if deterministic else dist.rsample()
        motion, motion_det = radial_squash(pre[..., :3])
        power = torch.tanh(pre[..., 3:4])
        continuous = torch.cat([motion, power], dim=-1)
        power_det = torch.log(1.0 - power.square() + 1e-6).sum(
            dim=-1, keepdim=True
        )
        continuous_logp = (
            dist.log_prob(pre).sum(dim=-1, keepdim=True)
            - motion_det
            - power_det
        )

        logits = masked_logits(self.discrete(h), mask)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        if deterministic:
            indices = torch.argmax(logits, dim=-1)
            hard = F.one_hot(indices, logits.shape[-1]).to(logits.dtype)
            action = hard
        else:
            uniform = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
            gumbel = -torch.log(-torch.log(uniform))
            soft = F.softmax(
                (logits + gumbel)
                / float(CONFIG["masac_gumbel_temperature"]),
                dim=-1,
            )
            indices = torch.argmax(soft, dim=-1)
            hard = F.one_hot(indices, logits.shape[-1]).to(logits.dtype)
            action = hard - soft.detach() + soft
        discrete_logp = (hard * log_probs).sum(dim=-1, keepdim=True)
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
        return {
            "continuous": continuous,
            "continuous_logp": continuous_logp,
            "one_hot": action,
            "indices": indices,
            "discrete_logp": discrete_logp,
            "discrete_entropy": entropy,
        }


class Matd3Actor(nn.Module):
    def __init__(self, obs_dim, continuous_dim, discrete_dim, hidden):
        super().__init__()
        hidden = tuple(hidden)
        self.encoder = build_mlp(obs_dim, hidden[:-1], hidden[-1])
        self.continuous = nn.Linear(hidden[-1], continuous_dim)
        self.discrete = nn.Linear(hidden[-1], discrete_dim)

    def actions(self, obs, mask, straight_through=True):
        h = self.encoder(obs)
        raw = self.continuous(h)
        motion, _ = radial_squash(raw[..., :3])
        power = torch.tanh(raw[..., 3:4])
        continuous = torch.cat([motion, power], dim=-1)
        logits = masked_logits(self.discrete(h), mask)
        probs = F.softmax(logits, dim=-1)
        indices = torch.argmax(logits, dim=-1)
        hard = F.one_hot(indices, logits.shape[-1]).to(logits.dtype)
        action = hard
        if straight_through:
            action = hard - probs.detach() + probs
        return {
            "continuous": continuous,
            "one_hot": action,
            "indices": indices,
        }


class CentralizedQ(nn.Module):
    def __init__(self, state_dim, joint_action_dim, hidden):
        super().__init__()
        self.net = build_mlp(state_dim + joint_action_dim, hidden, 1)

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class GpuMASAC:
    def __init__(self, env: GpuBatchedUAVEnv):
        self.device = env.device
        self.U = env.U
        self.obs_dim = env.local_obs_dim
        self.state_dim = env.state_dim
        self.C = 4
        self.D = env.D
        joint_dim = self.U * (self.C + self.D)
        hidden = CONFIG["masac_hidden_dims"]
        self.actor = MasacActor(self.obs_dim, self.C, self.D, hidden).to(self.device)
        self.q1 = CentralizedQ(self.state_dim, joint_dim, hidden).to(self.device)
        self.q2 = CentralizedQ(self.state_dim, joint_dim, hidden).to(self.device)
        self.tq1 = copy.deepcopy(self.q1).to(self.device).eval()
        self.tq2 = copy.deepcopy(self.q2).to(self.device).eval()
        for net in (self.tq1, self.tq2):
            for p in net.parameters():
                p.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=float(CONFIG["masac_actor_lr"])
        )
        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=float(CONFIG["masac_critic_lr"]),
        )
        self.log_alpha_c = torch.tensor(
            math.log(float(CONFIG["masac_initial_alpha_continuous"])),
            device=self.device,
            requires_grad=True,
        )
        self.log_alpha_d = torch.tensor(
            math.log(float(CONFIG["masac_initial_alpha_discrete"])),
            device=self.device,
            requires_grad=True,
        )
        self.alpha_c_opt = torch.optim.Adam(
            [self.log_alpha_c], lr=float(CONFIG["masac_alpha_lr"])
        )
        self.alpha_d_opt = torch.optim.Adam(
            [self.log_alpha_d], lr=float(CONFIG["masac_alpha_lr"])
        )
        self.gamma = float(CONFIG["masac_gamma"])
        self.tau = float(CONFIG["masac_tau"])
        self.clip = float(CONFIG["masac_gradient_clip_norm"])
        self.target_entropy_c = (
            -float(CONFIG["masac_continuous_target_entropy_scale"])
            * self.U
            * self.C
        )
        self.target_entropy_d_ratio = float(
            CONFIG["masac_discrete_target_entropy_ratio"]
        )
        self.update_count = 0

    @property
    def alpha_c(self):
        return self.log_alpha_c.exp()

    @property
    def alpha_d(self):
        return self.log_alpha_d.exp()

    def _policy(self, obs, mask, deterministic=False):
        B = obs.shape[0]
        x = self.actor.sample(
            obs.reshape(B * self.U, self.obs_dim),
            mask.reshape(B * self.U, self.D),
            deterministic=deterministic,
        )
        return {
            "continuous": x["continuous"].reshape(B, self.U, self.C),
            "one_hot": x["one_hot"].reshape(B, self.U, self.D),
            "indices": x["indices"].reshape(B, self.U),
            "continuous_logp": x["continuous_logp"]
            .reshape(B, self.U, 1)
            .sum(dim=1),
            "discrete_logp": x["discrete_logp"]
            .reshape(B, self.U, 1)
            .sum(dim=1),
            "discrete_entropy": x["discrete_entropy"]
            .reshape(B, self.U, 1)
            .sum(dim=1),
        }

    def actions(self, obs, mask, deterministic=False):
        with torch.no_grad():
            p = self._policy(obs, mask, deterministic)
        return p["continuous"], p["indices"]

    def _joint(self, continuous, one_hot):
        return torch.cat([continuous, one_hot], dim=-1).reshape(
            continuous.shape[0], -1
        )

    def _soft_update(self):
        with torch.no_grad():
            for target, source in ((self.tq1, self.q1), (self.tq2, self.q2)):
                for tp, sp in zip(target.parameters(), source.parameters()):
                    tp.mul_(1.0 - self.tau).add_(sp, alpha=self.tau)

    def update(self, replay: GpuReplayBuffer, batch_size: int):
        b = replay.sample(batch_size)
        one_hot = F.one_hot(b["destination_indices"], self.D).to(torch.float32)
        replay_action = self._joint(b["continuous_actions"], one_hot)
        with torch.no_grad():
            nxt = self._policy(
                b["next_observations"], b["next_destination_masks"]
            )
            nq = torch.minimum(
                self.tq1(b["next_states"], self._joint(nxt["continuous"], nxt["one_hot"])),
                self.tq2(b["next_states"], self._joint(nxt["continuous"], nxt["one_hot"])),
            )
            adjusted = (
                nq
                - self.alpha_c.detach() * nxt["continuous_logp"]
                - self.alpha_d.detach() * nxt["discrete_logp"]
            )
            target = b["rewards"] + self.gamma * (1.0 - b["terminated"]) * adjusted

        q1 = self.q1(b["states"], replay_action)
        q2 = self.q2(b["states"], replay_action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.q_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()), self.clip
        )
        self.q_opt.step()

        for q in (self.q1, self.q2):
            for p in q.parameters():
                p.requires_grad_(False)
        pol = self._policy(b["observations"], b["destination_masks"])
        pq = torch.minimum(
            self.q1(b["states"], self._joint(pol["continuous"], pol["one_hot"])),
            self.q2(b["states"], self._joint(pol["continuous"], pol["one_hot"])),
        )
        actor_loss = (
            self.alpha_c.detach() * pol["continuous_logp"]
            - self.alpha_d.detach() * pol["discrete_entropy"]
            - pq
        ).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip)
        self.actor_opt.step()
        for q in (self.q1, self.q2):
            for p in q.parameters():
                p.requires_grad_(True)

        continuous_entropy = -pol["continuous_logp"].detach()
        valid_counts = b["destination_masks"].sum(dim=-1).clamp_min(1.0)
        target_d = self.target_entropy_d_ratio * torch.log(valid_counts).sum(
            dim=1, keepdim=True
        )
        entropy_d = pol["discrete_entropy"].detach()
        alpha_c_loss = (
            self.log_alpha_c
            * (continuous_entropy - self.target_entropy_c)
        ).mean()
        alpha_d_loss = (
            self.log_alpha_d * (entropy_d - target_d)
        ).mean()
        self.alpha_c_opt.zero_grad(set_to_none=True)
        alpha_c_loss.backward()
        self.alpha_c_opt.step()
        self.alpha_d_opt.zero_grad(set_to_none=True)
        alpha_d_loss.backward()
        self.alpha_d_opt.step()
        self._soft_update()
        self.update_count += 1
        return {
            "critic_loss": critic_loss.detach(),
            "actor_loss": actor_loss.detach(),
            "alpha_continuous": self.alpha_c.detach(),
            "alpha_discrete": self.alpha_d.detach(),
        }


class GpuMATD3:
    def __init__(self, env: GpuBatchedUAVEnv, seed: int = 44):
        self.device = env.device
        self.U = env.U
        self.obs_dim = env.local_obs_dim
        self.state_dim = env.state_dim
        self.C = 4
        self.D = env.D
        joint_dim = self.U * (self.C + self.D)
        hidden = CONFIG["matd3_hidden_dims"]
        self.actor = Matd3Actor(self.obs_dim, self.C, self.D, hidden).to(self.device)
        self.tactor = copy.deepcopy(self.actor).to(self.device).eval()
        self.q1 = CentralizedQ(self.state_dim, joint_dim, hidden).to(self.device)
        self.q2 = CentralizedQ(self.state_dim, joint_dim, hidden).to(self.device)
        self.tq1 = copy.deepcopy(self.q1).to(self.device).eval()
        self.tq2 = copy.deepcopy(self.q2).to(self.device).eval()
        for net in (self.tactor, self.tq1, self.tq2):
            for p in net.parameters():
                p.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=float(CONFIG["matd3_actor_lr"])
        )
        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=float(CONFIG["matd3_critic_lr"]),
        )
        self.gamma = float(CONFIG["matd3_gamma"])
        self.tau = float(CONFIG["matd3_tau"])
        self.delay = int(CONFIG["matd3_policy_delay"])
        self.target_noise = float(CONFIG["matd3_target_policy_noise"])
        self.noise_clip = float(CONFIG["matd3_target_noise_clip"])
        self.exploration_noise = float(CONFIG["matd3_exploration_noise"])
        self.grad_clip = float(CONFIG["matd3_gradient_clip_norm"])
        self.eps_start = float(CONFIG["matd3_discrete_epsilon_start"])
        self.eps_end = float(CONFIG["matd3_discrete_epsilon_end"])
        self.eps_steps = int(CONFIG["matd3_discrete_epsilon_decay_steps"])
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed) + 991)
        self.update_count = 0
        self.action_count = 0

    def _actor_policy(self, obs, mask, target=False, straight_through=True):
        B = obs.shape[0]
        actor = self.tactor if target else self.actor
        x = actor.actions(
            obs.reshape(B * self.U, self.obs_dim),
            mask.reshape(B * self.U, self.D),
            straight_through=straight_through,
        )
        return {
            "continuous": x["continuous"].reshape(B, self.U, self.C),
            "one_hot": x["one_hot"].reshape(B, self.U, self.D),
            "indices": x["indices"].reshape(B, self.U),
        }

    def _epsilon(self):
        frac = min(1.0, self.action_count / max(1, self.eps_steps))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def actions(self, obs, mask, deterministic=False):
        with torch.no_grad():
            p = self._actor_policy(obs, mask, straight_through=False)
            continuous = p["continuous"]
            indices = p["indices"]
            if not deterministic:
                noise = torch.randn(
                    continuous.shape,
                    device=self.device,
                    dtype=continuous.dtype,
                    generator=self.generator,
                ) * self.exploration_noise
                continuous = (continuous + noise).clamp(-1.0, 1.0)
                continuous = torch.cat(
                    [
                        project_motion_action(continuous[..., :3]),
                        continuous[..., 3:4],
                    ],
                    dim=-1,
                )
                eps = self._epsilon()
                random_gate = torch.rand(
                    indices.shape,
                    device=self.device,
                    generator=self.generator,
                ) < eps
                probs = mask / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
                random_indices = torch.multinomial(
                    probs.reshape(-1, self.D),
                    1,
                    replacement=True,
                    generator=self.generator,
                ).reshape_as(indices)
                indices = torch.where(random_gate, random_indices, indices)
                self.action_count += obs.shape[0]
        return continuous, indices

    def _joint(self, continuous, one_hot):
        return torch.cat([continuous, one_hot], dim=-1).reshape(
            continuous.shape[0], -1
        )

    def _soft_update(self, target, source):
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.mul_(1.0 - self.tau).add_(sp, alpha=self.tau)

    def update(self, replay: GpuReplayBuffer, batch_size: int):
        b = replay.sample(batch_size)
        one_hot = F.one_hot(b["destination_indices"], self.D).to(torch.float32)
        replay_action = self._joint(b["continuous_actions"], one_hot)
        with torch.no_grad():
            nxt = self._actor_policy(
                b["next_observations"],
                b["next_destination_masks"],
                target=True,
                straight_through=False,
            )
            noise = torch.randn(
                nxt["continuous"].shape,
                device=self.device,
                dtype=nxt["continuous"].dtype,
                generator=self.generator,
            )
            noise = (noise * self.target_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            target_cont = (nxt["continuous"] + noise).clamp(-1.0, 1.0)
            target_cont = torch.cat(
                [
                    project_motion_action(target_cont[..., :3]),
                    target_cont[..., 3:4],
                ],
                dim=-1,
            )
            next_action = self._joint(target_cont, nxt["one_hot"])
            target_q = torch.minimum(
                self.tq1(b["next_states"], next_action),
                self.tq2(b["next_states"], next_action),
            )
            target = b["rewards"] + self.gamma * (1.0 - b["terminated"]) * target_q

        q1 = self.q1(b["states"], replay_action)
        q2 = self.q2(b["states"], replay_action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.q_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            self.grad_clip,
        )
        self.q_opt.step()
        self.update_count += 1

        actor_loss = torch.zeros((), device=self.device)
        if self.update_count % self.delay == 0:
            for q in (self.q1, self.q2):
                for p in q.parameters():
                    p.requires_grad_(False)
            policy = self._actor_policy(
                b["observations"], b["destination_masks"], straight_through=True
            )
            actor_loss = -self.q1(
                b["states"], self._joint(policy["continuous"], policy["one_hot"])
            ).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.grad_clip
            )
            self.actor_opt.step()
            for q in (self.q1, self.q2):
                for p in q.parameters():
                    p.requires_grad_(True)
            self._soft_update(self.tactor, self.actor)
            self._soft_update(self.tq1, self.q1)
            self._soft_update(self.tq2, self.q2)

        return {
            "critic_loss": critic_loss.detach(),
            "actor_loss": actor_loss.detach(),
        }


def make_trainer(algorithm: str, env: GpuBatchedUAVEnv, seed: int):
    algorithm = algorithm.lower()
    if algorithm == "masac":
        return GpuMASAC(env)
    if algorithm == "matd3":
        return GpuMATD3(env, seed=seed)
    raise ValueError("algorithm must be masac or matd3")


def make_replay(algorithm: str, env: GpuBatchedUAVEnv, seed: int):
    capacity = int(
        CONFIG[
            "masac_replay_capacity"
            if algorithm.lower() == "masac"
            else "matd3_replay_capacity"
        ]
    )
    return GpuReplayBuffer(
        capacity,
        env.U,
        env.local_obs_dim,
        env.state_dim,
        4,
        env.D,
        env.device,
        seed,
    )


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_gpu_training(
    algorithm: str,
    total_transitions: int,
    num_envs: int,
    device: str,
    seed: int = 44,
    learning_starts: int | None = None,
    batch_size: int | None = None,
    replay_capacity: int | None = None,
    updates_per_transition: int = 1,
    train: bool = True,
):
    configure_determinism(seed)
    algorithm = algorithm.lower()
    env = GpuBatchedUAVEnv(num_envs=num_envs, device=device, seed=seed)
    trainer = make_trainer(algorithm, env, seed)
    replay = make_replay(algorithm, env, seed)
    if replay_capacity is not None and replay_capacity != replay.capacity:
        replay = GpuReplayBuffer(
            int(replay_capacity),
            env.U,
            env.local_obs_dim,
            env.state_dim,
            4,
            env.D,
            env.device,
            seed,
        )
    if learning_starts is None:
        learning_starts = int(
            CONFIG[
                "masac_learning_starts"
                if algorithm == "masac"
                else "matd3_learning_starts"
            ]
        )
    if batch_size is None:
        batch_size = int(
            CONFIG[
                "masac_batch_size"
                if algorithm == "masac"
                else "matd3_batch_size"
            ]
        )

    total_transitions = int(total_transitions)
    if total_transitions < num_envs:
        raise ValueError("total_transitions must be >= num_envs")
    steps = math.ceil(total_transitions / num_envs)
    effective_transitions = steps * num_envs
    obs = env.observations()
    state = env.global_state(obs)
    mask = env.destination_mask()
    update_count = 0
    latest_update = None
    episode_return = torch.zeros(num_envs, device=env.device)
    completed_returns = []
    completed_success = []

    _sync(env.device)
    start = time.perf_counter()
    for _ in range(steps):
        continuous, destination = trainer.actions(obs, mask)
        next_obs, next_state, reward, done, info = env.step(
            continuous, destination
        )
        next_mask = env.destination_mask()
        replay.add_batch(
            obs,
            state,
            continuous,
            destination,
            mask,
            reward,
            next_obs,
            next_state,
            next_mask,
            done,
        )
        previous = (_ * num_envs)
        current = previous + num_envs
        if train and current > learning_starts and len(replay) >= batch_size:
            due = current - max(previous, learning_starts)
            for __ in range(due * int(updates_per_transition)):
                latest_update = trainer.update(replay, batch_size)
                update_count += 1

        episode_return += reward
        if bool(done.any()):
            done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
            completed_returns.extend(
                episode_return[done_ids].detach().tolist()
            )
            completed_success.extend(
                info["success"][done_ids].detach().tolist()
            )
            episode_return[done_ids] = 0.0
            env.reset(done_ids)
            next_obs = env.observations()
            next_state = env.global_state(next_obs)
            next_mask = env.destination_mask()

        obs, state, mask = next_obs, next_state, next_mask

    _sync(env.device)
    wall = time.perf_counter() - start
    result = {
        "implementation": "gpu_native_torch",
        "algorithm": algorithm,
        "device": str(env.device),
        "num_envs": num_envs,
        "requested_transitions": total_transitions,
        "transitions": effective_transitions,
        "wall_seconds": wall,
        "transitions_per_second": effective_transitions / wall,
        "updates": update_count,
        "episodes_completed": len(completed_returns),
        "mean_completed_return": (
            sum(completed_returns) / len(completed_returns)
            if completed_returns
            else None
        ),
        "success_rate": (
            sum(float(v) for v in completed_success) / len(completed_success)
            if completed_success
            else None
        ),
        "gpu_memory_allocated_mb": (
            torch.cuda.memory_allocated(env.device) / 1024**2
            if env.device.type == "cuda"
            else 0.0
        ),
        "gpu_peak_memory_allocated_mb": (
            torch.cuda.max_memory_allocated(env.device) / 1024**2
            if env.device.type == "cuda"
            else 0.0
        ),
        "latest_update": (
            {
                k: float(v.detach().item())
                for k, v in latest_update.items()
            }
            if latest_update is not None
            else None
        ),
        "state_dims": {
            "observation": env.local_obs_dim,
            "global_state": env.state_dim,
        },
    }
    return result


def env_determinism_digest(
    num_envs: int,
    steps: int,
    device: str,
    seed: int,
):
    configure_determinism(seed)
    env = GpuBatchedUAVEnv(num_envs, device=device, seed=seed)
    gen = torch.Generator(device=env.device)
    gen.manual_seed(seed + 123456)
    obs = env.observations()
    state = env.global_state(obs)
    reward_sum = torch.zeros(num_envs, device=env.device)
    for _ in range(int(steps)):
        motion = torch.randn(
            (num_envs, env.U, 3),
            device=env.device,
            generator=gen,
        )
        motion = project_motion_action(motion)
        power = torch.rand(
            (num_envs, env.U, 1),
            device=env.device,
            generator=gen,
        ) * 2.0 - 1.0
        mask = env.destination_mask()
        probs = mask / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        destination = torch.multinomial(
            probs.reshape(-1, env.D),
            1,
            generator=gen,
        ).reshape(num_envs, env.U)
        obs, state, reward, done, _ = env.step(
            torch.cat([motion, power], dim=-1),
            destination,
        )
        reward_sum += reward
        if bool(done.any()):
            env.reset(torch.nonzero(done, as_tuple=False).squeeze(-1))

    # Tiny deterministic signature copied to CPU only after the rollout.
    values = torch.cat(
        [
            env.pos.reshape(-1),
            env.vel.reshape(-1),
            env.battery.reshape(-1),
            env.target_confirmed.to(env.dtype).reshape(-1),
            env.target_delivered.to(env.dtype).reshape(-1),
            reward_sum.reshape(-1),
            state.reshape(-1)[:2048],
        ]
    )
    import hashlib

    payload = values.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def self_test(device: str = "cpu"):
    configure_determinism(44)
    env = GpuBatchedUAVEnv(2, device=device, seed=44)
    obs = env.observations()
    state = env.global_state(obs)
    mask = env.destination_mask()
    assert obs.shape == (2, env.U, env.local_obs_dim)
    assert state.shape == (2, env.state_dim)
    assert mask.shape == (2, env.U, env.D)
    assert obs.device == env.device
    assert state.device == env.device
    assert mask.device == env.device
    gen = torch.Generator(device=env.device)
    gen.manual_seed(123)
    for _ in range(8):
        continuous = torch.rand(
            (2, env.U, 4), device=env.device, generator=gen
        ) * 2.0 - 1.0
        continuous[..., :3] = project_motion_action(continuous[..., :3])
        destination = torch.zeros(
            (2, env.U), device=env.device, dtype=torch.long
        )
        obs, state, reward, done, _info = env.step(
            continuous, destination
        )
        assert torch.isfinite(obs).all()
        assert torch.isfinite(state).all()
        assert torch.isfinite(reward).all()
        assert torch.isfinite(env.belief).all()
        assert ((env.belief >= 0.0) & (env.belief <= 1.0)).all()
        assert torch.linalg.vector_norm(env.vel, dim=-1).max() <= env.max_speed + 1e-4
        if bool(done.any()):
            env.reset(torch.nonzero(done, as_tuple=False).squeeze(-1))
    d1 = env_determinism_digest(2, 5, device, 44)
    d2 = env_determinism_digest(2, 5, device, 44)
    assert d1 == d2, (d1, d2)
    return {
        "device": device,
        "obs_shape": list(obs.shape),
        "state_shape": list(state.shape),
        "determinism_digest": d1,
        "pass": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["masac", "matd3"], default="masac")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--transitions", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--replay-capacity", type=int, default=None)
    parser.add_argument("--updates-per-transition", type=int, default=1)
    parser.add_argument("--env-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        print(json.dumps(self_test(args.device), indent=2, sort_keys=True))
        return

    result = run_gpu_training(
        args.algorithm,
        args.transitions,
        args.num_envs,
        args.device,
        seed=args.seed,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        updates_per_transition=args.updates_per_transition,
        train=not args.env_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
