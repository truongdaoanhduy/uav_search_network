"""Production source migrated from frozen fix_test_gpu.ipynb.

Do not import the notebook at runtime. The notebook is retained only as a
reference/regression artifact. Production changes belong in Python modules.
Migrated from notebook cells 221..225.
"""

from ..algorithms.matd3 import *  # noqa: F401,F403
from ..common import contact_graph_reachability_fraction_torch
from ..world.apf import apf_repulsion_torch, _peer_path_conflicts_torch

# --- frozen notebook cell 221 ---
class FullGpuUAVBatchEnv:
    """Fixed-shape batched Simple backend implemented with CUDA tensors."""

    observation_dim = 333
    state_dim = 2875
    continuous_dim = 4

    def __init__(self, num_envs: int, device: str = "cuda:0", seed: int = 44, strict_cuda: bool = True):
        self.device = torch.device(device)
        if strict_cuda:
            _require_cuda(self.device)
        self.num_envs = int(num_envs)
        self.num_uavs = int(CONFIG["num_uavs"])
        self.num_targets = int(CONFIG["num_targets"])
        self.num_obstacles = int(CONFIG["num_obstacles"])
        self.discrete_dim = self.num_uavs + 1
        self.grid_n = math.ceil(float(CONFIG["map_size"]) / float(CONFIG["grid_cell_m"]))
        nearest_obstacles = int(CONFIG["observation_nearest_obstacles"])
        patch_cells = int(CONFIG["belief_patch_cells"])
        coarse_cells = int(CONFIG["belief_coarse_cells"])
        critic_cells = int(CONFIG["critic_belief_grid_cells"])
        buffer_slots = int(
            float(CONFIG["buffer_bytes"])
            // float(CONFIG["report_bytes"])
        )
        pending_slots = int(
            float(CONFIG["pending_buffer_bytes"])
            // float(CONFIG["report_bytes"])
        )
        self.observation_dim = (
            9
            + 3
            + 5 * (self.num_uavs - 1)
            + 5 * nearest_obstacles
            + patch_cells * patch_cells
            + coarse_cells * coarse_cells
            + 5
            + self.discrete_dim
        )
        per_uav_state_dim = (
            self.observation_dim
            + 4 * buffer_slots
            + 4 * pending_slots
            + 2
            + critic_cells * critic_cells
        )
        self.state_dim = (
            self.num_uavs * per_uav_state_dim
            + 4 * self.num_targets
            + 4 * self.num_obstacles
            + 1
        )
        self.seed = int(seed)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.seed)

        self.map_size = float(CONFIG["map_size"])
        self.altitude_min = float(CONFIG["altitude_min"])
        self.altitude_max = float(CONFIG["altitude_max"])
        self.dt = float(CONFIG["dt"])
        self.max_speed = float(CONFIG["max_speed"])
        self.max_accel = float(CONFIG["max_accel"])
        self.cell_size = float(CONFIG["grid_cell_m"])
        self.max_steps = int(CONFIG["max_steps"])
        self.gcs = torch.tensor(CONFIG["gcs_position"], device=self.device, dtype=torch.float32)
        # Optional external mission-network adapter. The default Simple backend
        # remains fully tensorized; UavNetSim is attached explicitly by the
        # benchmark/training helpers because UavNetSim itself is CPU/SimPy.
        self.external_network = None
        self.network_model = str(
            CONFIG.get("full_gpu_network_model", "simple")
        ).strip().lower()
        if self.network_model not in {"simple", "packet_approx", "uavnetsim_gpu"}:
            raise ValueError(
                "full_gpu_network_model must be simple, packet_approx, or uavnetsim_gpu"
            )

        # At altitude_max with 90-degree full FOV, radius <= altitude.
        max_fov_radius = self.altitude_max * math.tan(
            math.radians(float(CONFIG["camera_full_fov_deg"])) / 2.0
        )
        max_radius_cells = math.ceil(max_fov_radius / self.cell_size) + 1
        offsets = torch.arange(-max_radius_cells, max_radius_cells + 1, device=self.device)
        oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
        self.cell_offsets = torch.stack((ox.reshape(-1), oy.reshape(-1)), dim=-1).long()

        # Communication choice table: 0=silent, last=GCS, middle entries are peers.
        choices = torch.full(
            (self.num_uavs, self.discrete_dim),
            -2,
            device=self.device,
            dtype=torch.long,
        )
        for sender in range(self.num_uavs):
            peers = [p for p in range(self.num_uavs) if p != sender]
            choices[sender, 1 : self.num_uavs] = torch.tensor(
                peers, device=self.device, dtype=torch.long
            )
            choices[sender, self.num_uavs] = -1
        self.choices = choices

        # Immutable tensors reused by every observation/network step.
        self.xyz_scale = torch.tensor(
            [
                self.map_size,
                self.map_size,
                max(self.altitude_max, 1.0),
            ],
            device=self.device,
            dtype=torch.float32,
        )
        peer_rows = [
            [peer for peer in range(self.num_uavs) if peer != sender]
            for sender in range(self.num_uavs)
        ]
        self.peer_index = torch.tensor(
            peer_rows,
            device=self.device,
            dtype=torch.long,
        )
        patch_n = int(CONFIG["belief_patch_cells"])
        patch_radius = patch_n // 2
        patch_axis = torch.arange(
            -patch_radius,
            patch_radius + 1,
            device=self.device,
            dtype=torch.long,
        )
        patch_y, patch_x = torch.meshgrid(
            patch_axis,
            patch_axis,
            indexing="ij",
        )
        self.belief_patch_offsets = torch.stack(
            (
                patch_x.reshape(-1),
                patch_y.reshape(-1),
            ),
            dim=-1,
        )
        self.target_ids = torch.arange(
            self.num_targets,
            device=self.device,
            dtype=torch.long,
        ).view(1, self.num_targets)

        self.reset()

    def attach_external_network(self, adapter):
        if self.num_envs != 1:
            raise ValueError(
                "external packet-level networking currently requires num_envs=1"
            )
        if self.external_network is not None:
            self.external_network.close()
        self.external_network = adapter
        self.external_network.reset_from_env()
        return self

    def detach_external_network(self):
        if self.external_network is not None:
            self.external_network.close()
        self.external_network = None

    def close(self):
        self.detach_external_network()

    def _rand(self, *shape):
        return torch.rand(*shape, device=self.device, generator=self.generator)


    def _sample_world(self):
        E, U, T, O = (
            self.num_envs,
            self.num_uavs,
            self.num_targets,
            self.num_obstacles,
        )
        pool = max(
            64,
            int(CONFIG.get("full_gpu_world_candidates", 256))
            * int(CONFIG.get("full_gpu_world_sampling_rounds", 4)),
        )

        # GPU rejection sampling that mirrors create_uav(): random positions
        # inside the launch disk, clipped only by map bounds and minimum
        # pairwise launch spacing.
        launch_radius = float(CONFIG["launch_radius_m"])
        launch_spacing = float(CONFIG["launch_min_spacing_m"])
        positions = self.gcs.view(1, 1, 3).repeat(E, U, 1)
        positions[..., 2] = self.altitude_min
        for uav_index in range(U):
            offsets = (self._rand(E, pool, 2) * 2.0 - 1.0) * launch_radius
            candidates = self.gcs[:2].view(1, 1, 2) + offsets
            base_valid = (
                (offsets.square().sum(-1) <= launch_radius * launch_radius)
                & (candidates[..., 0] >= 0.0)
                & (candidates[..., 0] <= self.map_size)
                & (candidates[..., 1] >= 0.0)
                & (candidates[..., 1] <= self.map_size)
            )
            if uav_index > 0:
                distances = torch.linalg.vector_norm(
                    candidates[:, :, None, :]
                    - positions[:, None, :uav_index, :2],
                    dim=-1,
                )
                spacing_score = distances.amin(-1)
                valid = base_valid & (spacing_score >= launch_spacing)
            else:
                spacing_score = torch.full(
                    (E, pool),
                    launch_spacing,
                    device=self.device,
                )
                valid = base_valid

            first_valid = valid.float().argmax(-1)
            # The fallback is still fully GPU-side and chooses the
            # best-separated in-disk candidate. With the project geometry the
            # configured pool has abundant valid samples; this only prevents a
            # silent all-false argmax from selecting an arbitrary invalid point.
            fallback_score = torch.where(
                base_valid,
                spacing_score,
                torch.full_like(spacing_score, -1.0),
            )
            fallback = fallback_score.argmax(-1)
            chosen_index = torch.where(valid.any(-1), first_valid, fallback)
            chosen = torch.gather(
                candidates,
                1,
                chosen_index[:, None, None].expand(-1, 1, 2),
            ).squeeze(1)
            positions[:, uav_index, :2] = chosen

        velocities = torch.zeros(E, U, 3, device=self.device)
        battery = torch.full(
            (E, U),
            float(CONFIG["battery_j"]),
            device=self.device,
        )
        active = torch.ones(E, U, device=self.device, dtype=torch.bool)

        rmin = float(CONFIG["obstacle_radius_min_m"])
        rmax = float(CONFIG["obstacle_radius_max_m"])
        hmin = float(CONFIG["obstacle_height_min_m"])
        hmax = float(CONFIG["obstacle_height_max_m"])
        obstacle_radius = rmin + (rmax - rmin) * self._rand(E, O)
        obstacle_height = hmin + (hmax - hmin) * self._rand(E, O)
        obstacle_xy = torch.zeros(E, O, 2, device=self.device)
        gcs_exclusion = float(CONFIG["gcs_exclusion_radius_m"])
        launch_margin = float(CONFIG["safety_distance"])
        obstacle_clearance = float(CONFIG["obstacle_clearance_m"])
        if 2.0 * (rmax + obstacle_clearance) > self.map_size:
            raise ValueError(
                "obstacle radius plus clearance must fit inside the map"
            )

        for obstacle_index in range(O):
            radius = obstacle_radius[:, obstacle_index]
            extent = radius + obstacle_clearance
            candidates = (
                extent[:, None, None]
                + self._rand(E, pool, 2)
                * (self.map_size - 2.0 * extent)[:, None, None]
            )
            gcs_distance = torch.linalg.vector_norm(
                candidates - self.gcs[:2].view(1, 1, 2),
                dim=-1,
            )
            launch_distance = torch.linalg.vector_norm(
                candidates[:, :, None, :]
                - positions[:, None, :, :2],
                dim=-1,
            )
            valid = (
                (gcs_distance > (gcs_exclusion + radius)[:, None])
                & (
                    launch_distance
                    > (radius[:, None, None] + launch_margin)
                ).all(-1)
            )
            margin_score = torch.minimum(
                gcs_distance - (gcs_exclusion + radius)[:, None],
                (
                    launch_distance
                    - (radius[:, None, None] + launch_margin)
                ).amin(-1),
            )
            if obstacle_index > 0:
                previous_distance = torch.linalg.vector_norm(
                    candidates[:, :, None, :]
                    - obstacle_xy[:, None, :obstacle_index, :],
                    dim=-1,
                )
                previous_margin = (
                    previous_distance
                    - radius[:, None, None]
                    - obstacle_radius[:, None, :obstacle_index]
                    - 2.0 * obstacle_clearance
                    - launch_margin
                )
                valid &= (previous_margin > 0.0).all(-1)
                margin_score = torch.minimum(
                    margin_score,
                    previous_margin.amin(-1),
                )

            first_valid = valid.float().argmax(-1)
            fallback = margin_score.argmax(-1)
            chosen_index = torch.where(valid.any(-1), first_valid, fallback)
            obstacle_xy[:, obstacle_index] = torch.gather(
                candidates,
                1,
                chosen_index[:, None, None].expand(-1, 1, 2),
            ).squeeze(1)

        target_xy = torch.zeros(E, T, 2, device=self.device)
        target_exclusion = float(CONFIG["target_exclusion_radius_m"])
        for target_index in range(T):
            candidates = self._rand(E, pool, 2) * self.map_size
            gcs_distance = torch.linalg.vector_norm(
                candidates - self.gcs[:2].view(1, 1, 2),
                dim=-1,
            )
            obstacle_distance = torch.linalg.vector_norm(
                candidates[:, :, None, :] - obstacle_xy[:, None, :, :],
                dim=-1,
            )
            obstacle_clearance = float(CONFIG["obstacle_clearance_m"])
            valid = (
                (gcs_distance > target_exclusion)
                & (
                    obstacle_distance
                    > (
                        obstacle_radius[:, None, :]
                        + obstacle_clearance
                    )
                ).all(-1)
            )
            obstacle_margin = (
                (
                    obstacle_distance
                    - obstacle_radius[:, None, :]
                    - obstacle_clearance
                ).amin(-1)
                if O > 0
                else torch.full_like(gcs_distance, float("inf"))
            )
            margin_score = torch.minimum(
                gcs_distance - target_exclusion,
                obstacle_margin,
            )
            candidate_cells = torch.floor(candidates / self.cell_size).long()
            if target_index > 0:
                previous_cells = torch.floor(
                    target_xy[:, :target_index] / self.cell_size
                ).long()
                duplicate_cell = (
                    candidate_cells[:, :, None, :]
                    == previous_cells[:, None, :, :]
                ).all(-1).any(-1)
                valid &= ~duplicate_cell
                margin_score = torch.where(
                    duplicate_cell,
                    torch.full_like(margin_score, -1.0),
                    margin_score,
                )

            first_valid = valid.float().argmax(-1)
            fallback = margin_score.argmax(-1)
            chosen_index = torch.where(valid.any(-1), first_valid, fallback)
            target_xy[:, target_index] = torch.gather(
                candidates,
                1,
                chosen_index[:, None, None].expand(-1, 1, 2),
            ).squeeze(1)

        return (
            positions,
            velocities,
            battery,
            active,
            target_xy,
            obstacle_xy,
            obstacle_radius,
            obstacle_height,
        )

    def reset(self, mask: torch.Tensor | None = None):
        sampled = self._sample_world()
        E, U, T = self.num_envs, self.num_uavs, self.num_targets

        if mask is None:
            self.positions, self.velocities, self.battery, self.active = sampled[:4]
            self.target_xy, self.obstacle_xy, self.obstacle_radius, self.obstacle_height = sampled[4:]
            self.confirmed = torch.zeros(E, T, device=self.device, dtype=torch.bool)
            self.delivered = torch.zeros(E, T, device=self.device, dtype=torch.bool)
            self.belief = torch.full(
                (E, U, self.grid_n, self.grid_n),
                float(CONFIG["belief_prior"]),
                device=self.device,
            )
            self.report_valid = torch.zeros(E, U, T, device=self.device, dtype=torch.bool)
            self.report_remaining = torch.zeros(E, U, T, device=self.device)
            self.report_created = torch.full(
                (E, U, T), self.max_steps + int(CONFIG["report_ttl"]) + 1,
                device=self.device, dtype=torch.long
            )
            self.report_source = torch.full(
                (E, U, T), -1, device=self.device, dtype=torch.long
            )
            self.peer_received = torch.zeros(
                E, U, U, T, device=self.device
            )
            self.peer_source = torch.full(
                (E, U, U, T), -1, device=self.device, dtype=torch.long
            )
            self.peer_created = torch.full(
                (E, U, U, T),
                self.max_steps + int(CONFIG["report_ttl"]) + 1,
                device=self.device,
                dtype=torch.long,
            )
            self.pending_valid = torch.zeros(E, U, T, device=self.device, dtype=torch.bool)
            self.pending_remaining = torch.zeros(E, U, T, device=self.device)
            self.pending_created = torch.full_like(
                self.report_created,
                self.max_steps + int(CONFIG["report_ttl"]) + 1,
            )
            self.step_count = torch.zeros(E, device=self.device, dtype=torch.long)
            self.target_seen = torch.zeros(
                E,
                T,
                device=self.device,
                dtype=torch.bool,
            )
            self.coverage_seen = torch.zeros(
                E,
                self.grid_n * self.grid_n,
                device=self.device,
                dtype=torch.bool,
            )
        else:
            mask = mask.to(device=self.device, dtype=torch.bool)
            if mask.ndim != 1 or mask.shape[0] != E:
                raise ValueError("reset mask must have shape (num_envs,)")
            m3 = mask[:, None, None]
            self.positions = torch.where(m3, sampled[0], self.positions)
            self.velocities = torch.where(m3, sampled[1], self.velocities)
            self.battery = torch.where(mask[:, None], sampled[2], self.battery)
            self.active = torch.where(mask[:, None], sampled[3], self.active)
            self.target_xy = torch.where(m3[:, :1].expand(-1, T, 2), sampled[4], self.target_xy)
            mo2 = mask[:, None, None]
            self.obstacle_xy = torch.where(mo2, sampled[5], self.obstacle_xy)
            self.obstacle_radius = torch.where(mask[:, None], sampled[6], self.obstacle_radius)
            self.obstacle_height = torch.where(mask[:, None], sampled[7], self.obstacle_height)
            self.confirmed = torch.where(mask[:, None], torch.zeros_like(self.confirmed), self.confirmed)
            self.delivered = torch.where(mask[:, None], torch.zeros_like(self.delivered), self.delivered)
            self.belief = torch.where(
                mask[:, None, None, None],
                torch.full_like(self.belief, float(CONFIG["belief_prior"])),
                self.belief,
            )
            self.report_valid = torch.where(
                mask[:, None, None], torch.zeros_like(self.report_valid), self.report_valid
            )
            self.report_remaining = torch.where(
                mask[:, None, None], torch.zeros_like(self.report_remaining), self.report_remaining
            )
            self.report_created = torch.where(
                mask[:, None, None],
                torch.full_like(self.report_created, self.max_steps + int(CONFIG["report_ttl"]) + 1),
                self.report_created,
            )
            self.report_source = torch.where(
                mask[:, None, None],
                torch.full_like(self.report_source, -1),
                self.report_source,
            )
            self.peer_received = torch.where(
                mask[:, None, None, None],
                torch.zeros_like(self.peer_received),
                self.peer_received,
            )
            self.peer_source = torch.where(
                mask[:, None, None, None],
                torch.full_like(self.peer_source, -1),
                self.peer_source,
            )
            self.peer_created = torch.where(
                mask[:, None, None, None],
                torch.full_like(
                    self.peer_created,
                    self.max_steps + int(CONFIG["report_ttl"]) + 1,
                ),
                self.peer_created,
            )
            self.pending_valid = torch.where(
                mask[:, None, None], torch.zeros_like(self.pending_valid), self.pending_valid
            )
            self.pending_remaining = torch.where(
                mask[:, None, None], torch.zeros_like(self.pending_remaining), self.pending_remaining
            )
            self.pending_created = torch.where(
                mask[:, None, None],
                torch.full_like(self.pending_created, self.max_steps + int(CONFIG["report_ttl"]) + 1),
                self.pending_created,
            )
            self.step_count = torch.where(mask, torch.zeros_like(self.step_count), self.step_count)
            self.target_seen = torch.where(
                mask[:, None],
                torch.zeros_like(self.target_seen),
                self.target_seen,
            )
            self.coverage_seen = torch.where(
                mask[:, None],
                torch.zeros_like(self.coverage_seen),
                self.coverage_seen,
            )

        entropy_map = self._belief_entropy()
        self.belief_potential = self._belief_potential(
            entropy_map
        )
        if self.external_network is not None:
            self.external_network.reset_from_env()
        obs, mask_out = self._observation(
            entropy_map=entropy_map
        )
        state = self._state(
            obs,
            entropy_map=entropy_map,
        )
        return obs, state, mask_out

    def _belief_entropy(self):
        p = self.belief.clamp(1e-6, 1.0 - 1e-6)
        return (
            -p * torch.log2(p)
            - (1.0 - p) * torch.log2(1.0 - p)
        )

    def _belief_potential(self, entropy_map=None):
        if entropy_map is None:
            entropy_map = self._belief_entropy()
        return (
            1.0
            - entropy_map.mean(dim=(1, 2, 3))
        ).clamp(0.0, 1.0)

    def _coverage_potential(self):
        return self.coverage_seen.float().mean(-1)

    def _communication_progress_potential(self):
        report_size = float(CONFIG["report_bytes"])
        report_progress = torch.where(
            self.report_valid,
            (
                1.0
                - self.report_remaining
                / max(report_size, 1.0)
            ).clamp(0.0, 1.0),
            torch.zeros_like(self.report_remaining),
        )
        best_progress = report_progress.amax(dim=1)
        best_progress = torch.maximum(
            best_progress,
            self.delivered.float(),
        )
        return best_progress.mean(-1)

    @staticmethod
    def _project_unit_ball(x):
        x = x.clamp(-1.0, 1.0)
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1.0)
        return x / norm

    def _segment_cylinder_blocked(
        self,
        start,
        end,
        obstacle_xy=None,
        obstacle_radius=None,
        obstacle_height=None,
    ):
        """Exact vectorized counterpart of CPU segment_intersects_obstacle().

        start/end have shape [E, ..., 3]. Obstacles are the environment's
        vertical finite cylinders [E, O]. The returned tensor has shape
        [E, ...] and is True when the segment intersects any cylinder.
        """
        if obstacle_xy is None:
            obstacle_xy = self.obstacle_xy
        if obstacle_radius is None:
            obstacle_radius = self.obstacle_radius
        if obstacle_height is None:
            obstacle_height = self.obstacle_height

        start = start.to(dtype=self.positions.dtype)
        end = end.to(dtype=self.positions.dtype)
        extra = start.ndim - 2
        E = start.shape[0]
        O = obstacle_xy.shape[1]
        obs_xy = obstacle_xy.reshape(
            E, *([1] * extra), O, 2
        )
        obs_r = obstacle_radius.reshape(
            E, *([1] * extra), O
        )
        obs_h = obstacle_height.reshape(
            E, *([1] * extra), O
        )
        s = start.unsqueeze(-2)
        e = end.unsqueeze(-2)
        d = e - s
        dxy = d[..., :2]
        dz = d[..., 2]
        rel_xy = s[..., :2] - obs_xy
        eps = torch.finfo(start.dtype).eps

        dz_small = dz.abs() <= eps
        horizontal_z_ok = (s[..., 2] >= 0.0) & (s[..., 2] <= obs_h)
        safe_dz = torch.where(dz_small, torch.ones_like(dz), dz)
        t_ground = (0.0 - s[..., 2]) / safe_dz
        t_top = (obs_h - s[..., 2]) / safe_dz
        z_enter = torch.maximum(
            torch.zeros_like(t_ground),
            torch.minimum(t_ground, t_top),
        )
        z_exit = torch.minimum(
            torch.ones_like(t_ground),
            torch.maximum(t_ground, t_top),
        )
        z_enter = torch.where(dz_small, torch.zeros_like(z_enter), z_enter)
        z_exit = torch.where(dz_small, torch.ones_like(z_exit), z_exit)
        z_valid = torch.where(
            dz_small,
            horizontal_z_ok,
            z_enter <= z_exit,
        )

        a = (dxy * dxy).sum(-1)
        b = 2.0 * (rel_xy * dxy).sum(-1)
        c = (rel_xy * rel_xy).sum(-1) - obs_r * obs_r
        a_small = a <= eps
        discriminant = b * b - 4.0 * a * c
        root = discriminant.clamp_min(0.0).sqrt()
        safe_two_a = torch.where(
            a_small, torch.ones_like(a), 2.0 * a
        )
        t1 = (-b - root) / safe_two_a
        t2 = (-b + root) / safe_two_a
        xy_enter = torch.maximum(
            torch.zeros_like(t1),
            torch.minimum(t1, t2),
        )
        xy_exit = torch.minimum(
            torch.ones_like(t1),
            torch.maximum(t1, t2),
        )
        xy_enter = torch.where(a_small, torch.zeros_like(xy_enter), xy_enter)
        xy_exit = torch.where(a_small, torch.ones_like(xy_exit), xy_exit)
        xy_valid = torch.where(
            a_small,
            c <= 0.0,
            (discriminant >= 0.0) & (xy_enter <= xy_exit),
        )

        enter = torch.maximum(z_enter, xy_enter)
        exit_ = torch.minimum(z_exit, xy_exit)
        intersects = z_valid & xy_valid & (enter <= exit_)
        return intersects.any(-1)


    def _motion(self, motion):
        """Apply nominal MARL acceleration through the risk-aware APF shield."""
        old_pos = self.positions
        old_vel = self.velocities
        projected = self._project_unit_ball(motion)

        nominal_accel = projected * self.max_accel
        nominal_velocity = old_vel + nominal_accel * self.dt
        nominal_speed = torch.linalg.vector_norm(
            nominal_velocity,
            dim=-1,
            keepdim=True,
        )
        nominal_velocity = nominal_velocity * torch.minimum(
            torch.ones_like(nominal_speed),
            torch.as_tensor(
                self.max_speed,
                dtype=nominal_velocity.dtype,
                device=self.device,
            )
            / nominal_speed.clamp_min(1e-8),
        )

        # Match the boundary-limited path before APF predicts peer conflicts.
        nominal_pos = old_pos + nominal_velocity * self.dt
        bounded_nominal_pos = nominal_pos.clone()
        bounded_nominal_pos[..., :2].clamp_(0.0, self.map_size)
        bounded_nominal_pos[..., 2].clamp_(self.altitude_min, self.altitude_max)
        nominal_velocity = torch.where(
            nominal_pos != bounded_nominal_pos,
            (bounded_nominal_pos - old_pos) / self.dt,
            nominal_velocity,
        )
        apf = apf_repulsion_torch(
            old_pos,
            old_vel,
            self.active,
            self.obstacle_xy,
            self.obstacle_radius,
            self.obstacle_height,
            nominal_velocities=nominal_velocity,
            max_accel=self.max_accel,
            safety_distance=float(CONFIG["safety_distance"]),
            obstacle_clearance=float(
                CONFIG["obstacle_clearance_m"]
            ),
            soft_gain=float(
                CONFIG.get("apf_soft_gain", 1.5)
            ),
            lookahead_s=float(
                CONFIG.get("apf_lookahead_s", 3.0)
            ),
            braking_margin=float(
                CONFIG.get("apf_braking_margin", 1.5)
            ),
            control_dt=self.dt,
            enabled=bool(CONFIG.get("apf_enabled", True)),
        )

        soft_desired_accel = (
            nominal_accel
            + apf["soft_acceleration_mps2"]
        )
        desired_accel = torch.where(
            apf["emergency"].unsqueeze(-1),
            apf["emergency_acceleration_mps2"],
            soft_desired_accel,
        )
        applied_motion = self._project_unit_ball(
            desired_accel
            / max(self.max_accel, 1e-12)
        )
        accel = applied_motion * self.max_accel
        apf_correction_accel = accel - nominal_accel

        candidate_velocity = old_vel + accel * self.dt
        speed = torch.linalg.vector_norm(
            candidate_velocity,
            dim=-1,
            keepdim=True,
        )
        candidate_velocity = candidate_velocity * torch.minimum(
            torch.ones_like(speed),
            torch.tensor(
                self.max_speed,
                device=self.device,
            )
            / speed.clamp_min(1e-8),
        )
        raw_pos = old_pos + candidate_velocity * self.dt
        candidate_pos = raw_pos.clone()
        candidate_pos[..., 0].clamp_(0.0, self.map_size)
        candidate_pos[..., 1].clamp_(0.0, self.map_size)
        candidate_pos[..., 2].clamp_(
            self.altitude_min,
            self.altitude_max,
        )
        hclip = (
            (raw_pos[..., :2] - candidate_pos[..., :2])
            .abs()
            .amax(-1)
            > 1e-6
        )
        zclip = (
            (raw_pos[..., 2] - candidate_pos[..., 2]).abs()
            > 1e-6
        )
        hclip &= self.active
        zclip &= self.active

        clipped_axes = (raw_pos - candidate_pos).abs() > 1e-6
        boundary_velocity = (candidate_pos - old_pos) / self.dt
        candidate_velocity = torch.where(
            clipped_axes,
            boundary_velocity,
            candidate_velocity,
        )

        if bool(CONFIG.get("apf_enabled", True)):
            clearance = float(CONFIG["obstacle_clearance_m"])
            needs_refinement = self._segment_cylinder_blocked(
                old_pos,
                candidate_pos,
                obstacle_radius=self.obstacle_radius + clearance,
                obstacle_height=self.obstacle_height + clearance,
            ) & self.active
            if bool(needs_refinement.any()):
                refined_apf = apf_repulsion_torch(
                    old_pos,
                    old_vel,
                    self.active,
                    self.obstacle_xy,
                    self.obstacle_radius,
                    self.obstacle_height,
                    nominal_velocities=candidate_velocity,
                    max_accel=self.max_accel,
                    safety_distance=float(CONFIG["safety_distance"]),
                    obstacle_clearance=clearance,
                    soft_gain=float(CONFIG.get("apf_soft_gain", 1.5)),
                    lookahead_s=float(CONFIG.get("apf_lookahead_s", 3.0)),
                    braking_margin=float(CONFIG.get("apf_braking_margin", 1.5)),
                    control_dt=self.dt,
                    enabled=True,
                )
                refined_desired = torch.where(
                    refined_apf["emergency"].unsqueeze(-1),
                    refined_apf["emergency_acceleration_mps2"],
                    nominal_accel + refined_apf["soft_acceleration_mps2"],
                )
                refined_motion = self._project_unit_ball(
                    refined_desired / max(self.max_accel, 1e-12)
                )
                refined_accel = refined_motion * self.max_accel
                refined_velocity = old_vel + refined_accel * self.dt
                refined_speed = torch.linalg.vector_norm(
                    refined_velocity,
                    dim=-1,
                    keepdim=True,
                )
                refined_velocity = refined_velocity * torch.minimum(
                    torch.ones_like(refined_speed),
                    torch.as_tensor(
                        self.max_speed,
                        dtype=refined_velocity.dtype,
                        device=self.device,
                    ) / refined_speed.clamp_min(1e-8),
                )
                refined_raw_pos = old_pos + refined_velocity * self.dt
                refined_pos = refined_raw_pos.clone()
                refined_pos[..., :2].clamp_(0.0, self.map_size)
                refined_pos[..., 2].clamp_(self.altitude_min, self.altitude_max)
                refined_hclip = (
                    (refined_raw_pos[..., :2] - refined_pos[..., :2])
                    .abs()
                    .amax(-1)
                    > 1e-6
                ) & self.active
                refined_zclip = (
                    (refined_raw_pos[..., 2] - refined_pos[..., 2]).abs()
                    > 1e-6
                ) & self.active
                refined_axes = (refined_raw_pos - refined_pos).abs() > 1e-6
                refined_boundary_velocity = (refined_pos - old_pos) / self.dt
                refined_velocity = torch.where(
                    refined_axes,
                    refined_boundary_velocity,
                    refined_velocity,
                )
                select = needs_refinement.unsqueeze(-1)
                candidate_pos = torch.where(select, refined_pos, candidate_pos)
                candidate_velocity = torch.where(
                    select,
                    refined_velocity,
                    candidate_velocity,
                )
                hclip = torch.where(needs_refinement, refined_hclip, hclip)
                zclip = torch.where(needs_refinement, refined_zclip, zclip)
                applied_motion = torch.where(select, refined_motion, applied_motion)
                accel = applied_motion * self.max_accel
                apf_correction_accel = accel - nominal_accel
                for key, refined_value in refined_apf.items():
                    current_value = apf.get(key)
                    if (
                        torch.is_tensor(current_value)
                        and current_value.shape[:2] == needs_refinement.shape
                    ):
                        selector = needs_refinement.reshape(
                            *needs_refinement.shape,
                            *([1] * (current_value.ndim - 2)),
                        )
                        apf[key] = torch.where(
                            selector,
                            refined_value,
                            current_value,
                        )

            safety_distance = float(CONFIG["safety_distance"])
            obstacle_unsafe = self._segment_cylinder_blocked(
                old_pos, candidate_pos,
                obstacle_radius=self.obstacle_radius + clearance,
                obstacle_height=self.obstacle_height + clearance,
            ) & self.active
            peer_unsafe = _peer_path_conflicts_torch(
                old_pos, candidate_pos, self.active, safety_distance,
            ).any(-1)
            if bool((obstacle_unsafe | peer_unsafe).any()):
                # Commit one UAV at a time; each trial sees the paths already
                # accepted for its peers, including earlier fallback choices.
                for i in range(self.num_uavs):
                    needs_fallback = self.active[:, i] & (
                        obstacle_unsafe[:, i]
                        | _peer_path_conflicts_torch(
                            old_pos, candidate_pos, self.active, safety_distance,
                        )[:, i].any(-1)
                    )
                    if not bool(needs_fallback.any()):
                        continue
                    brake_motion = self._project_unit_ball(
                        -old_vel[:, i] / max(self.max_accel * self.dt, 1e-12)
                    )
                    for fallback_motion in (torch.zeros_like(brake_motion), brake_motion):
                        fallback_velocity = old_vel[:, i] + fallback_motion * self.max_accel * self.dt
                        speed = torch.linalg.vector_norm(fallback_velocity, dim=-1, keepdim=True)
                        fallback_velocity = fallback_velocity * (
                            self.max_speed / speed.clamp_min(1e-8)
                        ).clamp_max(1.0)
                        raw_pos = old_pos[:, i] + fallback_velocity * self.dt
                        fallback_pos = raw_pos.clone()
                        fallback_pos[..., :2].clamp_(0.0, self.map_size)
                        fallback_pos[..., 2].clamp_(self.altitude_min, self.altitude_max)
                        clipped_axes = (raw_pos - fallback_pos).abs() > 1e-6
                        fallback_velocity = torch.where(
                            clipped_axes, (fallback_pos - old_pos[:, i]) / self.dt,
                            fallback_velocity,
                        )
                        trial_pos = candidate_pos.clone()
                        trial_pos[:, i] = fallback_pos
                        unsafe_obstacle = self._segment_cylinder_blocked(
                            old_pos[:, i], fallback_pos,
                            obstacle_radius=self.obstacle_radius + clearance,
                            obstacle_height=self.obstacle_height + clearance,
                        )
                        unsafe_peer = _peer_path_conflicts_torch(
                            old_pos, trial_pos, self.active, safety_distance,
                        )[:, i].any(-1)
                        accept = needs_fallback & ~unsafe_obstacle & ~unsafe_peer
                        select = accept.unsqueeze(-1)
                        candidate_pos[:, i] = torch.where(select, fallback_pos, candidate_pos[:, i])
                        candidate_velocity[:, i] = torch.where(
                            select, fallback_velocity, candidate_velocity[:, i],
                        )
                        hclip[:, i] = torch.where(accept, clipped_axes[..., :2].any(-1), hclip[:, i])
                        zclip[:, i] = torch.where(accept, clipped_axes[..., 2], zclip[:, i])
                        applied_motion[:, i] = torch.where(select, fallback_motion, applied_motion[:, i])
                        needs_fallback = needs_fallback & ~accept
                        if not bool(needs_fallback.any()):
                            break
                accel = applied_motion * self.max_accel
                apf_correction_accel = accel - nominal_accel

        new_pos = torch.where(
            self.active.unsqueeze(-1),
            candidate_pos,
            old_pos,
        )
        new_vel = torch.where(
            self.active.unsqueeze(-1),
            candidate_velocity,
            torch.zeros_like(candidate_velocity),
        )
        realized_accel = (
            new_vel - old_vel
        ) / self.dt

        blocked = torch.zeros_like(self.active)
        blocked_by_obstacle = torch.zeros_like(self.active)
        blocked_by_peer = torch.zeros_like(self.active)

        self.positions = new_pos
        self.velocities = new_vel
        return {
            "blocked": blocked,
            "blocked_by_obstacle": blocked_by_obstacle,
            "blocked_by_peer": blocked_by_peer,
            "apf_active": apf["active"],
            "apf_peer_active": apf["peer_active"],
            "apf_obstacle_active": (
                apf["obstacle_active"]
            ),
            "apf_emergency": apf["emergency"],
            "apf_peer_emergency": apf["peer_emergency"],
            "apf_obstacle_emergency": (
                apf["obstacle_emergency"]
            ),
            "apf_acceleration_mps2": (
                apf_correction_accel
            ),
            "apf_peer_acceleration_mps2": (
                apf["peer_acceleration_mps2"]
            ),
            "apf_obstacle_acceleration_mps2": (
                apf["obstacle_acceleration_mps2"]
            ),
            "apf_min_peer_clearance_m": (
                apf["min_peer_clearance_m"]
            ),
            "apf_min_obstacle_clearance_m": (
                apf["min_obstacle_clearance_m"]
            ),
            "boundary": hclip | zclip,
            "horizontal_boundary": hclip,
            "altitude_boundary": zclip,
            "realized_accel": realized_accel,
            "commanded_motion": applied_motion,
            "policy_motion": projected,
        }

    def _sensing_profile(self):
        z = self.positions[..., 2]
        anchors = torch.tensor(CONFIG["altitude_anchors"], device=self.device)
        pdv = torch.tensor(CONFIG["pd"], device=self.device)
        pfv = torch.tensor(CONFIG["pf"], device=self.device)
        # Piecewise linear interpolation over four anchors.
        idx = torch.bucketize(z.contiguous(), anchors[1:-1])
        z0, z1 = anchors[idx], anchors[idx + 1]
        w = ((z - z0) / (z1 - z0).clamp_min(1e-6)).clamp(0.0, 1.0)
        pd = pdv[idx] + w * (pdv[idx + 1] - pdv[idx])
        pf = pfv[idx] + w * (pfv[idx + 1] - pfv[idx])
        half = math.radians(float(CONFIG["camera_full_fov_deg"]) / 2.0)
        radius = z * math.tan(half)
        return pd, pf, radius

    @staticmethod
    def _circle_sqrt_integral(x, radius):
        radius_safe = radius.clamp_min(torch.finfo(x.dtype).eps)
        x = torch.maximum(torch.minimum(x, radius), -radius)
        root = (radius * radius - x * x).clamp_min(0.0).sqrt()
        ratio = (x / radius_safe).clamp(-1.0, 1.0)
        return 0.5 * (x * root + radius * radius * torch.asin(ratio))

    def _cell_coverage_fraction(self, circle_x, circle_y, radius, gx, gy):
        original_shape = gx.shape
        circle_x = circle_x.reshape(-1)
        circle_y = circle_y.reshape(-1)
        radius = radius.reshape(-1)
        gx = gx.reshape(-1).float()
        gy = gy.reshape(-1).float()

        x_min = gx * self.cell_size - circle_x
        x_max = ((gx + 1.0) * self.cell_size).clamp(max=self.map_size) - circle_x
        y_min = gy * self.cell_size - circle_y
        y_max = ((gy + 1.0) * self.cell_size).clamp(max=self.map_size) - circle_y
        left = torch.maximum(x_min, -radius)
        right = torch.minimum(x_max, radius)
        base_valid = (
            (radius > 0.0)
            & (right > left)
            & (y_max > -radius)
            & (y_min < radius)
        )

        def crossings(y_edge):
            has = y_edge.abs() < radius
            x_cross = (radius * radius - y_edge * y_edge).clamp_min(0.0).sqrt()
            negative = torch.where(has, -x_cross, left)
            positive = torch.where(has, x_cross, left)
            negative = torch.maximum(left, torch.minimum(right, negative))
            positive = torch.maximum(left, torch.minimum(right, positive))
            return negative, positive

        y_min_neg, y_min_pos = crossings(y_min)
        y_max_neg, y_max_pos = crossings(y_max)
        cuts = torch.stack(
            (left, right, y_min_neg, y_min_pos, y_max_neg, y_max_pos), dim=-1
        ).sort(dim=-1).values
        a, b = cuts[:, :-1], cuts[:, 1:]
        width = b - a
        midpoint = 0.5 * (a + b)
        radius_2d = radius[:, None]
        y_min_2d = y_min[:, None]
        y_max_2d = y_max[:, None]
        half_height = (
            radius_2d * radius_2d - midpoint * midpoint
        ).clamp_min(0.0).sqrt()
        upper = torch.minimum(y_max_2d, half_height)
        lower = torch.maximum(y_min_2d, -half_height)
        arc = self._circle_sqrt_integral(b, radius_2d) - self._circle_sqrt_integral(
            a, radius_2d
        )
        upper_integral = torch.where(
            y_max_2d < half_height, y_max_2d * width, arc
        )
        lower_integral = torch.where(
            y_min_2d > -half_height, y_min_2d * width, -arc
        )
        area = torch.where(
            (width > 0.0) & (upper > lower),
            upper_integral - lower_integral,
            torch.zeros_like(width),
        ).sum(-1).clamp_min(0.0)
        cell_area = ((x_max - x_min) * (y_max - y_min)).clamp_min(
            torch.finfo(area.dtype).eps
        )
        fraction = (area / cell_area).clamp(0.0, 1.0)
        return torch.where(base_valid, fraction, torch.zeros_like(fraction)).reshape(
            original_shape
        )

    def _sense(self):
        E, U, T = self.num_envs, self.num_uavs, self.num_targets
        pd, pf, radius = self._sensing_profile()
        base = torch.floor(self.positions[..., :2] / self.cell_size).long()
        cand = base.unsqueeze(2) + self.cell_offsets.view(1, 1, -1, 2)
        gx, gy = cand[..., 0], cand[..., 1]
        valid = (gx >= 0) & (gx < self.grid_n) & (gy >= 0) & (gy < self.grid_n)
        gx_c = gx.clamp(0, self.grid_n - 1)
        gy_c = gy.clamp(0, self.grid_n - 1)
        centers = (torch.stack((gx_c, gy_c), dim=-1).float() + 0.5) * self.cell_size

        coverage = self._cell_coverage_fraction(
            self.positions[..., 0].unsqueeze(-1).expand_as(gx_c),
            self.positions[..., 1].unsqueeze(-1).expand_as(gy_c),
            radius.unsqueeze(-1).expand_as(gx_c),
            gx_c,
            gy_c,
        )
        visible = valid & (coverage > 0.0)
        visible &= (self.positions[..., 2] > 0.0).unsqueeze(-1) & self.active.unsqueeze(-1)

        if bool(CONFIG["sensing_obstacle_occlusion"]):
            start = self.positions.unsqueeze(2).expand(-1, -1, centers.shape[2], -1)
            end = torch.cat(
                (centers, torch.zeros(*centers.shape[:-1], 1, device=self.device)), dim=-1
            )
            blocked = self._segment_cylinder_blocked(start, end)
            visible &= ~blocked

        target_cell = torch.floor(self.target_xy / self.cell_size).long().clamp(0, self.grid_n - 1)
        cell_match = (
            (gx_c.unsqueeze(-1) == target_cell[:, None, None, :, 0])
            & (gy_c.unsqueeze(-1) == target_cell[:, None, None, :, 1])
        )
        target_distance = torch.linalg.vector_norm(
            self.target_xy[:, None, :, :] - self.positions[:, :, None, :2], dim=-1
        )
        target_in_fov = target_distance <= radius[:, :, None]

        if bool(CONFIG["sensing_obstacle_occlusion"]):
            target_end = torch.cat(
                (
                    self.target_xy[:, None, :, :].expand(-1, U, -1, -1),
                    torch.zeros(E, U, self.num_targets, 1, device=self.device),
                ),
                dim=-1,
            )
            target_start = self.positions[:, :, None, :].expand(
                -1, -1, self.num_targets, -1
            )
            target_blocked = self._segment_cylinder_blocked(
                target_start, target_end
            )
            target_in_fov &= ~target_blocked

        target_visible_per_cell = (
            cell_match
            & target_in_fov[:, :, None, :]
        )
        target_seen_step = (
            target_visible_per_cell
            & visible.unsqueeze(-1)
        ).any(dim=(1, 2))
        self.target_seen |= target_seen_step
        has_target = target_visible_per_cell.any(-1) & visible

        effective_pf = (
            1.0 - torch.pow(1.0 - pf.unsqueeze(-1), coverage)
        ).clamp(0.0, 1.0)
        effective_pd = (
            coverage * pd.unsqueeze(-1)
            + (1.0 - coverage) * effective_pf
        ).clamp(0.0, 1.0)
        positive_prob = torch.where(
            has_target, pd.unsqueeze(-1).expand_as(effective_pf), effective_pf
        )
        observation = self._rand(*positive_prob.shape) < positive_prob
        observation &= visible

        flat_idx = gy_c * self.grid_n + gx_c
        sentinel_index = self.grid_n * self.grid_n
        coverage_count_before = self.coverage_seen.sum(-1)
        coverage_sensed_cell_events = visible.sum(dim=(1, 2))
        coverage_padded = torch.cat(
            (
                self.coverage_seen,
                torch.zeros(
                    E,
                    1,
                    device=self.device,
                    dtype=torch.bool,
                ),
            ),
            dim=1,
        )
        coverage_scatter_idx = torch.where(
            visible,
            flat_idx,
            torch.full_like(
                flat_idx,
                sentinel_index,
            ),
        ).reshape(E, -1)
        coverage_padded.scatter_(
            1,
            coverage_scatter_idx,
            torch.ones_like(
                coverage_scatter_idx,
                dtype=torch.bool,
            ),
        )
        self.coverage_seen.copy_(
            coverage_padded[
                :,
                :sentinel_index,
            ]
        )
        coverage_new_unique_cells = (
            self.coverage_seen.sum(-1)
            - coverage_count_before
        )

        belief_flat = self.belief.view(E, U, -1)
        padded_belief = torch.cat(
            (
                belief_flat,
                torch.full(
                    (E, U, 1),
                    float(CONFIG["belief_prior"]),
                    device=self.device,
                    dtype=belief_flat.dtype,
                ),
            ),
            dim=2,
        )
        scatter_idx = torch.where(
            valid,
            flat_idx,
            torch.full_like(flat_idx, sentinel_index),
        )
        prior = torch.gather(
            padded_belief,
            2,
            scatter_idx,
        )
        eps = 1e-6
        prior_c = prior.clamp(eps, 1.0 - eps)
        pos_num = effective_pd * prior_c
        pos_den = pos_num + effective_pf * (1.0 - prior_c)
        neg_num = (1.0 - effective_pd) * prior_c
        neg_den = neg_num + (1.0 - effective_pf) * (1.0 - prior_c)
        posterior = torch.where(
            observation,
            pos_num / pos_den.clamp_min(eps),
            neg_num / neg_den.clamp_min(eps),
        ).clamp(eps, 1.0 - eps)
        posterior = torch.where(visible, posterior, prior)
        padded_belief.scatter_(
            2,
            scatter_idx,
            posterior,
        )
        belief_flat.copy_(
            padded_belief[..., :sentinel_index]
        )

        fine = self.positions[..., 2] <= float(CONFIG["fine_altitude"])
        confirm_cell = (
            visible
            & observation
            & (posterior >= float(CONFIG["confirmation_threshold"]))
            & fine.unsqueeze(-1)
            & has_target
        )
        confirm_target = (
            confirm_cell.unsqueeze(-1) & target_visible_per_cell
        ).any(2)  # [E,U,T]
        any_confirm = confirm_target.any(1)
        newly_confirmed = any_confirm & ~self.confirmed
        self.confirmed |= any_confirm

        # CPU semantics permit each confirming source UAV to create its own
        # report copy. A later confirmation can also create a new source copy
        # if that UAV no longer has that target buffered/pending.
        already_local = self.report_valid | self.pending_valid
        create_mask = (
            confirm_target
            & ~self.delivered[:, None, :]
            & ~already_local
        )
        capacity = int(
            float(CONFIG["buffer_bytes"])
            // float(CONFIG["report_bytes"])
        )
        pending_capacity = int(
            float(CONFIG["pending_buffer_bytes"])
            // float(CONFIG["report_bytes"])
        )
        free_slots = (
            capacity - self.report_valid.sum(-1)
        ).clamp_min(0)
        direct_rank = create_mask.long().cumsum(-1)
        direct = (
            create_mask
            & (direct_rank <= free_slots.unsqueeze(-1))
        )
        remaining_create = create_mask & ~direct
        pending_free_slots = (
            pending_capacity - self.pending_valid.sum(-1)
        ).clamp_min(0)
        pending_rank = remaining_create.long().cumsum(-1)
        pending = (
            remaining_create
            & (
                pending_rank
                <= pending_free_slots.unsqueeze(-1)
            )
        )
        dropped = create_mask & ~direct & ~pending

        self.report_valid |= direct
        self.report_remaining = torch.where(
            direct,
            torch.full_like(
                self.report_remaining,
                float(CONFIG["report_bytes"]),
            ),
            self.report_remaining,
        )
        self.report_created = torch.where(
            direct,
            self.step_count[:, None, None].expand_as(
                self.report_created
            ),
            self.report_created,
        )
        holder_ids = torch.arange(
            U,
            device=self.device,
            dtype=torch.long,
        ).view(1, U, 1).expand(E, U, T)
        self.report_source = torch.where(
            direct,
            holder_ids,
            self.report_source,
        )
        self.pending_valid |= pending
        self.pending_remaining = torch.where(
            pending,
            torch.full_like(
                self.pending_remaining,
                float(CONFIG["report_bytes"]),
            ),
            self.pending_remaining,
        )
        self.pending_created = torch.where(
            pending,
            self.step_count[:, None, None].expand_as(
                self.pending_created
            ),
            self.pending_created,
        )

        false_mask = (
            visible
            & observation
            & (
                posterior
                >= float(CONFIG["confirmation_threshold"])
            )
            & fine.unsqueeze(-1)
            & ~has_target
        )
        false_confirmation = false_mask.sum(dim=(1, 2))
        corrected = torch.where(
            false_mask,
            torch.full_like(
                posterior,
                float(CONFIG["verified_empty_belief"]),
            ),
            torch.gather(
                padded_belief,
                2,
                scatter_idx,
            ),
        )
        padded_belief.scatter_(
            2,
            scatter_idx,
            corrected,
        )
        belief_flat.copy_(
            padded_belief[..., :sentinel_index]
        )

        return {
            "new_confirmed": newly_confirmed.sum(-1),
            "false_confirmation": false_confirmation,
            "reports_created": create_mask.sum(dim=(1, 2)),
            "dropped": dropped.sum(dim=(1, 2)),
            "coverage_sensed_cell_events": coverage_sensed_cell_events,
            "coverage_new_unique_cells": coverage_new_unique_cells,
        }


    def _flush_pending(self):
        capacity = int(
            float(CONFIG["buffer_bytes"])
            // float(CONFIG["report_bytes"])
        )
        sentinel = (
            self.max_steps
            + int(CONFIG["report_ttl"])
            + 1
        )
        holder_ids = torch.arange(
            self.num_uavs,
            device=self.device,
            dtype=torch.long,
        ).view(1, self.num_uavs, 1).expand(
            self.num_envs,
            self.num_uavs,
            self.num_targets,
        )
        # A completed relay may have filled this target slot while its local
        # confirmation was pending. Preserve the existing copy and its GCS
        # progress, matching flush_pending_reports() duplicate handling.
        source_already_buffered = (
            self.report_valid[:, None, :, :]
            & (
                self.report_source[:, None, :, :]
                == holder_ids[:, :, None, :]
            )
        ).any(dim=2)
        discard = (
            self.report_valid
            | source_already_buffered
            | self.delivered[:, None, :]
        )
        self.pending_valid &= ~discard
        self.pending_remaining = torch.where(
            self.pending_valid,
            self.pending_remaining,
            torch.zeros_like(self.pending_remaining),
        )
        for _ in range(capacity):
            free = (
                (self.report_valid.sum(-1) < capacity)
                & self.active
            )
            created = torch.where(
                self.pending_valid,
                self.pending_created,
                torch.full_like(
                    self.pending_created,
                    sentinel,
                ),
            )
            target = created.argmin(-1)
            selected = torch.gather(
                self.pending_valid,
                2,
                target.unsqueeze(-1),
            ).squeeze(-1)
            move = free & selected
            target_oh = F.one_hot(
                target,
                self.num_targets,
            ).bool()
            move_mask = move.unsqueeze(-1) & target_oh
            self.report_valid |= move_mask
            self.report_remaining = torch.where(
                move_mask,
                self.pending_remaining,
                self.report_remaining,
            )
            self.report_created = torch.where(
                move_mask,
                self.pending_created,
                self.report_created,
            )
            self.report_source = torch.where(
                move_mask,
                holder_ids,
                self.report_source,
            )
            self.pending_valid &= ~move_mask
            self.pending_remaining = torch.where(
                self.pending_valid,
                self.pending_remaining,
                torch.zeros_like(
                    self.pending_remaining
                ),
            )

    def _destination_mask(self):
        E, U = self.num_envs, self.num_uavs
        has_report = self.report_valid.any(-1) & self.active
        mask = torch.zeros(E, U, self.discrete_dim, device=self.device)
        mask[..., 0] = 1.0

        diff = self.positions.unsqueeze(2) - self.positions.unsqueeze(1)
        peer_dist = torch.linalg.vector_norm(diff, dim=-1)
        for sender in range(U):
            peers = self.choices[sender, 1:U]
            ok = (
                peer_dist[:, sender, peers] <= float(CONFIG["peer_contact_range_m"])
            ) & self.active[:, peers] & has_report[:, sender, None]
            mask[:, sender, 1:U] = ok.float()
        gcs_dist = torch.linalg.vector_norm(self.positions - self.gcs.view(1, 1, 3), dim=-1)
        mask[..., U] = (
            (gcs_dist <= float(CONFIG["gcs_contact_range_m"])) & has_report
        ).float()
        return mask


    def _commit_completed_peer_receipts(self):
        """Retry complete peer transfers when receiver buffer space exists."""
        E, U, T = (
            self.num_envs,
            self.num_uavs,
            self.num_targets,
        )
        report_size = float(CONFIG["report_bytes"])
        capacity = int(
            float(CONFIG["buffer_bytes"]) // report_size
        )
        for sender in range(U):
            for receiver in range(U):
                if sender == receiver:
                    continue
                for target in range(T):
                    complete = (
                        self.peer_received[
                            :, sender, receiver, target
                        ]
                        >= report_size
                    )
                    source = self.peer_source[
                        :, sender, receiver, target
                    ]
                    created = self.peer_created[
                        :, sender, receiver, target
                    ]
                    generation_valid = (
                        source >= 0
                    ) & (
                        created
                        <= self.step_count
                    )
                    free = (
                        self.report_valid[:, receiver, :]
                        .sum(-1)
                        < capacity
                    )
                    add = (
                        complete
                        & generation_valid
                        & ~self.delivered[:, target]
                        & ~self.report_valid[
                            :, receiver, target
                        ]
                        & free
                    )
                    # known GCS progress is the best progress among all
                    # copies of the same report generation.
                    same_generation = (
                        self.report_valid[:, :, target]
                        & (
                            self.report_source[
                                :, :, target
                            ]
                            == source[:, None]
                        )
                        & (
                            self.report_created[
                                :, :, target
                            ]
                            == created[:, None]
                        )
                    )
                    known_remaining = torch.where(
                        same_generation,
                        self.report_remaining[
                            :, :, target
                        ],
                        torch.full_like(
                            self.report_remaining[
                                :, :, target
                            ],
                            report_size,
                        ),
                    ).amin(-1)
                    self.report_valid[
                        :, receiver, target
                    ] |= add
                    self.report_remaining[
                        :, receiver, target
                    ] = torch.where(
                        add,
                        known_remaining,
                        self.report_remaining[
                            :, receiver, target
                        ],
                    )
                    self.report_created[
                        :, receiver, target
                    ] = torch.where(
                        add,
                        created,
                        self.report_created[
                            :, receiver, target
                        ],
                    )
                    self.report_source[
                        :, receiver, target
                    ] = torch.where(
                        add,
                        source,
                        self.report_source[
                            :, receiver, target
                        ],
                    )

    def _uavnetsim_gpu_radio(
        self,
        selected_dest,
        tx_power,
        attempted_mask,
    ):
        """Vectorized UavNetSim-v2 A2A PHY/SINR surrogate on one CUDA device.

        The external simulator remains a SimPy discrete-event engine. This
        routine mirrors the configured A2A path-gain law and simultaneous-link
        SINR calculation in tensor form so hundreds/thousands of training
        environments do not leave CUDA for packet/radio calculations.
        """
        E, U = self.num_envs, self.num_uavs
        is_gcs = selected_dest == -1
        is_peer = selected_dest >= 0
        receiver = selected_dest.clamp(0, max(0, U - 1))
        peer_pos = torch.gather(
            self.positions,
            1,
            receiver.unsqueeze(-1).expand(-1, -1, 3),
        )
        dest_pos = torch.where(
            is_gcs.unsqueeze(-1),
            self.gcs.view(1, 1, 3),
            peer_pos,
        )
        min_distance = 1.0
        distance = torch.linalg.vector_norm(
            dest_pos - self.positions,
            dim=-1,
        ).clamp_min(min_distance)
        receiver_active = torch.gather(
            self.active,
            1,
            receiver,
        )
        contact = torch.where(
            is_gcs,
            distance <= float(CONFIG["gcs_contact_range_m"]),
            torch.where(
                is_peer,
                distance <= float(CONFIG["peer_contact_range_m"]),
                torch.zeros_like(is_peer),
            ),
        )
        contact &= torch.where(
            is_peer,
            receiver_active,
            torch.ones_like(receiver_active),
        )

        frequency = float(CONFIG["uavnetsim_carrier_frequency_hz"])
        wavelength = 299_792_458.0 / frequency
        reference_gain = (wavelength / (4.0 * math.pi)) ** 2
        los_model = str(CONFIG["uavnetsim_los_model"]).lower()
        nlos_model = str(CONFIG["uavnetsim_nlos_model"]).lower()
        los_params = {
            "free_space": (2.0, 0.0),
            "log_distance": (2.1, 1.0),
        }
        nlos_params = {
            "urban": (3.0, 12.0),
            "suburban": (2.6, 8.0),
        }
        if los_model not in los_params:
            raise ValueError(f"unsupported UavNetSim LoS model: {los_model}")
        if nlos_model not in nlos_params:
            raise ValueError(f"unsupported UavNetSim NLoS model: {nlos_model}")
        los_exp, los_excess = los_params[los_model]
        nlos_exp, nlos_excess = nlos_params[nlos_model]

        # CSMA/CA normally serializes contenders. Applying every attempted
        # sender as simultaneous interference would badly under-estimate its
        # throughput; the expected backoff/contention model below accounts for
        # sharing. Explicit simultaneous interference is reserved for ALOHA-like
        # access where overlapping transmissions are part of the MAC semantics.
        mac_name = str(CONFIG["uavnetsim_mac_protocol"]).strip().upper()
        interference_enabled = (
            bool(CONFIG.get("full_gpu_network_interference", True))
            and mac_name in {"PURE_ALOHA", "ALOHA"}
        )
        if interference_enabled:
            tx_start = self.positions[:, :, None, :].expand(
                -1, -1, U, -1
            )
            rx_end = dest_pos[:, None, :, :].expand(
                -1, U, -1, -1
            )
            pair_distance = torch.linalg.vector_norm(
                rx_end - tx_start,
                dim=-1,
            ).clamp_min(min_distance)
            pair_nlos = self._segment_cylinder_blocked(
                tx_start,
                rx_end,
            )
            exponent = torch.where(
                pair_nlos,
                torch.full_like(pair_distance, nlos_exp),
                torch.full_like(pair_distance, los_exp),
            )
            excess_db = torch.where(
                pair_nlos,
                torch.full_like(pair_distance, nlos_excess),
                torch.full_like(pair_distance, los_excess),
            )
            gain = (
                reference_gain
                * torch.pow(pair_distance, -exponent)
                * torch.pow(
                    torch.full_like(pair_distance, 10.0),
                    -excess_db / 10.0,
                )
            )
            # UavNetSim treats a transmitter that is itself the selected
            # receiver as unit-gain self-interference (half-duplex conflict).
            tx_ids = torch.arange(U, device=self.device).view(1, U, 1)
            selected_receivers = receiver[:, None, :]
            self_interference = is_peer[:, None, :] & (
                tx_ids == selected_receivers
            )
            gain = torch.where(
                self_interference,
                torch.ones_like(gain),
                gain,
            )
            rx_matrix = tx_power[:, :, None] * gain
            desired = torch.diagonal(
                rx_matrix, dim1=1, dim2=2
            )
            active_tx = attempted_mask[:, :, None].to(rx_matrix.dtype)
            total_received = (rx_matrix * active_tx).sum(dim=1)
            interference = (
                total_received
                - desired * attempted_mask.to(desired.dtype)
            ).clamp_min(0.0)
            selected_nlos = torch.diagonal(
                pair_nlos, dim1=1, dim2=2
            )
        else:
            selected_nlos = self._segment_cylinder_blocked(
                self.positions,
                dest_pos,
            )
            exponent = torch.where(
                selected_nlos,
                torch.full_like(distance, nlos_exp),
                torch.full_like(distance, los_exp),
            )
            excess_db = torch.where(
                selected_nlos,
                torch.full_like(distance, nlos_excess),
                torch.full_like(distance, los_excess),
            )
            gain = (
                reference_gain
                * torch.pow(distance, -exponent)
                * torch.pow(
                    torch.full_like(distance, 10.0),
                    -excess_db / 10.0,
                )
            )
            desired = tx_power * gain
            interference = torch.zeros_like(desired)

        thermal_dbm = (
            float(CONFIG["uavnetsim_thermal_noise_density_dbm_hz"])
            + 10.0 * math.log10(float(CONFIG["uavnetsim_bandwidth_hz"]))
            + float(CONFIG["uavnetsim_receiver_noise_figure_db"])
        )
        noise_w = 10.0 ** ((thermal_dbm - 30.0) / 10.0)
        sinr = desired / (noise_w + interference).clamp_min(1e-30)
        sinr_db = 10.0 * torch.log10(sinr.clamp_min(1e-20))
        link_ok = (
            attempted_mask
            & contact
            & (
                sinr_db
                >= float(CONFIG["uavnetsim_sinr_threshold_db"])
            )
        )
        rate = torch.where(
            link_ok,
            torch.full_like(
                sinr_db,
                float(CONFIG["uavnetsim_bit_rate_bps"]),
            ),
            torch.zeros_like(sinr_db),
        )
        return {
            "receiver": receiver,
            "dest_pos": dest_pos,
            "distance": distance,
            "contact": contact,
            "nlos": selected_nlos,
            "signal_w": desired,
            "interference_w": interference,
            "sinr_db": sinr_db,
            "link_ok": link_ok,
            "rate_bps": rate,
        }


    def _network(
        self,
        destination_idx,
        power_action,
        energy_budget_j=None,
    ):
        E, U, T = (
            self.num_envs,
            self.num_uavs,
            self.num_targets,
        )
        report_size = float(CONFIG["report_bytes"])
        selected_dest = torch.gather(
            self.choices.unsqueeze(0).expand(
                E, -1, -1
            ),
            2,
            destination_idx.unsqueeze(-1),
        ).squeeze(-1)
        tx_power = float(CONFIG["tx_power_min_w"]) + (
            power_action.squeeze(-1)
            .clamp(-1.0, 1.0)
            + 1.0
        ) * 0.5 * (
            float(CONFIG["tx_power_max_w"])
            - float(CONFIG["tx_power_min_w"])
        )

        if energy_budget_j is None:
            energy_budget = torch.full(
                (E, U),
                float("inf"),
                device=self.device,
                dtype=self.positions.dtype,
            )
        else:
            energy_budget = torch.as_tensor(
                energy_budget_j,
                device=self.device,
                dtype=self.positions.dtype,
            )
            if energy_budget.shape != (E, U):
                raise ValueError(
                    "energy_budget_j has wrong shape"
                )
            if (
                not torch.isfinite(
                    energy_budget
                ).all()
                or (energy_budget < 0.0).any()
            ):
                raise ValueError(
                    "energy_budget_j must be finite and >= 0"
                )

        affordable_duration = torch.minimum(
            torch.full_like(
                tx_power,
                self.dt,
            ),
            energy_budget
            / tx_power.clamp_min(1e-12),
        ).clamp_min(0.0)
        energy_byte_budget = torch.floor(
            affordable_duration
            * float(
                CONFIG[
                    "comm_max_link_rate_bps"
                ]
            )
            / 8.0
        )

        attempted_all = torch.zeros(
            E, U, device=self.device, dtype=torch.bool
        )
        actual_all = torch.zeros(
            E, U, device=self.device
        )
        comm_energy_all = torch.zeros(
            E, U, device=self.device
        )
        attempted_bytes_all = torch.zeros(
            E, U, device=self.device
        )
        delivery_latency_sum_s = torch.zeros(
            E, device=self.device
        )
        delivery_latency_count = torch.zeros(
            E, device=self.device, dtype=torch.long
        )
        delivery_latency_max_s = torch.zeros(
            E, device=self.device
        )
        new_delivery_total = torch.zeros(
            E, device=self.device, dtype=torch.long
        )
        sentinel = (
            self.max_steps
            + int(CONFIG["report_ttl"])
            + 1
        )
        target_ids = self.target_ids

        # Freeze the head report for every sender before any same-slot mutation.
        all_is_gcs = selected_dest == -1
        all_is_peer = selected_dest >= 0
        all_receivers = selected_dest.clamp(0, max(0, U - 1))
        receipt_index = all_receivers[:, :, None, None].expand(-1, -1, 1, T)
        initial_peer_received = torch.gather(
            self.peer_received, 2, receipt_index
        ).squeeze(2)
        initial_peer_source = torch.gather(
            self.peer_source, 2, receipt_index
        ).squeeze(2)
        initial_peer_created = torch.gather(
            self.peer_created, 2, receipt_index
        ).squeeze(2)
        initial_peer_complete = (
            (initial_peer_source == self.report_source)
            & (initial_peer_created == self.report_created)
            & (initial_peer_received >= report_size)
        )
        candidates = torch.where(
            all_is_gcs[:, :, None],
            self.report_valid & (self.report_remaining > 0.0),
            torch.where(
                all_is_peer[:, :, None],
                self.report_valid & ~initial_peer_complete,
                torch.zeros_like(self.report_valid),
            ),
        )
        intent_key = torch.where(
            candidates,
            self.report_created * (T + 1) + target_ids,
            torch.full_like(self.report_created, sentinel * (T + 1) + T),
        )
        intent_targets = intent_key.argmin(-1)
        intent_valid = (
            candidates.any(-1)
            & self.active
            & (energy_byte_budget > 0.0)
        )
        self._commit_completed_peer_receipts()

        network_model = self.network_model

        if network_model in {"packet_approx", "uavnetsim_gpu"}:
            pre_attempt = intent_valid
            channel_contenders = (
                pre_attempt.sum(-1)
                .float()
            )
            packet_payload_bytes = float(
                CONFIG[
                    "uavnetsim_payload_bytes"
                ]
            )
            queue_budget_bytes = (
                packet_payload_bytes
                * float(
                    CONFIG[
                        "uavnetsim_max_queue_size"
                    ]
                )
            )
            bit_rate = float(
                CONFIG[
                    "comm_max_link_rate_bps"
                ]
            )
            data_airtime_s = (
                packet_payload_bytes
                * 8.0
                / bit_rate
            )
            ack_airtime_s = (
                float(
                    CONFIG[
                        "full_gpu_packet_ack_bits"
                    ]
                )
                / bit_rate
            )
            avg_backoff_s = (
                (
                    float(
                        CONFIG[
                            "full_gpu_packet_cw_min"
                        ]
                    )
                    - 1.0
                )
                * 0.5
                * float(
                    CONFIG[
                        "full_gpu_packet_slot_us"
                    ]
                )
                * 1e-6
            )
            fixed_mac_overhead_s = (
                (
                    float(
                        CONFIG[
                            "full_gpu_packet_difs_us"
                        ]
                    )
                    + float(
                        CONFIG[
                            "full_gpu_packet_sifs_us"
                        ]
                    )
                )
                * 1e-6
                + ack_airtime_s
                + avg_backoff_s
            )
            packet_service_time_s = (
                data_airtime_s
                + fixed_mac_overhead_s
            )
            mac_service_budget_bytes = (
                math.floor(
                    self.dt
                    / max(
                        packet_service_time_s,
                        1e-12,
                    )
                )
                * packet_payload_bytes
            )
        else:
            channel_contenders = intent_valid.sum(-1).float()
            packet_payload_bytes = float(
                CONFIG[
                    "uavnetsim_payload_bytes"
                ]
            )
            queue_budget_bytes = float("inf")
            mac_service_budget_bytes = float(
                "inf"
            )

        uavnetsim_radio = None
        if network_model == "uavnetsim_gpu":
            # Pre-compute all selected-link radio quantities in one batched
            # CUDA pass. Report mutation below remains ordered per sender so
            # custody/delivery semantics stay identical and deterministic.
            uavnetsim_radio = self._uavnetsim_gpu_radio(
                selected_dest,
                tx_power,
                pre_attempt,
            )

        for sender in range(U):
            dest = selected_dest[:, sender]
            is_gcs = dest == -1
            is_peer = dest >= 0
            receiver = dest.clamp(
                0,
                max(0, U - 1),
            )

            report_valid = self.report_valid[
                :, sender, :
            ]
            peer_received = torch.gather(
                self.peer_received[:, sender, :, :],
                1,
                receiver[:, None, None].expand(
                    -1, 1, T
                ),
            ).squeeze(1)
            peer_source = torch.gather(
                self.peer_source[:, sender, :, :],
                1,
                receiver[:, None, None].expand(
                    -1, 1, T
                ),
            ).squeeze(1)
            peer_created = torch.gather(
                self.peer_created[:, sender, :, :],
                1,
                receiver[:, None, None].expand(
                    -1, 1, T
                ),
            ).squeeze(1)
            target = intent_targets[:, sender]
            selected_valid = (
                intent_valid[:, sender]
                & torch.gather(
                    report_valid, 1, target[:, None]
                ).squeeze(1)
            )
            attempted = (
                selected_valid
                & self.active[:, sender]
                & (is_gcs | is_peer)
            )
            attempted_all[:, sender] = attempted

            if uavnetsim_radio is not None:
                receiver = uavnetsim_radio["receiver"][:, sender]
                dest_pos = uavnetsim_radio["dest_pos"][:, sender, :]
                distance = uavnetsim_radio["distance"][:, sender]
                contact = uavnetsim_radio["contact"][:, sender]
                nlos = uavnetsim_radio["nlos"][:, sender]
                snr_db = uavnetsim_radio["sinr_db"][:, sender]
                rate = uavnetsim_radio["rate_bps"][:, sender]
                link_ok = (
                    uavnetsim_radio["link_ok"][:, sender]
                    & attempted
                )
                capacity_bytes = torch.where(
                    link_ok,
                    torch.floor(rate * self.dt / 8.0),
                    torch.zeros_like(rate),
                )
            else:
                peer_pos = torch.gather(
                    self.positions,
                    1,
                    receiver[:, None, None].expand(
                        -1, 1, 3
                    ),
                ).squeeze(1)
                dest_pos = torch.where(
                    is_gcs[:, None],
                    self.gcs.view(1, 3),
                    peer_pos,
                )
                distance = torch.linalg.vector_norm(
                    dest_pos - self.positions[:, sender, :],
                    dim=-1,
                ).clamp_min(
                    float(CONFIG["comm_reference_distance_m"])
                )
                contact = torch.where(
                    is_gcs,
                    distance <= float(CONFIG["gcs_contact_range_m"]),
                    torch.where(
                        is_peer,
                        distance <= float(CONFIG["peer_contact_range_m"]),
                        torch.zeros_like(is_peer),
                    ),
                )
                receiver_active = torch.gather(
                    self.active,
                    1,
                    receiver[:, None],
                ).squeeze(1)
                contact &= torch.where(
                    is_peer,
                    receiver_active,
                    torch.ones_like(receiver_active),
                )
                nlos = self._segment_cylinder_blocked(
                    self.positions[:, sender, :],
                    dest_pos,
                )
                gain_db = (
                    float(CONFIG["comm_reference_gain_db"])
                    - 10.0
                    * float(CONFIG["comm_path_loss_exponent"])
                    * torch.log10(
                        distance
                        / float(CONFIG["comm_reference_distance_m"])
                    )
                    - nlos.float()
                    * float(CONFIG["comm_nlos_additional_loss_db"])
                )
                rx_w = tx_power[:, sender] * torch.pow(
                    torch.tensor(10.0, device=self.device),
                    gain_db / 10.0,
                )
                noise_w = 10.0 ** (
                    (float(CONFIG["comm_noise_power_dbm"]) - 30.0)
                    / 10.0
                )
                snr = rx_w / noise_w
                snr_db = 10.0 * torch.log10(
                    snr.clamp_min(1e-20)
                )
                rate = float(CONFIG["comm_bandwidth_hz"]) * torch.log2(
                    1.0 + snr
                )
                rate = rate.clamp_max(
                    float(CONFIG["comm_max_link_rate_bps"])
                )
                link_ok = (
                    attempted
                    & contact
                    & (
                        snr_db
                        >= float(CONFIG["comm_snr_threshold_db"])
                    )
                )
                capacity_bytes = torch.where(
                    link_ok,
                    torch.floor(rate * self.dt / 8.0),
                    torch.zeros_like(rate),
                )
            if network_model in {"packet_approx", "uavnetsim_gpu"}:
                capacity_bytes = torch.floor(
                    torch.minimum(
                        torch.minimum(
                            capacity_bytes,
                            torch.full_like(
                                capacity_bytes,
                                queue_budget_bytes,
                            ),
                        ),
                        torch.full_like(
                            capacity_bytes,
                            float(
                                mac_service_budget_bytes
                            ),
                        ),
                    )
                    * torch.pow(
                        torch.full_like(
                            channel_contenders,
                            float(
                                CONFIG[
                                    "full_gpu_packet_contention_decay"
                                ]
                            ),
                        ),
                        (
                            channel_contenders
                            - 1.0
                        ).clamp_min(0.0),
                    )
                )

            capacity_bytes = torch.minimum(
                capacity_bytes,
                energy_byte_budget[:, sender],
            )

            target_oh = F.one_hot(
                target,
                T,
            ).bool()
            selected_source = torch.gather(
                self.report_source[
                    :, sender, :
                ],
                1,
                target[:, None],
            ).squeeze(1)
            selected_created = torch.gather(
                self.report_created[
                    :, sender, :
                ],
                1,
                target[:, None],
            ).squeeze(1)

            # GCS transfer: delivered-byte progress is shared by report copies
            # of the same generation.
            selected_remaining = torch.gather(
                self.report_remaining[
                    :, sender, :
                ],
                1,
                target[:, None],
            ).squeeze(1)
            gcs_actual = torch.minimum(
                capacity_bytes,
                selected_remaining,
            )
            gcs_actual = torch.where(
                is_gcs & selected_valid,
                gcs_actual,
                torch.zeros_like(gcs_actual),
            )
            gcs_new_remaining = (
                selected_remaining - gcs_actual
            ).clamp_min(0.0)
            copy_match = (
                self.report_valid
                & target_oh[:, None, :]
                & (
                    self.report_source
                    == selected_source[
                        :, None, None
                    ]
                )
                & (
                    self.report_created
                    == selected_created[
                        :, None, None
                    ]
                )
            )
            gcs_update = (
                is_gcs
                & link_ok
                & selected_valid
            )
            self.report_remaining = torch.where(
                copy_match
                & gcs_update[:, None, None],
                gcs_new_remaining[
                    :, None, None
                ],
                self.report_remaining,
            )
            gcs_complete = (
                gcs_update
                & (gcs_new_remaining <= 0.0)
            )
            delivered_target = (
                target_oh
                & gcs_complete[:, None]
            )
            new_delivery = (
                delivered_target
                & ~self.delivered
            )
            new_delivery_total += (
                new_delivery.sum(-1)
            )
            new_delivery_event = new_delivery.any(-1)
            delivery_latency_s = (
                (
                    self.step_count
                    - selected_created
                )
                .clamp_min(0)
                .float()
                * self.dt
            )
            delivery_latency_sum_s += torch.where(
                new_delivery_event,
                delivery_latency_s,
                torch.zeros_like(delivery_latency_s),
            )
            delivery_latency_count += new_delivery_event.long()
            delivery_latency_max_s = torch.maximum(
                delivery_latency_max_s,
                torch.where(
                    new_delivery_event,
                    delivery_latency_s,
                    torch.zeros_like(delivery_latency_s),
                ),
            )
            self.delivered |= delivered_target

            # Keep mission cleanup entirely tensorized: no CUDA->CPU scalar
            # synchronization is needed when no delivery completed.
            clear_target = delivered_target
            self.report_valid &= ~clear_target[
                :, None, :
            ]
            self.report_remaining = torch.where(
                self.report_valid,
                self.report_remaining,
                torch.zeros_like(
                    self.report_remaining
                ),
            )
            self.report_source = torch.where(
                self.report_valid,
                self.report_source,
                torch.full_like(
                    self.report_source,
                    -1,
                ),
            )
            self.pending_valid &= ~clear_target[
                :, None, :
            ]
            self.pending_remaining = torch.where(
                self.pending_valid,
                self.pending_remaining,
                torch.zeros_like(
                    self.pending_remaining
                ),
            )
            peer_clear = clear_target[
                :, None, None, :
            ]
            self.peer_received = torch.where(
                peer_clear,
                torch.zeros_like(
                    self.peer_received
                ),
                self.peer_received,
            )
            self.peer_source = torch.where(
                peer_clear,
                torch.full_like(
                    self.peer_source,
                    -1,
                ),
                self.peer_source,
            )

            # Peer transfer: progress is per sender-recipient-target and does
            # not consume/remove the sender's report.
            current_peer_received = torch.gather(
                peer_received,
                1,
                target[:, None],
            ).squeeze(1)
            current_peer_source = torch.gather(
                peer_source,
                1,
                target[:, None],
            ).squeeze(1)
            current_peer_created = torch.gather(
                peer_created,
                1,
                target[:, None],
            ).squeeze(1)
            generation_match = (
                (
                    current_peer_source
                    == selected_source
                )
                & (
                    current_peer_created
                    == selected_created
                )
            )
            base_peer_received = torch.where(
                generation_match,
                current_peer_received,
                torch.zeros_like(
                    current_peer_received
                ),
            )
            peer_remaining = (
                report_size - base_peer_received
            ).clamp_min(0.0)

            requested_bytes = torch.where(
                is_gcs,
                selected_remaining,
                torch.where(
                    is_peer,
                    peer_remaining,
                    torch.zeros_like(
                        selected_remaining
                    ),
                ),
            )
            max_attempt_bytes = float(
                math.floor(
                    float(
                        CONFIG[
                            "comm_max_link_rate_bps"
                        ]
                    )
                    * self.dt
                    / 8.0
                )
            )
            attempted_bytes = torch.minimum(
                torch.minimum(
                    requested_bytes,
                    torch.full_like(
                        requested_bytes,
                        max_attempt_bytes,
                    ),
                ),
                energy_byte_budget[:, sender],
            )
            attempted_bytes_all[:, sender] = torch.where(
                attempted,
                attempted_bytes,
                torch.zeros_like(attempted_bytes),
            )
            tx_duration = torch.minimum(
                torch.full_like(
                    attempted_bytes,
                    self.dt,
                ),
                attempted_bytes
                * 8.0
                / float(
                    CONFIG[
                        "comm_max_link_rate_bps"
                    ]
                ),
            )
            comm_energy_all[:, sender] = torch.where(
                attempted,
                tx_power[:, sender]
                * tx_duration,
                torch.zeros_like(tx_duration),
            )

            peer_actual = torch.minimum(
                capacity_bytes,
                peer_remaining,
            )
            peer_actual = torch.where(
                is_peer
                & link_ok
                & selected_valid,
                peer_actual,
                torch.zeros_like(peer_actual),
            )
            new_peer_received = (
                base_peer_received + peer_actual
            ).clamp_max(report_size)
            receiver_oh = F.one_hot(
                receiver,
                U,
            ).bool()
            pair_mask = (
                receiver_oh[:, :, None]
                & target_oh[:, None, :]
                & (
                    is_peer
                    & selected_valid
                )[:, None, None]
            )
            peer_slice = self.peer_received[
                :, sender, :, :
            ]
            source_slice = self.peer_source[
                :, sender, :, :
            ]
            created_slice = self.peer_created[
                :, sender, :, :
            ]
            self.peer_received[
                :, sender, :, :
            ] = torch.where(
                pair_mask,
                new_peer_received[
                    :, None, None
                ],
                peer_slice,
            )
            self.peer_source[
                :, sender, :, :
            ] = torch.where(
                pair_mask,
                selected_source[
                    :, None, None
                ],
                source_slice,
            )
            self.peer_created[
                :, sender, :, :
            ] = torch.where(
                pair_mask,
                selected_created[
                    :, None, None
                ],
                created_slice,
            )
            actual_all[:, sender] = (
                gcs_actual + peer_actual
            )

        # Same-step custody commit after all radio transfers.
        self._commit_completed_peer_receipts()
        packets_tx = torch.where(
            actual_all > 0.0,
            torch.ceil(
                actual_all
                / packet_payload_bytes
            ),
            torch.zeros_like(actual_all),
        )
        return {
            "new_delivery": new_delivery_total,
            "comm_energy": comm_energy_all,
            "attempted": attempted_all,
            "bytes_attempted": attempted_bytes_all,
            "bytes_tx": actual_all,
            "packets_tx": packets_tx,
            "delivery_latency_sum_s": delivery_latency_sum_s,
            "delivery_latency_count": delivery_latency_count,
            "delivery_latency_max_s": delivery_latency_max_s,
            "channel_contenders": (
                channel_contenders
            ),
            "link_ok": (
                (uavnetsim_radio["link_ok"] & attempted_all)
                if uavnetsim_radio is not None
                else (actual_all > 0.0)
            ),
            "sinr_db": (
                uavnetsim_radio["sinr_db"]
                if uavnetsim_radio is not None
                else torch.full_like(actual_all, float("nan"))
            ),
            "nlos": (
                uavnetsim_radio["nlos"]
                if uavnetsim_radio is not None
                else torch.zeros_like(attempted_all)
            ),
            "interference_w": (
                uavnetsim_radio["interference_w"]
                if uavnetsim_radio is not None
                else torch.zeros_like(actual_all)
            ),
        }


    def _propulsion_energy(
        self,
        motion_result,
        active_mask=None,
    ):
        speed2 = (
            self.velocities * self.velocities
        ).sum(-1)
        realized = motion_result[
            "realized_accel"
        ]
        realized_norm = torch.linalg.vector_norm(
            realized,
            dim=-1,
            keepdim=True,
        )
        realized = realized * torch.minimum(
            torch.ones_like(realized_norm),
            torch.tensor(
                self.max_accel,
                device=self.device,
            )
            / realized_norm.clamp_min(1e-8),
        )
        accel2 = (realized * realized).sum(-1)
        commanded_accel = (
            torch.linalg.vector_norm(
                motion_result["commanded_motion"],
                dim=-1,
            )
            * self.max_accel
        )
        ground_idle = (
            (self.positions[..., 2] <= 0.0)
            & (speed2 <= 1e-12)
            & (commanded_accel <= 1e-12)
        )
        base = torch.where(
            ground_idle,
            torch.full_like(
                speed2,
                float(CONFIG["energy_idle_power_w"]),
            ),
            torch.full_like(
                speed2,
                float(CONFIG["energy_hover_power_w"]),
            ),
        )
        propulsion = (
            base
            + float(
                CONFIG["energy_speed_sq_coeff"]
            )
            * speed2
            + float(
                CONFIG["energy_accel_sq_coeff"]
            )
            * accel2
        ) * self.dt
        propulsion = torch.where(
            ground_idle,
            torch.full_like(
                speed2,
                float(
                    CONFIG["energy_idle_power_w"]
                )
                * self.dt,
            ),
            propulsion,
        )
        if active_mask is None:
            active_mask = self.active
        return (
            propulsion
            * active_mask.float()
        )

    def _energy(
        self,
        motion_result,
        comm_energy,
        propulsion_override=None,
        active_before=None,
    ):
        if active_before is None:
            active_before = self.active
        active_before = active_before.bool()

        if propulsion_override is None:
            propulsion = self._propulsion_energy(
                motion_result
            )
        else:
            propulsion = torch.as_tensor(
                propulsion_override,
                device=self.device,
                dtype=self.positions.dtype,
            )
            if propulsion.shape != (
                self.num_envs,
                self.num_uavs,
            ):
                raise ValueError(
                    "propulsion_override has wrong shape"
                )
            if (
                not torch.isfinite(
                    propulsion
                ).all()
                or (propulsion < 0.0).any()
            ):
                raise ValueError(
                    "propulsion_override must be finite and >= 0"
                )

        comm_energy = torch.as_tensor(
            comm_energy,
            device=self.device,
            dtype=self.positions.dtype,
        )
        if comm_energy.shape != (
            self.num_envs,
            self.num_uavs,
        ):
            raise ValueError(
                "comm_energy has wrong shape"
            )
        if (
            not torch.isfinite(
                comm_energy
            ).all()
            or (comm_energy < 0.0).any()
        ):
            raise ValueError(
                "comm_energy must be finite and >= 0"
            )

        available = self.battery.clamp_min(0.0)
        propulsion = torch.minimum(
            propulsion,
            available,
        )
        comm_energy = torch.minimum(
            comm_energy,
            (
                available
                - propulsion
            ).clamp_min(0.0),
        )
        total = (
            propulsion + comm_energy
        ) * active_before.float()
        self.battery = (
            self.battery - total
        ).clamp_min(0.0)
        self.active &= self.battery > 1e-12
        self.velocities = torch.where(
            self.active.unsqueeze(-1),
            self.velocities,
            torch.zeros_like(self.velocities),
        )
        return total


    def _expire_reports(self):
        ttl = float(CONFIG["report_ttl"])
        age = (
            self.step_count[:, None, None]
            - self.report_created
        )
        expired = (
            self.report_valid
            & (age.float() * self.dt >= ttl)
        )
        pending_age = (
            self.step_count[:, None, None]
            - self.pending_created
        )
        pending_expired = (
            self.pending_valid
            & (
                pending_age.float()
                * self.dt
                >= ttl
            )
        )
        expired_targets = (
            expired | pending_expired
        ).any(dim=1).sum(-1)

        self.report_valid &= ~expired
        self.pending_valid &= ~pending_expired
        self.report_remaining = torch.where(
            self.report_valid,
            self.report_remaining,
            torch.zeros_like(
                self.report_remaining
            ),
        )
        self.report_source = torch.where(
            self.report_valid,
            self.report_source,
            torch.full_like(
                self.report_source,
                -1,
            ),
        )
        self.pending_remaining = torch.where(
            self.pending_valid,
            self.pending_remaining,
            torch.zeros_like(
                self.pending_remaining
            ),
        )

        peer_age = (
            self.step_count[:, None, None, None]
            - self.peer_created
        )
        peer_expired = (
            (self.peer_received > 0.0)
            & (
                peer_age.float()
                * self.dt
                >= ttl
            )
        )
        self.peer_received = torch.where(
            peer_expired,
            torch.zeros_like(self.peer_received),
            self.peer_received,
        )
        self.peer_source = torch.where(
            peer_expired,
            torch.full_like(
                self.peer_source,
                -1,
            ),
            self.peer_source,
        )
        return expired_targets

    def _observation(self, entropy_map=None):
        E, U, O = self.num_envs, self.num_uavs, self.num_obstacles
        p = self.positions
        v = self.velocities
        time_frac = (self.step_count.float() / max(1, self.max_steps)).clamp(0.0, 1.0)
        self_state = torch.cat(
            (
                (p[..., :2] / self.map_size).clamp(0.0, 1.0),
                ((p[..., 2:3] - self.altitude_min) / max(1e-6, self.altitude_max - self.altitude_min)).clamp(0.0, 1.0),
                (v / self.max_speed).clamp(-1.0, 1.0),
                (self.battery / float(CONFIG["battery_j"])).unsqueeze(-1).clamp(0.0, 1.0),
                self.active.float().unsqueeze(-1),
                time_frac[:, None, None].expand(E, U, 1),
            ),
            dim=-1,
        )

        gcs_rel = (
            (self.gcs.view(1, 1, 3) - p)
            / self.xyz_scale.view(1, 1, 3)
        ).clamp(-1.0, 1.0)

        # Vectorized peer features while preserving CPU peer-ID order.
        all_relative = (
            p[:, None, :, :]
            - p[:, :, None, :]
        )
        gather_index = self.peer_index.view(
            1, U, U - 1, 1
        ).expand(E, -1, -1, 3)
        relative = torch.gather(
            all_relative,
            2,
            gather_index,
        )
        distance = torch.linalg.vector_norm(
            relative,
            dim=-1,
        )
        peer_active = torch.gather(
            self.active[:, None, :].expand(
                E, U, U
            ),
            2,
            self.peer_index.view(
                1, U, U - 1
            ).expand(E, -1, -1),
        )
        visible = (
            self.active[:, :, None]
            & peer_active
            & (
                distance
                <= float(
                    CONFIG["peer_contact_range_m"]
                )
            )
        )
        neighbors = torch.cat(
            (
                (
                    relative
                    / self.xyz_scale.view(
                        1, 1, 1, 3
                    )
                ).clamp(-1.0, 1.0),
                (
                    distance
                    / float(
                        CONFIG[
                            "peer_contact_range_m"
                        ]
                    )
                )
                .clamp(0.0, 1.0)
                .unsqueeze(-1),
                visible.float().unsqueeze(-1),
            ),
            dim=-1,
        )
        neighbors = torch.where(
            visible.unsqueeze(-1),
            neighbors,
            torch.zeros_like(neighbors),
        )

        # Nearest 3 obstacles.
        odelta = self.obstacle_xy[:, None, :, :] - p[:, :, None, :2]
        odist = torch.linalg.vector_norm(odelta, dim=-1)
        ok = min(int(CONFIG["observation_nearest_obstacles"]), O)
        _, oi = torch.topk(odist, k=ok, dim=-1, largest=False)
        oxy = torch.gather(
            self.obstacle_xy[:, None, :, :].expand(E, U, O, 2),
            2, oi.unsqueeze(-1).expand(-1, -1, -1, 2)
        )
        orad = torch.gather(self.obstacle_radius[:, None, :].expand(E, U, O), 2, oi)
        oh = torch.gather(self.obstacle_height[:, None, :].expand(E, U, O), 2, oi)
        orel = oxy - p[:, :, None, :2]
        center_distance = torch.linalg.vector_norm(orel, dim=-1)
        obstacles = torch.cat(
            (
                (orel / self.map_size).clamp(-1.0, 1.0),
                (center_distance / (math.sqrt(2.0) * self.map_size))
                .clamp(0.0, 1.0)
                .unsqueeze(-1),
                (orad / self.map_size).clamp(0.0, 1.0).unsqueeze(-1),
                (oh / max(self.altitude_max, 1.0)).clamp(0.0, 1.0).unsqueeze(-1),
            ),
            dim=-1,
        )

        missing_obstacles = int(CONFIG["observation_nearest_obstacles"]) - ok
        if missing_obstacles > 0:
            obstacles = F.pad(obstacles, (0, 0, 0, missing_obstacles))

        # 13x13 local belief patch.
        center = torch.floor(
            p[..., :2] / self.cell_size
        ).long()
        pc = (
            center.unsqueeze(2)
            + self.belief_patch_offsets.view(
                1, 1, -1, 2
            )
        )
        pxg, pyg = pc[..., 0], pc[..., 1]
        pvalid = (pxg >= 0) & (pxg < self.grid_n) & (pyg >= 0) & (pyg < self.grid_n)
        pidx = pyg.clamp(0, self.grid_n - 1) * self.grid_n + pxg.clamp(0, self.grid_n - 1)
        patch = torch.gather(self.belief.view(E, U, -1), 2, pidx)
        patch = torch.where(pvalid, patch, torch.full_like(patch, float(CONFIG["belief_prior"])))

        coarse_n = int(CONFIG["belief_coarse_cells"])
        if entropy_map is None:
            entropy_map = self._belief_entropy()
        coarse = F.adaptive_avg_pool2d(
            entropy_map.view(E * U, 1, self.grid_n, self.grid_n),
            (coarse_n, coarse_n),
        ).view(E, U, -1).clamp(0.0, 1.0)

        valid_count = self.report_valid.sum(-1).float()
        oldest = torch.where(
            self.report_valid,
            self.report_created,
            torch.full_like(self.report_created, self.max_steps + int(CONFIG["report_ttl"]) + 1),
        ).amin(-1)
        age = torch.where(
            valid_count > 0,
            (self.step_count[:, None] - oldest).float() * self.dt / max(float(CONFIG["report_ttl"]), 1.0),
            torch.zeros_like(valid_count),
        ).clamp(0.0, 1.0)
        capacity_reports = max(
            1.0, float(CONFIG["buffer_bytes"]) / float(CONFIG["report_bytes"])
        )
        used_fraction = (valid_count / capacity_reports).clamp(0.0, 1.0)
        report_fraction = used_fraction
        pending_fraction = (
            self.pending_valid.sum(-1).float()
            / max(
                1.0,
                float(CONFIG["pending_buffer_bytes"]) / float(CONFIG["report_bytes"]),
            )
        ).clamp(0.0, 1.0)
        ordered_created = torch.where(
            self.report_valid,
            self.report_created,
            torch.full_like(
                self.report_created,
                self.max_steps + int(CONFIG["report_ttl"]) + 1,
            ),
        )
        head_target = ordered_created.argmin(-1)
        head_remaining = torch.gather(
            self.report_remaining, 2, head_target.unsqueeze(-1)
        ).squeeze(-1)
        head_delivery = torch.where(
            valid_count > 0,
            (1.0 - head_remaining / float(CONFIG["report_bytes"])).clamp(0.0, 1.0),
            torch.zeros_like(valid_count),
        )
        buffer_features = torch.stack(
            (used_fraction, report_fraction, age, pending_fraction, head_delivery),
            dim=-1,
        )

        dest_mask = self._destination_mask()
        obs = torch.cat(
            (
                self_state,
                gcs_rel,
                neighbors.reshape(E, U, -1),
                obstacles.reshape(E, U, -1),
                patch,
                coarse,
                buffer_features,
                dest_mask,
            ),
            dim=-1,
        ).float()
        if obs.shape[-1] != self.observation_dim:
            raise RuntimeError(f"GPU observation dim {obs.shape[-1]} != {self.observation_dim}")
        return obs, dest_mask

    def _report_slots(self, valid, remaining, created, slot_count=None):
        T = valid.shape[-1]
        if slot_count is None:
            slot_count = int(float(CONFIG["buffer_bytes"]) // float(CONFIG["report_bytes"]))
        target_ids = torch.arange(T, device=self.device).view(1, 1, T)
        sentinel = self.max_steps + int(CONFIG["report_ttl"]) + 1
        key = torch.where(
            valid,
            created * (T + 1) + target_ids,
            torch.full_like(created, sentinel * (T + 1) + T),
        )
        order = torch.argsort(key, dim=-1)[..., :slot_count]
        chosen_valid = torch.gather(valid, 2, order)
        chosen_remaining = torch.gather(remaining, 2, order)
        chosen_created = torch.gather(created, 2, order)
        age = (
            (self.step_count[:, None, None] - chosen_created).float()
            * self.dt
            / max(float(CONFIG["report_ttl"]), 1.0)
        ).clamp(0.0, 1.0)
        delivery = (1.0 - chosen_remaining / float(CONFIG["report_bytes"])).clamp(
            0.0, 1.0
        )
        target_fraction = order.float() / max(1.0, float(T - 1))
        slots = torch.stack(
            (
                chosen_valid.float(),
                target_fraction,
                age,
                delivery,
            ),
            dim=-1,
        )
        slots = torch.where(chosen_valid.unsqueeze(-1), slots, torch.zeros_like(slots))
        if slots.shape[-2] < slot_count:
            slots = F.pad(slots, (0, 0, 0, slot_count - slots.shape[-2]))
        return slots

    def _state(self, obs, entropy_map=None):
        E, U = self.num_envs, self.num_uavs
        buffer_slots = self._report_slots(
            self.report_valid, self.report_remaining, self.report_created
        )
        pending_slots = self._report_slots(
            self.pending_valid, self.pending_remaining, self.pending_created,
            slot_count=int(float(CONFIG["pending_buffer_bytes"]) // float(CONFIG["report_bytes"])),
        )
        if entropy_map is None:
            entropy_map = self._belief_entropy()
        entropy = entropy_map
        summary = torch.stack(
            (
                entropy.mean(dim=(-2, -1)).clamp(0.0, 1.0),
                ((self.belief - float(CONFIG["belief_prior"])).abs() > 0.05)
                .float()
                .mean(dim=(-2, -1)),
            ),
            dim=-1,
        )
        critic_cells = int(CONFIG["critic_belief_grid_cells"])
        pooled_belief = F.adaptive_avg_pool2d(
            self.belief.reshape(E * U, 1, self.grid_n, self.grid_n),
            (critic_cells, critic_cells),
        ).reshape(E, U, -1)

        per_uav = torch.cat(
            (
                obs,
                buffer_slots.reshape(E, U, -1),
                pending_slots.reshape(E, U, -1),
                summary,
                pooled_belief,
            ),
            dim=-1,
        ).reshape(E, -1)

        targets = torch.cat(
            (
                (self.target_xy / self.map_size).clamp(0.0, 1.0),
                self.confirmed.float().unsqueeze(-1),
                self.delivered.float().unsqueeze(-1),
            ),
            dim=-1,
        ).reshape(E, -1)
        obstacles = torch.cat(
            (
                (self.obstacle_xy / self.map_size).clamp(0.0, 1.0),
                (self.obstacle_radius / self.map_size).clamp(0.0, 1.0).unsqueeze(-1),
                (self.obstacle_height / max(self.altitude_max, 1.0))
                .clamp(0.0, 1.0)
                .unsqueeze(-1),
            ),
            dim=-1,
        ).reshape(E, -1)
        time_fraction = (
            self.step_count.float() / max(1, self.max_steps)
        ).clamp(0.0, 1.0).unsqueeze(-1)
        state = torch.cat((per_uav, targets, obstacles, time_fraction), dim=-1)
        if state.shape[-1] != self.state_dim:
            raise RuntimeError(
                f"GPU global state dim {state.shape[-1]} != {self.state_dim}"
            )
        return state.float()

    @torch.no_grad()
    def step(self, continuous_action, destination_idx):
        if continuous_action.shape != (
            self.num_envs,
            self.num_uavs,
            self.continuous_dim,
        ):
            raise ValueError("continuous_action has wrong shape")
        if destination_idx.shape != (self.num_envs, self.num_uavs):
            raise ValueError("destination_idx has wrong shape")

        old_coverage_potential = self._coverage_potential()
        old_communication_progress = (
            self._communication_progress_potential()
        )
        self.step_count += 1
        expired = self._expire_reports()
        self._flush_pending()
        old_potential = self.belief_potential
        old_confirmed = self.confirmed.clone()
        old_delivered = self.delivered.clone()

        positions_before_motion = self.positions.clone()
        velocities_before_motion = self.velocities.clone()
        battery_before_step = self.battery.clone()
        active_before_step = self.active.clone()

        motion = self._motion(
            continuous_action[..., :3]
        )
        propulsion_required_full = (
            self._propulsion_energy(
                motion,
                active_mask=(
                    active_before_step
                ),
            )
        )
        battery_fraction = torch.where(
            propulsion_required_full > 1e-12,
            (
                battery_before_step
                / propulsion_required_full.clamp_min(
                    1e-12
                )
            ).clamp(
                0.0,
                1.0,
            ),
            torch.ones_like(
                propulsion_required_full
            ),
        )
        battery_limited = (
            active_before_step
            & (
                propulsion_required_full
                > battery_before_step + 1e-6
            )
        )
        self.positions = torch.where(
            battery_limited.unsqueeze(-1),
            positions_before_motion
            + battery_fraction.unsqueeze(-1)
            * (
                self.positions
                - positions_before_motion
            ),
            self.positions,
        )
        self.velocities = torch.where(
            battery_limited.unsqueeze(-1),
            self.velocities
            * battery_fraction.unsqueeze(-1),
            self.velocities,
        )

        # APF remains the only collision-avoidance mechanism after battery
        # scaling. Do not reintroduce the old hard-stop peer resolver.
        motion["realized_accel"] = (
            self.velocities
            - velocities_before_motion
        ) / self.dt

        propulsion_required_final = (
            self._propulsion_energy(
                motion,
                active_mask=(
                    active_before_step
                ),
            )
        )
        newly_energy_limited = (
            active_before_step
            & ~battery_limited
            & (
                propulsion_required_final
                > battery_before_step
                + 1e-6
            )
        )
        battery_limited |= (
            newly_energy_limited
        )
        battery_fraction = torch.where(
            newly_energy_limited,
            torch.zeros_like(
                battery_fraction
            ),
            battery_fraction,
        )
        self.positions = torch.where(
            newly_energy_limited.unsqueeze(-1),
            positions_before_motion,
            self.positions,
        )
        self.velocities = torch.where(
            newly_energy_limited.unsqueeze(-1),
            torch.zeros_like(
                self.velocities
            ),
            self.velocities,
        )
        motion[
            "realized_accel"
        ] = (
            self.velocities
            - velocities_before_motion
        ) / self.dt
        motion[
            "battery_limited"
        ] = battery_limited
        motion[
            "battery_fraction"
        ] = battery_fraction

        propulsion_charged = torch.where(
            battery_limited,
            battery_before_step,
            torch.minimum(
                propulsion_required_final,
                battery_before_step,
            ),
        )
        self.active = (
            active_before_step
            & ~battery_limited
        )
        communication_budget = (
            battery_before_step
            - propulsion_charged
        ).clamp_min(0.0)
        communication_budget = torch.where(
            self.active,
            communication_budget,
            torch.zeros_like(
                communication_budget
            ),
        )

        sensing = self._sense()
        if self.external_network is None:
            network = self._network(
                destination_idx.long(),
                continuous_action[..., 3:4],
                energy_budget_j=(
                    communication_budget
                ),
            )
        else:
            network = self.external_network.step(
                destination_idx.long(),
                continuous_action[..., 3:4],
                energy_budget_j=(
                    communication_budget
                ),
            )
        network["comm_energy"] = torch.minimum(
            network["comm_energy"],
            communication_budget,
        )
        self._flush_pending()
        total_energy = self._energy(
            motion,
            network["comm_energy"],
            propulsion_override=(
                propulsion_charged
            ),
            active_before=(
                active_before_step
            ),
        )
        entropy_map = self._belief_entropy()
        new_potential = self._belief_potential(
            entropy_map
        )
        self.belief_potential = new_potential

        newly_confirmed = (self.confirmed & ~old_confirmed).sum(-1)
        newly_delivered = (self.delivered & ~old_delivered).sum(-1)
        success = self.delivered.all(-1)
        inactive = ~self.active.any(-1)
        horizon = self.step_count >= self.max_steps
        done = success | inactive | horizon

        # Exclusive episode-end reason:
        # 1 = mission success, 2 = all UAVs inactive,
        # 3 = max-step horizon.
        end_success = success
        end_inactive = (
            ~end_success
            & inactive
        )
        end_horizon = (
            ~end_success
            & ~end_inactive
            & horizon
        )
        end_reason_code = torch.where(
            end_success,
            torch.ones_like(
                self.step_count
            ),
            torch.where(
                end_inactive,
                torch.full_like(
                    self.step_count,
                    2,
                ),
                torch.where(
                    end_horizon,
                    torch.full_like(
                        self.step_count,
                        3,
                    ),
                    torch.zeros_like(
                        self.step_count
                    ),
                ),
            ),
        )
        shaping_gamma = float(CONFIG["reward_shaping_gamma"])
        next_phi = torch.where(
            done,
            torch.zeros_like(new_potential),
            new_potential,
        )
        coverage_potential = self._coverage_potential()
        communication_progress = (
            self._communication_progress_potential()
        )
        next_coverage_potential = torch.where(
            done,
            torch.zeros_like(coverage_potential),
            coverage_potential,
        )
        next_communication_progress = torch.where(
            done,
            torch.zeros_like(communication_progress),
            communication_progress,
        )
        shaping = float(CONFIG["reward_info_gain"]) * (
            shaping_gamma * next_phi - old_potential
        )
        reward_coverage_shaping = float(
            CONFIG.get(
                "reward_coverage_shaping",
                0.0,
            )
        ) * (
            shaping_gamma
            * next_coverage_potential
            - old_coverage_potential
        )
        reward_communication_progress = float(
            CONFIG.get(
                "reward_communication_progress_shaping",
                0.0,
            )
        ) * (
            shaping_gamma
            * next_communication_progress
            - old_communication_progress
        )
        reward_information_gain = shaping
        reward_confirmation = (
            float(CONFIG["reward_confirmation"])
            * newly_confirmed.float()
        )
        reward_delivery = (
            float(CONFIG["reward_delivery"])
            * newly_delivered.float()
        )
        reward_false_confirmation = (
            -float(
                CONFIG[
                    "reward_false_confirmation"
                ]
            )
            * sensing[
                "false_confirmation"
            ].float()
        )
        safety_denominator = (
            float(max(1, self.num_uavs))
            if bool(
                CONFIG.get(
                    "reward_normalize_safety_by_uavs",
                    False,
                )
            )
            else 1.0
        )
        reward_blocked_motion = (
            -float(
                CONFIG["reward_blocked"]
            )
            * motion[
                "blocked"
            ].sum(-1).float()
            / safety_denominator
        )
        reward_boundary = (
            -float(
                CONFIG["reward_boundary"]
            )
            * motion[
                "boundary"
            ].sum(-1).float()
            / safety_denominator
        )
        reward_expired_report = (
            -float(
                CONFIG[
                    "reward_expired_report"
                ]
            )
            * expired.float()
        )
        reward_dropped_report = (
            -float(
                CONFIG[
                    "reward_dropped_report"
                ]
            )
            * sensing[
                "dropped"
            ].float()
        )
        reward_energy = (
            -float(
                CONFIG[
                    "reward_energy_per_kj"
                ]
            )
            * total_energy.sum(-1)
            / 1000.0
        )
        reward_step = torch.full_like(
            shaping,
            -float(
                CONFIG[
                    "reward_step_penalty"
                ]
            ),
        )
        reward_success_bonus = torch.where(
            success,
            torch.full_like(
                shaping,
                float(
                    CONFIG[
                        "reward_all_delivered_bonus"
                    ]
                ),
            ),
            torch.zeros_like(
                shaping
            ),
        )
        reward_search = (
            reward_information_gain
            + reward_coverage_shaping
            + reward_confirmation
            + reward_false_confirmation
        )
        reward_communication = (
            reward_communication_progress
            + reward_delivery
            + reward_expired_report
            + reward_dropped_report
        )
        reward_safety = (
            reward_blocked_motion
            + reward_boundary
        )
        reward_mission = (
            reward_step
            + reward_success_bonus
        )
        reward = (
            reward_information_gain
            + reward_coverage_shaping
            + reward_communication_progress
            + reward_confirmation
            + reward_delivery
            + reward_false_confirmation
            + reward_blocked_motion
            + reward_boundary
            + reward_expired_report
            + reward_dropped_report
            + reward_energy
            + reward_step
            + reward_success_bonus
        )

        obs, mask = self._observation(
            entropy_map=entropy_map
        )
        state = self._state(
            obs,
            entropy_map=entropy_map,
        )
        metrics = {
            "success": success,
            "end_success": end_success,
            "end_all_uavs_inactive": (
                end_inactive
            ),
            "end_horizon": end_horizon,
            "end_reason_code": (
                end_reason_code
            ),
            "episode_step": (
                self.step_count.clone()
            ),
            "new_confirmed": newly_confirmed,
            "new_delivered": newly_delivered,
            "blocked": motion["blocked"].sum(-1),
            "blocked_by_peer": motion[
                "blocked_by_peer"
            ].sum(-1),
            "blocked_by_obstacle": motion[
                "blocked_by_obstacle"
            ].sum(-1),
            "apf_active": motion[
                "apf_active"
            ].sum(-1),
            "apf_peer_active": motion[
                "apf_peer_active"
            ].sum(-1),
            "apf_obstacle_active": motion[
                "apf_obstacle_active"
            ].sum(-1),
            "apf_emergency": motion[
                "apf_emergency"
            ].sum(-1),
            "apf_correction_norm_sum": (
                torch.linalg.vector_norm(
                    motion["apf_acceleration_mps2"],
                    dim=-1,
                ).sum(-1)
            ),
            "boundary": motion["boundary"].sum(-1),
            "distance_m": torch.linalg.vector_norm(
                self.positions - positions_before_motion,
                dim=-1,
            ).sum(-1),
            "false_confirmations": sensing[
                "false_confirmation"
            ].float(),
            "reports_created": sensing[
                "reports_created"
            ].float(),
            "expired_reports": expired.float(),
            "dropped_reports": sensing[
                "dropped"
            ].float(),
            "comm_energy_j": network[
                "comm_energy"
            ].sum(-1),
            "network_bytes_attempted": network[
                "bytes_attempted"
            ].sum(-1),
            "network_bytes_tx": network[
                "bytes_tx"
            ].sum(-1),
            "network_tx_attempts": network[
                "attempted"
            ].sum(-1).float(),
            "network_link_successes": network[
                "link_ok"
            ].sum(-1).float(),
            "network_nlos_attempts": (
                network["nlos"]
                & network["attempted"]
            ).sum(-1).float(),
            "delivery_latency_sum_s": network[
                "delivery_latency_sum_s"
            ],
            "delivery_latency_count": network[
                "delivery_latency_count"
            ].float(),
            "delivery_latency_max_s": network[
                "delivery_latency_max_s"
            ],
            "gcs_in_range_uav_fraction": (
                (
                    (
                        torch.linalg.vector_norm(
                            self.positions
                            - self.gcs.view(1, 1, 3),
                            dim=-1,
                        )
                        <= float(
                            CONFIG[
                                "gcs_contact_range_m"
                            ]
                        )
                    )
                    & self.active
                )
                .float()
                .sum(-1)
                / self.active.float().sum(-1).clamp_min(1.0)
            ),
            "gcs_contact_graph_reachability_fraction": (
                contact_graph_reachability_fraction_torch(
                    self.positions,
                    self.active,
                    self.gcs,
                    peer_range_m=float(CONFIG["peer_contact_range_m"]),
                    gcs_range_m=float(CONFIG["gcs_contact_range_m"]),
                )
            ),
            "coverage_sensed_cell_events": sensing[
                "coverage_sensed_cell_events"
            ].float(),
            "coverage_new_unique_cells": sensing[
                "coverage_new_unique_cells"
            ].float(),
            "information_gain_bits": (
                (
                    (
                        -float(CONFIG["belief_prior"])
                        * math.log2(
                            max(
                                float(CONFIG["belief_prior"]),
                                1e-12,
                            )
                        )
                        - (
                            1.0
                            - float(CONFIG["belief_prior"])
                        )
                        * math.log2(
                            max(
                                1.0
                                - float(CONFIG["belief_prior"]),
                                1e-12,
                            )
                        )
                    )
                    - (
                        1.0
                        - new_potential
                    )
                )
                * float(
                    self.num_uavs
                    * self.grid_n
                    * self.grid_n
                )
            ),
            "energy_j": total_energy.sum(-1),
            "reward_total": reward,
            "reward_search": reward_search,
            "reward_communication": (
                reward_communication
            ),
            "reward_safety": reward_safety,
            "reward_energy": reward_energy,
            "reward_mission": reward_mission,
            "reward_information_gain": (
                reward_information_gain
            ),
            "reward_coverage_shaping": (
                reward_coverage_shaping
            ),
            "reward_communication_progress": (
                reward_communication_progress
            ),
            "reward_confirmation": (
                reward_confirmation
            ),
            "reward_delivery": reward_delivery,
            "reward_false_confirmation": (
                reward_false_confirmation
            ),
            "reward_blocked_motion": (
                reward_blocked_motion
            ),
            "reward_boundary": reward_boundary,
            "reward_expired_report": (
                reward_expired_report
            ),
            "reward_dropped_report": (
                reward_dropped_report
            ),
            "reward_step": reward_step,
            "reward_success_bonus": (
                reward_success_bonus
            ),
            "confirmed_count": (
                self.confirmed.sum(-1)
            ),
            "delivered_count": (
                self.delivered.sum(-1)
            ),
        }
        return obs, state, mask, reward.float(), done, metrics

    def assert_all_state_on_cuda(self):
        tensors = {
            k: v
            for k, v in self.__dict__.items()
            if isinstance(v, torch.Tensor)
        }
        bad = [k for k, v in tensors.items() if v.device.type != "cuda"]
        if bad:
            raise AssertionError(f"state tensors not on CUDA: {bad}")
        return len(tensors)


# --- frozen notebook cell 223 ---
class GpuReplayBuffer:
    """Tensor replay storage with portable checkpoints and unique ring writes."""

    _tensor_fields = (
        "observations", "next_observations", "states", "next_states",
        "continuous_actions", "destination_indices", "destination_masks",
        "next_destination_masks", "rewards", "terminated", "truncated",
    )
    _dimension_fields = (
        "capacity", "num_agents", "observation_dim", "state_dim",
        "continuous_dim", "discrete_dim",
    )

    def __init__(self, capacity, num_agents, observation_dim, state_dim,
                 continuous_dim, discrete_dim, device, seed=44, strict_cuda=True):
        self.capacity = int(capacity)
        self.num_agents = int(num_agents)
        self.observation_dim = int(observation_dim)
        self.state_dim = int(state_dim)
        self.continuous_dim = int(continuous_dim)
        self.discrete_dim = int(discrete_dim)
        if any(getattr(self, name) < 1 for name in self._dimension_fields):
            raise ValueError("replay dimensions and capacity must be >= 1")
        self.device = torch.device(device)
        if strict_cuda:
            _require_cuda(self.device)
        C, U, O, S, A, D = (self.capacity, self.num_agents, self.observation_dim,
                            self.state_dim, self.continuous_dim, self.discrete_dim)
        self.observations = torch.empty(C, U, O, device=self.device)
        self.next_observations = torch.empty_like(self.observations)
        self.states = torch.empty(C, S, device=self.device)
        self.next_states = torch.empty_like(self.states)
        self.continuous_actions = torch.empty(C, U, A, device=self.device)
        self.destination_indices = torch.empty(C, U, device=self.device, dtype=torch.long)
        self.destination_masks = torch.empty(C, U, D, device=self.device)
        self.next_destination_masks = torch.empty_like(self.destination_masks)
        self.rewards = torch.empty(C, 1, device=self.device)
        self.terminated = torch.empty(C, 1, device=self.device)
        self.truncated = torch.zeros(C, 1, device=self.device)
        self.position = 0
        self.size = 0
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))

    def __len__(self):
        return int(self.size)

    @torch.no_grad()
    def add_batch(self, observations, states, continuous_actions, destination_indices,
                  destination_masks, rewards, next_observations, next_states,
                  next_destination_masks, terminated, truncated=None):
        batch_size = int(observations.shape[0])
        if batch_size == 0:
            return
        if truncated is None:
            truncated = torch.zeros_like(terminated)
        values = {
            "observations": observations,
            "next_observations": next_observations,
            "states": states,
            "next_states": next_states,
            "continuous_actions": continuous_actions,
            "destination_indices": destination_indices,
            "destination_masks": destination_masks,
            "next_destination_masks": next_destination_masks,
            "rewards": rewards.reshape(batch_size, 1),
            "terminated": terminated.reshape(batch_size, 1),
            "truncated": truncated.reshape(batch_size, 1),
        }
        for name, value in values.items():
            expected = (batch_size,) + tuple(getattr(self, name).shape[1:])
            if tuple(value.shape) != expected:
                raise ValueError(f"replay {name} has wrong shape: {tuple(value.shape)} != {expected}")
        # An oversized insertion must retain the newest capacity rows. Dropping
        # its overwritten prefix makes destination indices unique on CUDA.
        skip = max(0, batch_size - self.capacity)
        retained = batch_size - skip
        indices = (torch.arange(retained, device=self.device)
                   + self.position + skip) % self.capacity
        for name, value in values.items():
            destination = getattr(self, name)
            destination[indices] = value[skip:].to(device=self.device, dtype=destination.dtype)
        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.capacity, self.size + batch_size)

    def sample(self, batch_size, device):
        device = torch.device(device)
        if device != self.device:
            raise ValueError("GpuReplayBuffer sample device must equal storage device")
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.size < batch_size:
            raise ValueError("not enough replay samples")
        indices = torch.randint(0, self.size, (batch_size,),
                                generator=self.generator, device=self.device)
        return {name: getattr(self, name)[indices] for name in self._tensor_fields}

    def state_dict(self):
        state = {name: int(getattr(self, name)) for name in self._dimension_fields}
        state.update(format_version=2, storage="torch", position=int(self.position),
                     size=int(self.size), generator_device_type=self.device.type,
                     generator_state=self.generator.get_state().cpu().clone())
        for name in self._tensor_fields:
            state[name] = getattr(self, name)[:self.size].detach().cpu().clone()
        return state

    @torch.no_grad()
    def load_state_dict(self, state):
        if not isinstance(state, dict):
            raise TypeError("replay state must be a dict")
        if int(state.get("format_version", -1)) != 2 or state.get("storage") != "torch":
            raise ValueError("unsupported GPU replay checkpoint format")
        for name in self._dimension_fields:
            if int(state[name]) != int(getattr(self, name)):
                raise ValueError(f"replay {name} mismatch")
        size, position = int(state["size"]), int(state["position"])
        if not 0 <= size <= self.capacity or not 0 <= position < self.capacity:
            raise ValueError("invalid replay size or position")
        if size < self.capacity and position != size:
            raise ValueError("partial replay position must equal size")
        if state.get("generator_device_type") != self.device.type:
            raise ValueError("exact replay RNG restoration requires the same device type")
        generator_state = state["generator_state"]
        if not isinstance(generator_state, torch.Tensor):
            raise TypeError("generator_state must be a tensor")
        # Validate the complete checkpoint before mutating storage.
        restored_generator = torch.Generator(device=self.device)
        restored_generator.set_state(generator_state.detach().cpu())
        for name in self._tensor_fields:
            value = state[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"replay {name} must be a tensor")
            if tuple(value.shape) != tuple(getattr(self, name)[:size].shape):
                raise ValueError(f"replay {name} has wrong shape")
        for name in self._tensor_fields:
            destination = getattr(self, name)
            destination[:size].copy_(state[name].to(device=self.device, dtype=destination.dtype))
        self.size, self.position = size, position
        self.generator = restored_generator


# --- frozen notebook cell 225 ---
import importlib
import os


# Export public and internal helpers required by downstream production layers.
__all__ = [name for name in globals() if not name.startswith("__")]
