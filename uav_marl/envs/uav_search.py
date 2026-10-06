"""Production source migrated from frozen fix_test_gpu.ipynb.

Do not import the notebook at runtime. The notebook is retained only as a
reference/regression artifact. Production changes belong in Python modules.
Migrated from notebook cells 139..163.
"""

from ..world.observations import *  # noqa: F401,F403
from ..common import contact_graph_reachability_fraction_numpy

# --- frozen notebook cell 139 ---
class UAVSearchEnv(gym.Env):
    """Cooperative multi-UAV search environment with structured MARL actions."""

    metadata: ClassVar[dict] = {
        "render_modes": [],
    }

    def __init__(
        self,
        backend_name=None,
        network_backend_kwargs=None,
    ):
        super().__init__()

        self.num_uavs = int(
            CONFIG["num_uavs"]
        )
        self.num_targets = int(
            CONFIG["num_targets"]
        )

        if self.num_uavs < 1:
            raise ValueError(
                "num_uavs must be >= 1"
            )

        patch_cells = int(
            CONFIG["belief_patch_cells"]
        )

        if (
            patch_cells < 1
            or patch_cells % 2 == 0
        ):
            raise ValueError(
                "belief_patch_cells must be "
                "a positive odd integer"
            )

        self.agent_ids = tuple(
            f"uav_{uav_id}"
            for uav_id
            in range(self.num_uavs)
        )

        self.backend_name = (
            CONFIG.get(
                "network_backend",
                "simple",
            )
            if backend_name is None
            else backend_name
        )

        self.network_backend_kwargs = dict(
            network_backend_kwargs
            or {}
        )

        destination_count = (
            self.num_uavs + 1
        )

        self.action_space = (
            gym.spaces.Dict(
                {
                    agent_id: (
                        gym.spaces.Dict(
                            {
                                "motion": (
                                    gym.spaces.Box(
                                        low=-1.0,
                                        high=1.0,
                                        shape=(3,),
                                        dtype=np.float32,
                                    )
                                ),
                                "destination": (
                                    gym.spaces.Discrete(
                                        destination_count
                                    )
                                ),
                                "power": (
                                    gym.spaces.Box(
                                        low=-1.0,
                                        high=1.0,
                                        shape=(1,),
                                        dtype=np.float32,
                                    )
                                ),
                            }
                        )
                    )
                    for agent_id
                    in self.agent_ids
                }
            )
        )

        self.observation_space = (
            gym.spaces.Dict(
                {
                    agent_id: (
                        gym.spaces.Dict(
                            {
                                "self_state": (
                                    gym.spaces.Box(
                                        low=-1.0,
                                        high=1.0,
                                        shape=(9,),
                                        dtype=np.float32,
                                    )
                                ),
                                "gcs_relative": (
                                    gym.spaces.Box(
                                        low=-1.0,
                                        high=1.0,
                                        shape=(3,),
                                        dtype=np.float32,
                                    )
                                ),
                                "neighbors": (
                                    gym.spaces.Box(
                                        low=-1.0,
                                        high=1.0,
                                        shape=(
                                            self.num_uavs
                                            - 1,
                                            5,
                                        ),
                                        dtype=np.float32,
                                    )
                                ),
                                "obstacles": (
                                    gym.spaces.Box(
                                        low=-1.0,
                                        high=1.0,
                                        shape=(
                                            int(
                                                CONFIG[
                                                    "observation_nearest_obstacles"
                                                ]
                                            ),
                                            5,
                                        ),
                                        dtype=np.float32,
                                    )
                                ),
                                "belief_patch": (
                                    gym.spaces.Box(
                                        low=0.0,
                                        high=1.0,
                                        shape=(
                                            patch_cells,
                                            patch_cells,
                                        ),
                                        dtype=np.float32,
                                    )
                                ),
                                "belief_coarse": (
                                    gym.spaces.Box(
                                        low=0.0,
                                        high=1.0,
                                        shape=(
                                            int(
                                                CONFIG[
                                                    "belief_coarse_cells"
                                                ]
                                            ),
                                            int(
                                                CONFIG[
                                                    "belief_coarse_cells"
                                                ]
                                            ),
                                        ),
                                        dtype=np.float32,
                                    )
                                ),
                                "buffer": (
                                    gym.spaces.Box(
                                        low=0.0,
                                        high=1.0,
                                        shape=(5,),
                                        dtype=np.float32,
                                    )
                                ),
                                "destination_mask": (
                                    gym.spaces.MultiBinary(
                                        destination_count
                                    )
                                ),
                            }
                        )
                    )
                    for agent_id
                    in self.agent_ids
                }
            )
        )

        local_observation_size = (
            9
            + 3
            + (
                self.num_uavs
                - 1
            )
            * 5
            + int(
                CONFIG[
                    "observation_nearest_obstacles"
                ]
            )
            * 5
            + patch_cells
            * patch_cells
            + int(
                CONFIG[
                    "belief_coarse_cells"
                ]
            )
            ** 2
            + 5
            + destination_count
        )

        report_slots_per_uav = (
            int(
                CONFIG[
                    "buffer_bytes"
                ]
                // CONFIG[
                    "report_bytes"
                ]
            )
            + int(
                CONFIG[
                    "pending_buffer_bytes"
                ]
                // CONFIG[
                    "report_bytes"
                ]
            )
        )

        state_size = (
            self.num_uavs
            * (
                local_observation_size
                + report_slots_per_uav
                * 4
                + 2
                + int(
                    CONFIG[
                        "critic_belief_grid_cells"
                    ]
                )
                ** 2
            )
            + self.num_targets
            * 4
            + int(
                CONFIG[
                    "num_obstacles"
                ]
            )
            * 4
            + 1
        )

        self.state_space = (
            gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(state_size,),
                dtype=np.float32,
            )
        )

        self.current_step = 0
        self.episode_seed = None
        self.rng = None
        self.uavs = None
        self.targets = None
        self.obstacles = None
        self.belief_maps = None
        self.report_buffers = None
        self.pending_reports = None
        self.gcs_received_target_ids = None
        self.coverage_seen = None
        self.network_backend = None
        self._episode_done = False
        self._pending_checkpoint_state = (
            None
        )

    def _make_network_backend(
        self,
        seed,
    ):
        kwargs = dict(
            self.network_backend_kwargs
        )

        normalized = str(
            self.backend_name
        ).strip().lower()

        if normalized in {
            "uavnetsim",
            "uav_net_sim",
        }:
            kwargs.setdefault(
                "seed",
                int(seed),
            )

        return create_network_backend(
            self.backend_name,
            **kwargs,
        )

    def _canonical_actions(
        self,
        actions,
    ):
        if not isinstance(
            actions,
            dict,
        ):
            raise TypeError(
                "actions must be a dict "
                "keyed by UAV agent id"
            )

        if set(actions) != set(
            self.agent_ids
        ):
            raise ValueError(
                "actions must contain exactly "
                "one entry for every UAV"
            )

        canonical = {}

        for agent_id in self.agent_ids:
            action = actions[
                agent_id
            ]

            if not isinstance(
                action,
                dict,
            ):
                raise TypeError(
                    f"{agent_id} action "
                    "must be a dict"
                )

            if set(action) != {
                "motion",
                "destination",
                "power",
            }:
                raise ValueError(
                    f"{agent_id} action must "
                    "contain exactly motion, "
                    "destination, power"
                )

            destination = action[
                "destination"
            ]

            if isinstance(
                destination,
                (bool, np.bool_),
            ):
                raise TypeError(
                    "destination must be "
                    "an integer"
                )

            if not isinstance(
                destination,
                (int, np.integer),
            ):
                raise TypeError(
                    "destination must be "
                    "an integer"
                )

            motion = np.asarray(
                action["motion"],
                dtype=np.float32,
            )

            power = np.asarray(
                action["power"],
                dtype=np.float32,
            )

            if power.shape == ():
                power = power.reshape(
                    1
                )

            candidate = {
                "motion": motion,
                "destination": int(
                    destination
                ),
                "power": power,
            }

            subspace = (
                self.action_space.spaces[
                    agent_id
                ]
            )

            if not subspace.contains(
                candidate
            ):
                raise ValueError(
                    f"{agent_id} action is "
                    "outside action_space"
                )

            canonical[
                agent_id
            ] = candidate

        return canonical

    def _decode_actions(
        self,
        actions,
    ):
        return [
            decode_hybrid_action(
                uav_id,
                actions[
                    self.agent_ids[
                        uav_id
                    ]
                ],
                self.num_uavs,
            )
            for uav_id
            in range(self.num_uavs)
        ]

    def _observations(self):
        observations = {
            agent_id: (
                build_agent_observation(
                    uav_id,
                    self.uavs,
                    self.belief_maps,
                    self.obstacles,
                    self.report_buffers,
                    self.pending_reports,
                    self.current_step,
                )
            )
            for uav_id, agent_id
            in enumerate(
                self.agent_ids
            )
        }

        if not self.observation_space.contains(
            observations
        ):
            raise RuntimeError(
                "generated observation is "
                "outside observation_space"
            )

        return observations

    def global_state(
        self,
        observations=None,
    ):
        if self.uavs is None:
            raise RuntimeError(
                "reset must be called before "
                "global_state"
            )

        state = build_global_state(
            self.uavs,
            self.targets,
            self.belief_maps,
            self.obstacles,
            self.report_buffers,
            self.pending_reports,
            self.gcs_received_target_ids,
            self.current_step,
            observations=observations,
        )

        if not self.state_space.contains(
            state
        ):
            raise RuntimeError(
                "generated global state is "
                "outside state_space"
            )

        return state

    def _coverage_potential(self):
        if self.coverage_seen is None:
            return 0.0
        coverage = np.asarray(
            self.coverage_seen,
            dtype=bool,
        )
        if coverage.size == 0:
            return 0.0
        return float(
            np.count_nonzero(coverage)
            / coverage.size
        )

    def _communication_progress_potential(self):
        """Fractional GCS-delivery progress averaged across mission targets."""
        progress = np.zeros(
            self.num_targets,
            dtype=np.float64,
        )
        for target_id in self.gcs_received_target_ids or ():
            if 0 <= int(target_id) < self.num_targets:
                progress[int(target_id)] = 1.0

        buffers = list(self.report_buffers or [])
        if self.pending_reports is not None:
            buffers.append(self.pending_reports)
        for buffer in buffers:
            for report in buffer:
                target_id = int(report.target_id)
                if not 0 <= target_id < self.num_targets:
                    continue
                fraction = (
                    float(report.delivered_bytes)
                    / max(1.0, float(report.size_bytes))
                )
                progress[target_id] = max(
                    progress[target_id],
                    float(np.clip(fraction, 0.0, 1.0)),
                )
        return float(np.mean(progress)) if progress.size else 0.0

    def checkpoint_state(self):
        """Return exact simple-backend state in a torch-safe representation."""
        if self.uavs is None:
            raise RuntimeError(
                "reset must be called before "
                "checkpoint_state"
            )

        normalized = str(
            self.backend_name
        ).strip().lower()
        if (
            normalized != "simple"
            or not isinstance(
                self.network_backend,
                SimpleNetworkBackend,
            )
        ):
            raise ValueError(
                "exact environment checkpointing "
                "supports simple backend only"
            )

        state = {
            "backend_name": "simple",
            "current_step": int(
                self.current_step
            ),
            "episode_seed": int(
                self.episode_seed
            ),
            "episode_done": bool(
                self._episode_done
            ),
            "belief_potential": float(
                self._belief_potential
            ),
            "coverage_seen": self.coverage_seen,
            "rng_state_json": json.dumps(
                self.rng.bit_generator.state
            ),
            "uavs": self.uavs,
            "targets": self.targets,
            "obstacles": self.obstacles,
            "belief_maps": (
                self.belief_maps
            ),
            "report_buffers": (
                self.report_buffers
            ),
            "pending_reports": (
                self.pending_reports
            ),
            "gcs_received_target_ids": (
                self.gcs_received_target_ids
            ),
            "network_peer_transfer_states": (
                self.network_backend
                ._peer_transfer_states
            ),
            "network_metric_state": (
                self.network_backend
                ._metric_state
            ),
            "network_last_energy": (
                self.network_backend
                ._last_step_comm_energy_by_uav
            ),
        }
        return {
            "format_version": 1,
            "payload": _checkpoint_pack(
                state
            ),
        }

    def restore_checkpoint_state(
        self,
        checkpoint_state,
    ):
        """Restore a state returned by checkpoint_state without resetting it."""
        if not isinstance(
            checkpoint_state,
            dict,
        ):
            raise TypeError(
                "checkpoint_state must be "
                "a dict"
            )

        if int(
            checkpoint_state.get(
                "format_version",
                -1,
            )
        ) != 1:
            raise ValueError(
                "unsupported environment "
                "checkpoint format"
            )

        state = _checkpoint_unpack(
            checkpoint_state["payload"]
        )
        if (
            state.get("backend_name")
            != "simple"
            or str(
                self.backend_name
            ).strip().lower()
            != "simple"
        ):
            raise ValueError(
                "environment checkpoint backend "
                "does not match simple backend"
            )

        if self.network_backend is not None:
            close_method = getattr(
                self.network_backend,
                "close",
                None,
            )
            if callable(close_method):
                close_method()

        self.current_step = int(
            state["current_step"]
        )
        self.episode_seed = int(
            state["episode_seed"]
        )
        self._episode_done = bool(
            state["episode_done"]
        )
        self._belief_potential = float(
            state["belief_potential"]
        )
        grid_n = int(
            np.ceil(
                float(CONFIG["map_size"])
                / float(CONFIG["grid_cell_m"])
            )
        )
        restored_coverage = state.get(
            "coverage_seen"
        )
        self.coverage_seen = (
            np.zeros((grid_n, grid_n), dtype=bool)
            if restored_coverage is None
            else np.asarray(
                restored_coverage,
                dtype=bool,
            ).copy()
        )
        self.uavs = state["uavs"]
        self.targets = state["targets"]
        self.obstacles = state["obstacles"]
        self.belief_maps = np.asarray(
            state["belief_maps"],
            dtype=np.float64,
        ).copy()
        self.report_buffers = (
            state["report_buffers"]
        )
        self.pending_reports = (
            state["pending_reports"]
        )
        self.gcs_received_target_ids = (
            state[
                "gcs_received_target_ids"
            ]
        )

        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = (
            json.loads(
                state["rng_state_json"]
            )
        )
        super().reset(
            seed=self.episode_seed
        )

        self.network_backend = (
            self._make_network_backend(
                self.episode_seed
            )
        )
        if not isinstance(
            self.network_backend,
            SimpleNetworkBackend,
        ):
            raise TypeError(
                "restored backend must be "
                "SimpleNetworkBackend"
            )

        self.network_backend.reset(
            uavs=self.uavs,
            gcs_position=CONFIG[
                "gcs_position"
            ],
            obstacles=self.obstacles,
            report_buffers=(
                self.report_buffers
            ),
            pending_reports=(
                self.pending_reports
            ),
            gcs_received_target_ids=(
                self.gcs_received_target_ids
            ),
        )
        self.network_backend._peer_transfer_states = (
            state[
                "network_peer_transfer_states"
            ]
        )
        self.network_backend._metric_state = (
            state["network_metric_state"]
        )
        self.network_backend._last_step_comm_energy_by_uav = (
            np.asarray(
                state[
                    "network_last_energy"
                ],
                dtype=np.float64,
            ).copy()
        )

        observations = self._observations()
        info = {
            "step": self.current_step,
            "seed": self.episode_seed,
            "backend": str(
                self.backend_name
            ),
            "state": self.global_state(
                observations=observations
            ),
        }
        return observations, info

    def restore_pending_checkpoint_state(
        self,
    ):
        """Worker-callable bridge for per-env AsyncVectorEnv restoration."""
        if (
            self._pending_checkpoint_state
            is None
        ):
            raise RuntimeError(
                "no pending checkpoint state"
            )

        checkpoint_state = (
            self._pending_checkpoint_state
        )
        self._pending_checkpoint_state = (
            None
        )
        return self.restore_checkpoint_state(
            checkpoint_state
        )

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        del options

        if seed is None:
            seed = int(
                CONFIG.get(
                    "seed",
                    44,
                )
            )

        seed = int(seed)

        super().reset(
            seed=seed
        )

        if self.network_backend is not None:
            close_method = getattr(
                self.network_backend,
                "close",
                None,
            )

            if callable(
                close_method
            ):
                close_method()

        # The world owns its RNG; resetting it must not reseed the learner.

        (
            self.rng,
            self.uavs,
            self.targets,
            self.obstacles,
        ) = create_world(seed)

        self.belief_maps = (
            create_belief_maps()
        )
        self._belief_potential = (
            belief_certainty_potential(
                self.belief_maps
            )
        )
        self.report_buffers = (
            create_report_buffers()
        )
        self.pending_reports = (
            create_pending_reports()
        )
        self.gcs_received_target_ids = (
            create_gcs_received_target_ids()
        )
        grid_n = int(
            np.ceil(
                float(CONFIG["map_size"])
                / float(CONFIG["grid_cell_m"])
            )
        )
        self.coverage_seen = np.zeros(
            (grid_n, grid_n),
            dtype=bool,
        )

        self.current_step = 0
        self.episode_seed = seed
        self._episode_done = False

        self.network_backend = (
            self._make_network_backend(
                seed
            )
        )

        self.network_backend.reset(
            uavs=self.uavs,
            gcs_position=CONFIG[
                "gcs_position"
            ],
            obstacles=self.obstacles,
            report_buffers=(
                self.report_buffers
            ),
            pending_reports=(
                self.pending_reports
            ),
            gcs_received_target_ids=(
                self.gcs_received_target_ids
            ),
        )

        observations = (
            self._observations()
        )

        info = {
            "step": 0,
            "seed": seed,
            "backend": str(
                self.backend_name
            ),
            "state": (
                self.global_state(
                    observations=observations,
                )
            ),
        }

        return observations, info

    def step(
        self,
        actions,
    ):
        if self.uavs is None:
            raise RuntimeError(
                "reset must be called before step"
            )

        if self._episode_done:
            raise RuntimeError(
                "episode is done; call reset "
                "before stepping again"
            )

        canonical_actions = (
            self._canonical_actions(
                actions
            )
        )
        hybrid_actions = (
            self._decode_actions(
                canonical_actions
            )
        )

        next_step = (
            self.current_step + 1
        )
        coverage_potential_before = (
            self._coverage_potential()
        )
        communication_progress_before = (
            self._communication_progress_potential()
        )

        expired_target_ids = set()

        for buffer in self.report_buffers:
            expired_target_ids.update(
                report.target_id
                for report
                in remove_expired_reports(
                    buffer,
                    next_step,
                )
            )

        expired_target_ids.update(
            report.target_id
            for report
            in remove_expired_reports(
                self.pending_reports,
                next_step,
            )
        )

        flush_before = (
            flush_pending_reports(
                self.pending_reports,
                self.report_buffers,
                next_step,
                self.gcs_received_target_ids,
                self.uavs,
            )
        )

        expired_target_ids.update(
            flush_before[
                "expired_target_ids"
            ]
        )

        confirmed_before = {
            target.id
            for target in self.targets
            if target.confirmed
        }

        delivered_before = set(
            self.gcs_received_target_ids
        )

        motion_actions = np.stack(
            [
                action.movement
                for action
                in hybrid_actions
            ],
            axis=0,
        )

        positions_before_motion = np.stack(
            [
                uav.position.copy()
                for uav in self.uavs
            ],
            axis=0,
        )
        velocities_before_motion = np.stack(
            [
                uav.velocity.copy()
                for uav in self.uavs
            ],
            axis=0,
        )
        battery_before_step = np.asarray(
            [
                float(uav.battery_j)
                for uav in self.uavs
            ],
            dtype=np.float64,
        )
        active_before_step = np.asarray(
            [
                bool(uav.active)
                for uav in self.uavs
            ],
            dtype=bool,
        )

        motion_result = (
            apply_swarm_motion(
                self.uavs,
                motion_actions,
                obstacles=self.obstacles,
                dt=CONFIG["dt"],
            )
        )
        battery_motion = (
            apply_battery_limited_motion(
                self.uavs,
                hybrid_actions,
                motion_result,
                positions_before_motion,
                velocities_before_motion,
                battery_before_step,
                active_before_step,
                dt=CONFIG["dt"],
            )
        )

        sensing_record_count = 0
        coverage_new_unique_cells = 0
        positive_observation_count = 0
        target_positive_observation_count = 0
        targets_in_fov_ids = set()
        information_gain_bits = 0.0
        confirmation_events = []

        belief_potential_before = float(
            self._belief_potential
        )

        for uav_id, uav in enumerate(
            self.uavs
        ):
            records = sense_and_update(
                uav,
                self.belief_maps[
                    uav_id
                ],
                self.targets,
                self.rng,
                obstacles=(
                    self.obstacles
                ),
            )

            sensing_record_count += len(
                records
            )
            for record in records:
                gy = int(record["gy"])
                gx = int(record["gx"])
                coverage_new_unique_cells += int(
                    not self.coverage_seen[gy, gx]
                )
                self.coverage_seen[gy, gx] = True
            information_gain_bits += (
                sensing_information_gain_bits(
                    records
                )
            )

            for record in records:
                observation_positive = bool(
                    record.get(
                        "observation",
                        0,
                    )
                )
                has_target = bool(
                    record.get(
                        "has_target",
                        False,
                    )
                )
                positive_observation_count += int(
                    observation_positive
                )
                target_positive_observation_count += int(
                    observation_positive
                    and has_target
                )
                if has_target:
                    targets_in_fov_ids.update(
                        int(target_id)
                        for target_id in record.get(
                            "target_ids",
                            [],
                        )
                    )
                confirmation_events.extend(
                    create_confirmation_events(
                        uav,
                        record,
                        next_step,
                    )
                )

        _, event_log = (
            process_confirmation_events(
                confirmation_events,
                self.targets,
                self.report_buffers,
                self.pending_reports,
                self.gcs_received_target_ids,
            )
        )

        for event in event_log:
            if (
                event.confirmation_type
                != "false_confirmation"
            ):
                continue

            gx, gy = event.cell

            self.belief_maps[
                int(
                    event.uav_id
                ),
                int(gy),
                int(gx),
            ] = float(
                CONFIG[
                    "verified_empty_belief"
                ]
            )

        belief_potential_after = (
            belief_certainty_potential(
                self.belief_maps
            )
        )
        self._belief_potential = float(
            belief_potential_after
        )

        flush_after_confirmation = (
            flush_pending_reports(
                self.pending_reports,
                self.report_buffers,
                next_step,
                self.gcs_received_target_ids,
                self.uavs,
            )
        )

        expired_target_ids.update(
            flush_after_confirmation[
                "expired_target_ids"
            ]
        )

        intents = []

        peer_transfer_states = (
            self.network_backend
            .peer_transfer_states
        )

        for uav_id, (
            uav,
            hybrid_action,
        ) in enumerate(
            zip(
                self.uavs,
                hybrid_actions,
            )
        ):
            if not uav.active:
                continue

            intent = (
                build_transmission_intent(
                    uav_id,
                    hybrid_action,
                    self.report_buffers[
                        uav_id
                    ],
                    next_step,
                    peer_transfer_states=(
                        peer_transfer_states
                    ),
                )
            )

            intent = (
                transmission_intent_with_energy_budget(
                    intent,
                    battery_motion[
                        "communication_budget_j"
                    ][uav_id],
                    CONFIG["dt"],
                    self.backend_name,
                )
            )

            if intent is not None:
                intents.append(
                    intent
                )

        self.network_backend.sync_positions(
            self.uavs
        )
        set_energy_budget = getattr(
            self.network_backend,
            "set_step_energy_budget_j",
            None,
        )
        if callable(set_energy_budget):
            set_energy_budget(
                battery_motion[
                    "communication_budget_j"
                ]
            )

        network_results = (
            self.network_backend.step(
                intents,
                dt=CONFIG["dt"],
                current_step=next_step,
            )
        )

        communication_edges = []
        for intent, result in zip(
            intents,
            network_results,
        ):
            tx_bytes = int(
                result.get(
                    "tx_bytes",
                    0,
                )
            )
            if tx_bytes <= 0:
                continue
            sender_id = int(
                intent.sender
            )
            recipient_id = int(
                intent.recipient
            )
            sender_position = (
                np.asarray(
                    self.uavs[
                        sender_id
                    ].position,
                    dtype=np.float64,
                ).copy()
            )
            if recipient_id == GCS_NODE:
                recipient_position = (
                    np.asarray(
                        CONFIG[
                            "gcs_position"
                        ],
                        dtype=np.float64,
                    ).copy()
                )
            elif (
                0
                <= recipient_id
                < len(
                    self.uavs
                )
            ):
                recipient_position = (
                    np.asarray(
                        self.uavs[
                            recipient_id
                        ].position,
                        dtype=np.float64,
                    ).copy()
                )
            else:
                recipient_position = None
            communication_edges.append(
                {
                    "sender": sender_id,
                    "recipient": recipient_id,
                    "target_id": int(
                        intent.target_id
                    ),
                    "tx_bytes": tx_bytes,
                    "status": str(
                        result.get(
                            "status",
                            "",
                        )
                    ),
                    "step": int(
                        next_step
                    ),
                    "sender_position": (
                        sender_position
                    ),
                    "recipient_position": (
                        recipient_position
                    ),
                }
            )

        flush_after_network = (
            flush_pending_reports(
                self.pending_reports,
                self.report_buffers,
                next_step,
                self.gcs_received_target_ids,
                self.uavs,
            )
        )

        expired_target_ids.update(
            flush_after_network[
                "expired_target_ids"
            ]
        )

        communication_energy = (
            self.network_backend
            .last_step_communication_energy_by_uav()
        )
        communication_energy = np.minimum(
            np.asarray(
                communication_energy,
                dtype=np.float64,
            ),
            battery_motion[
                "communication_budget_j"
            ],
        )

        energy_result = (
            apply_uav_energy_budget(
                self.uavs,
                hybrid_actions,
                communication_energy,
                dt=CONFIG["dt"],
                realized_accelerations_mps2=(
                    motion_result[
                        "realized_acceleration_mps2"
                    ]
                ),
                propulsion_energy_override_j=(
                    battery_motion[
                        "propulsion_charged_j"
                    ]
                ),
            )
        )
        energy_result[
            "depleted_uav_ids"
        ] = sorted(
            set(
                int(value)
                for value in energy_result[
                    "depleted_uav_ids"
                ]
            )
            | set(
                int(value)
                for value in np.flatnonzero(
                    battery_motion[
                        "battery_limited"
                    ]
                )
            )
        )

        confirmed_after = {
            target.id
            for target in self.targets
            if target.confirmed
        }

        delivered_after = set(
            self.gcs_received_target_ids
        )

        newly_confirmed = sorted(
            confirmed_after
            - confirmed_before
        )
        newly_delivered = sorted(
            delivered_after
            - delivered_before
        )

        false_confirmation_count = sum(
            event.confirmation_type
            == "false_confirmation"
            for event in event_log
        )

        dropped_report_count = sum(
            event.report_status
            == "dropped"
            for event in event_log
        )

        blocked_motion_count = int(
            np.count_nonzero(
                motion_result.get(
                    "blocked",
                    np.zeros(
                        self.num_uavs,
                        dtype=bool,
                    ),
                )
            )
        )
        boundary_count = int(
            np.count_nonzero(
                motion_result[
                    "boundary_clipped"
                ]
            )
        )

        success = (
            len(
                self.gcs_received_target_ids
            )
            == self.num_targets
        )

        all_inactive = not any(
            uav.active
            for uav in self.uavs
        )

        episode_will_end = bool(
            success
            or all_inactive
            or next_step
            >= int(
                CONFIG[
                    "max_steps"
                ]
            )
        )

        shaping_gamma = float(
            CONFIG["reward_shaping_gamma"]
        )
        shaping_next_potential = (
            0.0
            if episode_will_end
            else belief_potential_after
        )
        coverage_potential_after = (
            self._coverage_potential()
        )
        communication_progress_after = (
            self._communication_progress_potential()
        )
        shaping_next_coverage = (
            0.0
            if episode_will_end
            else coverage_potential_after
        )
        shaping_next_communication = (
            0.0
            if episode_will_end
            else communication_progress_after
        )

        information_shaping = float(
            CONFIG["reward_info_gain"]
        ) * (
            shaping_gamma
            * shaping_next_potential
            - belief_potential_before
        )
        coverage_shaping = float(
            CONFIG.get(
                "reward_coverage_shaping",
                0.0,
            )
        ) * (
            shaping_gamma
            * shaping_next_coverage
            - coverage_potential_before
        )
        communication_progress_shaping = float(
            CONFIG.get(
                "reward_communication_progress_shaping",
                0.0,
            )
        ) * (
            shaping_gamma
            * shaping_next_communication
            - communication_progress_before
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

        reward_components = {
            "information_gain": (
                information_shaping
            ),
            "coverage_shaping": (
                coverage_shaping
            ),
            "communication_progress": (
                communication_progress_shaping
            ),
            "confirmation": (
                float(
                    CONFIG[
                        "reward_confirmation"
                    ]
                )
                * len(
                    newly_confirmed
                )
            ),
            "delivery": (
                float(
                    CONFIG[
                        "reward_delivery"
                    ]
                )
                * len(
                    newly_delivered
                )
            ),
            "false_confirmation": (
                -float(
                    CONFIG[
                        "reward_false_confirmation"
                    ]
                )
                * false_confirmation_count
            ),
            "blocked_motion": (
                -float(
                    CONFIG[
                        "reward_blocked"
                    ]
                )
                * blocked_motion_count
                / safety_denominator
            ),
            "boundary": (
                -float(
                    CONFIG[
                        "reward_boundary"
                    ]
                )
                * boundary_count
                / safety_denominator
            ),
            "expired_report": (
                -float(
                    CONFIG[
                        "reward_expired_report"
                    ]
                )
                * len(
                    expired_target_ids
                )
            ),
            "dropped_report": (
                -float(
                    CONFIG[
                        "reward_dropped_report"
                    ]
                )
                * dropped_report_count
            ),
            "energy": (
                -float(
                    CONFIG[
                        "reward_energy_per_kj"
                    ]
                )
                * float(
                    np.sum(
                        energy_result[
                            "total_j"
                        ]
                    )
                )
                / 1000.0
            ),
            "step": (
                -float(
                    CONFIG[
                        "reward_step_penalty"
                    ]
                )
            ),
            "success_bonus": (
                float(
                    CONFIG[
                        "reward_all_delivered_bonus"
                    ]
                )
                if success
                else 0.0
            ),
        }

        reward = float(
            sum(
                reward_components.values()
            )
        )

        self.current_step = next_step

        horizon_reached = bool(
            self.current_step
            >= int(
                CONFIG[
                    "max_steps"
                ]
            )
        )
        # max_steps is the finite mission horizon, not an
        # external TimeLimit wrapper. The remaining-time fraction
        # is already part of each actor observation and the critic
        # state, so reaching the horizon is an MDP terminal event.
        terminated = bool(
            success
            or all_inactive
            or horizon_reached
        )
        truncated = False

        self._episode_done = bool(
            terminated
            or truncated
        )

        observations = (
            self._observations()
        )
        state = self.global_state(
            observations=observations,
        )

        agent_rewards = {
            agent_id: reward
            for agent_id
            in self.agent_ids
        }

        gcs_contact_graph_reachability_fraction = (
            contact_graph_reachability_fraction_numpy(
                np.stack(
                    [
                        np.asarray(uav.position, dtype=np.float64)
                        for uav in self.uavs
                    ],
                    axis=0,
                ),
                np.asarray(
                    [bool(uav.active) for uav in self.uavs],
                    dtype=bool,
                ),
                np.asarray(CONFIG["gcs_position"], dtype=np.float64),
                peer_range_m=float(CONFIG["peer_contact_range_m"]),
                gcs_range_m=float(CONFIG["gcs_contact_range_m"]),
            )
        )

        info = {
            "step": self.current_step,
            "backend": str(
                self.backend_name
            ),
            "state": state,
            "agent_rewards": (
                agent_rewards
            ),
            "reward_components": (
                reward_components
            ),
            "motion": motion_result,
            "network_results": (
                make_info_pickle_safe(
                    network_results
                )
            ),
            "communication_edges": (
                communication_edges
            ),
            "network_metrics": (
                self.network_backend
                .metrics()
            ),
            "sensing_record_count": (
                sensing_record_count
            ),
            "coverage_sensed_cell_events": int(
                sensing_record_count
            ),
            "coverage_new_unique_cells": int(
                coverage_new_unique_cells
            ),
            "gcs_contact_graph_reachability_fraction": float(
                gcs_contact_graph_reachability_fraction
            ),
            "positive_observation_count": int(
                positive_observation_count
            ),
            "target_positive_observation_count": int(
                target_positive_observation_count
            ),
            "targets_in_fov_ids": sorted(
                targets_in_fov_ids
            ),
            "information_gain_bits": (
                information_gain_bits
            ),
            "belief_certainty_before": (
                belief_potential_before
            ),
            "belief_certainty_after": (
                belief_potential_after
            ),
            "information_shaping": (
                information_shaping
            ),
            "newly_confirmed_target_ids": (
                newly_confirmed
            ),
            "newly_delivered_target_ids": (
                newly_delivered
            ),
            "false_confirmation_count": (
                int(
                    false_confirmation_count
                )
            ),
            "dropped_report_count": (
                int(
                    dropped_report_count
                )
            ),
            "expired_target_ids": sorted(
                expired_target_ids
            ),
            "energy": energy_result,
            "success": bool(success),
            "all_uavs_inactive": bool(
                all_inactive
            ),
        }

        return (
            observations,
            reward,
            terminated,
            truncated,
            info,
        )

    def close(self):
        if self.network_backend is None:
            return

        close_method = getattr(
            self.network_backend,
            "close",
            None,
        )

        if callable(close_method):
            close_method()

        self.network_backend = None


# --- frozen notebook cell 140 ---
import concurrent.futures


# --- frozen notebook cell 141 ---
import queue


# --- frozen notebook cell 142 ---
import threading


def _get_torch_sensing_tls():
    # Resolve process-local state lazily instead of capturing an unpicklable
    # threading.local in cloudpickle's dependency graph for the DDP worker.
    local_state = globals().get("_TORCH_SENSING_TLS")
    if local_state is None:
        local_state = threading.local()
        globals()["_TORCH_SENSING_TLS"] = local_state
    return local_state


_get_torch_sensing_tls()


# --- frozen notebook cell 143 ---
_CPU_SENSE_AND_UPDATE = sense_and_update


# --- frozen notebook cell 144 ---
class TorchBatchedSensingEngine:
    """Batch the expensive FOV/LOS/Bayes sensing math with torch.

    The mission/world/report/network semantics stay in UAVSearchEnv. Only the
    per-UAV sensing kernel is vectorized. CPU sensing remains the reference
    implementation and is used whenever no coordinator is active.
    """

    def __init__(
        self,
        device="auto",
        dtype="float32",
    ):
        if str(device).strip().lower() == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        normalized_dtype = str(dtype).strip().lower()
        if normalized_dtype == "float64":
            self.dtype = torch.float64
        elif normalized_dtype == "float32":
            self.dtype = torch.float32
        else:
            raise ValueError(
                "training_torch_sensing_dtype must be "
                "'float32' or 'float64'"
            )

        self.batch_count = 0
        self.request_count = 0
        self.candidate_count = 0
        self.kernel_seconds = 0.0

    @staticmethod
    def _circle_sqrt_integral_torch(
        x,
        radius,
    ):
        radius_safe = torch.clamp(
            radius,
            min=torch.finfo(x.dtype).eps,
        )
        x = torch.maximum(
            torch.minimum(x, radius),
            -radius,
        )
        root = torch.sqrt(
            torch.clamp(
                radius * radius - x * x,
                min=0.0,
            )
        )
        ratio = torch.clamp(
            x / radius_safe,
            -1.0,
            1.0,
        )
        return 0.5 * (
            x * root
            + radius * radius
            * torch.asin(ratio)
        )

    def _coverage_fraction(
        self,
        circle_x,
        circle_y,
        radius,
        gx,
        gy,
    ):
        cell_size = float(CONFIG["grid_cell_m"])
        map_size = float(CONFIG["map_size"])

        x_min = gx * cell_size
        x_max = torch.clamp(
            (gx + 1.0) * cell_size,
            max=map_size,
        )
        y_min = gy * cell_size
        y_max = torch.clamp(
            (gy + 1.0) * cell_size,
            max=map_size,
        )

        x_min = x_min - circle_x
        x_max = x_max - circle_x
        y_min = y_min - circle_y
        y_max = y_max - circle_y

        left = torch.maximum(
            x_min,
            -radius,
        )
        right = torch.minimum(
            x_max,
            radius,
        )

        base_valid = (
            (radius > 0.0)
            & (right > left)
            & (y_max > -radius)
            & (y_min < radius)
        )

        def _crossings(y_edge):
            has_crossing = (
                torch.abs(y_edge)
                < radius
            )
            x_cross = torch.sqrt(
                torch.clamp(
                    radius * radius
                    - y_edge * y_edge,
                    min=0.0,
                )
            )
            negative = torch.where(
                has_crossing,
                -x_cross,
                left,
            )
            positive = torch.where(
                has_crossing,
                x_cross,
                left,
            )
            negative = torch.maximum(
                left,
                torch.minimum(
                    right,
                    negative,
                ),
            )
            positive = torch.maximum(
                left,
                torch.minimum(
                    right,
                    positive,
                ),
            )
            return negative, positive

        y_min_neg, y_min_pos = _crossings(
            y_min
        )
        y_max_neg, y_max_pos = _crossings(
            y_max
        )

        cuts = torch.stack(
            (
                left,
                right,
                y_min_neg,
                y_min_pos,
                y_max_neg,
                y_max_pos,
            ),
            dim=-1,
        )
        cuts = torch.sort(
            cuts,
            dim=-1,
        ).values

        a = cuts[:, :-1]
        b = cuts[:, 1:]
        width = b - a
        interval_valid = width > 0.0
        midpoint = 0.5 * (a + b)

        radius_2d = radius[:, None]
        y_min_2d = y_min[:, None]
        y_max_2d = y_max[:, None]

        half_height = torch.sqrt(
            torch.clamp(
                radius_2d * radius_2d
                - midpoint * midpoint,
                min=0.0,
            )
        )
        upper = torch.minimum(
            y_max_2d,
            half_height,
        )
        lower = torch.maximum(
            y_min_2d,
            -half_height,
        )

        arc_integral = (
            self._circle_sqrt_integral_torch(
                b,
                radius_2d,
            )
            - self._circle_sqrt_integral_torch(
                a,
                radius_2d,
            )
        )

        upper_integral = torch.where(
            y_max_2d < half_height,
            y_max_2d * width,
            arc_integral,
        )
        lower_integral = torch.where(
            y_min_2d > -half_height,
            y_min_2d * width,
            -arc_integral,
        )

        area_terms = torch.where(
            interval_valid
            & (upper > lower),
            upper_integral - lower_integral,
            torch.zeros_like(width),
        )
        intersection_area = torch.clamp(
            area_terms.sum(dim=-1),
            min=0.0,
        )

        cell_area = torch.clamp(
            (x_max - x_min)
            * (y_max - y_min),
            min=torch.finfo(
                intersection_area.dtype
            ).eps,
        )
        fraction = torch.clamp(
            intersection_area / cell_area,
            0.0,
            1.0,
        )
        return torch.where(
            base_valid,
            fraction,
            torch.zeros_like(fraction),
        )

    @staticmethod
    def _segment_hits_any_obstacle(
        start,
        end,
        centers,
        radii,
        heights,
        obstacle_valid,
    ):
        if centers.shape[1] == 0:
            return torch.zeros(
                start.shape[0],
                dtype=torch.bool,
                device=start.device,
            )

        eps = 1e-12
        direction = end - start
        dz = direction[:, 2:3]
        start_z = start[:, 2:3]

        horizontal_z = (
            torch.abs(dz) <= eps
        )
        safe_dz = torch.where(
            horizontal_z,
            torch.ones_like(dz),
            dz,
        )

        t_ground = (
            -start_z / safe_dz
        )
        t_top = (
            heights - start_z
        ) / safe_dz

        z_enter_dynamic = torch.maximum(
            torch.zeros_like(t_top),
            torch.minimum(
                t_ground,
                t_top,
            ),
        )
        z_exit_dynamic = torch.minimum(
            torch.ones_like(t_top),
            torch.maximum(
                t_ground,
                t_top,
            ),
        )

        z_enter = torch.where(
            horizontal_z,
            torch.zeros_like(
                z_enter_dynamic
            ),
            z_enter_dynamic,
        )
        z_exit = torch.where(
            horizontal_z,
            torch.ones_like(
                z_exit_dynamic
            ),
            z_exit_dynamic,
        )
        z_valid = torch.where(
            horizontal_z,
            (
                (start_z >= 0.0)
                & (start_z <= heights)
            ),
            z_enter <= z_exit,
        )

        relative_xy = (
            start[:, None, :2]
            - centers
        )
        direction_xy = (
            direction[:, None, :2]
        )
        a = (
            direction_xy
            * direction_xy
        ).sum(dim=-1)
        c = (
            relative_xy
            * relative_xy
        ).sum(dim=-1) - radii * radii

        stationary_xy = a <= eps
        safe_a = torch.where(
            stationary_xy,
            torch.ones_like(a),
            a,
        )

        b = 2.0 * (
            relative_xy
            * direction_xy
        ).sum(dim=-1)
        discriminant = (
            b * b
            - 4.0 * safe_a * c
        )
        root = torch.sqrt(
            torch.clamp(
                discriminant,
                min=0.0,
            )
        )

        t1 = (
            -b - root
        ) / (2.0 * safe_a)
        t2 = (
            -b + root
        ) / (2.0 * safe_a)

        xy_enter_dynamic = torch.maximum(
            torch.zeros_like(t1),
            torch.minimum(t1, t2),
        )
        xy_exit_dynamic = torch.minimum(
            torch.ones_like(t1),
            torch.maximum(t1, t2),
        )

        xy_enter = torch.where(
            stationary_xy,
            torch.zeros_like(
                xy_enter_dynamic
            ),
            xy_enter_dynamic,
        )
        xy_exit = torch.where(
            stationary_xy,
            torch.ones_like(
                xy_exit_dynamic
            ),
            xy_exit_dynamic,
        )
        xy_valid = torch.where(
            stationary_xy,
            c <= 0.0,
            (
                (discriminant >= 0.0)
                & (
                    xy_enter_dynamic
                    <= xy_exit_dynamic
                )
            ),
        )

        enter = torch.maximum(
            z_enter,
            xy_enter,
        )
        exit_ = torch.minimum(
            z_exit,
            xy_exit,
        )

        intersects = (
            obstacle_valid
            & z_valid
            & xy_valid
            & (enter <= exit_)
        )
        return torch.any(
            intersects,
            dim=-1,
        )

    def _request_obstacle_tensors(
        self,
        requests,
    ):
        max_obstacles = max(
            (
                len(
                    request["obstacles"]
                )
                for request in requests
            ),
            default=0,
        )
        count = len(requests)

        centers = np.zeros(
            (
                count,
                max_obstacles,
                2,
            ),
            dtype=np.float64,
        )
        radii = np.zeros(
            (
                count,
                max_obstacles,
            ),
            dtype=np.float64,
        )
        heights = np.zeros(
            (
                count,
                max_obstacles,
            ),
            dtype=np.float64,
        )
        valid = np.zeros(
            (
                count,
                max_obstacles,
            ),
            dtype=np.bool_,
        )

        for request_index, request in enumerate(
            requests
        ):
            for obstacle_index, obstacle in enumerate(
                request["obstacles"]
            ):
                centers[
                    request_index,
                    obstacle_index,
                ] = obstacle.position
                radii[
                    request_index,
                    obstacle_index,
                ] = float(
                    obstacle.radius
                )
                heights[
                    request_index,
                    obstacle_index,
                ] = float(
                    obstacle.height
                )
                valid[
                    request_index,
                    obstacle_index,
                ] = True

        return (
            torch.as_tensor(
                centers,
                dtype=self.dtype,
                device=self.device,
            ),
            torch.as_tensor(
                radii,
                dtype=self.dtype,
                device=self.device,
            ),
            torch.as_tensor(
                heights,
                dtype=self.dtype,
                device=self.device,
            ),
            torch.as_tensor(
                valid,
                dtype=torch.bool,
                device=self.device,
            ),
        )

    def sense_batch(
        self,
        requests,
    ):
        if not requests:
            return []

        kernel_start = time.perf_counter()

        results = [
            []
            for _ in requests
        ]

        candidate_request = []
        candidate_gx = []
        candidate_gy = []
        candidate_circle_x = []
        candidate_circle_y = []
        candidate_radius = []
        candidate_base_pd = []
        candidate_base_pf = []
        candidate_start = []
        target_id = []
        target_xy = []

        target_maps = []
        request_profiles = []

        for request in requests:
            target_map = {}
            for target in request["targets"]:
                target_map[
                    world_to_grid(
                        target.position
                    )
                ] = (
                    int(target.id),
                    np.asarray(
                        target.position,
                        dtype=np.float64,
                    ),
                )
            target_maps.append(
                target_map
            )

        for request_index, request in enumerate(
            requests
        ):
            uav = request["uav"]

            if (
                not uav.active
                or float(
                    uav.position[2]
                )
                <= 0.0
            ):
                request_profiles.append(
                    None
                )
                continue

            (
                base_pd,
                base_pf,
                fov_radius,
            ) = sensing_profile(
                float(
                    uav.position[2]
                )
            )
            request_profiles.append(
                (
                    base_pd,
                    base_pf,
                    fov_radius,
                )
            )

            cell_size = float(
                CONFIG["grid_cell_m"]
            )
            grid_n = int(
                np.ceil(
                    float(
                        CONFIG[
                            "map_size"
                        ]
                    )
                    / cell_size
                )
            )
            uav_x = float(
                uav.position[0]
            )
            uav_y = float(
                uav.position[1]
            )

            gx_min = max(
                0,
                int(
                    np.floor(
                        (
                            uav_x
                            - fov_radius
                        )
                        / cell_size
                    )
                ),
            )
            gx_max = min(
                grid_n - 1,
                int(
                    np.floor(
                        (
                            uav_x
                            + fov_radius
                        )
                        / cell_size
                    )
                ),
            )
            gy_min = max(
                0,
                int(
                    np.floor(
                        (
                            uav_y
                            - fov_radius
                        )
                        / cell_size
                    )
                ),
            )
            gy_max = min(
                grid_n - 1,
                int(
                    np.floor(
                        (
                            uav_y
                            + fov_radius
                        )
                        / cell_size
                    )
                ),
            )

            for gy in range(
                gy_min,
                gy_max + 1,
            ):
                for gx in range(
                    gx_min,
                    gx_max + 1,
                ):
                    candidate_request.append(
                        request_index
                    )
                    candidate_gx.append(gx)
                    candidate_gy.append(gy)
                    candidate_circle_x.append(
                        uav_x
                    )
                    candidate_circle_y.append(
                        uav_y
                    )
                    candidate_radius.append(
                        fov_radius
                    )
                    candidate_base_pd.append(
                        base_pd
                    )
                    candidate_base_pf.append(
                        base_pf
                    )
                    candidate_start.append(
                        np.asarray(
                            uav.position,
                            dtype=np.float64,
                        )
                    )

                    target_entry = (
                        target_maps[
                            request_index
                        ].get(
                            (gx, gy)
                        )
                    )
                    if target_entry is None:
                        target_id.append(-1)
                        target_xy.append(
                            np.zeros(
                                2,
                                dtype=np.float64,
                            )
                        )
                    else:
                        (
                            current_target_id,
                            current_target_xy,
                        ) = target_entry
                        target_id.append(
                            current_target_id
                        )
                        target_xy.append(
                            current_target_xy
                        )

        if not candidate_request:
            self.batch_count += 1
            self.request_count += len(
                requests
            )
            self.kernel_seconds += (
                time.perf_counter()
                - kernel_start
            )
            return results

        request_index_np = np.asarray(
            candidate_request,
            dtype=np.int64,
        )
        gx_np = np.asarray(
            candidate_gx,
            dtype=np.int64,
        )
        gy_np = np.asarray(
            candidate_gy,
            dtype=np.int64,
        )
        target_id_np = np.asarray(
            target_id,
            dtype=np.int64,
        )

        request_index = torch.as_tensor(
            request_index_np,
            dtype=torch.long,
            device=self.device,
        )
        gx = torch.as_tensor(
            gx_np,
            dtype=self.dtype,
            device=self.device,
        )
        gy = torch.as_tensor(
            gy_np,
            dtype=self.dtype,
            device=self.device,
        )
        circle_x = torch.as_tensor(
            candidate_circle_x,
            dtype=self.dtype,
            device=self.device,
        )
        circle_y = torch.as_tensor(
            candidate_circle_y,
            dtype=self.dtype,
            device=self.device,
        )
        radius = torch.as_tensor(
            candidate_radius,
            dtype=self.dtype,
            device=self.device,
        )
        base_pd = torch.as_tensor(
            candidate_base_pd,
            dtype=self.dtype,
            device=self.device,
        )
        base_pf = torch.as_tensor(
            candidate_base_pf,
            dtype=self.dtype,
            device=self.device,
        )
        start = torch.as_tensor(
            np.asarray(
                candidate_start,
                dtype=np.float64,
            ),
            dtype=self.dtype,
            device=self.device,
        )
        target_xy_tensor = torch.as_tensor(
            np.asarray(
                target_xy,
                dtype=np.float64,
            ),
            dtype=self.dtype,
            device=self.device,
        )
        target_id_tensor = torch.as_tensor(
            target_id_np,
            dtype=torch.long,
            device=self.device,
        )

        coverage = (
            self._coverage_fraction(
                circle_x,
                circle_y,
                radius,
                gx,
                gy,
            )
        )
        visible = coverage > 0.0

        cell_size = float(
            CONFIG["grid_cell_m"]
        )
        map_size = float(
            CONFIG["map_size"]
        )
        cell_center = torch.stack(
            (
                torch.clamp(
                    (
                        gx + 0.5
                    )
                    * cell_size,
                    max=map_size,
                ),
                torch.clamp(
                    (
                        gy + 0.5
                    )
                    * cell_size,
                    max=map_size,
                ),
                torch.zeros_like(gx),
            ),
            dim=-1,
        )

        (
            obstacle_centers,
            obstacle_radii,
            obstacle_heights,
            obstacle_valid,
        ) = self._request_obstacle_tensors(
            requests
        )
        candidate_centers = (
            obstacle_centers[
                request_index
            ]
        )
        candidate_radii = (
            obstacle_radii[
                request_index
            ]
        )
        candidate_heights = (
            obstacle_heights[
                request_index
            ]
        )
        candidate_obstacle_valid = (
            obstacle_valid[
                request_index
            ]
        )

        occlusion_enabled = bool(
            CONFIG[
                "sensing_obstacle_occlusion"
            ]
        )

        if occlusion_enabled:
            cell_blocked = (
                self._segment_hits_any_obstacle(
                    start,
                    cell_center,
                    candidate_centers,
                    candidate_radii,
                    candidate_heights,
                    candidate_obstacle_valid,
                )
            )
        else:
            cell_blocked = torch.zeros_like(
                visible
            )

        cell_valid = (
            visible
            & ~cell_blocked
        )

        target_present = (
            target_id_tensor >= 0
        )
        target_delta = (
            target_xy_tensor
            - start[:, :2]
        )
        target_in_footprint = (
            torch.linalg.vector_norm(
                target_delta,
                dim=-1,
            )
            <= radius
        )

        target_end = torch.cat(
            (
                target_xy_tensor,
                torch.zeros(
                    (
                        target_xy_tensor.shape[
                            0
                        ],
                        1,
                    ),
                    dtype=self.dtype,
                    device=self.device,
                ),
            ),
            dim=-1,
        )

        if occlusion_enabled:
            target_blocked = (
                self._segment_hits_any_obstacle(
                    start,
                    target_end,
                    candidate_centers,
                    candidate_radii,
                    candidate_heights,
                    candidate_obstacle_valid,
                )
            )
        else:
            target_blocked = (
                torch.zeros_like(
                    target_present
                )
            )

        target_visible = (
            target_present
            & target_in_footprint
            & ~target_blocked
        )
        has_target = target_visible

        effective_pf = (
            1.0
            - torch.pow(
                1.0 - base_pf,
                coverage,
            )
        )
        effective_pd = (
            coverage * base_pd
            + (
                1.0 - coverage
            )
            * effective_pf
        )
        positive_probability = (
            torch.where(
                has_target,
                base_pd,
                effective_pf,
            )
        )

        valid_np = (
            cell_valid
            .detach()
            .cpu()
            .numpy()
            .astype(np.bool_)
        )

        random_values = np.zeros(
            len(candidate_request),
            dtype=np.float64,
        )
        prior_values = np.zeros(
            len(candidate_request),
            dtype=np.float64,
        )

        for request_id, request in enumerate(
            requests
        ):
            indices = np.flatnonzero(
                valid_np
                & (
                    request_index_np
                    == request_id
                )
            )
            if indices.size == 0:
                continue

            random_values[
                indices
            ] = request[
                "rng"
            ].random(
                indices.size
            )
            prior_values[
                indices
            ] = request[
                "belief_map"
            ][
                gy_np[indices],
                gx_np[indices],
            ]

        random_tensor = torch.as_tensor(
            random_values,
            dtype=self.dtype,
            device=self.device,
        )
        prior = torch.as_tensor(
            prior_values,
            dtype=self.dtype,
            device=self.device,
        )

        observation = (
            random_tensor
            < positive_probability
        )
        eps = 1e-8
        prior_clamped = torch.clamp(
            prior,
            eps,
            1.0 - eps,
        )

        positive_numerator = (
            effective_pd
            * prior_clamped
        )
        positive_denominator = (
            positive_numerator
            + effective_pf
            * (
                1.0
                - prior_clamped
            )
        )

        negative_numerator = (
            (
                1.0
                - effective_pd
            )
            * prior_clamped
        )
        negative_denominator = (
            negative_numerator
            + (
                1.0
                - effective_pf
            )
            * (
                1.0
                - prior_clamped
            )
        )

        numerator = torch.where(
            observation,
            positive_numerator,
            negative_numerator,
        )
        denominator = torch.where(
            observation,
            positive_denominator,
            negative_denominator,
        )
        denominator = torch.clamp(
            denominator,
            min=eps,
        )
        posterior = torch.clamp(
            numerator / denominator,
            eps,
            1.0 - eps,
        )

        coverage_np = (
            coverage.detach()
            .cpu()
            .numpy()
        )
        effective_pf_np = (
            effective_pf.detach()
            .cpu()
            .numpy()
        )
        effective_pd_np = (
            effective_pd.detach()
            .cpu()
            .numpy()
        )
        base_pd_np = (
            base_pd.detach()
            .cpu()
            .numpy()
        )
        base_pf_np = (
            base_pf.detach()
            .cpu()
            .numpy()
        )
        target_visible_np = (
            target_visible.detach()
            .cpu()
            .numpy()
            .astype(np.bool_)
        )
        observation_np = (
            observation.detach()
            .cpu()
            .numpy()
            .astype(np.int8)
        )
        posterior_np = (
            posterior.detach()
            .cpu()
            .numpy()
        )

        for request_id, request in enumerate(
            requests
        ):
            indices = np.flatnonzero(
                valid_np
                & (
                    request_index_np
                    == request_id
                )
            )
            if indices.size == 0:
                continue

            request[
                "belief_map"
            ][
                gy_np[indices],
                gx_np[indices],
            ] = posterior_np[
                indices
            ]

            sensing_log = []
            for index in indices:
                current_target_ids = (
                    [
                        int(
                            target_id_np[
                                index
                            ]
                        )
                    ]
                    if target_visible_np[
                        index
                    ]
                    else []
                )
                sensing_log.append(
                    {
                        "gx": int(
                            gx_np[index]
                        ),
                        "gy": int(
                            gy_np[index]
                        ),
                        "target_ids": (
                            current_target_ids
                        ),
                        "has_target": bool(
                            target_visible_np[
                                index
                            ]
                        ),
                        "observation": int(
                            observation_np[
                                index
                            ]
                        ),
                        "prior": float(
                            prior_values[
                                index
                            ]
                        ),
                        "posterior": float(
                            posterior_np[
                                index
                            ]
                        ),
                        "pd": float(
                            effective_pd_np[
                                index
                            ]
                        ),
                        "pf": float(
                            effective_pf_np[
                                index
                            ]
                        ),
                        "base_pd": float(
                            base_pd_np[
                                index
                            ]
                        ),
                        "base_pf": float(
                            base_pf_np[
                                index
                            ]
                        ),
                        "coverage_fraction": float(
                            coverage_np[
                                index
                            ]
                        ),
                    }
                )
            results[
                request_id
            ] = sensing_log

        if (
            self.device.type
            == "cuda"
        ):
            torch.cuda.synchronize(
                self.device
            )

        self.batch_count += 1
        self.request_count += len(
            requests
        )
        self.candidate_count += len(
            candidate_request
        )
        self.kernel_seconds += (
            time.perf_counter()
            - kernel_start
        )
        return results

    def metrics(self):
        return {
            "device": str(
                self.device
            ),
            "dtype": str(
                self.dtype
            ).replace(
                "torch.",
                "",
            ),
            "batches": int(
                self.batch_count
            ),
            "requests": int(
                self.request_count
            ),
            "candidates": int(
                self.candidate_count
            ),
            "kernel_seconds": float(
                self.kernel_seconds
            ),
        }


# --- frozen notebook cell 145 ---
class TorchSensingCoordinator:
    def __init__(
        self,
        num_envs,
        device="auto",
        dtype="float32",
        batch_timeout_s=0.002,
    ):
        self.num_envs = int(
            num_envs
        )
        self.batch_timeout_s = float(
            batch_timeout_s
        )
        self.engine = (
            TorchBatchedSensingEngine(
                device=device,
                dtype=dtype,
            )
        )
        self._queue = queue.Queue()
        self._sentinel = object()
        self._closed = False
        self._thread = threading.Thread(
            target=self._worker,
            name="torch-sensing-batcher",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        *,
        env_index,
        call_index,
        uav,
        belief_map,
        targets,
        rng,
        obstacles,
    ):
        if self._closed:
            raise RuntimeError(
                "sensing coordinator is closed"
            )

        future = (
            concurrent.futures.Future()
        )
        self._queue.put(
            (
                int(env_index),
                int(call_index),
                {
                    "uav": uav,
                    "belief_map": (
                        belief_map
                    ),
                    "targets": targets,
                    "rng": rng,
                    "obstacles": (
                        []
                        if obstacles is None
                        else obstacles
                    ),
                },
                future,
            )
        )
        return future.result()

    def _worker(self):
        while True:
            first = self._queue.get()
            if (
                first is self._sentinel
            ):
                return

            batch = [first]
            deadline = (
                time.perf_counter()
                + self.batch_timeout_s
            )

            while (
                len(batch)
                < self.num_envs
            ):
                remaining = (
                    deadline
                    - time.perf_counter()
                )
                if remaining <= 0.0:
                    break

                try:
                    item = self._queue.get(
                        timeout=remaining
                    )
                except queue.Empty:
                    break

                if (
                    item
                    is self._sentinel
                ):
                    self._queue.put(
                        self._sentinel
                    )
                    break
                batch.append(item)

            batch.sort(
                key=lambda item: (
                    item[1],
                    item[0],
                )
            )
            requests = [
                item[2]
                for item in batch
            ]
            futures = [
                item[3]
                for item in batch
            ]

            try:
                results = (
                    self.engine.sense_batch(
                        requests
                    )
                )
            except (
                RuntimeError,
                ValueError,
                TypeError,
                IndexError,
                KeyError,
            ) as exc:
                for future in futures:
                    future.set_exception(
                        exc
                    )
                continue

            for future, result in zip(
                futures,
                results,
            ):
                future.set_result(
                    result
                )

    def metrics(self):
        return self.engine.metrics()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._queue.put(
            self._sentinel
        )
        self._thread.join(
            timeout=5.0
        )


# --- frozen notebook cell 146 ---
def sense_and_update(
    uav,
    belief_map,
    targets,
    rng,
    obstacles=None,
):
    coordinator = getattr(
        _get_torch_sensing_tls(),
        "coordinator",
        None,
    )
    if coordinator is None:
        return _CPU_SENSE_AND_UPDATE(
            uav,
            belief_map,
            targets,
            rng,
            obstacles=obstacles,
        )

    env_index = getattr(
        _get_torch_sensing_tls(),
        "env_index",
        None,
    )
    if env_index is None:
        raise RuntimeError(
            "torch sensing coordinator active "
            "without env_index"
        )

    call_index = int(
        getattr(
            _get_torch_sensing_tls(),
            "call_index",
            0,
        )
    )
    _get_torch_sensing_tls().call_index = (
        call_index + 1
    )

    return coordinator.submit(
        env_index=env_index,
        call_index=call_index,
        uav=uav,
        belief_map=belief_map,
        targets=targets,
        rng=rng,
        obstacles=obstacles,
    )


# --- frozen notebook cell 147 ---
def _zero_like_vector_value(value):
    if isinstance(
        value,
        np.ndarray,
    ):
        return np.zeros_like(
            value
        )
    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return False
    if isinstance(
        value,
        (int, np.integer),
    ):
        return 0
    if isinstance(
        value,
        (float, np.floating),
    ):
        return 0.0
    if isinstance(value, dict):
        return {
            key: _zero_like_vector_value(
                item
            )
            for key, item
            in value.items()
        }
    return None


# --- frozen notebook cell 148 ---
def _stack_vector_values(values):
    if all(
        isinstance(value, dict)
        for value in values
    ):
        return _stack_vector_dicts(
            values
        )

    if all(
        isinstance(value, np.ndarray)
        for value in values
    ):
        shapes = {
            value.shape
            for value in values
        }
        if len(shapes) == 1:
            return np.stack(
                values,
                axis=0,
            )

    if all(
        isinstance(
            value,
            (
                bool,
                np.bool_,
                int,
                np.integer,
                float,
                np.floating,
            ),
        )
        for value in values
    ):
        return np.asarray(
            values
        )

    result = np.empty(
        len(values),
        dtype=object,
    )
    for index, value in enumerate(
        values
    ):
        result[index] = (
            copy.deepcopy(
                value
            )
        )
    return result


# --- frozen notebook cell 149 ---
def _stack_vector_dicts(
    dictionaries,
):
    keys = sorted(
        {
            key
            for dictionary
            in dictionaries
            for key
            in dictionary
        },
        key=str,
    )
    result = {}

    for key in keys:
        present = np.asarray(
            [
                key in dictionary
                for dictionary
                in dictionaries
            ],
            dtype=np.bool_,
        )
        prototype = next(
            dictionary[key]
            for dictionary
            in dictionaries
            if key in dictionary
        )
        values = [
            (
                dictionary[key]
                if key in dictionary
                else _zero_like_vector_value(
                    prototype
                )
            )
            for dictionary
            in dictionaries
        ]
        result[key] = (
            _stack_vector_values(
                values
            )
        )
        if not np.all(present):
            result[
                f"_{key}"
            ] = present

    return result


# --- frozen notebook cell 150 ---
class TorchThreadedSimpleVectorEnv:
    """In-process vector env that batches sensing requests on one torch device."""

    def __init__(
        self,
        env_fns,
        *,
        sensing_device="auto",
        sensing_dtype="float32",
        sensing_batch_timeout_s=0.002,
    ):
        self.envs = [
            env_fn()
            for env_fn in env_fns
        ]
        if not self.envs:
            raise ValueError(
                "env_fns must not be empty"
            )

        self.num_envs = len(
            self.envs
        )
        self.coordinator = (
            TorchSensingCoordinator(
                self.num_envs,
                device=sensing_device,
                dtype=sensing_dtype,
                batch_timeout_s=(
                    sensing_batch_timeout_s
                ),
            )
        )
        self._executor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=self.num_envs,
                thread_name_prefix=(
                    "uav-env"
                ),
            )
        )
        self._driver_executor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=(
                    "uav-vector-driver"
                ),
            )
        )
        self._last_observations = [
            None
        ] * self.num_envs
        self._last_infos = [
            None
        ] * self.num_envs
        self._async_future = None
        self.backend_name = (
            "torch_threaded_simple"
        )

    @property
    def sensing_metrics(self):
        return self.coordinator.metrics()

    def _run_step(
        self,
        env_index,
        action,
    ):
        sensing_tls = _get_torch_sensing_tls()
        sensing_tls.coordinator = (
            self.coordinator
        )
        sensing_tls.env_index = int(
            env_index
        )
        sensing_tls.call_index = 0
        try:
            return self.envs[
                env_index
            ].step(action)
        finally:
            for name in (
                "coordinator",
                "env_index",
                "call_index",
            ):
                if hasattr(
                    sensing_tls,
                    name,
                ):
                    delattr(
                        sensing_tls,
                        name,
                    )

    def _action_at(
        self,
        vector_actions,
        index,
    ):
        result = {}
        for agent_id, action in (
            vector_actions.items()
        ):
            result[agent_id] = {
                "motion": np.asarray(
                    action["motion"][
                        index
                    ],
                    dtype=np.float32,
                ).copy(),
                "destination": int(
                    np.asarray(
                        action[
                            "destination"
                        ]
                    )[index]
                ),
                "power": np.asarray(
                    action["power"][
                        index
                    ],
                    dtype=np.float32,
                ).copy(),
            }
        return result

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        if options is None:
            options = {}

        reset_mask = options.get(
            "reset_mask"
        )
        if reset_mask is None:
            reset_mask = np.ones(
                self.num_envs,
                dtype=np.bool_,
            )
        else:
            reset_mask = np.asarray(
                reset_mask,
                dtype=np.bool_,
            )
            if reset_mask.shape != (
                self.num_envs,
            ):
                raise ValueError(
                    "reset_mask has invalid shape"
                )

        if seed is None:
            seeds = [
                None
            ] * self.num_envs
        elif isinstance(
            seed,
            (int, np.integer),
        ):
            seeds = [
                int(seed) + index
                for index in range(
                    self.num_envs
                )
            ]
        else:
            seeds = list(seed)
            if len(seeds) != self.num_envs:
                raise ValueError(
                    "seed sequence length must "
                    "match num_envs"
                )

        reset_infos = [
            {}
            for _ in range(
                self.num_envs
            )
        ]

        for index in range(
            self.num_envs
        ):
            if not reset_mask[index]:
                if (
                    self._last_observations[
                        index
                    ]
                    is None
                ):
                    raise RuntimeError(
                        "partial reset before "
                        "initial reset"
                    )
                continue

            observation, info = (
                self.envs[
                    index
                ].reset(
                    seed=seeds[index]
                )
            )
            self._last_observations[
                index
            ] = observation
            self._last_infos[
                index
            ] = info
            reset_infos[index] = info

        return (
            _stack_vector_dicts(
                self._last_observations
            ),
            _stack_vector_dicts(
                reset_infos
            ),
        )

    def step(
        self,
        vector_actions,
    ):
        futures = [
            self._executor.submit(
                self._run_step,
                index,
                self._action_at(
                    vector_actions,
                    index,
                ),
            )
            for index in range(
                self.num_envs
            )
        ]
        results = [
            future.result()
            for future in futures
        ]

        observations = [
            item[0]
            for item in results
        ]
        rewards = np.asarray(
            [
                item[1]
                for item in results
            ],
            dtype=np.float64,
        )
        terminated = np.asarray(
            [
                item[2]
                for item in results
            ],
            dtype=np.bool_,
        )
        truncated = np.asarray(
            [
                item[3]
                for item in results
            ],
            dtype=np.bool_,
        )
        infos = [
            item[4]
            for item in results
        ]

        self._last_observations = (
            observations
        )
        self._last_infos = infos

        return (
            _stack_vector_dicts(
                observations
            ),
            rewards,
            terminated,
            truncated,
            _stack_vector_dicts(
                infos
            ),
        )

    def step_async(
        self,
        vector_actions,
    ):
        if self._async_future is not None:
            raise RuntimeError(
                "step_async called twice"
            )
        self._async_future = (
            self._driver_executor.submit(
                self.step,
                vector_actions,
            )
        )

    def step_wait(self):
        if self._async_future is None:
            raise RuntimeError(
                "step_wait without step_async"
            )
        future = self._async_future
        self._async_future = None
        return future.result()

    def close(self):
        if self._async_future is not None:
            try:
                self._async_future.result()
            finally:
                self._async_future = None

        self.coordinator.close()
        self._executor.shutdown(
            wait=True
        )
        self._driver_executor.shutdown(
            wait=True
        )

        for env in self.envs:
            env.close()


# --- frozen notebook cell 151 ---
def resolve_training_vector_runtime(
    device,
):
    if device is None:
        resolved_device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        resolved_device = torch.device(
            device
        )

    backend = str(
        CONFIG.get(
            "training_vector_backend",
            "auto",
        )
    ).strip().lower()

    if backend not in {
        "auto",
        "async_cpu",
        "torch_threaded",
    }:
        raise ValueError(
            "training_vector_backend must be "
            "'auto', 'async_cpu', or "
            "'torch_threaded'"
        )

    torch_enabled = bool(
        CONFIG.get(
            "training_torch_sensing_enabled",
            True,
        )
    )

    if backend == "torch_threaded":
        use_torch_vector = True
    elif backend == "async_cpu":
        use_torch_vector = False
    else:
        use_torch_vector = bool(
            torch_enabled
            and resolved_device.type
            == "cuda"
        )

    # Worker count follows the learner device for either collector.
    if resolved_device.type == "cuda":
        num_envs = int(
            CONFIG.get(
                "training_cuda_num_envs",
                CONFIG.get(
                    "training_num_envs",
                    1,
                ),
            )
        )
    else:
        num_envs = int(
            CONFIG.get(
                "training_num_envs",
                1,
            )
        )

    num_envs = max(
        1,
        num_envs,
    )

    sensing_device = CONFIG.get(
        "training_torch_sensing_device",
        "auto",
    )
    if (
        str(
            sensing_device
        ).strip().lower()
        == "auto"
    ):
        sensing_device = str(
            resolved_device
        )

    if use_torch_vector:
        overlap = bool(
            CONFIG.get(
                "training_torch_overlap_env_and_updates",
                False,
            )
        )
    else:
        overlap = bool(
            CONFIG.get(
                "training_overlap_env_and_updates",
                True,
            )
        )

    return {
        "resolved_device": (
            resolved_device
        ),
        "backend": (
            "torch_threaded"
            if use_torch_vector
            else "async_cpu"
        ),
        "use_torch_vector": bool(
            use_torch_vector
        ),
        "num_envs": int(
            num_envs
        ),
        "overlap": bool(
            overlap
        ),
        "sensing_device": str(
            sensing_device
        ),
        "sensing_dtype": str(
            CONFIG.get(
                "training_torch_sensing_dtype",
                "float64",
            )
        ),
        "sensing_batch_timeout_s": float(
            CONFIG.get(
                "training_torch_sensing_batch_timeout_s",
                0.002,
            )
        ),
    }


# --- frozen notebook cell 152 ---
AGENT_OBSERVATION_VECTOR_KEYS = (
    "self_state",
    "gcs_relative",
    "neighbors",
    "obstacles",
    "belief_patch",
    "belief_coarse",
    "buffer",
    "destination_mask",
)


# --- frozen notebook cell 153 ---
def flatten_agent_observation(
    observation,
):
    if not isinstance(
        observation,
        dict,
    ):
        raise TypeError(
            "observation must be a dict"
        )

    if set(observation) != set(
        AGENT_OBSERVATION_VECTOR_KEYS
    ):
        raise ValueError(
            "observation keys do not match "
            "the MASAC observation contract"
        )

    parts = []

    for key in (
        AGENT_OBSERVATION_VECTOR_KEYS
    ):
        values = np.asarray(
            observation[key],
            dtype=np.float32,
        ).reshape(-1)

        if not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                f"{key} contains non-finite values"
            )

        parts.append(
            values
        )

    return np.concatenate(
        parts,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )


# --- frozen notebook cell 154 ---
def observations_to_masac_arrays(
    observations,
    agent_ids,
):
    if set(observations) != set(
        agent_ids
    ):
        raise ValueError(
            "observations must contain exactly "
            "the configured agent ids"
        )

    observation_vectors = np.stack(
        [
            flatten_agent_observation(
                observations[agent_id]
            )
            for agent_id
            in agent_ids
        ],
        axis=0,
    ).astype(
        np.float32
    )

    destination_masks = np.stack(
        [
            np.asarray(
                observations[
                    agent_id
                ][
                    "destination_mask"
                ],
                dtype=np.float32,
            )
            for agent_id
            in agent_ids
        ],
        axis=0,
    )

    if (
        destination_masks.ndim != 2
        or np.any(
            destination_masks.sum(
                axis=-1
            )
            < 1.0
        )
    ):
        raise ValueError(
            "every agent needs at least "
            "one valid destination"
        )

    return (
        observation_vectors,
        destination_masks,
    )


# --- frozen notebook cell 155 ---
def env_actions_to_masac_arrays(
    actions,
    agent_ids,
):
    continuous = []
    destination_indices = []

    for agent_id in agent_ids:
        action = actions[
            agent_id
        ]

        motion = np.asarray(
            project_motion_action_np(
                action["motion"]
            ),
            dtype=np.float32,
        )

        power = np.asarray(
            action["power"],
            dtype=np.float32,
        ).reshape(-1)

        if (
            motion.shape != (3,)
            or power.shape != (1,)
        ):
            raise ValueError(
                "invalid hybrid action shape"
            )

        continuous.append(
            np.concatenate(
                [
                    motion,
                    power,
                ],
                axis=0,
            )
        )
        destination_indices.append(
            int(
                action["destination"]
            )
        )

    return (
        np.stack(
            continuous,
            axis=0,
        ).astype(
            np.float32
        ),
        np.asarray(
            destination_indices,
            dtype=np.int64,
        ),
    )


# --- frozen notebook cell 156 ---
def masac_arrays_to_env_actions(
    continuous_actions,
    destination_indices,
    agent_ids,
):
    continuous_actions = np.asarray(
        continuous_actions,
        dtype=np.float32,
    )
    destination_indices = np.asarray(
        destination_indices,
        dtype=np.int64,
    )

    if continuous_actions.shape != (
        len(agent_ids),
        4,
    ):
        raise ValueError(
            "continuous_actions must have "
            "shape (num_agents, 4)"
        )

    if destination_indices.shape != (
        len(agent_ids),
    ):
        raise ValueError(
            "destination_indices must have "
            "shape (num_agents,)"
        )

    motion_actions = np.asarray(
        project_motion_action_np(
            continuous_actions[:, :3]
        ),
        dtype=np.float32,
    )

    return {
        agent_id: {
            "motion": (
                motion_actions[
                    index
                ].copy()
            ),
            "destination": int(
                destination_indices[
                    index
                ]
            ),
            "power": np.asarray(
                [
                    continuous_actions[
                        index,
                        3,
                    ]
                ],
                dtype=np.float32,
            ),
        }
        for index, agent_id
        in enumerate(agent_ids)
    }


# --- frozen notebook cell 157 ---
def sample_uniform_unit_ball_motion(rng):
    direction = rng.normal(
        size=3
    )
    direction_norm = max(
        float(
            np.linalg.norm(direction)
        ),
        float(
            np.finfo(np.float64).eps
        ),
    )
    radius = float(
        rng.random() ** (1.0 / 3.0)
    )

    return (
        direction
        / direction_norm
        * radius
    ).astype(
        np.float32
    )


# --- frozen notebook cell 158 ---
def sample_valid_random_actions(
    observations,
    agent_ids,
    rng,
):
    actions = {}

    for agent_id in agent_ids:
        mask = np.asarray(
            observations[
                agent_id
            ][
                "destination_mask"
            ],
            dtype=np.int8,
        )

        valid = np.flatnonzero(
            mask
        )

        if valid.size == 0:
            raise RuntimeError(
                "destination mask has "
                "no valid action"
            )

        actions[
            agent_id
        ] = {
            "motion": (
                sample_uniform_unit_ball_motion(
                    rng
                )
            ),
            "destination": int(
                rng.choice(
                    valid
                )
            ),
            "power": rng.uniform(
                -1.0,
                1.0,
                size=1,
            ).astype(
                np.float32
            ),
        }

    return actions


# --- frozen notebook cell 159 ---
def build_mlp(
    input_dim,
    hidden_dims,
    output_dim,
    activation_factory=nn.ReLU,
):
    dims = (
        int(input_dim),
        *tuple(
            int(value)
            for value
            in hidden_dims
        ),
        int(output_dim),
    )

    layers = []

    for index in range(
        len(dims) - 2
    ):
        layers.extend(
            [
                nn.Linear(
                    dims[index],
                    dims[index + 1],
                ),
                activation_factory(),
            ]
        )

    layers.append(
        nn.Linear(
            dims[-2],
            dims[-1],
        )
    )

    return nn.Sequential(
        *layers
    )


# --- frozen notebook cell 160 ---
def masked_categorical_logits(
    logits,
    destination_mask,
):
    if logits.shape != (
        destination_mask.shape
    ):
        raise ValueError(
            "logits and destination_mask "
            "must have the same shape"
        )

    mask = (
        destination_mask > 0.5
    )

    if torch.any(
        mask.sum(
            dim=-1
        )
        < 1
    ):
        raise ValueError(
            "each categorical row needs "
            "at least one valid action"
        )

    negative_large = torch.finfo(
        logits.dtype
    ).min

    return logits.masked_fill(
        ~mask,
        negative_large,
    )


# --- frozen notebook cell 161 ---
def straight_through_categorical_sample(
    logits,
    destination_mask,
    temperature,
    deterministic=False,
):
    temperature = float(
        temperature
    )

    if temperature <= 0.0:
        raise ValueError(
            "temperature must be > 0"
        )

    masked_logits = (
        masked_categorical_logits(
            logits,
            destination_mask,
        )
    )

    log_probabilities = (
        F.log_softmax(
            masked_logits,
            dim=-1,
        )
    )
    probabilities = torch.exp(
        log_probabilities
    )

    if deterministic:
        indices = torch.argmax(
            masked_logits,
            dim=-1,
        )
        hard = F.one_hot(
            indices,
            num_classes=(
                masked_logits.shape[-1]
            ),
        ).to(
            masked_logits.dtype
        )
        action = hard
    else:
        uniform = torch.rand_like(
            masked_logits
        ).clamp_(
            1e-6,
            1.0 - 1e-6,
        )

        gumbel_noise = -torch.log(
            -torch.log(
                uniform
            )
        )

        soft = F.softmax(
            (
                masked_logits
                + gumbel_noise
            )
            / temperature,
            dim=-1,
        )

        indices = torch.argmax(
            soft,
            dim=-1,
        )
        hard = F.one_hot(
            indices,
            num_classes=(
                masked_logits.shape[-1]
            ),
        ).to(
            masked_logits.dtype
        )

        # Straight-through estimator:
        # forward value is hard one-hot,
        # backward gradient follows soft.
        action = (
            hard
            - soft.detach()
            + soft
        )

    selected_log_probability = (
        hard
        * log_probabilities
    ).sum(
        dim=-1,
        keepdim=True,
    )

    entropy = -(
        probabilities
        * log_probabilities
    ).sum(
        dim=-1,
        keepdim=True,
    )

    return {
        "action": action,
        "hard_action": hard,
        "indices": indices,
        "log_probability": (
            selected_log_probability
        ),
        "entropy": entropy,
        "probabilities": probabilities,
        "log_probabilities": (
            log_probabilities
        ),
    }


# --- frozen notebook cell 162 ---
def radial_squash_motion_action(
    pre_squash,
    eps=1e-6,
):
    """Invertibly squash R^3 into the open 3-D unit ball."""
    if pre_squash.shape[-1] != 3:
        raise ValueError(
            "pre_squash must have trailing dimension 3"
        )

    radius = torch.linalg.vector_norm(
        pre_squash,
        dim=-1,
        keepdim=True,
    )
    squashed_radius = torch.tanh(
        radius
    )
    small = radius <= eps
    scale_regular = (
        squashed_radius
        / radius.clamp_min(eps)
    )
    scale_series = (
        1.0
        - radius.pow(2) / 3.0
    )
    scale = torch.where(
        small,
        scale_series,
        scale_regular,
    )
    action = pre_squash * scale

    radial_derivative = (
        1.0
        - squashed_radius.pow(2)
    ).clamp_min(eps)
    tangential_scale = scale.clamp_min(eps)
    log_abs_det_jacobian = (
        torch.log(radial_derivative)
        + 2.0
        * torch.log(tangential_scale)
    )

    return action, log_abs_det_jacobian


# --- frozen notebook cell 163 ---
def project_motion_action_tensor(action):
    """Project already-bounded 3-D actions onto the closed unit ball."""
    if action.shape[-1] != 3:
        raise ValueError(
            "motion action tensor must have trailing dimension 3"
        )

    norm = torch.linalg.vector_norm(
        action,
        dim=-1,
        keepdim=True,
    )
    scale = torch.where(
        norm > 1.0,
        1.0 / norm.clamp_min(1e-12),
        torch.ones_like(norm),
    )
    return action * scale


# Export public and internal helpers required by downstream production layers.
__all__ = [name for name in globals() if not name.startswith("__")]
