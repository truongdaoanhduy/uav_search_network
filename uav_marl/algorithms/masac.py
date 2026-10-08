"""Production source migrated from frozen fix_test_gpu.ipynb.

Do not import the notebook at runtime. The notebook is retained only as a
reference/regression artifact. Production changes belong in Python modules.
Migrated from notebook cells 164..187.
"""

from ..envs.uav_search import *  # noqa: F401,F403


def transform_actor_observations(
    observations,
    discrete_dim,
):
    """Center belief features around the uninformative mission prior."""
    num_uavs = int(discrete_dim) - 1
    prefix = (
        9
        + 3
        + 5 * (num_uavs - 1)
        + 5 * int(
            CONFIG["observation_nearest_obstacles"]
        )
    )
    patch_count = int(
        CONFIG["belief_patch_cells"]
    ) ** 2
    coarse_count = int(
        CONFIG["belief_coarse_cells"]
    ) ** 2
    patch_end = prefix + patch_count
    coarse_end = patch_end + coarse_count
    expected_dim = (
        coarse_end
        + 5
        + int(discrete_dim)
    )

    if observations.shape[-1] != expected_dim:
        raise ValueError(
            "actor observation dimension does not match "
            "the configured UAV observation contract"
        )

    belief_prior = float(
        CONFIG["belief_prior"]
    )
    return torch.cat(
        (
            observations[..., :prefix],
            2.0
            * (
                observations[
                    ..., prefix:patch_end
                ]
                - belief_prior
            ),
            1.0
            - observations[
                ..., patch_end:coarse_end
            ],
            observations[..., coarse_end:],
        ),
        dim=-1,
    )


def build_actor_encoder(
    observation_dim,
    hidden_dims,
):
    return build_mlp(
        observation_dim,
        hidden_dims[:-1],
        hidden_dims[-1],
        activation_factory=(
            lambda: nn.LeakyReLU(
                negative_slope=0.01
            )
        ),
    )


def clip_grad_norm_finite(
    parameters,
    max_norm,
    label,
):
    """Clip gradients and fail before an optimizer can apply NaN/Inf values."""
    try:
        return torch.nn.utils.clip_grad_norm_(
            list(parameters),
            max_norm,
            error_if_nonfinite=True,
        )
    except RuntimeError as exc:
        if "non-finite" not in str(exc):
            raise
        raise FloatingPointError(
            f"non-finite {label} gradient"
        ) from exc


# --- frozen notebook cell 164 ---
class HybridMASACActor(nn.Module):
    def __init__(
        self,
        observation_dim,
        continuous_dim,
        discrete_dim,
        hidden_dims,
    ):
        super().__init__()

        hidden_dims = tuple(
            int(value)
            for value
            in hidden_dims
        )

        if not hidden_dims:
            raise ValueError(
                "hidden_dims must not be empty"
            )

        self.observation_dim = int(
            observation_dim
        )
        self.continuous_dim = int(
            continuous_dim
        )
        self.discrete_dim = int(
            discrete_dim
        )

        self.encoder = build_actor_encoder(
            self.observation_dim,
            hidden_dims,
        )

        feature_dim = (
            hidden_dims[-1]
        )

        self.mean_head = nn.Linear(
            feature_dim,
            self.continuous_dim,
        )
        self.log_std_head = nn.Linear(
            feature_dim,
            self.continuous_dim,
        )
        self.discrete_head = nn.Linear(
            feature_dim,
            self.discrete_dim,
        )

    def forward(
        self,
        observations,
    ):
        features = self.encoder(
            transform_actor_observations(
                observations,
                self.discrete_dim,
            )
        )

        mean = self.mean_head(
            features
        )

        log_std = (
            self.log_std_head(
                features
            ).clamp(
                min=float(
                    CONFIG[
                        "masac_log_std_min"
                    ]
                ),
                max=float(
                    CONFIG[
                        "masac_log_std_max"
                    ]
                ),
            )
        )

        logits = self.discrete_head(
            features
        )

        return (
            mean,
            log_std,
            logits,
        )

    def sample(
        self,
        observations,
        destination_mask,
        deterministic=False,
    ):
        (
            mean,
            log_std,
            logits,
        ) = self(
            observations
        )

        std = torch.exp(
            log_std
        )
        distribution = Normal(
            mean,
            std,
        )

        if deterministic:
            pre_tanh = mean
        else:
            pre_tanh = (
                distribution.rsample()
            )

        motion_action, motion_log_det = (
            radial_squash_motion_action(
                pre_tanh[..., :3]
            )
        )
        power_action = torch.tanh(
            pre_tanh[..., 3:4]
        )
        continuous_action = torch.cat(
            [
                motion_action,
                power_action,
            ],
            dim=-1,
        )

        power_log_det = torch.distributions.transforms.TanhTransform().log_abs_det_jacobian(
            pre_tanh[..., 3:4], power_action
        ).sum(
            dim=-1,
            keepdim=True,
        )
        log_probability = (
            distribution.log_prob(
                pre_tanh
            ).sum(
                dim=-1,
                keepdim=True,
            )
            - motion_log_det
            - power_log_det
        )

        discrete = (
            straight_through_categorical_sample(
                logits,
                destination_mask,
                temperature=CONFIG[
                    "masac_gumbel_temperature"
                ],
                deterministic=deterministic,
            )
        )

        return {
            "continuous": (
                continuous_action
            ),
            "continuous_log_probability": (
                log_probability
            ),
            "destination_one_hot": (
                discrete["action"]
            ),
            "destination_hard_one_hot": (
                discrete[
                    "hard_action"
                ]
            ),
            "destination_index": (
                discrete["indices"]
            ),
            "destination_log_probability": (
                discrete[
                    "log_probability"
                ]
            ),
            "destination_entropy": (
                discrete["entropy"]
            ),
        }


# --- frozen notebook cell 165 ---
class CentralizedQNetwork(nn.Module):
    def __init__(
        self,
        state_dim,
        joint_action_dim,
        hidden_dims,
    ):
        super().__init__()

        self.state_dim = int(
            state_dim
        )
        self.joint_action_dim = int(
            joint_action_dim
        )

        self.network = build_mlp(
            self.state_dim
            + self.joint_action_dim,
            hidden_dims,
            1,
        )

    def forward(
        self,
        state,
        joint_action,
    ):
        if state.ndim != 2:
            raise ValueError(
                "state must be rank 2"
            )

        if joint_action.ndim != 2:
            raise ValueError(
                "joint_action must be rank 2"
            )

        values = torch.cat(
            [
                state,
                joint_action,
            ],
            dim=-1,
        )

        return self.network(
            values
        )


# --- frozen notebook cell 166 ---
class HybridReplayBuffer:
    def __init__(
        self,
        capacity,
        num_agents,
        observation_dim,
        state_dim,
        continuous_dim,
        discrete_dim,
        seed,
    ):
        self.capacity = int(
            capacity
        )
        self.num_agents = int(
            num_agents
        )
        self.observation_dim = int(
            observation_dim
        )
        self.state_dim = int(
            state_dim
        )
        self.continuous_dim = int(
            continuous_dim
        )
        self.discrete_dim = int(
            discrete_dim
        )

        if self.capacity < 1:
            raise ValueError(
                "capacity must be >= 1"
            )

        self.rng = (
            np.random.default_rng(
                int(seed)
            )
        )

        self.observations = np.zeros(
            (
                self.capacity,
                self.num_agents,
                self.observation_dim,
            ),
            dtype=np.float32,
        )
        self.next_observations = (
            np.zeros_like(
                self.observations
            )
        )
        self.states = np.zeros(
            (
                self.capacity,
                self.state_dim,
            ),
            dtype=np.float32,
        )
        self.next_states = (
            np.zeros_like(
                self.states
            )
        )
        self.continuous_actions = np.zeros(
            (
                self.capacity,
                self.num_agents,
                self.continuous_dim,
            ),
            dtype=np.float32,
        )
        self.destination_indices = np.zeros(
            (
                self.capacity,
                self.num_agents,
            ),
            dtype=np.int64,
        )
        self.destination_masks = np.zeros(
            (
                self.capacity,
                self.num_agents,
                self.discrete_dim,
            ),
            dtype=np.float32,
        )
        self.next_destination_masks = (
            np.zeros_like(
                self.destination_masks
            )
        )
        self.rewards = np.zeros(
            (
                self.capacity,
                1,
            ),
            dtype=np.float32,
        )
        self.terminated = np.zeros(
            (
                self.capacity,
                1,
            ),
            dtype=np.float32,
        )
        self.truncated = np.zeros(
            (
                self.capacity,
                1,
            ),
            dtype=np.float32,
        )

        self.position = 0
        self.size = 0

    def __len__(self):
        return int(
            self.size
        )

    def add(
        self,
        observations,
        state,
        continuous_actions,
        destination_indices,
        destination_masks,
        reward,
        next_observations,
        next_state,
        next_destination_masks,
        terminated,
        truncated,
    ):
        observations = np.asarray(
            observations,
            dtype=np.float32,
        )
        next_observations = np.asarray(
            next_observations,
            dtype=np.float32,
        )
        state = np.asarray(
            state,
            dtype=np.float32,
        )
        next_state = np.asarray(
            next_state,
            dtype=np.float32,
        )
        continuous_actions = np.asarray(
            continuous_actions,
            dtype=np.float32,
        )
        destination_indices = np.asarray(
            destination_indices,
            dtype=np.int64,
        )
        destination_masks = np.asarray(
            destination_masks,
            dtype=np.float32,
        )
        next_destination_masks = np.asarray(
            next_destination_masks,
            dtype=np.float32,
        )

        expected_observation_shape = (
            self.num_agents,
            self.observation_dim,
        )
        expected_action_shape = (
            self.num_agents,
            self.continuous_dim,
        )
        expected_mask_shape = (
            self.num_agents,
            self.discrete_dim,
        )

        if observations.shape != (
            expected_observation_shape
        ):
            raise ValueError(
                "observations have wrong shape"
            )

        if next_observations.shape != (
            expected_observation_shape
        ):
            raise ValueError(
                "next_observations have wrong shape"
            )

        if state.shape != (
            self.state_dim,
        ):
            raise ValueError(
                "state has wrong shape"
            )

        if next_state.shape != (
            self.state_dim,
        ):
            raise ValueError(
                "next_state has wrong shape"
            )

        if continuous_actions.shape != (
            expected_action_shape
        ):
            raise ValueError(
                "continuous_actions have wrong shape"
            )

        if destination_indices.shape != (
            self.num_agents,
        ):
            raise ValueError(
                "destination_indices have wrong shape"
            )

        if destination_masks.shape != (
            expected_mask_shape
        ):
            raise ValueError(
                "destination_masks have wrong shape"
            )

        if next_destination_masks.shape != (
            expected_mask_shape
        ):
            raise ValueError(
                "next_destination_masks have wrong shape"
            )

        if np.any(
            destination_indices < 0
        ) or np.any(
            destination_indices
            >= self.discrete_dim
        ):
            raise ValueError(
                "destination index out of range"
            )

        slot = int(
            self.position
        )

        self.observations[
            slot
        ] = observations
        self.next_observations[
            slot
        ] = next_observations
        self.states[
            slot
        ] = state
        self.next_states[
            slot
        ] = next_state
        self.continuous_actions[
            slot
        ] = continuous_actions
        self.destination_indices[
            slot
        ] = destination_indices
        self.destination_masks[
            slot
        ] = destination_masks
        self.next_destination_masks[
            slot
        ] = next_destination_masks
        self.rewards[
            slot,
            0,
        ] = float(
            reward
        )
        self.terminated[
            slot,
            0,
        ] = float(
            bool(
                terminated
            )
        )
        self.truncated[
            slot,
            0,
        ] = float(
            bool(
                truncated
            )
        )

        self.position = (
            slot + 1
        ) % self.capacity
        self.size = min(
            self.size + 1,
            self.capacity,
        )

    def state_dict(self):
        size = int(
            self.size
        )

        def packed_tensor(
            array,
        ):
            return torch.from_numpy(
                array[:size].copy()
            )

        return {
            "format_version": 1,
            "capacity": int(
                self.capacity
            ),
            "num_agents": int(
                self.num_agents
            ),
            "observation_dim": int(
                self.observation_dim
            ),
            "state_dim": int(
                self.state_dim
            ),
            "continuous_dim": int(
                self.continuous_dim
            ),
            "discrete_dim": int(
                self.discrete_dim
            ),
            "position": int(
                self.position
            ),
            "size": size,
            "rng_state_json": json.dumps(
                self.rng
                .bit_generator
                .state
            ),
            "observations": packed_tensor(
                self.observations
            ),
            "next_observations": (
                packed_tensor(
                    self.next_observations
                )
            ),
            "states": packed_tensor(
                self.states
            ),
            "next_states": packed_tensor(
                self.next_states
            ),
            "continuous_actions": (
                packed_tensor(
                    self.continuous_actions
                )
            ),
            "destination_indices": (
                packed_tensor(
                    self.destination_indices
                )
            ),
            "destination_masks": (
                packed_tensor(
                    self.destination_masks
                )
            ),
            "next_destination_masks": (
                packed_tensor(
                    self.next_destination_masks
                )
            ),
            "rewards": packed_tensor(
                self.rewards
            ),
            "terminated": packed_tensor(
                self.terminated
            ),
            "truncated": packed_tensor(
                self.truncated
            ),
        }

    def load_state_dict(
        self,
        state,
    ):
        if not isinstance(
            state,
            dict,
        ):
            raise TypeError(
                "replay state must be a dict"
            )

        expected = {
            "capacity": self.capacity,
            "num_agents": self.num_agents,
            "observation_dim": (
                self.observation_dim
            ),
            "state_dim": self.state_dim,
            "continuous_dim": (
                self.continuous_dim
            ),
            "discrete_dim": (
                self.discrete_dim
            ),
        }

        for key, expected_value in (
            expected.items()
        ):
            actual = int(
                state[key]
            )

            if actual != int(
                expected_value
            ):
                raise ValueError(
                    f"replay {key} mismatch: "
                    f"{actual} != "
                    f"{expected_value}"
                )

        size = int(
            state["size"]
        )
        position = int(
            state["position"]
        )

        if not 0 <= size <= self.capacity:
            raise ValueError(
                "invalid replay size"
            )

        if not 0 <= position < self.capacity:
            raise ValueError(
                "invalid replay position"
            )

        if (
            size < self.capacity
            and position != size
        ):
            raise ValueError(
                "partial replay position "
                "must equal size"
            )

        tensor_keys = (
            "observations",
            "next_observations",
            "states",
            "next_states",
            "continuous_actions",
            "destination_indices",
            "destination_masks",
            "next_destination_masks",
            "rewards",
            "terminated",
            "truncated",
        )

        destination_arrays = {
            "observations": (
                self.observations
            ),
            "next_observations": (
                self.next_observations
            ),
            "states": self.states,
            "next_states": (
                self.next_states
            ),
            "continuous_actions": (
                self.continuous_actions
            ),
            "destination_indices": (
                self.destination_indices
            ),
            "destination_masks": (
                self.destination_masks
            ),
            "next_destination_masks": (
                self.next_destination_masks
            ),
            "rewards": self.rewards,
            "terminated": (
                self.terminated
            ),
            "truncated": (
                self.truncated
            ),
        }

        for key in tensor_keys:
            destination = (
                destination_arrays[
                    key
                ]
            )
            destination.fill(
                0
            )

            source_tensor = (
                state[key]
            )

            if not isinstance(
                source_tensor,
                torch.Tensor,
            ):
                raise TypeError(
                    f"replay {key} must "
                    "be a tensor"
                )

            source = (
                source_tensor
                .detach()
                .cpu()
                .numpy()
            )

            expected_shape = (
                destination[:size]
                .shape
            )

            if source.shape != (
                expected_shape
            ):
                raise ValueError(
                    f"replay {key} has "
                    "wrong shape"
                )

            destination[
                :size
            ] = source.astype(
                destination.dtype,
                copy=False,
            )

        self.position = position
        self.size = size

        self.rng.bit_generator.state = (
            json.loads(
                state[
                    "rng_state_json"
                ]
            )
        )

    def sample(
        self,
        batch_size,
        device,
    ):
        batch_size = int(
            batch_size
        )

        if batch_size < 1:
            raise ValueError(
                "batch_size must be >= 1"
            )

        if self.size < batch_size:
            raise ValueError(
                "not enough replay samples"
            )

        indices = self.rng.integers(
            0,
            self.size,
            size=batch_size,
        )

        def tensor(
            array,
            dtype=torch.float32,
        ):
            return torch.as_tensor(
                array[indices],
                dtype=dtype,
                device=device,
            )

        return {
            "observations": tensor(
                self.observations
            ),
            "next_observations": tensor(
                self.next_observations
            ),
            "states": tensor(
                self.states
            ),
            "next_states": tensor(
                self.next_states
            ),
            "continuous_actions": tensor(
                self.continuous_actions
            ),
            "destination_indices": tensor(
                self.destination_indices,
                dtype=torch.long,
            ),
            "destination_masks": tensor(
                self.destination_masks
            ),
            "next_destination_masks": tensor(
                self.next_destination_masks
            ),
            "rewards": tensor(
                self.rewards
            ),
            "terminated": tensor(
                self.terminated
            ),
            "truncated": tensor(
                self.truncated
            ),
        }


# --- frozen notebook cell 167 ---
class HybridMASAC:
    def __init__(
        self,
        env,
        device=None,
    ):
        if not isinstance(
            env,
            UAVSearchEnv,
        ):
            raise TypeError(
                "env must be UAVSearchEnv"
            )

        self.agent_ids = tuple(
            env.agent_ids
        )
        self.num_agents = len(
            self.agent_ids
        )

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(
            device
        )

        agent_space = (
            env.observation_space
            .spaces[
                self.agent_ids[0]
            ]
        )

        self.observation_dim = int(
            sum(
                np.prod(
                    agent_space.spaces[
                        key
                    ].shape,
                    dtype=np.int64,
                )
                for key
                in AGENT_OBSERVATION_VECTOR_KEYS
            )
        )
        self.state_dim = int(
            np.prod(
                env.state_space.shape,
                dtype=np.int64,
            )
        )
        self.continuous_dim = 4
        self.discrete_dim = int(
            env.action_space
            .spaces[
                self.agent_ids[0]
            ]
            .spaces[
                "destination"
            ]
            .n
        )

        self.joint_action_dim = (
            self.num_agents
            * (
                self.continuous_dim
                + self.discrete_dim
            )
        )

        hidden_dims = tuple(
            CONFIG[
                "masac_hidden_dims"
            ]
        )

        self.actor = (
            HybridMASACActor(
                self.observation_dim,
                self.continuous_dim,
                self.discrete_dim,
                hidden_dims,
            ).to(
                self.device
            )
        )

        self.critic_1 = (
            CentralizedQNetwork(
                self.state_dim,
                self.joint_action_dim,
                hidden_dims,
            ).to(
                self.device
            )
        )
        self.critic_2 = (
            CentralizedQNetwork(
                self.state_dim,
                self.joint_action_dim,
                hidden_dims,
            ).to(
                self.device
            )
        )

        self.target_critic_1 = (
            copy.deepcopy(
                self.critic_1
            ).to(
                self.device
            )
        )
        self.target_critic_2 = (
            copy.deepcopy(
                self.critic_2
            ).to(
                self.device
            )
        )

        for network in (
            self.target_critic_1,
            self.target_critic_2,
        ):
            network.eval()

            for parameter in (
                network.parameters()
            ):
                parameter.requires_grad_(
                    False
                )

        self.actor_optimizer = (
            torch.optim.Adam(
                self.actor.parameters(),
                lr=float(
                    CONFIG[
                        "masac_actor_lr"
                    ]
                ),
            )
        )

        self.critic_optimizer = (
            torch.optim.Adam(
                list(
                    self.critic_1.parameters()
                )
                + list(
                    self.critic_2.parameters()
                ),
                lr=float(
                    CONFIG[
                        "masac_critic_lr"
                    ]
                ),
            )
        )

        initial_alpha_continuous = float(
            CONFIG[
                "masac_initial_alpha_continuous"
            ]
        )
        initial_alpha_discrete = float(
            CONFIG[
                "masac_initial_alpha_discrete"
            ]
        )

        if (
            initial_alpha_continuous <= 0.0
            or initial_alpha_discrete <= 0.0
        ):
            raise ValueError(
                "initial alpha values must be > 0"
            )

        self.log_alpha_continuous = (
            torch.tensor(
                np.log(
                    initial_alpha_continuous
                ),
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            )
        )
        self.log_alpha_discrete = (
            torch.tensor(
                np.log(
                    initial_alpha_discrete
                ),
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            )
        )

        self.alpha_continuous_optimizer = (
            torch.optim.Adam(
                [
                    self.log_alpha_continuous
                ],
                lr=float(
                    CONFIG[
                        "masac_alpha_lr"
                    ]
                ),
            )
        )
        self.alpha_discrete_optimizer = (
            torch.optim.Adam(
                [
                    self.log_alpha_discrete
                ],
                lr=float(
                    CONFIG[
                        "masac_alpha_lr"
                    ]
                ),
            )
        )

        self.gamma = float(
            CONFIG["masac_gamma"]
        )
        self.tau = float(
            CONFIG["masac_tau"]
        )
        self.gradient_clip_norm = float(
            CONFIG[
                "masac_gradient_clip_norm"
            ]
        )

        self.continuous_target_entropy = (
            -float(
                CONFIG[
                    "masac_continuous_target_entropy_scale"
                ]
            )
            * self.num_agents
            * self.continuous_dim
        )

        self.discrete_target_entropy_ratio = float(
            CONFIG[
                "masac_discrete_target_entropy_ratio"
            ]
        )

        if not (
            0.0
            <= self.discrete_target_entropy_ratio
            <= 1.0
        ):
            raise ValueError(
                "discrete target entropy ratio "
                "must be within [0, 1]"
            )

        self.update_count = 0
        amp_dtype_name = str(
            CONFIG.get(
                "training_amp_dtype",
                "float16",
            )
        ).strip().lower()
        if amp_dtype_name == "float16":
            self.amp_dtype = torch.float16
        elif amp_dtype_name == "bfloat16":
            self.amp_dtype = torch.bfloat16
        else:
            raise ValueError(
                "training_amp_dtype must be float16 or bfloat16"
            )
        self.amp_enabled = (
            bool(
                CONFIG.get(
                    "training_amp_enabled",
                    False,
                )
            )
            and self.device.type == "cuda"
        )
        self.grad_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.amp_enabled,
        )

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_enabled,
        )

    @property
    def alpha_continuous(self):
        return torch.exp(
            self.log_alpha_continuous
        )

    @property
    def alpha_discrete(self):
        return torch.exp(
            self.log_alpha_discrete
        )

    def make_replay_buffer(
        self,
        seed=None,
    ):
        if seed is None:
            seed = int(
                CONFIG["seed"]
            )

        return HybridReplayBuffer(
            capacity=CONFIG[
                "masac_replay_capacity"
            ],
            num_agents=(
                self.num_agents
            ),
            observation_dim=(
                self.observation_dim
            ),
            state_dim=(
                self.state_dim
            ),
            continuous_dim=(
                self.continuous_dim
            ),
            discrete_dim=(
                self.discrete_dim
            ),
            seed=seed,
        )

    def _joint_action_tensor(
        self,
        continuous_actions,
        destination_one_hot,
    ):
        if continuous_actions.ndim != 3:
            raise ValueError(
                "continuous_actions must "
                "have rank 3"
            )

        if destination_one_hot.ndim != 3:
            raise ValueError(
                "destination_one_hot must "
                "have rank 3"
            )

        combined = torch.cat(
            [
                continuous_actions,
                destination_one_hot,
            ],
            dim=-1,
        )

        return combined.reshape(
            combined.shape[0],
            -1,
        )

    def _sample_joint_policy(
        self,
        observations,
        destination_masks,
        deterministic=False,
    ):
        if observations.ndim != 3:
            raise ValueError(
                "observations must have rank 3"
            )

        batch_size = (
            observations.shape[0]
        )

        flat_observations = (
            observations.reshape(
                batch_size
                * self.num_agents,
                self.observation_dim,
            )
        )
        flat_masks = (
            destination_masks.reshape(
                batch_size
                * self.num_agents,
                self.discrete_dim,
            )
        )

        sampled = self.actor.sample(
            flat_observations,
            flat_masks,
            deterministic=deterministic,
        )

        continuous = sampled[
            "continuous"
        ].reshape(
            batch_size,
            self.num_agents,
            self.continuous_dim,
        )
        destination_one_hot = (
            sampled[
                "destination_one_hot"
            ].reshape(
                batch_size,
                self.num_agents,
                self.discrete_dim,
            )
        )
        destination_indices = (
            sampled[
                "destination_index"
            ].reshape(
                batch_size,
                self.num_agents,
            )
        )

        continuous_log_probability = (
            sampled[
                "continuous_log_probability"
            ].reshape(
                batch_size,
                self.num_agents,
                1,
            ).sum(
                dim=1
            )
        )
        destination_log_probability = (
            sampled[
                "destination_log_probability"
            ].reshape(
                batch_size,
                self.num_agents,
                1,
            ).sum(
                dim=1
            )
        )
        destination_entropy = (
            sampled[
                "destination_entropy"
            ].reshape(
                batch_size,
                self.num_agents,
                1,
            ).sum(
                dim=1
            )
        )

        return {
            "continuous": continuous,
            "destination_one_hot": (
                destination_one_hot
            ),
            "destination_indices": (
                destination_indices
            ),
            "continuous_log_probability": (
                continuous_log_probability
            ),
            "destination_log_probability": (
                destination_log_probability
            ),
            "destination_entropy": (
                destination_entropy
            ),
        }

    def select_actions(
        self,
        observations,
        deterministic=False,
    ):
        (
            observation_vectors,
            destination_masks,
        ) = observations_to_masac_arrays(
            observations,
            self.agent_ids,
        )

        observation_tensor = (
            torch.as_tensor(
                observation_vectors,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(
                0
            )
        )
        mask_tensor = torch.as_tensor(
            destination_masks,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(
            0
        )

        with torch.no_grad():
            sampled = (
                self._sample_joint_policy(
                    observation_tensor,
                    mask_tensor,
                    deterministic=(
                        deterministic
                    ),
                )
            )

        continuous = (
            sampled["continuous"][
                0
            ]
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )
        destination_indices = (
            sampled[
                "destination_indices"
            ][
                0
            ]
            .cpu()
            .numpy()
            .astype(
                np.int64
            )
        )

        return masac_arrays_to_env_actions(
            continuous,
            destination_indices,
            self.agent_ids,
        )

    def _soft_update_targets(
        self,
    ):
        with torch.no_grad():
            for target, source in (
                (
                    self.target_critic_1,
                    self.critic_1,
                ),
                (
                    self.target_critic_2,
                    self.critic_2,
                ),
            ):
                for (
                    target_parameter,
                    source_parameter,
                ) in zip(
                    target.parameters(),
                    source.parameters(),
                ):
                    target_parameter.mul_(
                        1.0 - self.tau
                    )
                    target_parameter.add_(
                        self.tau
                        * source_parameter
                    )

    def update(
        self,
        replay_buffer,
        batch_size=None,
        collect_metrics=True,
    ):
        if not isinstance(
            replay_buffer,
            HybridReplayBuffer,
        ):
            raise TypeError(
                "replay_buffer must be "
                "HybridReplayBuffer"
            )

        if batch_size is None:
            batch_size = int(
                CONFIG[
                    "masac_batch_size"
                ]
            )

        batch = replay_buffer.sample(
            batch_size,
            self.device,
        )

        destination_one_hot = (
            F.one_hot(
                batch[
                    "destination_indices"
                ],
                num_classes=(
                    self.discrete_dim
                ),
            ).to(
                torch.float32
            )
        )

        replay_joint_action = (
            self._joint_action_tensor(
                batch[
                    "continuous_actions"
                ],
                destination_one_hot,
            )
        )

        with torch.no_grad(), self._autocast():
            next_policy = (
                self._sample_joint_policy(
                    batch[
                        "next_observations"
                    ],
                    batch[
                        "next_destination_masks"
                    ],
                    deterministic=False,
                )
            )

            next_joint_action = (
                self._joint_action_tensor(
                    next_policy[
                        "continuous"
                    ],
                    next_policy[
                        "destination_one_hot"
                    ],
                )
            )

            target_q = torch.minimum(
                self.target_critic_1(
                    batch[
                        "next_states"
                    ],
                    next_joint_action,
                ),
                self.target_critic_2(
                    batch[
                        "next_states"
                    ],
                    next_joint_action,
                ),
            )

            entropy_adjusted_target = (
                target_q
                - self.alpha_continuous.detach()
                * next_policy[
                    "continuous_log_probability"
                ]
                - self.alpha_discrete.detach()
                * next_policy[
                    "destination_log_probability"
                ]
            )

            # Gymnasium truncation is a time-limit event,
            # so only true termination blocks bootstrapping.
            critic_target = (
                batch["rewards"]
                + self.gamma
                * (
                    1.0
                    - batch[
                        "terminated"
                    ]
                )
                * entropy_adjusted_target
            )

        with self._autocast():
            critic_1_value = (
                self.critic_1(
                    batch["states"],
                    replay_joint_action,
                )
            )
            critic_2_value = (
                self.critic_2(
                    batch["states"],
                    replay_joint_action,
                )
            )

            critic_loss = (
                F.mse_loss(
                    critic_1_value,
                    critic_target,
                )
                + F.mse_loss(
                    critic_2_value,
                    critic_target,
                )
            )

        self.critic_optimizer.zero_grad(
            set_to_none=True
        )
        if self.amp_enabled:
            self.grad_scaler.scale(
                critic_loss
            ).backward()
            self.grad_scaler.unscale_(
                self.critic_optimizer
            )
        else:
            critic_loss.backward()

        critic_grad_norm = (
            clip_grad_norm_finite(
                list(
                    self.critic_1.parameters()
                )
                + list(
                    self.critic_2.parameters()
                ),
                self.gradient_clip_norm,
                "MASAC critic",
            )
        )

        if self.amp_enabled:
            self.grad_scaler.step(
                self.critic_optimizer
            )
        else:
            self.critic_optimizer.step()

        for critic in (
            self.critic_1,
            self.critic_2,
        ):
            for parameter in (
                critic.parameters()
            ):
                parameter.requires_grad_(
                    False
                )

        with self._autocast():
            policy = self._sample_joint_policy(
                batch["observations"],
                batch[
                    "destination_masks"
                ],
                deterministic=False,
            )

            policy_joint_action = (
                self._joint_action_tensor(
                    policy[
                        "continuous"
                    ],
                    policy[
                        "destination_one_hot"
                    ],
                )
            )

            policy_q = torch.minimum(
                self.critic_1(
                    batch["states"],
                    policy_joint_action,
                ),
                self.critic_2(
                    batch["states"],
                    policy_joint_action,
                ),
            )

            # Exact categorical entropy retains the probability-weight gradient.
            actor_loss = (
                self.alpha_continuous.detach()
                * policy["continuous_log_probability"]
                - self.alpha_discrete.detach()
                * policy["destination_entropy"]
                - policy_q
            ).mean()

        self.actor_optimizer.zero_grad(
            set_to_none=True
        )
        if self.amp_enabled:
            self.grad_scaler.scale(
                actor_loss
            ).backward()
            self.grad_scaler.unscale_(
                self.actor_optimizer
            )
        else:
            actor_loss.backward()

        actor_grad_norm = (
            clip_grad_norm_finite(
                self.actor.parameters(),
                self.gradient_clip_norm,
                "MASAC actor",
            )
        )

        if self.amp_enabled:
            self.grad_scaler.step(
                self.actor_optimizer
            )
            self.grad_scaler.update()
        else:
            self.actor_optimizer.step()

        for critic in (
            self.critic_1,
            self.critic_2,
        ):
            for parameter in (
                critic.parameters()
            ):
                parameter.requires_grad_(
                    True
                )

        continuous_entropy = -policy[
            "continuous_log_probability"
        ].detach()

        continuous_target = torch.full_like(
            continuous_entropy,
            float(
                self.continuous_target_entropy
            ),
        )

        valid_counts = (
            batch[
                "destination_masks"
            ].sum(
                dim=-1
            ).clamp(
                min=1.0
            )
        )

        discrete_target = (
            self.discrete_target_entropy_ratio
            * torch.log(
                valid_counts
            ).sum(
                dim=1,
                keepdim=True,
            )
        )

        discrete_entropy = policy[
            "destination_entropy"
        ].detach()

        alpha_continuous_loss = (
            self.log_alpha_continuous
            * (
                continuous_entropy
                - continuous_target
            )
        ).mean()

        alpha_discrete_loss = (
            self.log_alpha_discrete
            * (
                discrete_entropy
                - discrete_target
            )
        ).mean()

        self.alpha_continuous_optimizer.zero_grad(
            set_to_none=True
        )
        alpha_continuous_loss.backward()
        self.alpha_continuous_optimizer.step()

        self.alpha_discrete_optimizer.zero_grad(
            set_to_none=True
        )
        alpha_discrete_loss.backward()
        self.alpha_discrete_optimizer.step()

        self._soft_update_targets()

        self.update_count += 1

        if not collect_metrics:
            return None

        metrics = {
            "critic_loss": float(
                critic_loss.detach().cpu()
            ),
            "actor_loss": float(
                actor_loss.detach().cpu()
            ),
            "alpha_continuous_loss": float(
                alpha_continuous_loss
                .detach()
                .cpu()
            ),
            "alpha_discrete_loss": float(
                alpha_discrete_loss
                .detach()
                .cpu()
            ),
            "alpha_continuous": float(
                self.alpha_continuous
                .detach()
                .cpu()
            ),
            "alpha_discrete": float(
                self.alpha_discrete
                .detach()
                .cpu()
            ),
            "continuous_entropy": float(
                continuous_entropy
                .mean()
                .cpu()
            ),
            "discrete_entropy": float(
                discrete_entropy
                .mean()
                .cpu()
            ),
            "target_q_mean": float(
                critic_target
                .mean()
                .detach()
                .cpu()
            ),
            "critic_grad_norm": float(
                torch.as_tensor(
                    critic_grad_norm
                )
                .detach()
                .cpu()
            ),
            "actor_grad_norm": float(
                torch.as_tensor(
                    actor_grad_norm
                )
                .detach()
                .cpu()
            ),
            "update_count": int(
                self.update_count
            ),
        }

        if not all(
            np.isfinite(value)
            for key, value
            in metrics.items()
            if key
            != "update_count"
        ):
            raise FloatingPointError(
                "non-finite MASAC metric"
            )

        return metrics

    @staticmethod
    def _move_optimizer_state_to_device(
        optimizer,
        device,
    ):
        for state in optimizer.state.values():
            for key, value in list(
                state.items()
            ):
                if isinstance(
                    value,
                    torch.Tensor,
                ):
                    state[key] = value.to(
                        device
                    )

    def state_dict(self):
        return {
            "actor": (
                self.actor.state_dict()
            ),
            "critic_1": (
                self.critic_1.state_dict()
            ),
            "critic_2": (
                self.critic_2.state_dict()
            ),
            "target_critic_1": (
                self.target_critic_1
                .state_dict()
            ),
            "target_critic_2": (
                self.target_critic_2
                .state_dict()
            ),
            "actor_optimizer": (
                self.actor_optimizer
                .state_dict()
            ),
            "critic_optimizer": (
                self.critic_optimizer
                .state_dict()
            ),
            "log_alpha_continuous": (
                self.log_alpha_continuous
                .detach()
                .cpu()
            ),
            "log_alpha_discrete": (
                self.log_alpha_discrete
                .detach()
                .cpu()
            ),
            "alpha_continuous_optimizer": (
                self.alpha_continuous_optimizer
                .state_dict()
            ),
            "alpha_discrete_optimizer": (
                self.alpha_discrete_optimizer
                .state_dict()
            ),
            "update_count": int(
                self.update_count
            ),
        }

    def load_state_dict(
        self,
        state,
    ):
        self.actor.load_state_dict(
            state["actor"]
        )
        self.critic_1.load_state_dict(
            state["critic_1"]
        )
        self.critic_2.load_state_dict(
            state["critic_2"]
        )
        self.target_critic_1.load_state_dict(
            state[
                "target_critic_1"
            ]
        )
        self.target_critic_2.load_state_dict(
            state[
                "target_critic_2"
            ]
        )

        self.actor_optimizer.load_state_dict(
            state[
                "actor_optimizer"
            ]
        )
        self.critic_optimizer.load_state_dict(
            state[
                "critic_optimizer"
            ]
        )

        with torch.no_grad():
            self.log_alpha_continuous.copy_(
                state[
                    "log_alpha_continuous"
                ].to(
                    self.device
                )
            )
            self.log_alpha_discrete.copy_(
                state[
                    "log_alpha_discrete"
                ].to(
                    self.device
                )
            )

        self.alpha_continuous_optimizer.load_state_dict(
            state[
                "alpha_continuous_optimizer"
            ]
        )
        self.alpha_discrete_optimizer.load_state_dict(
            state[
                "alpha_discrete_optimizer"
            ]
        )

        for optimizer in (
            self.actor_optimizer,
            self.critic_optimizer,
            self.alpha_continuous_optimizer,
            self.alpha_discrete_optimizer,
        ):
            self._move_optimizer_state_to_device(
                optimizer,
                self.device,
            )

        self.update_count = int(
            state.get(
                "update_count",
                0,
            )
        )


# --- frozen notebook cell 168 ---
def add_environment_transition_to_replay(
    replay_buffer,
    observations,
    state,
    actions,
    reward,
    next_observations,
    next_state,
    terminated,
    truncated,
    agent_ids,
):
    (
        observation_vectors,
        destination_masks,
    ) = observations_to_masac_arrays(
        observations,
        agent_ids,
    )

    (
        next_observation_vectors,
        next_destination_masks,
    ) = observations_to_masac_arrays(
        next_observations,
        agent_ids,
    )

    (
        continuous_actions,
        destination_indices,
    ) = env_actions_to_masac_arrays(
        actions,
        agent_ids,
    )

    replay_buffer.add(
        observations=(
            observation_vectors
        ),
        state=state,
        continuous_actions=(
            continuous_actions
        ),
        destination_indices=(
            destination_indices
        ),
        destination_masks=(
            destination_masks
        ),
        reward=reward,
        next_observations=(
            next_observation_vectors
        ),
        next_state=next_state,
        next_destination_masks=(
            next_destination_masks
        ),
        terminated=terminated,
        truncated=truncated,
    )


# --- frozen notebook cell 169 ---
def train_masac(
    env,
    total_steps,
    seed=None,
    trainer=None,
    replay_buffer=None,
):
    if not isinstance(
        env,
        UAVSearchEnv,
    ):
        raise TypeError(
            "env must be UAVSearchEnv"
        )

    total_steps = int(
        total_steps
    )

    if total_steps < 1:
        raise ValueError(
            "total_steps must be >= 1"
        )

    if seed is None:
        seed = int(
            CONFIG["seed"]
        )

    seed = int(
        seed
    )

    set_seed(
        seed
    )

    if trainer is None:
        trainer = HybridMASAC(
            env
        )

    if replay_buffer is None:
        replay_buffer = (
            trainer.make_replay_buffer(
                seed=seed,
            )
        )

    rng = np.random.default_rng(
        seed
    )

    observations, info = env.reset(
        seed=seed
    )
    state = info[
        "state"
    ]

    episode_return = 0.0
    episode_length = 0
    episode_index = 0
    completed_episodes = []
    latest_update_metrics = None

    learning_starts = int(
        CONFIG[
            "masac_learning_starts"
        ]
    )
    updates_per_step = int(
        CONFIG[
            "masac_updates_per_step"
        ]
    )
    batch_size = int(
        CONFIG[
            "masac_batch_size"
        ]
    )

    for global_step in range(
        1,
        total_steps + 1,
    ):
        if global_step <= learning_starts:
            actions = (
                sample_valid_random_actions(
                    observations,
                    env.agent_ids,
                    rng,
                )
            )
        else:
            actions = (
                trainer.select_actions(
                    observations,
                    deterministic=False,
                )
            )

        (
            next_observations,
            reward,
            terminated,
            truncated,
            next_info,
        ) = env.step(
            actions
        )

        next_state = next_info[
            "state"
        ]

        add_environment_transition_to_replay(
            replay_buffer,
            observations,
            state,
            actions,
            reward,
            next_observations,
            next_state,
            terminated,
            truncated,
            env.agent_ids,
        )

        episode_return += float(
            reward
        )
        episode_length += 1

        if (
            global_step > learning_starts
            and len(
                replay_buffer
            )
            >= batch_size
        ):
            for _ in range(
                updates_per_step
            ):
                latest_update_metrics = (
                    trainer.update(
                        replay_buffer,
                        batch_size=(
                            batch_size
                        ),
                    )
                )

        observations = (
            next_observations
        )
        state = next_state

        if terminated or truncated:
            completed_episodes.append(
                {
                    "episode": (
                        episode_index
                    ),
                    "return": float(
                        episode_return
                    ),
                    "length": int(
                        episode_length
                    ),
                    "terminated": bool(
                        terminated
                    ),
                    "truncated": bool(
                        truncated
                    ),
                    "success": bool(
                        next_info[
                            "success"
                        ]
                    ),
                }
            )

            episode_index += 1
            episode_return = 0.0
            episode_length = 0

            if global_step < total_steps:
                observations, info = (
                    env.reset(
                        seed=(
                            seed
                            + episode_index
                        )
                    )
                )
                state = info[
                    "state"
                ]

    return {
        "trainer": trainer,
        "replay_buffer": replay_buffer,
        "completed_episodes": (
            completed_episodes
        ),
        "latest_update_metrics": (
            latest_update_metrics
        ),
        "total_steps": total_steps,
    }


# --- frozen notebook cell 170 ---
def evaluate_masac(
    env,
    trainer,
    episodes=3,
    seed=None,
):
    if seed is None:
        seed = int(
            CONFIG["seed"]
        ) + 10_000

    episodes = int(
        episodes
    )

    if episodes < 1:
        raise ValueError(
            "episodes must be >= 1"
        )

    results = []

    for episode_index in range(
        episodes
    ):
        observations, _ = env.reset(
            seed=(
                int(seed)
                + episode_index
            )
        )

        episode_return = 0.0
        diagnostics = (
            new_episode_diagnostics(env=env)
        )

        while True:
            actions = trainer.select_actions(
                observations,
                deterministic=True,
            )
            update_action_diagnostics(
                diagnostics,
                actions,
            )

            (
                observations,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                actions
            )

            episode_return += float(
                reward
            )

            update_episode_diagnostics(
                diagnostics,
                info,
                env=env,
            )

            if (
                terminated
                or truncated
            ):
                episode_metrics = (
                    finalize_episode_diagnostics(
                        diagnostics,
                        env,
                        episode_return=(
                            episode_return
                        ),
                        episode_length=(
                            info["step"]
                        ),
                        success=(
                            info["success"]
                        ),
                    )
                )

                results.append(
                    {
                        "episode": (
                            episode_index
                        ),
                        **episode_metrics,
                        "terminated": bool(
                            terminated
                        ),
                        "truncated": bool(
                            truncated
                        ),
                    }
                )
                break

    return results


# --- frozen notebook cell 171 ---
def _checkpoint_config_snapshot():
    snapshot = {}

    for key, value in CONFIG.items():
        if isinstance(
            value,
            tuple,
        ):
            snapshot[key] = list(
                value
            )
        elif isinstance(
            value,
            np.ndarray,
        ):
            snapshot[key] = (
                value.tolist()
            )
        elif isinstance(
            value,
            np.generic,
        ):
            snapshot[key] = (
                value.item()
            )
        else:
            snapshot[key] = value

    return snapshot


# --- frozen notebook cell 172 ---
def _validate_resume_config(saved_config, algorithm):
    """Reject changes to the mission or learner during an exact resume."""
    current = _checkpoint_config_snapshot()
    other_algorithm = "matd3" if algorithm == "masac" else "masac"
    mismatches = []
    for key, value in current.items():
        if (
            key.startswith(("training_", "kaggle_", other_algorithm + "_"))
            or key in {"matd3_training_output_dir", "matd3_wandb_project"}
        ):
            continue
        if key not in saved_config or saved_config[key] != value:
            mismatches.append(key)
    if mismatches:
        raise ValueError(
            "exact resume config mismatch: " + ", ".join(sorted(mismatches))
        )


# --- frozen notebook cell 173 ---
def save_masac_checkpoint(
    path,
    trainer,
    replay_buffer=None,
    training_state=None,
    include_replay=None,
):
    if not isinstance(
        trainer,
        HybridMASAC,
    ):
        raise TypeError(
            "trainer must be HybridMASAC"
        )

    if include_replay is None:
        include_replay = bool(
            CONFIG[
                "training_checkpoint_include_replay"
            ]
        )

    include_replay = bool(
        include_replay
    )

    if (
        include_replay
        and replay_buffer is None
    ):
        raise ValueError(
            "replay_buffer is required "
            "when include_replay=True"
        )

    if (
        replay_buffer is not None
        and not isinstance(
            replay_buffer,
            HybridReplayBuffer,
        )
    ):
        raise TypeError(
            "replay_buffer must be "
            "HybridReplayBuffer"
        )

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if training_state is None:
        training_state = {}

    if not isinstance(
        training_state,
        dict,
    ):
        raise TypeError(
            "training_state must be a dict"
        )

    payload = {
        "format_version": 1,
        "trainer_state": (
            trainer.state_dict()
        ),
        "training_state": (
            copy.deepcopy(
                training_state
            )
        ),
        "config_snapshot": (
            _checkpoint_config_snapshot()
        ),
        "replay_state": (
            replay_buffer.state_dict()
            if include_replay
            else None
        ),
        "torch_rng_state": (
            torch.get_rng_state()
        ),
        "cuda_rng_states": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else []
        ),
    }

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        payload,
        temporary_path,
    )

    temporary_path.replace(
        path
    )

    return path


# --- frozen notebook cell 174 ---
def load_masac_checkpoint(
    path,
    env,
    device=None,
    load_replay=True,
    restore_torch_rng=True,
):
    if not isinstance(
        env,
        UAVSearchEnv,
    ):
        raise TypeError(
            "env must be UAVSearchEnv"
        )

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            str(path)
        )

    # weights_only=True is sufficient because this project checkpoint
    # contains tensors and basic Python containers, not serialized model
    # objects.
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if int(
        payload.get(
            "format_version",
            -1,
        )
    ) != 1:
        raise ValueError(
            "unsupported checkpoint "
            "format_version"
        )

    trainer = HybridMASAC(
        env,
        device=device,
    )
    trainer.load_state_dict(
        payload[
            "trainer_state"
        ]
    )

    replay_buffer = None

    replay_state = payload.get(
        "replay_state"
    )

    if (
        load_replay
        and replay_state is not None
    ):
        training_state = payload.get(
            "training_state",
            {},
        )
        replay_seed = int(
            training_state.get(
                "seed",
                CONFIG["seed"],
            )
        )

        replay_buffer = (
            trainer.make_replay_buffer(
                seed=replay_seed,
            )
        )
        replay_buffer.load_state_dict(
            replay_state
        )

    if restore_torch_rng:
        torch.set_rng_state(
            payload[
                "torch_rng_state"
            ]
        )

        cuda_states = payload.get(
            "cuda_rng_states",
            [],
        )

        if (
            torch.cuda.is_available()
            and cuda_states
        ):
            torch.cuda.set_rng_state_all(
                cuda_states
            )

    return {
        "trainer": trainer,
        "replay_buffer": replay_buffer,
        "training_state": (
            copy.deepcopy(
                payload.get(
                    "training_state",
                    {},
                )
            )
        ),
        "config_snapshot": (
            copy.deepcopy(
                payload.get(
                    "config_snapshot",
                    {},
                )
            )
        ),
        "path": path,
    }


# --- frozen notebook cell 175 ---
def git_experiment_metadata(
    repo_path=".",
):
    repo_path = Path(
        repo_path
    ).resolve()

    def git_output(
        *args,
    ):
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_path),
                *args,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            return None

        value = result.stdout.strip()

        return (
            value
            if value
            else None
        )

    return {
        "git_commit": git_output(
            "rev-parse",
            "HEAD",
        ),
        "git_branch": git_output(
            "branch",
            "--show-current",
        ),
        "git_remote": git_output(
            "remote",
            "get-url",
            "origin",
        ),
    }


# --- frozen notebook cell 176 ---
def modeled_collision_violations(env):
    """Count post-step modeled collision/separation violations.

    Peer violations use the project's safety_distance because UAV physical
    radii are not modeled separately. Obstacle violations use penetration of
    the actual obstacle cylinder (without the extra planning clearance).
    Counts are instantaneous post-step violations, not avoidance interventions.
    """
    peer_count = 0
    obstacle_count = 0

    positions = np.stack(
        [
            np.asarray(uav.position, dtype=np.float64)
            for uav in env.uavs
        ],
        axis=0,
    )

    safety_distance = float(CONFIG["safety_distance"])
    tolerance = 1e-9

    for i in range(len(env.uavs)):
        for j in range(i + 1, len(env.uavs)):
            distance = float(
                np.linalg.norm(
                    positions[i] - positions[j]
                )
            )
            if distance < safety_distance - tolerance:
                peer_count += 1

    for uav_index, position in enumerate(positions):
        xy = position[:2]
        altitude = float(position[2])
        for obstacle in env.obstacles:
            horizontal_distance = float(
                np.linalg.norm(
                    xy
                    - np.asarray(
                        obstacle.position,
                        dtype=np.float64,
                    )
                )
            )
            if (
                horizontal_distance
                < float(obstacle.radius) - tolerance
                and altitude
                <= float(obstacle.height) + tolerance
            ):
                obstacle_count += 1

    return {
        "peer_collision_violations": int(peer_count),
        "obstacle_collision_violations": int(obstacle_count),
        "collision_violations": int(
            peer_count + obstacle_count
        ),
    }


# --- frozen notebook cell 177 ---
def new_episode_diagnostics(env=None):
    grid_n = int(
        np.ceil(
            float(CONFIG["map_size"])
            / float(CONFIG["grid_cell_m"])
        )
    )
    return {
        "information_gain_bits": 0.0,
        "sensing_records": 0,
        "coverage_sensed_cell_events": 0,
        "coverage_new_unique_cells": 0,
        "gcs_contact_graph_reachability_fraction_sum": 0.0,
        "positive_observations": 0,
        "target_positive_observations": 0,
        "targets_ever_in_fov_ids": set(),
        "false_confirmations": 0,
        "expired_reports": 0,
        "dropped_reports": 0,
        "blocked_motion": 0,
        "obstacle_blocks": 0,
        "peer_safety_blocks": 0,
        "apf_interventions": 0,
        "apf_peer_interventions": 0,
        "apf_obstacle_interventions": 0,
        "boundary_clips": 0,
        "horizontal_boundary_clips": 0,
        "altitude_clips": 0,
        "communication_energy_j": 0.0,
        "total_energy_j": 0.0,
        "peer_collision_violations": 0,
        "obstacle_collision_violations": 0,
        "collision_steps": 0,
        "network_snapshot": {},
        "distance_total_m": 0.0,
        "previous_uav_positions": (
            None
            if env is None
            else np.stack(
                [
                    np.asarray(
                        uav.position,
                        dtype=np.float64,
                    ).copy()
                    for uav in env.uavs
                ],
                axis=0,
            )
        ),
        "first_confirmation_step": None,
        "all_confirmation_step": None,
        "first_delivery_step": None,
        "target_confirmation_steps": {},
        "confirmation_to_delivery_latency_sum_s": 0.0,
        "confirmation_to_delivery_latency_max_s": 0.0,
        "confirmation_to_delivery_latency_count": 0,
        "coverage_grid": np.zeros(
            (grid_n, grid_n),
            dtype=bool,
        ),
        "reward_components": {
            "information_gain": 0.0,
            "coverage_shaping": 0.0,
            "communication_progress": 0.0,
            "confirmation": 0.0,
            "delivery": 0.0,
            "false_confirmation": 0.0,
            "blocked_motion": 0.0,
            "boundary": 0.0,
            "expired_report": 0.0,
            "dropped_report": 0.0,
            "energy": 0.0,
            "step": 0.0,
            "success_bonus": 0.0,
        },
        "trajectory_xyz": [],
        "trace_steps": [],
        "coverage_history": [],
        "target_status_history": [],
        "communication_edges": [],
        # Evaluation-only actor diagnostics. These retain enough information
        # to detect policy collapse (boundary-biased motion or always-silent
        # communication) without changing the environment or reward.
        "action_count": 0,
        "action_motion_sum": np.zeros(3, dtype=np.float64),
        "action_motion_sq_sum": np.zeros(3, dtype=np.float64),
        "action_motion_norm_sum": 0.0,
        "action_motion_saturation_count": 0,
        "action_destination_silent_count": 0,
        "action_destination_peer_count": 0,
        "action_destination_gcs_count": 0,
        "action_power_sum": 0.0,
        "action_power_sq_sum": 0.0,
        "action_tx_power_w_sum": 0.0,
    }


# --- frozen notebook cell 178 ---
def update_action_diagnostics(
    diagnostics,
    actions,
):
    """Accumulate deterministic evaluation actor outputs without side effects."""
    num_uavs = int(CONFIG["num_uavs"])
    for sender_id in range(num_uavs):
        action = actions[f"uav_{sender_id}"]
        motion = np.asarray(
            action["motion"],
            dtype=np.float64,
        ).reshape(3)
        destination_index = int(
            action["destination"]
        )
        power_action = float(
            np.asarray(
                action["power"],
                dtype=np.float64,
            ).reshape(-1)[0]
        )

        diagnostics["action_count"] += 1
        diagnostics["action_motion_sum"] += motion
        diagnostics["action_motion_sq_sum"] += motion ** 2
        motion_norm = float(
            np.linalg.norm(motion)
        )
        diagnostics["action_motion_norm_sum"] += motion_norm
        diagnostics["action_motion_saturation_count"] += int(
            motion_norm >= 0.95
        )

        # communication_choices = (silent, peers..., GCS), therefore index 0
        # is silent and the last index (num_uavs) is GCS for every sender.
        if destination_index == 0:
            diagnostics["action_destination_silent_count"] += 1
        elif destination_index == num_uavs:
            diagnostics["action_destination_gcs_count"] += 1
        else:
            diagnostics["action_destination_peer_count"] += 1

        diagnostics["action_power_sum"] += power_action
        diagnostics["action_power_sq_sum"] += power_action ** 2
        diagnostics["action_tx_power_w_sum"] += float(
            decode_tx_power_w(
                np.asarray([power_action], dtype=np.float32)
            )
        )


# --- frozen notebook cell 179 ---
def update_episode_diagnostics(
    diagnostics,
    info,
    env=None,
):
    diagnostics[
        "information_gain_bits"
    ] += float(
        info[
            "information_gain_bits"
        ]
    )
    diagnostics[
        "sensing_records"
    ] += int(
        info.get(
            "sensing_record_count",
            0,
        )
    )
    diagnostics[
        "coverage_sensed_cell_events"
    ] += int(
        info.get(
            "coverage_sensed_cell_events",
            info.get("sensing_record_count", 0),
        )
    )
    diagnostics[
        "coverage_new_unique_cells"
    ] += int(
        info.get(
            "coverage_new_unique_cells",
            0,
        )
    )
    diagnostics[
        "gcs_contact_graph_reachability_fraction_sum"
    ] += float(
        info.get(
            "gcs_contact_graph_reachability_fraction",
            0.0,
        )
    )
    diagnostics[
        "positive_observations"
    ] += int(
        info.get(
            "positive_observation_count",
            0,
        )
    )
    diagnostics[
        "target_positive_observations"
    ] += int(
        info.get(
            "target_positive_observation_count",
            0,
        )
    )
    diagnostics[
        "targets_ever_in_fov_ids"
    ].update(
        int(target_id)
        for target_id in info.get(
            "targets_in_fov_ids",
            [],
        )
    )
    diagnostics[
        "false_confirmations"
    ] += int(
        info[
            "false_confirmation_count"
        ]
    )
    diagnostics[
        "expired_reports"
    ] += len(
        info[
            "expired_target_ids"
        ]
    )
    diagnostics[
        "dropped_reports"
    ] += int(
        info[
            "dropped_report_count"
        ]
    )

    step = int(info.get("step", 0))
    dt_seconds = float(CONFIG["dt"])
    newly_confirmed_ids = [
        int(target_id)
        for target_id in info.get(
            "newly_confirmed_target_ids",
            [],
        )
    ]
    newly_delivered_ids = [
        int(target_id)
        for target_id in info.get(
            "newly_delivered_target_ids",
            [],
        )
    ]

    for target_id in newly_confirmed_ids:
        diagnostics[
            "target_confirmation_steps"
        ].setdefault(
            target_id,
            step,
        )
    if (
        newly_confirmed_ids
        and diagnostics[
            "first_confirmation_step"
        ] is None
    ):
        diagnostics[
            "first_confirmation_step"
        ] = step

    if (
        newly_delivered_ids
        and diagnostics[
            "first_delivery_step"
        ] is None
    ):
        diagnostics[
            "first_delivery_step"
        ] = step
    for target_id in newly_delivered_ids:
        confirmation_step = diagnostics[
            "target_confirmation_steps"
        ].get(target_id)
        if confirmation_step is None:
            continue
        latency_s = max(
            0.0,
            float(step - confirmation_step)
            * dt_seconds,
        )
        diagnostics[
            "confirmation_to_delivery_latency_sum_s"
        ] += latency_s
        diagnostics[
            "confirmation_to_delivery_latency_max_s"
        ] = max(
            float(
                diagnostics[
                    "confirmation_to_delivery_latency_max_s"
                ]
            ),
            latency_s,
        )
        diagnostics[
            "confirmation_to_delivery_latency_count"
        ] += 1

    for key, value in info.get(
        "reward_components",
        {},
    ).items():
        diagnostics[
            "reward_components"
        ].setdefault(
            str(key),
            0.0,
        )
        diagnostics[
            "reward_components"
        ][str(key)] += float(
            value
        )

    motion = info[
        "motion"
    ]

    diagnostics[
        "blocked_motion"
    ] += int(
        np.count_nonzero(
            motion[
                "blocked"
            ]
        )
    )
    diagnostics[
        "obstacle_blocks"
    ] += int(
        np.count_nonzero(
            motion[
                "blocked_by_obstacle"
            ]
        )
    )
    diagnostics[
        "peer_safety_blocks"
    ] += int(
        np.count_nonzero(
            motion[
                "blocked_by_peer"
            ]
        )
    )
    diagnostics[
        "apf_interventions"
    ] += int(
        np.count_nonzero(
            motion.get(
                "apf_active",
                np.zeros_like(
                    motion["blocked"],
                    dtype=bool,
                ),
            )
        )
    )
    diagnostics[
        "apf_peer_interventions"
    ] += int(
        np.count_nonzero(
            motion.get(
                "apf_peer_active",
                np.zeros_like(
                    motion["blocked"],
                    dtype=bool,
                ),
            )
        )
    )
    diagnostics[
        "apf_obstacle_interventions"
    ] += int(
        np.count_nonzero(
            motion.get(
                "apf_obstacle_active",
                np.zeros_like(
                    motion["blocked"],
                    dtype=bool,
                ),
            )
        )
    )
    diagnostics[
        "boundary_clips"
    ] += int(
        np.count_nonzero(
            motion[
                "boundary_clipped"
            ]
        )
    )
    diagnostics[
        "horizontal_boundary_clips"
    ] += int(
        np.count_nonzero(
            motion[
                "horizontal_boundary_clipped"
            ]
        )
    )
    diagnostics[
        "altitude_clips"
    ] += int(
        np.count_nonzero(
            motion[
                "altitude_clipped"
            ]
        )
    )

    energy = info[
        "energy"
    ]

    diagnostics[
        "communication_energy_j"
    ] += float(
        np.sum(
            energy[
                "communication_j"
            ]
        )
    )
    diagnostics[
        "total_energy_j"
    ] += float(
        np.sum(
            energy[
                "total_j"
            ]
        )
    )

    diagnostics[
        "communication_edges"
    ].extend(
        copy.deepcopy(
            info.get(
                "communication_edges",
                [],
            )
        )
    )

    if env is not None:
        current_positions = np.stack(
            [
                np.asarray(
                    uav.position,
                    dtype=np.float64,
                )
                for uav in env.uavs
            ],
            axis=0,
        )
        previous_positions = diagnostics.get(
            "previous_uav_positions"
        )
        if previous_positions is not None:
            diagnostics[
                "distance_total_m"
            ] += float(
                np.linalg.norm(
                    current_positions
                    - np.asarray(
                        previous_positions,
                        dtype=np.float64,
                    ),
                    axis=1,
                ).sum()
            )
        diagnostics[
            "previous_uav_positions"
        ] = current_positions.copy()

        if (
            diagnostics[
                "all_confirmation_step"
            ] is None
            and env.targets
            and all(
                bool(target.confirmed)
                for target in env.targets
            )
        ):
            diagnostics[
                "all_confirmation_step"
            ] = step

        collision = modeled_collision_violations(
            env
        )
        diagnostics[
            "peer_collision_violations"
        ] += int(
            collision[
                "peer_collision_violations"
            ]
        )
        diagnostics[
            "obstacle_collision_violations"
        ] += int(
            collision[
                "obstacle_collision_violations"
            ]
        )
        diagnostics[
            "collision_steps"
        ] += int(
            collision[
                "collision_violations"
            ]
            > 0
        )

        coverage_grid = diagnostics[
            "coverage_grid"
        ]
        cell_size = float(
            CONFIG["grid_cell_m"]
        )
        grid_n = int(
            coverage_grid.shape[0]
        )
        for uav in env.uavs:
            if not bool(
                uav.active
            ):
                continue
            altitude = float(
                uav.position[2]
            )
            if altitude <= 0.0:
                continue
            _, _, fov_radius = (
                sensing_profile(
                    altitude
                )
            )
            ux = float(
                uav.position[0]
            )
            uy = float(
                uav.position[1]
            )
            gx_min = max(
                0,
                int(
                    np.floor(
                        (
                            ux
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
                            ux
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
                            uy
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
                            uy
                            + fov_radius
                        )
                        / cell_size
                    )
                ),
            )
            if (
                gx_min > gx_max
                or gy_min > gy_max
            ):
                continue
            xs = (
                (
                    np.arange(
                        gx_min,
                        gx_max + 1,
                        dtype=np.float64,
                    )
                    + 0.5
                )
                * cell_size
            )
            ys = (
                (
                    np.arange(
                        gy_min,
                        gy_max + 1,
                        dtype=np.float64,
                    )
                    + 0.5
                )
                * cell_size
            )
            xx, yy = np.meshgrid(
                xs,
                ys,
                indexing="xy",
            )
            visible = (
                (xx - ux) ** 2
                + (yy - uy) ** 2
                <= float(
                    fov_radius
                ) ** 2
            )
            coverage_grid[
                gy_min : gy_max + 1,
                gx_min : gx_max + 1,
            ] |= visible

        stride = max(
            1,
            int(
                CONFIG.get(
                    "training_visualization_trajectory_stride",
                    10,
                )
            ),
        )
        step = int(
            info.get(
                "step",
                0,
            )
        )
        if (
            step == 1
            or step % stride == 0
        ):
            diagnostics[
                "trajectory_xyz"
            ].append(
                np.stack(
                    [
                        np.asarray(
                            uav.position,
                            dtype=np.float64,
                        ).copy()
                        for uav in env.uavs
                    ],
                    axis=0,
                )
            )
            diagnostics[
                "trace_steps"
            ].append(
                int(step)
            )
            diagnostics[
                "coverage_history"
            ].append(
                coverage_grid.copy()
            )
            diagnostics[
                "target_status_history"
            ].append(
                {
                    "confirmed": [
                        int(target.id)
                        for target in env.targets
                        if bool(
                            target.confirmed
                        )
                    ],
                    "delivered": sorted(
                        int(target_id)
                        for target_id in (
                            env.gcs_received_target_ids
                        )
                    ),
                }
            )

    diagnostics[
        "network_snapshot"
    ] = copy.deepcopy(
        info[
            "network_metrics"
        ]
    )


# --- frozen notebook cell 180 ---
def extract_network_kpis(
    snapshot,
):
    metrics = {}

    nested = snapshot.get(
        "network_metrics",
        {},
    )

    key_map = {
        "pdr_percent": (
            "network_pdr_percent"
        ),
        "e2e_delay_ms": (
            "network_e2e_delay_ms"
        ),
        "throughput_kbps": (
            "network_throughput_kbps"
        ),
        "average_hops": (
            "network_average_hops"
        ),
        "phy_success_percent": (
            "network_phy_success_percent"
        ),
        "collisions": (
            "network_collisions"
        ),
    }

    for (
        source_key,
        metric_key,
    ) in key_map.items():
        value = nested.get(
            source_key
        )

        if isinstance(
            value,
            (bool, int, float, np.generic),
        ):
            metrics[
                metric_key
            ] = float(value)

    numeric_counters = {
        "requests": (
            "network_requests"
        ),
        "packets_injected": (
            "network_packets_injected"
        ),
        "packets_delivered": (
            "network_packets_delivered"
        ),
        "packets_failed": (
            "network_packets_failed"
        ),
        "payload_bytes_injected": (
            "network_payload_bytes_injected"
        ),
        "payload_bytes_delivered": (
            "network_payload_bytes_delivered"
        ),
        "payload_bytes_deferred": (
            "network_payload_bytes_deferred"
        ),
        "tx_bytes": (
            "network_tx_bytes"
        ),
    }
    for (
        source_key,
        metric_key,
    ) in numeric_counters.items():
        value = snapshot.get(
            source_key
        )
        if isinstance(
            value,
            (bool, int, float, np.generic),
        ):
            metrics[
                metric_key
            ] = float(value)

    injected = snapshot.get(
        "payload_bytes_injected"
    )
    delivered = snapshot.get(
        "payload_bytes_delivered"
    )

    if (
        isinstance(
            injected,
            (int, float, np.generic),
        )
        and isinstance(
            delivered,
            (int, float, np.generic),
        )
    ):
        injected_value = float(
            injected
        )
        delivered_value = float(
            delivered
        )
        metrics[
            "network_payload_delivery_ratio"
        ] = (
            delivered_value
            / injected_value
            if injected_value > 0.0
            else 0.0
        )
        metrics[
            "network_traffic_active"
        ] = float(
            injected_value > 0.0
        )

    if (
        "network_traffic_active"
        not in metrics
    ):
        requests = float(
            metrics.get(
                "network_requests",
                0.0,
            )
        )
        tx_bytes = float(
            metrics.get(
                "network_tx_bytes",
                0.0,
            )
        )
        metrics[
            "network_traffic_active"
        ] = float(
            requests > 0.0
            or tx_bytes > 0.0
        )

    return metrics


# --- frozen notebook cell 181 ---
def finalize_episode_diagnostics(
    diagnostics,
    env,
    episode_return,
    episode_length,
    success,
):
    delivered_targets = len(
        env.gcs_received_target_ids
    )
    confirmed_targets = sum(
        target.confirmed
        for target
        in env.targets
    )

    target_count = max(
        1,
        int(
            env.num_targets
        ),
    )
    obstacle_count = int(
        len(env.obstacles)
    )
    targets_ever_in_fov_count = int(
        len(
            diagnostics[
                "targets_ever_in_fov_ids"
            ]
        )
    )
    possible_agent_steps = max(
        1,
        int(
            episode_length
        )
        * int(
            env.num_uavs
        ),
    )

    coverage_grid = np.asarray(
        diagnostics[
            "coverage_grid"
        ],
        dtype=bool,
    )
    coverage_percent = float(
        100.0
        * np.count_nonzero(
            coverage_grid
        )
        / max(
            1,
            coverage_grid.size,
        )
    )

    battery_values = np.asarray(
        [
            max(
                0.0,
                float(uav.battery_j),
            )
            for uav in env.uavs
        ],
        dtype=np.float64,
    )
    battery_capacity_j = max(
        1e-12,
        float(CONFIG["battery_j"]),
    )
    battery_mean_j = float(
        np.mean(battery_values)
    )

    first_confirmation_step = diagnostics[
        "first_confirmation_step"
    ]
    all_confirmation_step = diagnostics[
        "all_confirmation_step"
    ]
    first_delivery_step = diagnostics[
        "first_delivery_step"
    ]
    dt_seconds = float(CONFIG["dt"])
    confirmation_delivery_count = int(
        diagnostics[
            "confirmation_to_delivery_latency_count"
        ]
    )

    reward_components = {
        str(key): float(value)
        for key, value in diagnostics[
            "reward_components"
        ].items()
    }
    reward_search = (
        reward_components.get(
            "information_gain",
            0.0,
        )
        + reward_components.get(
            "coverage_shaping",
            0.0,
        )
        + reward_components.get(
            "confirmation",
            0.0,
        )
        + reward_components.get(
            "false_confirmation",
            0.0,
        )
    )
    reward_communication = (
        reward_components.get(
            "communication_progress",
            0.0,
        )
        + reward_components.get(
            "delivery",
            0.0,
        )
        + reward_components.get(
            "expired_report",
            0.0,
        )
        + reward_components.get(
            "dropped_report",
            0.0,
        )
    )
    reward_safety = (
        reward_components.get(
            "blocked_motion",
            0.0,
        )
        + reward_components.get(
            "boundary",
            0.0,
        )
    )
    reward_energy = reward_components.get(
        "energy",
        0.0,
    )
    reward_mission = (
        reward_components.get(
            "step",
            0.0,
        )
        + reward_components.get(
            "success_bonus",
            0.0,
        )
    )

    trace = {
        "map_size": float(
            CONFIG["map_size"]
        ),
        "grid_cell_m": float(
            CONFIG["grid_cell_m"]
        ),
        "coverage_grid": coverage_grid,
        "trajectory_xyz": [
            np.asarray(
                value,
                dtype=np.float64,
            )
            for value in diagnostics[
                "trajectory_xyz"
            ]
        ],
        "trace_steps": [
            int(value)
            for value in diagnostics[
                "trace_steps"
            ]
        ],
        "coverage_history": [
            np.asarray(
                value,
                dtype=bool,
            )
            for value in diagnostics[
                "coverage_history"
            ]
        ],
        "target_status_history": copy.deepcopy(
            diagnostics[
                "target_status_history"
            ]
        ),
        "communication_edges": copy.deepcopy(
            diagnostics[
                "communication_edges"
            ]
        ),
        "gcs_position": np.asarray(
            CONFIG["gcs_position"],
            dtype=np.float64,
        ).copy(),
        "targets": [
            {
                "id": int(target.id),
                "position": np.asarray(
                    target.position,
                    dtype=np.float64,
                ).copy(),
                "confirmed": bool(
                    target.confirmed
                ),
                "delivered": bool(
                    target.id
                    in env.gcs_received_target_ids
                ),
            }
            for target in env.targets
        ],
        "obstacles": [
            {
                "position": np.asarray(
                    obstacle.position,
                    dtype=np.float64,
                ).copy(),
                "radius": float(
                    obstacle.radius
                ),
                "height": float(
                    obstacle.height
                ),
            }
            for obstacle in env.obstacles
        ],
        "final_uav_positions": np.stack(
            [
                np.asarray(
                    uav.position,
                    dtype=np.float64,
                )
                for uav in env.uavs
            ],
            axis=0,
        ),
        "coverage_percent": coverage_percent,
        "target_search_rate_percent": float(
            100.0
            * confirmed_targets
            / target_count
        ),
    }

    action_count = max(
        1,
        int(diagnostics["action_count"]),
    )
    action_motion_mean = (
        diagnostics["action_motion_sum"]
        / action_count
    )
    action_motion_variance = np.maximum(
        0.0,
        diagnostics["action_motion_sq_sum"]
        / action_count
        - action_motion_mean ** 2,
    )
    action_motion_std = np.sqrt(
        action_motion_variance
    )
    action_power_mean = float(
        diagnostics["action_power_sum"]
        / action_count
    )
    action_power_variance = max(
        0.0,
        float(
            diagnostics["action_power_sq_sum"]
            / action_count
            - action_power_mean ** 2
        ),
    )

    sensed_cell_events = int(
        diagnostics["coverage_sensed_cell_events"]
    )
    new_unique_cells = int(
        diagnostics["coverage_new_unique_cells"]
    )
    coverage_redundancy_percent = float(
        100.0 * (1.0 - new_unique_cells / sensed_cell_events)
        if sensed_cell_events > 0
        else 0.0
    )
    gcs_contact_graph_reachability_percent = float(
        100.0
        * diagnostics["gcs_contact_graph_reachability_fraction_sum"]
        / max(1, int(episode_length))
    )

    metrics = {
        "return": float(
            episode_return
        ),
        "reward_total": float(
            episode_return
        ),
        "reward_search": float(
            reward_search
        ),
        "reward_communication": float(
            reward_communication
        ),
        "reward_safety": float(
            reward_safety
        ),
        "reward_energy": float(
            reward_energy
        ),
        "reward_mission": float(
            reward_mission
        ),
        **{
            f"reward_component_{key}": float(
                value
            )
            for key, value in (
                reward_components.items()
            )
        },
        "coverage_percent": (
            coverage_percent
        ),
        "coverage_redundancy_percent": (
            coverage_redundancy_percent
        ),
        "gcs_contact_graph_reachability_percent": (
            gcs_contact_graph_reachability_percent
        ),
        "length": int(
            episode_length
        ),
        "episode_end_step": int(
            episode_length
        ),
        "success": float(
            bool(success)
        ),
        "targets_spawned": int(
            env.num_targets
        ),
        "obstacles_spawned": (
            obstacle_count
        ),
        "confirmed_targets": int(
            confirmed_targets
        ),
        "delivered_targets": int(
            delivered_targets
        ),
        "target_search_rate_percent": float(
            100.0
            * confirmed_targets
            / target_count
        ),
        "target_delivery_rate_percent": float(
            100.0
            * delivered_targets
            / target_count
        ),
        "targets_ever_in_fov_count": (
            targets_ever_in_fov_count
        ),
        "target_encounter_rate": float(
            targets_ever_in_fov_count
            / target_count
        ),
        "sensing_records": int(
            diagnostics[
                "sensing_records"
            ]
        ),
        "positive_observations": int(
            diagnostics[
                "positive_observations"
            ]
        ),
        "target_positive_observations": int(
            diagnostics[
                "target_positive_observations"
            ]
        ),
        "delivery_rate": float(
            delivered_targets
            / target_count
        ),
        "confirmation_rate": float(
            confirmed_targets
            / target_count
        ),
        "report_delivery_given_confirmation_rate_percent": float(
            100.0
            * delivered_targets
            / confirmed_targets
            if confirmed_targets > 0
            else 0.0
        ),
        "distance_total_m": float(
            diagnostics["distance_total_m"]
        ),
        "first_confirmation_observed": float(
            first_confirmation_step is not None
        ),
        "all_targets_confirmed": float(
            all_confirmation_step is not None
        ),
        "first_delivery_observed": float(
            first_delivery_step is not None
        ),
        "target_confirmation_to_delivery_latency_count": int(
            confirmation_delivery_count
        ),
        "false_confirmations": int(
            diagnostics[
                "false_confirmations"
            ]
        ),
        "expired_reports": int(
            diagnostics[
                "expired_reports"
            ]
        ),
        "dropped_reports": int(
            diagnostics[
                "dropped_reports"
            ]
        ),
        "blocked_motion_rate": float(
            diagnostics[
                "blocked_motion"
            ]
            / possible_agent_steps
        ),
        "obstacle_block_rate": float(
            diagnostics[
                "obstacle_blocks"
            ]
            / possible_agent_steps
        ),
        "peer_safety_block_rate": float(
            diagnostics[
                "peer_safety_blocks"
            ]
            / possible_agent_steps
        ),
        "apf_intervention_rate": float(
            diagnostics[
                "apf_interventions"
            ]
            / possible_agent_steps
        ),
        "apf_peer_intervention_rate": float(
            diagnostics[
                "apf_peer_interventions"
            ]
            / possible_agent_steps
        ),
        "apf_obstacle_intervention_rate": float(
            diagnostics[
                "apf_obstacle_interventions"
            ]
            / possible_agent_steps
        ),
        "boundary_clip_rate": float(
            diagnostics[
                "boundary_clips"
            ]
            / possible_agent_steps
        ),
        "horizontal_boundary_clip_rate": float(
            diagnostics[
                "horizontal_boundary_clips"
            ]
            / possible_agent_steps
        ),
        "altitude_clip_rate": float(
            diagnostics[
                "altitude_clips"
            ]
            / possible_agent_steps
        ),
        "total_energy_j": float(
            diagnostics[
                "total_energy_j"
            ]
        ),
        "communication_energy_j": float(
            diagnostics[
                "communication_energy_j"
            ]
        ),
        "battery_remaining_mean_j": (
            battery_mean_j
        ),
        "battery_remaining_mean_percent": float(
            100.0
            * battery_mean_j
            / battery_capacity_j
        ),
        "depleted_uav_count": int(
            np.count_nonzero(
                battery_values <= 1e-12
            )
        ),
        "collision_violation_count": int(
            diagnostics[
                "peer_collision_violations"
            ]
            + diagnostics[
                "obstacle_collision_violations"
            ]
        ),
        "collision_step_count": int(
            diagnostics[
                "collision_steps"
            ]
        ),
        "information_gain_bits": float(
            diagnostics[
                "information_gain_bits"
            ]
        ),
        "action_motion_mean_x": float(action_motion_mean[0]),
        "action_motion_mean_y": float(action_motion_mean[1]),
        "action_motion_mean_z": float(action_motion_mean[2]),
        "action_motion_std_x": float(action_motion_std[0]),
        "action_motion_std_y": float(action_motion_std[1]),
        "action_motion_std_z": float(action_motion_std[2]),
        "action_motion_norm_mean": float(
            diagnostics["action_motion_norm_sum"]
            / action_count
        ),
        "action_motion_saturation_rate": float(
            diagnostics["action_motion_saturation_count"]
            / action_count
        ),
        "action_destination_silent_rate": float(
            diagnostics["action_destination_silent_count"]
            / action_count
        ),
        "action_destination_peer_rate": float(
            diagnostics["action_destination_peer_count"]
            / action_count
        ),
        "action_destination_gcs_rate": float(
            diagnostics["action_destination_gcs_count"]
            / action_count
        ),
        "action_power_mean": action_power_mean,
        "action_power_std": float(
            np.sqrt(action_power_variance)
        ),
        "action_tx_power_w_mean": float(
            diagnostics["action_tx_power_w_sum"]
            / action_count
        ),
        "visualization_trace": trace,
    }

    if confirmed_targets > 0:
        metrics["energy_per_confirmed_target_j"] = float(
            diagnostics["total_energy_j"]
            / confirmed_targets
        )
    if delivered_targets > 0:
        metrics["energy_per_delivered_target_j"] = float(
            diagnostics["total_energy_j"]
            / delivered_targets
        )
    if first_confirmation_step is not None:
        metrics["time_to_first_confirm_s"] = float(
            first_confirmation_step * dt_seconds
        )
    if all_confirmation_step is not None:
        metrics["time_to_all_confirm_s"] = float(
            all_confirmation_step * dt_seconds
        )
    if first_delivery_step is not None:
        metrics["time_to_first_delivery_s"] = float(
            first_delivery_step * dt_seconds
        )
    if confirmation_delivery_count > 0:
        metrics[
            "target_confirmation_to_delivery_latency_s"
        ] = float(
            diagnostics[
                "confirmation_to_delivery_latency_sum_s"
            ]
            / confirmation_delivery_count
        )
        metrics[
            "target_confirmation_to_delivery_latency_max_s"
        ] = float(
            diagnostics[
                "confirmation_to_delivery_latency_max_s"
            ]
        )

    metrics.update(
        extract_network_kpis(
            diagnostics[
                "network_snapshot"
            ]
        )
    )

    return metrics


# --- frozen notebook cell 182 ---
def filter_training_metrics_for_wandb(
    algorithm,
    metrics,
):
    normalized = str(
        algorithm
    ).strip().lower()

    if normalized == "masac":
        keep = {
            "critic_loss",
            "actor_loss",
            "alpha_continuous",
            "alpha_discrete",
            "continuous_entropy",
            "discrete_entropy",
        }
    elif normalized == "matd3":
        keep = {
            "critic_loss",
            "actor_loss",
        }
    else:
        raise ValueError(
            "algorithm must be "
            "'masac' or 'matd3'"
        )

    return {
        key: value
        for key, value
        in metrics.items()
        if key in keep
        and value is not None
    }


# --- frozen notebook cell 183 ---
class MASACExperimentLogger:
    def __init__(
        self,
        run_dir,
        run_name,
        enable_csv=None,
        enable_tensorboard=None,
        enable_wandb=None,
        wandb_entity=None,
        wandb_project=None,
        wandb_mode=None,
        wandb_run_id=None,
        wandb_resume=None,
        algorithm=None,
        seed=None,
        backend_name=None,
        config_snapshot=None,
    ):
        self.run_dir = Path(
            run_dir
        )
        self.run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.run_name = str(
            run_name
        )
        self.algorithm = (
            None
            if algorithm is None
            else str(
                algorithm
            ).lower()
        )
        self.seed = (
            None
            if seed is None
            else int(seed)
        )
        self.backend_name = (
            None
            if backend_name is None
            else str(
                backend_name
            )
        )

        if enable_csv is None:
            enable_csv = CONFIG[
                "training_enable_csv"
            ]

        if enable_tensorboard is None:
            enable_tensorboard = CONFIG[
                "training_enable_tensorboard"
            ]

        if enable_wandb is None:
            enable_wandb = CONFIG[
                "training_enable_wandb"
            ]

        if wandb_entity is None:
            wandb_entity = CONFIG.get(
                "training_wandb_entity"
            )

        if wandb_project is None:
            wandb_project = CONFIG[
                "training_wandb_project"
            ]

        if wandb_mode is None:
            wandb_mode = CONFIG[
                "training_wandb_mode"
            ]

        self.enable_csv = bool(
            enable_csv
        )
        self.enable_tensorboard = bool(
            enable_tensorboard
        )
        self.enable_wandb = bool(
            enable_wandb
        )

        self.csv_path = (
            self.run_dir
            / "metrics.csv"
        )
        self._csv_file = None
        self._csv_writer = None
        self._tensorboard_writer = None
        self._wandb_run = None

        self.tensorboard_error = None
        self.wandb_error = None
        self.wandb_run_id = None
        self.wandb_run_url = None

        if config_snapshot is None:
            config_snapshot = (
                _checkpoint_config_snapshot()
            )

        self.config_snapshot = (
            copy.deepcopy(
                config_snapshot
            )
        )

        git_metadata = (
            git_experiment_metadata()
        )

        self.config_snapshot.update(
            {
                "monitor/algorithm": (
                    self.algorithm
                ),
                "monitor/seed": self.seed,
                "monitor/backend": (
                    self.backend_name
                ),
                **{
                    (
                        f"monitor/{key}"
                    ): value
                    for key, value
                    in git_metadata.items()
                },
            }
        )

        if self.enable_csv:
            file_exists = (
                self.csv_path.is_file()
                and self.csv_path.stat().st_size
                > 0
            )

            self._csv_file = open(  # noqa: SIM115 - kept open for the run
                self.csv_path,
                "a",
                newline="",
                encoding="utf-8",
            )
            self._csv_writer = (
                csv.writer(
                    self._csv_file
                )
            )

            if not file_exists:
                self._csv_writer.writerow(
                    [
                        "step",
                        "episode",
                        "metric",
                        "value",
                    ]
                )
                self._csv_file.flush()

        if self.enable_tensorboard:
            try:
                from torch.utils.tensorboard import (
                    SummaryWriter,
                )

                self._tensorboard_writer = (
                    SummaryWriter(
                        log_dir=str(
                            self.run_dir
                            / "tensorboard"
                        )
                    )
                )
            except Exception as exc:  # noqa: BLE001 - optional logger fallback
                self.tensorboard_error = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

        if self.enable_wandb:
            try:
                wandb = (
                    importlib.import_module(
                        "wandb"
                    )
                )

                import secrets
                resolved_wandb_run_id = (
                    str(wandb_run_id)
                    if wandb_run_id
                    else secrets.token_hex(4)
                )
                wandb_algorithm = (
                    str(self.algorithm).upper()
                    if self.algorithm
                    else "MARL"
                )
                wandb_seed = (
                    int(self.seed)
                    if self.seed is not None
                    else int(CONFIG["seed"])
                )
                wandb_name_format = str(
                    CONFIG.get(
                        "training_wandb_name_format",
                        "{algorithm}-seed{seed}-{id}",
                    )
                )
                initial_name = wandb_name_format.format(
                    algorithm=wandb_algorithm,
                    seed=wandb_seed,
                    id=resolved_wandb_run_id,
                )

                self._wandb_run = (
                    wandb.init(
                        entity=(
                            wandb_entity
                        ),
                        project=str(
                            wandb_project
                        ),
                        name=initial_name,
                        mode=str(
                            wandb_mode
                        ),
                        id=resolved_wandb_run_id,
                        resume=wandb_resume,
                        dir=str(
                            self.run_dir
                        ),
                        reinit="finish_previous",
                        settings=wandb.Settings(
                            x_disable_stats=bool(
                                CONFIG.get(
                                    "training_wandb_disable_system_stats",
                                    True,
                                )
                            )
                        ),
                        save_code=False,
                        tags=None,
                        group=None,
                        job_type=None,
                    )
                )
                wandb_display_name = initial_name

                self.wandb_run_id = (
                    self._wandb_run.id
                )
                self.wandb_run_url = (
                    self._wandb_run.url
                )

                self._configure_wandb_metrics()
            except Exception as exc:  # noqa: BLE001 - optional logger fallback
                self.wandb_error = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

    def _configure_wandb_metrics(
        self,
    ):
        if self._wandb_run is None:
            return

        self._wandb_run.define_metric(
            "global_step"
        )

        for pattern in (
            "train/*",
            "episode/*",
            "evaluation/*",
        ):
            self._wandb_run.define_metric(
                pattern,
                step_metric="global_step",
            )

        for metric in (
            "evaluation/success_rate",
            "evaluation/delivery_rate",
            "evaluation/confirmation_rate",
            "evaluation/return",
            "evaluation/network_pdr_percent",
            "evaluation/network_throughput_kbps",
        ):
            self._wandb_run.define_metric(
                metric,
                summary="max",
            )

        for metric in (
            "evaluation/total_energy_j",
            "evaluation/communication_energy_j",
            "evaluation/false_confirmations",
            "evaluation/expired_reports",
            "evaluation/dropped_reports",
            "evaluation/obstacle_block_rate",
            "evaluation/peer_safety_block_rate",
            "evaluation/boundary_clip_rate",
            "evaluation/network_e2e_delay_ms",
        ):
            self._wandb_run.define_metric(
                metric,
                summary="min",
            )

        self._wandb_run.define_metric(
            "train/critic_loss",
            summary="min",
        )

    @staticmethod
    def _scalar_metrics(
        metrics,
    ):
        if not isinstance(
            metrics,
            dict,
        ):
            raise TypeError(
                "metrics must be a dict"
            )

        scalar_metrics = {}

        for key, value in (
            metrics.items()
        ):
            if isinstance(
                value,
                torch.Tensor,
            ):
                if value.numel() != 1:
                    continue

                value = float(
                    value
                    .detach()
                    .cpu()
                    .item()
                )
            elif isinstance(
                value,
                np.generic,
            ):
                value = value.item()

            if isinstance(
                value,
                (bool, int, float),
            ):
                value = float(value)

                if np.isfinite(
                    value
                ):
                    scalar_metrics[
                        str(key)
                    ] = value

        return scalar_metrics

    def log_metrics(
        self,
        prefix,
        metrics,
        step,
        episode=None,
    ):
        prefix = str(
            prefix
        ).strip(
            "/"
        )
        step = int(
            step
        )

        scalar_metrics = (
            self._scalar_metrics(
                metrics
            )
        )

        tagged = {
            (
                f"{prefix}/{key}"
                if prefix
                else key
            ): value
            for key, value
            in scalar_metrics.items()
        }

        if (
            self._csv_writer
            is not None
        ):
            for key, value in (
                tagged.items()
            ):
                self._csv_writer.writerow(
                    [
                        step,
                        (
                            ""
                            if episode is None
                            else int(
                                episode
                            )
                        ),
                        key,
                        value,
                    ]
                )

            self._csv_file.flush()

        if (
            self._tensorboard_writer
            is not None
        ):
            for key, value in (
                tagged.items()
            ):
                self._tensorboard_writer.add_scalar(
                    key,
                    value,
                    step,
                )

        if self._wandb_run is not None:
            self._wandb_run.log(
                {
                    "global_step": step,
                    **tagged,
                }
            )

        return tagged

    def update_summary(
        self,
        metrics,
        prefix="summary",
    ):
        scalar_metrics = (
            self._scalar_metrics(
                metrics
            )
        )

        if self._wandb_run is not None:
            for key, value in (
                scalar_metrics.items()
            ):
                self._wandb_run.summary[
                    f"{prefix}/{key}"
                ] = value

        return scalar_metrics

    def log_model_artifact(
        self,
        path,
        aliases,
        metadata=None,
    ):
        if self._wandb_run is None:
            return None

        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(
                str(path)
            )

        wandb = importlib.import_module(
            "wandb"
        )

        artifact_name = (
            f"{self.run_name}-"
            f"{self.algorithm or 'marl'}-model"
        )

        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            metadata=(
                {}
                if metadata is None
                else copy.deepcopy(
                    metadata
                )
            ),
        )
        artifact.add_file(
            str(path),
            name=path.name,
        )

        self._wandb_run.log_artifact(
            artifact,
            aliases=list(
                aliases
            ),
        )

        return artifact_name

    def status(self):
        return {
            "csv": bool(
                self._csv_writer
                is not None
            ),
            "tensorboard": bool(
                self._tensorboard_writer
                is not None
            ),
            "wandb": bool(
                self._wandb_run
                is not None
            ),
            "wandb_run_id": (
                self.wandb_run_id
            ),
            "wandb_run_url": (
                self.wandb_run_url
            ),
            "algorithm": self.algorithm,
            "seed": self.seed,
            "backend": (
                self.backend_name
            ),
            "tensorboard_error": (
                self.tensorboard_error
            ),
            "wandb_error": (
                self.wandb_error
            ),
        }

    def close(self):
        if (
            self._tensorboard_writer
            is not None
        ):
            self._tensorboard_writer.flush()
            self._tensorboard_writer.close()
            self._tensorboard_writer = None

        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None

        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None


# --- frozen notebook cell 184 ---
def summarize_evaluation_results(
    results,
):
    if not results:
        raise ValueError(
            "evaluation results must "
            "not be empty"
        )

    summary = {
        "success_rate": float(
            np.mean(
                np.asarray(
                    [
                        item[
                            "success"
                        ]
                        for item
                        in results
                    ],
                    dtype=np.float64,
                )
            )
        ),
        "return": float(
            np.mean(
                np.asarray(
                    [
                        item[
                            "return"
                        ]
                        for item
                        in results
                    ],
                    dtype=np.float64,
                )
            )
        ),
        "length": float(
            np.mean(
                np.asarray(
                    [
                        item[
                            "length"
                        ]
                        for item
                        in results
                    ],
                    dtype=np.float64,
                )
            )
        ),
    }

    keys_to_average = (
        "reward_total",
        "reward_search",
        "reward_communication",
        "reward_safety",
        "reward_energy",
        "reward_mission",
        "reward_component_information_gain",
        "reward_component_confirmation",
        "reward_component_delivery",
        "reward_component_false_confirmation",
        "reward_component_blocked_motion",
        "reward_component_boundary",
        "reward_component_expired_report",
        "reward_component_dropped_report",
        "reward_component_energy",
        "reward_component_step",
        "reward_component_success_bonus",
        "coverage_percent",
        "coverage_redundancy_percent",
        "gcs_contact_graph_reachability_percent",
        "targets_spawned",
        "obstacles_spawned",
        "confirmed_targets",
        "delivered_targets",
        "target_search_rate_percent",
        "target_delivery_rate_percent",
        "report_delivery_given_confirmation_rate_percent",
        "distance_total_m",
        "first_confirmation_observed",
        "all_targets_confirmed",
        "first_delivery_observed",
        "target_confirmation_to_delivery_latency_count",
        "targets_ever_in_fov_count",
        "target_encounter_rate",
        "sensing_records",
        "positive_observations",
        "target_positive_observations",
        "delivery_rate",
        "confirmation_rate",
        "false_confirmations",
        "expired_reports",
        "dropped_reports",
        "blocked_motion_rate",
        "obstacle_block_rate",
        "peer_safety_block_rate",
        "apf_intervention_rate",
        "apf_peer_intervention_rate",
        "apf_obstacle_intervention_rate",
        "boundary_clip_rate",
        "horizontal_boundary_clip_rate",
        "altitude_clip_rate",
        "total_energy_j",
        "communication_energy_j",
        "battery_remaining_mean_j",
        "battery_remaining_mean_percent",
        "depleted_uav_count",
        "collision_violation_count",
        "collision_step_count",
        "episode_end_step",
        "information_gain_bits",
        "action_motion_mean_x",
        "action_motion_mean_y",
        "action_motion_mean_z",
        "action_motion_std_x",
        "action_motion_std_y",
        "action_motion_std_z",
        "action_motion_norm_mean",
        "action_motion_saturation_rate",
        "action_destination_silent_rate",
        "action_destination_peer_rate",
        "action_destination_gcs_rate",
        "action_power_mean",
        "action_power_std",
        "action_tx_power_w_mean",
        "network_pdr_percent",
        "network_e2e_delay_ms",
        "network_throughput_kbps",
        "network_average_hops",
        "network_phy_success_percent",
        "network_collisions",
        "network_payload_delivery_ratio",
        "network_requests",
        "network_packets_injected",
        "network_packets_delivered",
        "network_packets_failed",
        "network_payload_bytes_injected",
        "network_payload_bytes_delivered",
        "network_payload_bytes_deferred",
        "network_tx_bytes",
        "network_traffic_active",
    )

    for key in keys_to_average:
        values = [
            item[key]
            for item in results
            if key in item
        ]

        if len(values) != len(
            results
        ):
            continue

        summary[key] = float(
            np.mean(
                np.asarray(
                    values,
                    dtype=np.float64,
                )
            )
        )

    conditional_keys = (
        "energy_per_confirmed_target_j",
        "energy_per_delivered_target_j",
        "time_to_first_confirm_s",
        "time_to_all_confirm_s",
        "time_to_first_delivery_s",
        "target_confirmation_to_delivery_latency_s",
        "target_confirmation_to_delivery_latency_max_s",
    )
    for key in conditional_keys:
        values = [
            float(item[key])
            for item in results
            if key in item
            and item[key] is not None
            and np.isfinite(float(item[key]))
        ]
        if values:
            summary[key] = float(
                np.mean(
                    np.asarray(
                        values,
                        dtype=np.float64,
                    )
                )
            )

    return summary


# --- frozen notebook cell 185 ---
def evaluation_is_better(
    candidate,
    best,
):
    if best is None:
        return True

    candidate_key = (
        float(
            candidate[
                "success_rate"
            ]
        ),
        float(
            candidate.get(
                "delivery_rate",
                0.0,
            )
        ),
        float(
            candidate.get(
                "confirmation_rate",
                0.0,
            )
        ),
        float(
            candidate[
                "return"
            ]
        ),
    )
    best_key = (
        float(
            best[
                "success_rate"
            ]
        ),
        float(
            best.get(
                "delivery_rate",
                0.0,
            )
        ),
        float(
            best.get(
                "confirmation_rate",
                0.0,
            )
        ),
        float(
            best[
                "return"
            ]
        ),
    )

    return bool(
        candidate_key > best_key
    )


# --- frozen notebook cell 186 ---
def train_masac_experiment(
    env,
    total_steps,
    seed=None,
    run_dir=None,
    run_name=None,
    resume_checkpoint=None,
    device=None,
    enable_csv=None,
    enable_tensorboard=None,
    enable_wandb=None,
):
    if not isinstance(
        env,
        UAVSearchEnv,
    ):
        raise TypeError(
            "env must be UAVSearchEnv"
        )

    total_steps = int(
        total_steps
    )

    if total_steps < 1:
        raise ValueError(
            "total_steps must be >= 1"
        )

    if seed is None:
        seed = int(
            CONFIG["seed"]
        )

    seed = int(
        seed
    )

    if run_name is None:
        run_name = (
            make_experiment_run_name(
                "masac",
                seed,
            )
        )

    if run_dir is None:
        if resume_checkpoint is not None:
            run_dir = (
                Path(
                    resume_checkpoint
                )
                .resolve()
                .parent
                .parent
            )
        else:
            run_dir = (
                Path(
                    CONFIG[
                        "training_output_dir"
                    ]
                )
                / run_name
            )

    run_dir = Path(
        run_dir
    )
    checkpoint_dir = (
        run_dir / "checkpoints"
    )
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_seed(
        seed
    )

    if resume_checkpoint is None:
        trainer = HybridMASAC(
            env,
            device=device,
        )
        replay_buffer = (
            trainer.make_replay_buffer(
                seed=seed,
            )
        )
        training_state = {
            "seed": seed,
            "global_step": 0,
            "episode_index": 0,
            "best_evaluation": None,
            "last_eval_step": 0,
            "last_checkpoint_step": 0,
            "training_rng_state_json": None,
            "resume_safe": True,
        }
    else:
        restored = (
            load_masac_checkpoint(
                resume_checkpoint,
                env,
                device=device,
                load_replay=True,
                restore_torch_rng=True,
            )
        )
        trainer = restored[
            "trainer"
        ]
        replay_buffer = restored[
            "replay_buffer"
        ]
        training_state = restored[
            "training_state"
        ]

        if training_state.get("vectorized", False):
            raise ValueError(
                "vector checkpoint requires the saved multi-env runtime; "
                "do not resume it through the single-env loop"
            )
        _validate_resume_config(
            restored["config_snapshot"],
            "masac",
        )

        if replay_buffer is None:
            raise ValueError(
                "resume checkpoint does not "
                "contain replay state; set "
                "training_checkpoint_include_replay=True "
                "when creating resumable checkpoints"
            )

        if not bool(
            training_state.get(
                "resume_safe",
                False,
            )
        ):
            raise ValueError(
                "checkpoint is not marked "
                "resume_safe"
            )

        saved_seed = int(
            training_state.get(
                "seed",
                seed,
            )
        )

        if saved_seed != seed:
            raise ValueError(
                "resume seed does not match "
                "checkpoint seed"
            )

    global_step = int(
        training_state.get(
            "global_step",
            0,
        )
    )
    episode_index = int(
        training_state.get(
            "episode_index",
            0,
        )
    )
    best_evaluation = copy.deepcopy(
        training_state.get(
            "best_evaluation"
        )
    )
    last_eval_step = int(
        training_state.get(
            "last_eval_step",
            0,
        )
    )
    last_checkpoint_step = int(
        training_state.get(
            "last_checkpoint_step",
            0,
        )
    )

    if global_step > total_steps:
        raise ValueError(
            "total_steps is smaller than "
            "checkpoint global_step"
        )

    rng = np.random.default_rng(
        seed
    )

    saved_rng_state = (
        training_state.get(
            "training_rng_state_json"
        )
    )

    if saved_rng_state:
        rng.bit_generator.state = (
            json.loads(
                saved_rng_state
            )
        )

    logger = MASACExperimentLogger(
        run_dir=run_dir,
        run_name=run_name,
        enable_csv=enable_csv,
        enable_tensorboard=(
            enable_tensorboard
        ),
        enable_wandb=enable_wandb,
        wandb_entity=CONFIG.get(
            "training_wandb_entity"
        ),
        wandb_project=CONFIG[
            "training_wandb_project"
        ],
        algorithm="masac",
        seed=seed,
        backend_name=(
            env.backend_name
        ),
        config_snapshot=(
            _checkpoint_config_snapshot()
        ),
    )

    eval_env = UAVSearchEnv(
        backend_name=env.backend_name,
        network_backend_kwargs=(
            env.network_backend_kwargs
        ),
    )

    observations, info = env.reset(
        seed=(
            seed + episode_index
        )
    )
    state = info[
        "state"
    ]

    episode_return = 0.0
    episode_length = 0
    episode_diagnostics = (
        new_episode_diagnostics()
    )
    completed_episodes = []
    latest_update_metrics = None
    latest_evaluation = None

    learning_starts = int(
        CONFIG[
            "masac_learning_starts"
        ]
    )
    batch_size = int(
        CONFIG[
            "masac_batch_size"
        ]
    )
    updates_per_step = int(
        CONFIG[
            "masac_updates_per_step"
        ]
    )
    log_interval = max(
        1,
        int(
            CONFIG[
                "training_log_interval_steps"
            ]
        ),
    )
    eval_interval = max(
        1,
        int(
            CONFIG[
                "training_eval_interval_steps"
            ]
        ),
    )
    checkpoint_interval = max(
        1,
        int(
            CONFIG[
                "training_checkpoint_interval_steps"
            ]
        ),
    )
    eval_episodes = max(
        1,
        int(
            CONFIG[
                "training_eval_episodes"
            ]
        ),
    )

    try:
        while global_step < total_steps:
            next_global_step = (
                global_step + 1
            )

            if (
                next_global_step
                <= learning_starts
            ):
                actions = (
                    sample_valid_random_actions(
                        observations,
                        env.agent_ids,
                        rng,
                    )
                )
            else:
                actions = (
                    trainer.select_actions(
                        observations,
                        deterministic=False,
                    )
                )

            (
                next_observations,
                reward,
                terminated,
                truncated,
                next_info,
            ) = env.step(
                actions
            )

            next_state = next_info[
                "state"
            ]

            add_environment_transition_to_replay(
                replay_buffer,
                observations,
                state,
                actions,
                reward,
                next_observations,
                next_state,
                terminated,
                truncated,
                env.agent_ids,
            )

            global_step = (
                next_global_step
            )
            episode_return += float(
                reward
            )
            episode_length += 1

            update_episode_diagnostics(
                episode_diagnostics,
                next_info,
                env=env,
            )

            if (
                global_step
                > learning_starts
                and len(
                    replay_buffer
                )
                >= batch_size
            ):
                for _ in range(
                    updates_per_step
                ):
                    latest_update_metrics = (
                        trainer.update(
                            replay_buffer,
                            batch_size=(
                                batch_size
                            ),
                        )
                    )

            if (
                latest_update_metrics
                is not None
                and global_step
                % log_interval
                == 0
            ):
                logger.log_metrics(
                    "train",
                    filter_training_metrics_for_wandb(
                        "masac",
                        latest_update_metrics,
                    ),
                    step=global_step,
                    episode=episode_index,
                )

            observations = (
                next_observations
            )
            state = next_state

            should_evaluate = (
                global_step
                - last_eval_step
                >= eval_interval
                or global_step
                >= total_steps
            )

            if should_evaluate:
                evaluation_results = (
                    evaluate_masac(
                        eval_env,
                        trainer,
                        episodes=(
                            eval_episodes
                        ),
                        seed=(
                            seed
                            + int(
                                CONFIG[
                                    "training_eval_seed_offset"
                                ]
                            )
                        ),
                    )
                )
                latest_evaluation = (
                    summarize_evaluation_results(
                        evaluation_results
                    )
                )

                logger.log_metrics(
                    "evaluation",
                    latest_evaluation,
                    step=global_step,
                    episode=episode_index,
                )
                logger.update_summary(
                    latest_evaluation,
                    prefix=(
                        "evaluation_latest"
                    ),
                )

                last_eval_step = (
                    global_step
                )

                if evaluation_is_better(
                    latest_evaluation,
                    best_evaluation,
                ):
                    best_evaluation = (
                        copy.deepcopy(
                            latest_evaluation
                        )
                    )

                    best_state = {
                        "seed": seed,
                        "global_step": (
                            global_step
                        ),
                        "episode_index": (
                            episode_index
                        ),
                        "best_evaluation": (
                            best_evaluation
                        ),
                        "last_eval_step": (
                            last_eval_step
                        ),
                        "last_checkpoint_step": (
                            last_checkpoint_step
                        ),
                        "training_rng_state_json": (
                            json.dumps(
                                rng
                                .bit_generator
                                .state
                            )
                        ),
                        "resume_safe": False,
                    }

                    best_checkpoint_path = (
                        save_masac_checkpoint(
                            checkpoint_dir
                            / "best.pt",
                            trainer,
                            replay_buffer=(
                                replay_buffer
                            ),
                            training_state=(
                                best_state
                            ),
                            include_replay=False,
                        )
                    )
                    logger.update_summary(
                        best_evaluation,
                        prefix="best",
                    )
                    logger.log_model_artifact(
                        best_checkpoint_path,
                        aliases=[
                            "best",
                        ],
                        metadata=(
                            best_evaluation
                        ),
                    )

            if not (
                terminated
                or truncated
            ):
                continue

            episode_metrics = (
                finalize_episode_diagnostics(
                    episode_diagnostics,
                    env,
                    episode_return=(
                        episode_return
                    ),
                    episode_length=(
                        episode_length
                    ),
                    success=(
                        next_info[
                            "success"
                        ]
                    ),
                )
            )

            logger.log_metrics(
                "episode",
                episode_metrics,
                step=global_step,
                episode=episode_index,
            )

            completed_episodes.append(
                {
                    "episode": (
                        episode_index
                    ),
                    **episode_metrics,
                    "terminated": bool(
                        terminated
                    ),
                    "truncated": bool(
                        truncated
                    ),
                }
            )

            episode_index += 1

            should_checkpoint = (
                global_step
                - last_checkpoint_step
                >= checkpoint_interval
                or global_step
                >= total_steps
            )

            if should_checkpoint:
                last_checkpoint_step = (
                    global_step
                )

                include_replay = bool(
                    CONFIG[
                        "training_checkpoint_include_replay"
                    ]
                )

                checkpoint_state = {
                    "seed": seed,
                    "global_step": (
                        global_step
                    ),
                    "episode_index": (
                        episode_index
                    ),
                    "best_evaluation": (
                        copy.deepcopy(
                            best_evaluation
                        )
                    ),
                    "last_eval_step": (
                        last_eval_step
                    ),
                    "last_checkpoint_step": (
                        last_checkpoint_step
                    ),
                    "training_rng_state_json": (
                        json.dumps(
                            rng
                            .bit_generator
                            .state
                        )
                    ),
                    "resume_safe": (
                        include_replay
                    ),
                }

                save_masac_checkpoint(
                    checkpoint_dir
                    / "latest.pt",
                    trainer,
                    replay_buffer=(
                        replay_buffer
                    ),
                    training_state=(
                        checkpoint_state
                    ),
                    include_replay=(
                        include_replay
                    ),
                )

            episode_return = 0.0
            episode_length = 0
            episode_diagnostics = (
                new_episode_diagnostics()
            )

            if global_step < total_steps:
                observations, info = (
                    env.reset(
                        seed=(
                            seed
                            + episode_index
                        )
                    )
                )
                state = info[
                    "state"
                ]

        final_state = {
            "seed": seed,
            "global_step": (
                global_step
            ),
            "episode_index": (
                episode_index
            ),
            "best_evaluation": (
                copy.deepcopy(
                    best_evaluation
                )
            ),
            "last_eval_step": (
                last_eval_step
            ),
            "last_checkpoint_step": (
                last_checkpoint_step
            ),
            "training_rng_state_json": (
                json.dumps(
                    rng
                    .bit_generator
                    .state
                )
            ),
            "resume_safe": False,
        }

        # Always keep a lightweight final model checkpoint. If training
        # stops mid-episode it is valid for evaluation, but resume_safe
        # is false because environment state is intentionally not serialized.
        final_checkpoint_path = (
            save_masac_checkpoint(
                checkpoint_dir
                / "final.pt",
                trainer,
                replay_buffer=(
                    replay_buffer
                ),
                training_state=(
                    final_state
                ),
                include_replay=False,
            )
        )
        logger.log_model_artifact(
            final_checkpoint_path,
            aliases=[
                "final",
            ],
            metadata={
                "global_step": (
                    global_step
                ),
                "episode_index": (
                    episode_index
                ),
            },
        )
        logger.update_summary(
            {
                "global_step": (
                    global_step
                ),
                "episodes_completed": (
                    episode_index
                ),
                "trainer_updates": int(
                    trainer.update_count
                ),
            },
            prefix="training_final",
        )

        return {
            "trainer": trainer,
            "replay_buffer": replay_buffer,
            "completed_episodes": (
                completed_episodes
            ),
            "latest_update_metrics": (
                latest_update_metrics
            ),
            "latest_evaluation": (
                latest_evaluation
            ),
            "best_evaluation": (
                best_evaluation
            ),
            "training_state": (
                final_state
            ),
            "logger_status": (
                logger.status()
            ),
            "run_dir": run_dir,
            "checkpoint_dir": (
                checkpoint_dir
            ),
        }
    finally:
        eval_env.close()
        logger.close()


# --- frozen notebook cell 187 ---
def straight_through_masked_argmax(
    logits,
    destination_mask,
    temperature=None,
):
    if temperature is None:
        temperature = CONFIG[
            "matd3_gumbel_temperature"
        ]

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

    probabilities = F.softmax(
        masked_logits
        / temperature,
        dim=-1,
    )

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

    action = (
        hard
        - probabilities.detach()
        + probabilities
    )

    return {
        "action": action,
        "hard_action": hard,
        "indices": indices,
        "probabilities": probabilities,
    }


# Export public and internal helpers required by downstream production layers.
__all__ = [name for name in globals() if not name.startswith("__")]
