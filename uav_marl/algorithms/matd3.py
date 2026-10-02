"""Production source migrated from frozen fix_test_gpu.ipynb.

Do not import the notebook at runtime. The notebook is retained only as a
reference/regression artifact. Production changes belong in Python modules.
Migrated from notebook cells 188..220.
"""

from .masac import *  # noqa: F401,F403

# --- frozen notebook cell 188 ---
class HybridMATD3Actor(nn.Module):
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

        self.encoder = build_mlp(
            self.observation_dim,
            hidden_dims[:-1],
            hidden_dims[-1],
        )

        feature_dim = hidden_dims[-1]

        self.continuous_head = nn.Linear(
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
            observations
        )

        raw_continuous = self.continuous_head(
            features
        )
        motion, _ = radial_squash_motion_action(
            raw_continuous[..., :3]
        )
        power = torch.tanh(
            raw_continuous[..., 3:4]
        )
        continuous = torch.cat(
            [
                motion,
                power,
            ],
            dim=-1,
        )
        logits = self.discrete_head(
            features
        )

        return (
            continuous,
            logits,
        )

    def actions(
        self,
        observations,
        destination_mask,
        straight_through=True,
    ):
        (
            continuous,
            logits,
        ) = self(
            observations
        )

        discrete = (
            straight_through_masked_argmax(
                logits,
                destination_mask,
            )
        )

        destination_action = (
            discrete["action"]
            if straight_through
            else discrete[
                "hard_action"
            ]
        )

        return {
            "continuous": continuous,
            "destination_one_hot": (
                destination_action
            ),
            "destination_hard_one_hot": (
                discrete[
                    "hard_action"
                ]
            ),
            "destination_indices": (
                discrete["indices"]
            ),
            "destination_probabilities": (
                discrete[
                    "probabilities"
                ]
            ),
        }


# --- frozen notebook cell 189 ---
class HybridMATD3:
    def __init__(
        self,
        env,
        device=None,
        seed=None,
    ):
        if not isinstance(
            env,
            UAVSearchEnv,
        ):
            raise TypeError(
                "env must be UAVSearchEnv"
            )

        if seed is None:
            seed = int(
                CONFIG["seed"]
            )

        self.seed = int(seed)
        self.rng = (
            np.random.default_rng(
                self.seed
            )
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
                "matd3_hidden_dims"
            ]
        )

        self.actor = HybridMATD3Actor(
            self.observation_dim,
            self.continuous_dim,
            self.discrete_dim,
            hidden_dims,
        ).to(
            self.device
        )

        self.target_actor = copy.deepcopy(
            self.actor
        ).to(
            self.device
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
            self.target_actor,
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
                        "matd3_actor_lr"
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
                        "matd3_critic_lr"
                    ]
                ),
            )
        )

        self.gamma = float(
            CONFIG["matd3_gamma"]
        )
        self.tau = float(
            CONFIG["matd3_tau"]
        )
        self.policy_delay = int(
            CONFIG[
                "matd3_policy_delay"
            ]
        )
        self.target_policy_noise = float(
            CONFIG[
                "matd3_target_policy_noise"
            ]
        )
        self.target_noise_clip = float(
            CONFIG[
                "matd3_target_noise_clip"
            ]
        )
        self.exploration_noise = float(
            CONFIG[
                "matd3_exploration_noise"
            ]
        )
        self.gradient_clip_norm = float(
            CONFIG[
                "matd3_gradient_clip_norm"
            ]
        )

        if self.policy_delay < 1:
            raise ValueError(
                "matd3_policy_delay "
                "must be >= 1"
            )

        self.update_count = 0
        self.action_count = 0
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

    def make_replay_buffer(
        self,
        seed=None,
    ):
        if seed is None:
            seed = self.seed

        return HybridReplayBuffer(
            capacity=CONFIG[
                "matd3_replay_capacity"
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

    def _actor_joint_actions(
        self,
        observations,
        destination_masks,
        straight_through=True,
        target_actor=False,
    ):
        if observations.ndim != 3:
            raise ValueError(
                "observations must "
                "have rank 3"
            )

        batch_size = int(
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

        actor = (
            self.target_actor
            if target_actor
            else self.actor
        )

        sampled = actor.actions(
            flat_observations,
            flat_masks,
            straight_through=(
                straight_through
            ),
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
                "destination_indices"
            ].reshape(
                batch_size,
                self.num_agents,
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
        }

    def _discrete_epsilon(
        self,
        step,
    ):
        start = float(
            CONFIG[
                "matd3_discrete_epsilon_start"
            ]
        )
        end = float(
            CONFIG[
                "matd3_discrete_epsilon_end"
            ]
        )
        decay_steps = max(
            1,
            int(
                CONFIG[
                    "matd3_discrete_epsilon_decay_steps"
                ]
            ),
        )

        fraction = min(
            1.0,
            max(
                0.0,
                float(step)
                / decay_steps,
            ),
        )

        return float(
            start
            + fraction
            * (
                end - start
            )
        )

    def select_actions(
        self,
        observations,
        deterministic=False,
        exploration_step=None,
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
            )
        )
        mask_tensor = torch.as_tensor(
            destination_masks,
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            sampled = self.actor.actions(
                observation_tensor,
                mask_tensor,
                straight_through=False,
            )

        continuous = (
            sampled[
                "continuous"
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
            ]
            .cpu()
            .numpy()
            .astype(
                np.int64
            )
        )

        if not deterministic:
            noise = self.rng.normal(
                loc=0.0,
                scale=self.exploration_noise,
                size=continuous.shape,
            ).astype(
                np.float32
            )

            continuous = np.clip(
                continuous + noise,
                -1.0,
                1.0,
            ).astype(
                np.float32
            )
            continuous[:, :3] = np.asarray(
                project_motion_action_np(
                    continuous[:, :3]
                ),
                dtype=np.float32,
            )

            if exploration_step is None:
                exploration_step = (
                    self.action_count
                )

            epsilon = (
                self._discrete_epsilon(
                    exploration_step
                )
            )

            for agent_index in range(
                self.num_agents
            ):
                if (
                    self.rng.random()
                    >= epsilon
                ):
                    continue

                valid = np.flatnonzero(
                    destination_masks[
                        agent_index
                    ]
                    > 0.5
                )

                destination_indices[
                    agent_index
                ] = int(
                    self.rng.choice(
                        valid
                    )
                )

            self.action_count += 1

        return masac_arrays_to_env_actions(
            continuous,
            destination_indices,
            self.agent_ids,
        )

    def _soft_update(
        self,
        target,
        source,
    ):
        with torch.no_grad():
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
                    "matd3_batch_size"
                ]
            )

        batch = replay_buffer.sample(
            batch_size,
            self.device,
        )

        replay_destination_one_hot = (
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
                replay_destination_one_hot,
            )
        )

        with torch.no_grad(), self._autocast():
            target_policy = (
                self._actor_joint_actions(
                    batch[
                        "next_observations"
                    ],
                    batch[
                        "next_destination_masks"
                    ],
                    straight_through=False,
                    target_actor=True,
                )
            )

            noise = (
                torch.randn_like(
                    target_policy[
                        "continuous"
                    ]
                )
                * self.target_policy_noise
            ).clamp(
                -self.target_noise_clip,
                self.target_noise_clip,
            )

            target_continuous = (
                target_policy[
                    "continuous"
                ]
                + noise
            ).clamp(
                -1.0,
                1.0,
            )
            target_continuous = torch.cat(
                [
                    project_motion_action_tensor(
                        target_continuous[..., :3]
                    ),
                    target_continuous[..., 3:4],
                ],
                dim=-1,
            )

            target_joint_action = (
                self._joint_action_tensor(
                    target_continuous,
                    target_policy[
                        "destination_one_hot"
                    ],
                )
            )

            target_q = torch.minimum(
                self.target_critic_1(
                    batch[
                        "next_states"
                    ],
                    target_joint_action,
                ),
                self.target_critic_2(
                    batch[
                        "next_states"
                    ],
                    target_joint_action,
                ),
            )

            critic_target = (
                batch["rewards"]
                + self.gamma
                * (
                    1.0
                    - batch[
                        "terminated"
                    ]
                )
                * target_q
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
            torch.nn.utils.clip_grad_norm_(
                list(
                    self.critic_1.parameters()
                )
                + list(
                    self.critic_2.parameters()
                ),
                self.gradient_clip_norm,
            )
        )

        if self.amp_enabled:
            self.grad_scaler.step(
                self.critic_optimizer
            )
        else:
            self.critic_optimizer.step()

        self.update_count += 1

        actor_updated = bool(
            self.update_count
            % self.policy_delay
            == 0
        )
        actor_loss_value = None
        actor_grad_norm_value = None

        if actor_updated:
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
                policy = (
                    self._actor_joint_actions(
                        batch[
                            "observations"
                        ],
                        batch[
                            "destination_masks"
                        ],
                        straight_through=True,
                        target_actor=False,
                    )
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

                actor_loss = -self.critic_1(
                    batch["states"],
                    policy_joint_action,
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
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    self.gradient_clip_norm,
                )
            )

            if self.amp_enabled:
                self.grad_scaler.step(
                    self.actor_optimizer
                )
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

            self._soft_update(
                self.target_actor,
                self.actor,
            )
            self._soft_update(
                self.target_critic_1,
                self.critic_1,
            )
            self._soft_update(
                self.target_critic_2,
                self.critic_2,
            )

            actor_loss_value = (
                actor_loss.detach()
            )
            actor_grad_norm_value = (
                torch.as_tensor(
                    actor_grad_norm
                ).detach()
            )

        if self.amp_enabled:
            self.grad_scaler.update()

        if not collect_metrics:
            return None

        if actor_loss_value is not None:
            actor_loss_value = float(
                actor_loss_value.cpu()
            )
            actor_grad_norm_value = float(
                actor_grad_norm_value.cpu()
            )

        metrics = {
            "critic_loss": float(
                critic_loss
                .detach()
                .cpu()
            ),
            "critic_1_q_mean": float(
                critic_1_value
                .mean()
                .detach()
                .cpu()
            ),
            "critic_2_q_mean": float(
                critic_2_value
                .mean()
                .detach()
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
            "actor_updated": (
                actor_updated
            ),
            "actor_loss": (
                actor_loss_value
            ),
            "actor_grad_norm": (
                actor_grad_norm_value
            ),
            "update_count": int(
                self.update_count
            ),
        }

        for key, value in (
            metrics.items()
        ):
            if value is None:
                continue

            if isinstance(
                value,
                bool,
            ):
                continue

            if not np.isfinite(
                value
            ):
                raise FloatingPointError(
                    f"non-finite MATD3 metric: "
                    f"{key}"
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
            "target_actor": (
                self.target_actor
                .state_dict()
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
            "update_count": int(
                self.update_count
            ),
            "action_count": int(
                self.action_count
            ),
            "rng_state_json": json.dumps(
                self.rng
                .bit_generator
                .state
            ),
        }

    def load_state_dict(
        self,
        state,
    ):
        self.actor.load_state_dict(
            state["actor"]
        )
        self.target_actor.load_state_dict(
            state["target_actor"]
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

        for optimizer in (
            self.actor_optimizer,
            self.critic_optimizer,
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
        self.action_count = int(
            state.get(
                "action_count",
                0,
            )
        )

        self.rng.bit_generator.state = (
            json.loads(
                state[
                    "rng_state_json"
                ]
            )
        )


# --- frozen notebook cell 190 ---
def train_matd3(
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

    seed = int(seed)

    set_seed(seed)

    if trainer is None:
        trainer = HybridMATD3(
            env,
            seed=seed,
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
    state = info["state"]

    episode_return = 0.0
    episode_length = 0
    episode_index = 0
    completed_episodes = []
    latest_update_metrics = None

    learning_starts = int(
        CONFIG[
            "matd3_learning_starts"
        ]
    )
    updates_per_step = int(
        CONFIG[
            "matd3_updates_per_step"
        ]
    )
    batch_size = int(
        CONFIG[
            "matd3_batch_size"
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
                    exploration_step=(
                        global_step
                    ),
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


# --- frozen notebook cell 191 ---
def evaluate_matd3(
    env,
    trainer,
    episodes=3,
    seed=None,
):
    if not isinstance(
        trainer,
        HybridMATD3,
    ):
        raise TypeError(
            "trainer must be HybridMATD3"
        )

    if seed is None:
        seed = int(
            CONFIG["seed"]
        ) + 20_000

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
            new_episode_diagnostics()
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


# --- frozen notebook cell 192 ---
def save_matd3_checkpoint(
    path,
    trainer,
    replay_buffer=None,
    training_state=None,
    include_replay=None,
):
    if not isinstance(
        trainer,
        HybridMATD3,
    ):
        raise TypeError(
            "trainer must be HybridMATD3"
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

    if training_state is None:
        training_state = {}

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "format_version": 1,
        "algorithm": "hybrid_matd3",
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


# --- frozen notebook cell 193 ---
def load_matd3_checkpoint(
    path,
    env,
    device=None,
    load_replay=True,
    restore_torch_rng=True,
):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            str(path)
        )

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if payload.get(
        "algorithm"
    ) != "hybrid_matd3":
        raise ValueError(
            "checkpoint algorithm "
            "is not hybrid_matd3"
        )

    trainer = HybridMATD3(
        env,
        device=device,
        seed=int(
            payload.get(
                "training_state",
                {},
            ).get(
                "seed",
                CONFIG["seed"],
            )
        ),
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
        replay_buffer = (
            trainer.make_replay_buffer(
                seed=trainer.seed
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


# --- frozen notebook cell 194 ---
def train_matd3_experiment(
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

    seed = int(seed)

    if run_name is None:
        run_name = (
            make_experiment_run_name(
                "matd3",
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
                        "matd3_training_output_dir"
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

    set_seed(seed)

    if resume_checkpoint is None:
        trainer = HybridMATD3(
            env,
            device=device,
            seed=seed,
        )
        replay_buffer = (
            trainer.make_replay_buffer(
                seed=seed
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
            load_matd3_checkpoint(
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
            "matd3",
        )

        if replay_buffer is None:
            raise ValueError(
                "resume checkpoint does not "
                "contain replay state; set "
                "training_checkpoint_include_replay=True"
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

        if int(
            training_state.get(
                "seed",
                seed,
            )
        ) != seed:
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
            "matd3_wandb_project"
        ],
        algorithm="matd3",
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
            "matd3_learning_starts"
        ]
    )
    batch_size = int(
        CONFIG[
            "matd3_batch_size"
        ]
    )
    updates_per_step = int(
        CONFIG[
            "matd3_updates_per_step"
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
                        exploration_step=(
                            next_global_step
                        ),
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
                        "matd3",
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
                    evaluate_matd3(
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

                    best_checkpoint_path = (
                        save_matd3_checkpoint(
                            checkpoint_dir
                            / "best.pt",
                            trainer,
                            replay_buffer=(
                                replay_buffer
                            ),
                            training_state={
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
                            },
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

                save_matd3_checkpoint(
                    checkpoint_dir
                    / "latest.pt",
                    trainer,
                    replay_buffer=(
                        replay_buffer
                    ),
                    training_state={
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
                    },
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

        final_checkpoint_path = (
            save_matd3_checkpoint(
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


# --- frozen notebook cell 195 ---
def running_on_kaggle():
    return bool(
        os.environ.get(
            "KAGGLE_KERNEL_RUN_TYPE"
        )
        or Path(
            "/kaggle/working"
        ).is_dir()
    )


# --- frozen notebook cell 196 ---
def resolve_kaggle_account():
    """Return the authenticated Kaggle username, or the literal "None" off Kaggle."""
    if not running_on_kaggle():
        return "None"

    env_username = str(
        os.environ.get("KAGGLE_USERNAME", "")
    ).strip()
    if env_username:
        return env_username

    username = None
    try:
        from kagglehub.auth import get_username
        username = get_username()
    except Exception:
        username = None

    if not username:
        try:
            kagglehub = importlib.import_module("kagglehub")
            whoami = getattr(kagglehub, "whoami", None)
            if callable(whoami):
                identity = whoami(verbose=False)
                if isinstance(identity, dict):
                    username = identity.get("username")
        except Exception:
            username = None

    if username:
        return str(username).strip()

    if bool(
        CONFIG.get(
            "training_require_kaggle_account_name",
            True,
        )
    ):
        raise RuntimeError(
            "Running on Kaggle but the authenticated Kaggle account name "
            "could not be resolved. Ensure Kaggle notebook authentication is "
            "available or set KAGGLE_USERNAME before starting full-flow training."
        )
    return "None"


# --- frozen notebook cell 197 ---
def kaggle_runtime_status():
    cuda_available = bool(
        torch.cuda.is_available()
    )
    device_count = (
        torch.cuda.device_count()
        if cuda_available
        else 0
    )
    gpu_names = [
        torch.cuda.get_device_name(
            index
        )
        for index in range(
            device_count
        )
    ]

    return {
        "running_on_kaggle": (
            running_on_kaggle()
        ),
        "kaggle_account": resolve_kaggle_account(),
        "cuda_available": (
            cuda_available
        ),
        "cuda_device_count": int(
            device_count
        ),
        "device": (
            "cuda"
            if cuda_available
            else "cpu"
        ),
        "gpu_names": gpu_names,
        "wandb_api_key_present": bool(
            os.environ.get(
                "WANDB_API_KEY"
            )
        ),
    }


# --- frozen notebook cell 198 ---
def configure_kaggle_wandb(
    secret_names=None,
    verify_login=True,
):
    result = {
        "running_on_kaggle": (
            running_on_kaggle()
        ),
        "configured": False,
        "secret_name": None,
        "error": None,
    }

    if not result[
        "running_on_kaggle"
    ]:
        result["error"] = (
            "not running inside a "
            "Kaggle notebook"
        )
        return result

    if secret_names is None:
        secret_names = CONFIG[
            "kaggle_wandb_secret_names"
        ]

    try:
        from kaggle_secrets import (
            UserSecretsClient,
        )

        client = UserSecretsClient()
    except Exception as exc:  # noqa: BLE001 - Kaggle-only optional API
        result["error"] = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )
        return result

    api_key = os.environ.get(
        "WANDB_API_KEY"
    )
    selected_name = None

    if not api_key:
        for secret_name in (
            secret_names
        ):
            try:
                candidate = (
                    client.get_secret(
                        str(
                            secret_name
                        )
                    )
                )
            except Exception:  # noqa: BLE001,S112 - try the next configured label
                continue

            if candidate:
                api_key = candidate
                selected_name = str(
                    secret_name
                )
                break

    if not api_key:
        result["error"] = (
            "no attached W&B secret "
            "was found"
        )
        return result

    os.environ[
        "WANDB_API_KEY"
    ] = api_key

    try:
        wandb = importlib.import_module(
            "wandb"
        )

        if verify_login:
            wandb.login(
                key=api_key,
                relogin=True,
                verify=True,
            )

        result["configured"] = True
        result["secret_name"] = (
            selected_name
        )
    except Exception as exc:  # noqa: BLE001 - optional logger must not crash setup
        result["error"] = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    return result


# --- frozen notebook cell 199 ---
def prepare_kaggle_training(
    algorithm="matd3",
    configure_wandb=True,
):
    normalized = str(
        algorithm
    ).strip().lower()

    if normalized not in {
        "matd3",
        "masac",
    }:
        raise ValueError(
            "algorithm must be "
            "'matd3' or 'masac'"
        )

    status = (
        kaggle_runtime_status()
    )

    if running_on_kaggle():
        output_root = Path(
            "/kaggle/working"
        )

        if normalized == "matd3":
            CONFIG[
                "matd3_training_output_dir"
            ] = str(
                output_root
                / "outputs"
                / "matd3"
            )
        else:
            CONFIG[
                "training_output_dir"
            ] = str(
                output_root
                / "outputs"
                / "masac"
            )

    wandb_status = None

    if configure_wandb:
        wandb_status = (
            configure_kaggle_wandb()
        )

        if (
            wandb_status.get(
                "configured"
            )
        ):
            CONFIG[
                "training_enable_wandb"
            ] = True
            CONFIG[
                "training_wandb_mode"
            ] = "online"

    return {
        **status,
        "algorithm": normalized,
        "wandb": wandb_status,
        "output_dir": (
            CONFIG[
                "matd3_training_output_dir"
            ]
            if normalized == "matd3"
            else CONFIG[
                "training_output_dir"
            ]
        ),
    }


# --- frozen notebook cell 200 ---
def run_kaggle_matd3_experiment(
    total_steps=100_000,
    seed=None,
    run_name=None,
    backend_name="simple",
    enable_wandb=True,
):
    if not running_on_kaggle():
        raise RuntimeError(
            "run_kaggle_matd3_experiment "
            "must be called inside a "
            "Kaggle notebook"
        )

    if seed is None:
        seed = int(
            CONFIG["seed"]
        )

    seed = int(seed)

    if run_name is None:
        run_name = (
            make_experiment_run_name(
                "matd3",
                seed,
            )
        )

    preparation = (
        prepare_kaggle_training(
            algorithm="matd3",
            configure_wandb=(
                enable_wandb
            ),
        )
    )

    if (
        enable_wandb
        and not preparation[
            "wandb"
        ].get(
            "configured",
            False,
        )
    ):
        raise RuntimeError(
            "W&B was requested but no "
            "attached Kaggle secret could "
            "be configured. Add a Kaggle "
            "Secret named WANDB_API_KEY "
            "or wandb_key."
        )

    run_dir = (
        Path(
            preparation[
                "output_dir"
            ]
        )
        / run_name
    )

    result = run_training_experiment(
        "matd3",
        total_steps=total_steps,
        seed=seed,
        backend_name=backend_name,
        run_name=run_name,
        run_dir=run_dir,
        device=preparation[
            "device"
        ],
        enable_csv=True,
        enable_tensorboard=(
            CONFIG[
                "training_enable_tensorboard"
            ]
        ),
        enable_wandb=enable_wandb,
    )

    return {
        **result,
        "kaggle_status": preparation,
    }


# --- frozen notebook cell 201 ---
def _select_vector_policy_actions(
    trainer,
    observations_by_env,
    algorithm,
    exploration_steps,
):
    num_envs = len(observations_by_env)
    arrays = [
        observations_to_masac_arrays(
            observations,
            trainer.agent_ids,
        )
        for observations in observations_by_env
    ]
    observation_vectors = np.stack(
        [item[0] for item in arrays],
        axis=0,
    )
    destination_masks = np.stack(
        [item[1] for item in arrays],
        axis=0,
    )
    observation_tensor = torch.as_tensor(
        observation_vectors,
        dtype=torch.float32,
        device=trainer.device,
    )
    mask_tensor = torch.as_tensor(
        destination_masks,
        dtype=torch.float32,
        device=trainer.device,
    )

    if algorithm == "masac":
        with torch.no_grad():
            sampled = trainer._sample_joint_policy(
                observation_tensor,
                mask_tensor,
                deterministic=False,
            )
        continuous = (
            sampled["continuous"]
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        destination_indices = (
            sampled["destination_indices"]
            .cpu()
            .numpy()
            .astype(np.int64)
        )
    else:
        flat_observations = observation_tensor.reshape(
            num_envs * trainer.num_agents,
            trainer.observation_dim,
        )
        flat_masks = mask_tensor.reshape(
            num_envs * trainer.num_agents,
            trainer.discrete_dim,
        )
        with torch.no_grad():
            sampled = trainer.actor.actions(
                flat_observations,
                flat_masks,
                straight_through=False,
            )
        continuous = (
            sampled["continuous"]
            .reshape(
                num_envs,
                trainer.num_agents,
                trainer.continuous_dim,
            )
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        destination_indices = (
            sampled["destination_indices"]
            .reshape(
                num_envs,
                trainer.num_agents,
            )
            .cpu()
            .numpy()
            .astype(np.int64)
        )
        noise = trainer.rng.normal(
            loc=0.0,
            scale=trainer.exploration_noise,
            size=continuous.shape,
        ).astype(np.float32)
        continuous = np.clip(
            continuous + noise,
            -1.0,
            1.0,
        ).astype(np.float32)

        for env_index in range(num_envs):
            epsilon = trainer._discrete_epsilon(
                exploration_steps[env_index]
            )
            for agent_index in range(
                trainer.num_agents
            ):
                if trainer.rng.random() >= epsilon:
                    continue
                valid = np.flatnonzero(
                    destination_masks[
                        env_index,
                        agent_index,
                    ]
                    > 0.5
                )
                destination_indices[
                    env_index,
                    agent_index,
                ] = int(
                    trainer.rng.choice(valid)
                )
        trainer.action_count += num_envs

    return [
        masac_arrays_to_env_actions(
            continuous[index],
            destination_indices[index],
            trainer.agent_ids,
        )
        for index in range(num_envs)
    ]


# --- frozen notebook cell 202 ---
def _vector_batch_item(value, index, num_envs):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if str(key).startswith("_"):
                continue
            mask = value.get(f"_{key}")
            if (
                isinstance(mask, np.ndarray)
                and mask.shape == (num_envs,)
                and not bool(mask[index])
            ):
                continue
            result[key] = _vector_batch_item(
                item,
                index,
                num_envs,
            )
        return result

    if (
        isinstance(value, np.ndarray)
        and value.ndim >= 1
        and value.shape[0] == num_envs
    ):
        return copy.deepcopy(value[index])

    return copy.deepcopy(value)


# --- frozen notebook cell 203 ---
def _vector_observation_at(
    observations,
    index,
    num_envs,
):
    return _vector_batch_item(
        observations,
        index,
        num_envs,
    )


# --- frozen notebook cell 204 ---
def _vector_info_at(
    infos,
    index,
    num_envs,
):
    return _vector_batch_item(
        infos,
        index,
        num_envs,
    )


# --- frozen notebook cell 205 ---
def _stack_vector_actions(
    actions_by_env,
    agent_ids,
):
    return {
        agent_id: {
            "motion": np.stack(
                [
                    np.asarray(
                        actions[agent_id]["motion"],
                        dtype=np.float32,
                    )
                    for actions in actions_by_env
                ],
                axis=0,
            ),
            "destination": np.asarray(
                [
                    int(
                        actions[agent_id][
                            "destination"
                        ]
                    )
                    for actions in actions_by_env
                ],
                dtype=np.int64,
            ),
            "power": np.stack(
                [
                    np.asarray(
                        actions[agent_id]["power"],
                        dtype=np.float32,
                    )
                    for actions in actions_by_env
                ],
                axis=0,
            ),
        }
        for agent_id in agent_ids
    }


# --- frozen notebook cell 206 ---
def _new_vector_episode_tracker(
    episode_id,
):
    return {
        "episode_id": int(episode_id),
        "return": 0.0,
        "length": 0,
        "diagnostics": (
            new_episode_diagnostics()
        ),
        "confirmed_target_ids": set(),
        "delivered_target_ids": set(),
    }


# --- frozen notebook cell 207 ---
def _update_vector_episode_tracker(
    tracker,
    info,
    reward,
):
    tracker["return"] += float(reward)
    tracker["length"] += 1

    update_episode_diagnostics(
        tracker["diagnostics"],
        info,
    )

    tracker[
        "confirmed_target_ids"
    ].update(
        int(target_id)
        for target_id in info.get(
            "newly_confirmed_target_ids",
            [],
        )
    )
    tracker[
        "delivered_target_ids"
    ].update(
        int(target_id)
        for target_id in info.get(
            "newly_delivered_target_ids",
            [],
        )
    )


# --- frozen notebook cell 208 ---
def _finalize_vector_episode_tracker(
    tracker,
    success,
):
    diagnostics = tracker["diagnostics"]
    target_count = max(
        1,
        int(CONFIG["num_targets"]),
    )
    possible_agent_steps = max(
        1,
        int(tracker["length"])
        * int(CONFIG["num_uavs"]),
    )

    metrics = {
        "return": float(
            tracker["return"]
        ),
        "length": int(
            tracker["length"]
        ),
        "success": float(
            bool(success)
        ),
        "delivery_rate": float(
            len(
                tracker[
                    "delivered_target_ids"
                ]
            )
            / target_count
        ),
        "confirmation_rate": float(
            len(
                tracker[
                    "confirmed_target_ids"
                ]
            )
            / target_count
        ),
        "false_confirmations": int(
            diagnostics[
                "false_confirmations"
            ]
        ),
        "expired_reports": int(
            diagnostics["expired_reports"]
        ),
        "dropped_reports": int(
            diagnostics["dropped_reports"]
        ),
        "obstacle_block_rate": float(
            diagnostics["obstacle_blocks"]
            / possible_agent_steps
        ),
        "peer_safety_block_rate": float(
            diagnostics["peer_safety_blocks"]
            / possible_agent_steps
        ),
        "boundary_clip_rate": float(
            diagnostics["boundary_clips"]
            / possible_agent_steps
        ),
        "total_energy_j": float(
            diagnostics["total_energy_j"]
        ),
        "communication_energy_j": float(
            diagnostics[
                "communication_energy_j"
            ]
        ),
        "information_gain_bits": float(
            diagnostics[
                "information_gain_bits"
            ]
        ),
    }
    metrics.update(
        extract_network_kpis(
            diagnostics[
                "network_snapshot"
            ]
        )
    )
    return metrics


# --- frozen notebook cell 209 ---
def _vector_training_is_enabled(
    env,
    device=None,
):
    vector_runtime = (
        resolve_training_vector_runtime(
            device
        )
    )
    return bool(
        env.backend_name == "simple"
        and int(
            vector_runtime["num_envs"]
        )
        > 1
    )


# --- frozen notebook cell 210 ---
def _count_vector_updates_due(
    previous_global_step,
    num_envs,
    learning_starts,
    replay_len_before,
    replay_capacity,
    batch_size,
    updates_per_step,
):
    updates_due = 0

    for offset in range(
        int(num_envs)
    ):
        transition_step = (
            int(previous_global_step)
            + offset
            + 1
        )
        replay_size_after = min(
            int(replay_capacity),
            int(replay_len_before)
            + offset
            + 1,
        )

        if (
            transition_step
            > int(learning_starts)
            and replay_size_after
            >= int(batch_size)
        ):
            updates_due += int(
                updates_per_step
            )

    return int(updates_due)


# --- frozen notebook cell 211 ---
def _vector_system_metrics(
    trainer,
    global_step,
    wall_start,
    evaluation_seconds,
    env_collection_seconds,
    env_wait_seconds,
    policy_inference_seconds,
    update_seconds,
    num_envs,
    overlap,
    initial_global_step=0,
    prior_metrics=None,
):
    if prior_metrics is None:
        prior_metrics = {}

    session_wall_seconds = max(
        np.finfo(np.float64).eps,
        time.perf_counter()
        - float(wall_start),
    )
    session_core_seconds = max(
        np.finfo(np.float64).eps,
        session_wall_seconds
        - float(evaluation_seconds),
    )

    total_wall_seconds = (
        float(
            prior_metrics.get(
                "training_wall_time_seconds",
                0.0,
            )
        )
        + session_wall_seconds
    )
    total_evaluation_seconds = (
        float(
            prior_metrics.get(
                "evaluation_seconds",
                0.0,
            )
        )
        + float(evaluation_seconds)
    )
    core_training_seconds = max(
        np.finfo(np.float64).eps,
        total_wall_seconds
        - total_evaluation_seconds,
    )
    session_transitions = max(
        0,
        int(global_step)
        - int(initial_global_step),
    )

    metrics = {
        "transitions_per_second": (
            float(global_step)
            / core_training_seconds
        ),
        "session_transitions_per_second": (
            float(session_transitions)
            / session_core_seconds
        ),
        "training_wall_time_seconds": (
            total_wall_seconds
        ),
        "core_training_seconds": (
            core_training_seconds
        ),
        "evaluation_seconds": (
            total_evaluation_seconds
        ),
        "env_collection_seconds": (
            float(
                prior_metrics.get(
                    "env_collection_seconds",
                    0.0,
                )
            )
            + float(
                env_collection_seconds
            )
        ),
        "env_wait_seconds": (
            float(
                prior_metrics.get(
                    "env_wait_seconds",
                    0.0,
                )
            )
            + float(env_wait_seconds)
        ),
        "policy_inference_seconds": (
            float(
                prior_metrics.get(
                    "policy_inference_seconds",
                    0.0,
                )
            )
            + float(
                policy_inference_seconds
            )
        ),
        "update_seconds": (
            float(
                prior_metrics.get(
                    "update_seconds",
                    0.0,
                )
            )
            + float(update_seconds)
        ),
        "num_envs": float(num_envs),
        "overlap_enabled": float(
            bool(overlap)
        ),
    }

    device = getattr(
        trainer,
        "device",
        None,
    )

    if (
        isinstance(
            device,
            torch.device,
        )
        and device.type == "cuda"
    ):
        device_index = (
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        )
        metrics[
            "gpu_memory_allocated_mb"
        ] = float(
            torch.cuda.memory_allocated(
                device_index
            )
            / (1024.0 ** 2)
        )
        metrics[
            "gpu_peak_memory_allocated_mb"
        ] = float(
            torch.cuda.max_memory_allocated(
                device_index
            )
            / (1024.0 ** 2)
        )

    return metrics


# --- frozen notebook cell 212 ---
def _capture_vector_env_states(
    vector_env,
):
    if isinstance(
        vector_env,
        TorchThreadedSimpleVectorEnv,
    ):
        return [
            env.checkpoint_state()
            for env in vector_env.envs
        ]

    # Forked CPU workers must not enter the torch runtime. Environment
    # checkpoint arrays are therefore encoded as dtype/shape/raw bytes.
    return list(
        vector_env.call(
            "checkpoint_state"
        )
    )


# --- frozen notebook cell 213 ---
def _restore_vector_env_states(
    vector_env,
    env_states,
    num_envs,
):
    if len(env_states) != int(
        num_envs
    ):
        raise ValueError(
            "checkpoint environment count "
            "does not match vector runtime"
        )

    if isinstance(
        vector_env,
        TorchThreadedSimpleVectorEnv,
    ):
        restored = [
            env.restore_checkpoint_state(
                state
            )
            for env, state
            in zip(
                vector_env.envs,
                env_states,
            )
        ]
        vector_env._last_observations = [
            item[0]
            for item in restored
        ]
        vector_env._last_infos = [
            item[1]
            for item in restored
        ]
    else:
        vector_env.set_attr(
            "_pending_checkpoint_state",
            tuple(env_states),
        )
        restored = list(
            vector_env.call(
                "restore_pending_checkpoint_state"
            )
        )

    observations = _stack_vector_dicts(
        [
            item[0]
            for item in restored
        ]
    )
    states = np.stack(
        [
            np.asarray(
                item[1]["state"],
                dtype=np.float32,
            )
            for item in restored
        ],
        axis=0,
    )
    return observations, states


# --- frozen notebook cell 214 ---
def _make_vector_checkpoint_state(
    *,
    algorithm,
    seed,
    global_step,
    best_evaluation,
    latest_evaluation,
    last_eval_step,
    last_checkpoint_step,
    rng,
    num_envs,
    vector_backend,
    vector_env,
    trackers,
    completed_episodes,
    next_episode_id,
    system_metrics,
    wandb_run_id,
    resume_safe,
):
    return {
        "seed": int(seed),
        "algorithm": str(algorithm),
        "global_step": int(
            global_step
        ),
        "episode_index": len(
            completed_episodes
        ),
        "completed_episodes": (
            _checkpoint_pack(
                completed_episodes
            )
        ),
        "trackers": _checkpoint_pack(
            trackers
        ),
        "next_episode_id": int(
            next_episode_id
        ),
        "best_evaluation": (
            copy.deepcopy(
                best_evaluation
            )
        ),
        "latest_evaluation": (
            copy.deepcopy(
                latest_evaluation
            )
        ),
        "last_eval_step": int(
            last_eval_step
        ),
        "last_checkpoint_step": int(
            last_checkpoint_step
        ),
        "training_rng_state_json": (
            json.dumps(
                rng.bit_generator.state
            )
        ),
        "vector_env_states": (
            _capture_vector_env_states(
                vector_env
            )
        ),
        "system_metrics": (
            copy.deepcopy(
                system_metrics
            )
        ),
        "wandb_run_id": wandb_run_id,
        "resume_safe": bool(
            resume_safe
        ),
        "vectorized": True,
        "num_envs": int(num_envs),
        "vector_backend": str(
            vector_backend
        ),
    }


# --- frozen notebook cell 215 ---
def train_vectorized_experiment(
    algorithm,
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
    normalized = str(
        algorithm
    ).strip().lower()

    if normalized not in {
        "masac",
        "matd3",
    }:
        raise ValueError(
            "algorithm must be "
            "'masac' or 'matd3'"
        )

    if not isinstance(
        env,
        UAVSearchEnv,
    ):
        raise TypeError(
            "env must be UAVSearchEnv"
        )

    if env.backend_name != "simple":
        raise ValueError(
            "vectorized training currently "
            "supports simple backend only"
        )

    total_steps = int(total_steps)
    vector_runtime = (
        resolve_training_vector_runtime(
            device
        )
    )
    num_envs = int(
        vector_runtime["num_envs"]
    )

    if total_steps < num_envs:
        raise ValueError(
            "total_steps must be >= "
            f"the selected num_envs ({num_envs})"
        )

    if total_steps % num_envs != 0:
        raise ValueError(
            "vectorized total_steps must be "
            "divisible by selected num_envs "
            f"({num_envs})"
        )

    if seed is None:
        seed = int(CONFIG["seed"])
    seed = int(seed)

    if resume_checkpoint is not None:
        resume_checkpoint = Path(
            resume_checkpoint
        )
        if run_dir is None:
            run_dir = (
                resume_checkpoint
                .resolve()
                .parent
                .parent
            )

    if run_name is None:
        if (
            resume_checkpoint is not None
            and run_dir is not None
        ):
            run_name = Path(
                run_dir
            ).name
        else:
            run_name = (
                make_experiment_run_name(
                    normalized,
                    seed,
                )
            )

    if run_dir is None:
        output_key = (
            "matd3_training_output_dir"
            if normalized == "matd3"
            else "training_output_dir"
        )
        run_dir = (
            Path(CONFIG[output_key])
            / run_name
        )

    run_dir = Path(run_dir)
    checkpoint_dir = (
        run_dir / "checkpoints"
    )
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    context = str(
        CONFIG.get(
            "training_vector_context",
            "fork",
        )
    )
    shared_memory = bool(
        CONFIG.get(
            "training_vector_shared_memory",
            True,
        )
    )
    overlap = bool(
        vector_runtime["overlap"]
    )
    vector_backend = str(
        vector_runtime["backend"]
    )

    env_fns = [
        lambda: UAVSearchEnv(
            backend_name="simple",
            network_backend_kwargs=(
                env.network_backend_kwargs
            ),
        )
        for _ in range(num_envs)
    ]

    if vector_runtime[
        "use_torch_vector"
    ]:
        vector_env = (
            TorchThreadedSimpleVectorEnv(
                env_fns,
                sensing_device=(
                    vector_runtime[
                        "sensing_device"
                    ]
                ),
                sensing_dtype=(
                    vector_runtime[
                        "sensing_dtype"
                    ]
                ),
                sensing_batch_timeout_s=(
                    vector_runtime[
                        "sensing_batch_timeout_s"
                    ]
                ),
            )
        )
    else:
        vector_env = (
            gym.vector.AsyncVectorEnv(
                env_fns,
                context=context,
                shared_memory=(
                    shared_memory
                ),
                autoreset_mode=(
                    gym.vector
                    .AutoresetMode.DISABLED
                ),
            )
        )

    logger = None
    eval_env = None

    try:
        set_seed(seed)

        if normalized == "masac":
            batch_size = int(
                CONFIG[
                    "masac_batch_size"
                ]
            )
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
            evaluate_fn = (
                evaluate_masac
            )
            save_fn = (
                save_masac_checkpoint
            )
            load_fn = (
                load_masac_checkpoint
            )
            wandb_project = CONFIG[
                "training_wandb_project"
            ]
        else:
            batch_size = int(
                CONFIG[
                    "matd3_batch_size"
                ]
            )
            learning_starts = int(
                CONFIG[
                    "matd3_learning_starts"
                ]
            )
            updates_per_step = int(
                CONFIG[
                    "matd3_updates_per_step"
                ]
            )
            evaluate_fn = (
                evaluate_matd3
            )
            save_fn = (
                save_matd3_checkpoint
            )
            load_fn = (
                load_matd3_checkpoint
            )
            wandb_project = CONFIG[
                "matd3_wandb_project"
            ]

        resume_state = None
        if resume_checkpoint is None:
            if normalized == "masac":
                trainer = HybridMASAC(
                    env,
                    device=device,
                )
            else:
                trainer = HybridMATD3(
                    env,
                    device=device,
                    seed=seed,
                )
            replay_buffer = (
                trainer.make_replay_buffer(
                    seed=seed
                )
            )
        else:
            loaded = load_fn(
                resume_checkpoint,
                env,
                device=device,
                load_replay=True,
                restore_torch_rng=True,
            )
            trainer = loaded["trainer"]
            replay_buffer = loaded[
                "replay_buffer"
            ]
            resume_state = loaded[
                "training_state"
            ]

            _validate_resume_config(
                loaded["config_snapshot"],
                normalized,
            )
            saved_config = loaded["config_snapshot"]
            runtime_keys = (
                ("training_torch_sensing_dtype",
                 "training_torch_overlap_env_and_updates")
                if vector_backend == "torch_threaded"
                else ("training_overlap_env_and_updates",)
            )
            for key in runtime_keys:
                if saved_config.get(key) != CONFIG.get(key):
                    raise ValueError(
                        f"exact resume vector runtime mismatch: {key}"
                    )

            if replay_buffer is None:
                raise ValueError(
                    "resume checkpoint does "
                    "not contain replay state"
                )
            if not bool(
                resume_state.get(
                    "resume_safe",
                    False,
                )
            ):
                raise ValueError(
                    "checkpoint is not marked "
                    "resume_safe"
                )
            if not bool(
                resume_state.get(
                    "vectorized",
                    False,
                )
            ):
                raise ValueError(
                    "checkpoint is not a "
                    "vectorized training state"
                )
            if str(
                resume_state.get(
                    "algorithm",
                    "",
                )
            ) != normalized:
                raise ValueError(
                    "checkpoint algorithm "
                    "does not match"
                )
            if int(
                resume_state.get(
                    "seed",
                    -1,
                )
            ) != seed:
                raise ValueError(
                    "checkpoint seed does "
                    "not match"
                )
            if int(
                resume_state.get(
                    "num_envs",
                    -1,
                )
            ) != num_envs:
                raise ValueError(
                    "checkpoint num_envs "
                    "does not match runtime"
                )
            if str(
                resume_state.get(
                    "vector_backend",
                    "",
                )
            ) != vector_backend:
                raise ValueError(
                    "checkpoint vector backend "
                    "does not match runtime"
                )

        wandb_run_id = (
            None
            if resume_state is None
            else resume_state.get(
                "wandb_run_id"
            )
        )
        logger = MASACExperimentLogger(
            run_dir=run_dir,
            run_name=run_name,
            enable_csv=enable_csv,
            enable_tensorboard=(
                enable_tensorboard
            ),
            enable_wandb=(
                enable_wandb
            ),
            wandb_entity=CONFIG.get(
                "training_wandb_entity"
            ),
            wandb_project=(
                wandb_project
            ),
            wandb_run_id=(
                wandb_run_id
            ),
            wandb_resume=(
                "allow"
                if wandb_run_id
                else None
            ),
            algorithm=normalized,
            seed=seed,
            backend_name=(
                env.backend_name
            ),
            config_snapshot=(
                _checkpoint_config_snapshot()
            ),
        )

        eval_env = UAVSearchEnv(
            backend_name=(
                env.backend_name
            ),
            network_backend_kwargs=(
                env.network_backend_kwargs
            ),
        )
        rng = np.random.default_rng(
            seed
        )

        if resume_state is None:
            initial_episode_ids = list(
                range(num_envs)
            )
            next_episode_id = (
                num_envs
            )
            trackers = [
                _new_vector_episode_tracker(
                    episode_id
                )
                for episode_id
                in initial_episode_ids
            ]
            completed_episodes = []
            latest_evaluation = None
            best_evaluation = None
            global_step = 0
            last_eval_step = 0
            last_checkpoint_step = 0
            prior_system_metrics = {}

            observations, infos = (
                vector_env.reset(
                    seed=[
                        seed + episode_id
                        for episode_id
                        in initial_episode_ids
                    ]
                )
            )
            states = np.asarray(
                infos["state"],
                dtype=np.float32,
            ).copy()
        else:
            global_step = int(
                resume_state[
                    "global_step"
                ]
            )
            if global_step > total_steps:
                raise ValueError(
                    "total_steps is below "
                    "checkpoint global_step"
                )
            if global_step % num_envs != 0:
                raise ValueError(
                    "checkpoint global_step "
                    "is not a vector-step boundary"
                )

            rng.bit_generator.state = (
                json.loads(
                    resume_state[
                        "training_rng_state_json"
                    ]
                )
            )
            trackers = _checkpoint_unpack(
                resume_state["trackers"]
            )
            completed_episodes = (
                _checkpoint_unpack(
                    resume_state[
                        "completed_episodes"
                    ]
                )
            )
            next_episode_id = int(
                resume_state[
                    "next_episode_id"
                ]
            )
            latest_evaluation = (
                copy.deepcopy(
                    resume_state.get(
                        "latest_evaluation"
                    )
                )
            )
            best_evaluation = (
                copy.deepcopy(
                    resume_state.get(
                        "best_evaluation"
                    )
                )
            )
            last_eval_step = int(
                resume_state.get(
                    "last_eval_step",
                    0,
                )
            )
            last_checkpoint_step = (
                int(
                    resume_state.get(
                        "last_checkpoint_step",
                        global_step,
                    )
                )
            )
            prior_system_metrics = (
                copy.deepcopy(
                    resume_state.get(
                        "system_metrics",
                        {},
                    )
                )
            )

            if len(trackers) != num_envs:
                raise ValueError(
                    "checkpoint tracker count "
                    "does not match runtime"
                )
            observations, states = (
                _restore_vector_env_states(
                    vector_env,
                    resume_state[
                        "vector_env_states"
                    ],
                    num_envs,
                )
            )

        initial_global_step = (
            global_step
        )
        latest_update_metrics = None
        pending_updates = 0

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
        include_replay = bool(
            CONFIG[
                "training_checkpoint_include_replay"
            ]
        )

        wall_start = time.perf_counter()
        evaluation_seconds = 0.0
        env_collection_seconds = 0.0
        env_wait_seconds = 0.0
        policy_inference_seconds = 0.0
        update_seconds = 0.0

        while global_step < total_steps:
            observations_by_env = [
                _vector_observation_at(
                    observations,
                    index,
                    num_envs,
                )
                for index in range(
                    num_envs
                )
            ]

            exploration_steps = [
                global_step + index + 1
                for index in range(
                    num_envs
                )
            ]
            policy_inference_start = (
                time.perf_counter()
                if any(
                    step > learning_starts
                    for step
                    in exploration_steps
                )
                else None
            )

            if all(
                step <= learning_starts
                for step
                in exploration_steps
            ):
                actions_by_env = [
                    sample_valid_random_actions(
                        observations_by_env[
                            index
                        ],
                        env.agent_ids,
                        rng,
                    )
                    for index in range(
                        num_envs
                    )
                ]
            elif all(
                step > learning_starts
                for step
                in exploration_steps
            ):
                actions_by_env = (
                    _select_vector_policy_actions(
                        trainer,
                        observations_by_env,
                        normalized,
                        exploration_steps,
                    )
                )
            else:
                actions_by_env = []
                for (
                    index,
                    env_observations,
                ) in enumerate(
                    observations_by_env
                ):
                    transition_step = (
                        exploration_steps[
                            index
                        ]
                    )
                    if (
                        transition_step
                        <= learning_starts
                    ):
                        actions = (
                            sample_valid_random_actions(
                                env_observations,
                                env.agent_ids,
                                rng,
                            )
                        )
                    elif normalized == "masac":
                        actions = (
                            trainer.select_actions(
                                env_observations,
                                deterministic=False,
                            )
                        )
                    else:
                        actions = (
                            trainer.select_actions(
                                env_observations,
                                deterministic=False,
                                exploration_step=(
                                    transition_step
                                ),
                            )
                        )
                    actions_by_env.append(
                        actions
                    )

            if (
                policy_inference_start
                is not None
            ):
                policy_inference_seconds += (
                    time.perf_counter()
                    - policy_inference_start
                )

            vector_actions = (
                _stack_vector_actions(
                    actions_by_env,
                    env.agent_ids,
                )
            )
            env_collection_start = (
                time.perf_counter()
            )

            if overlap:
                vector_env.step_async(
                    vector_actions
                )
            else:
                next_step_result = (
                    vector_env.step(
                        vector_actions
                    )
                )

            if (
                pending_updates > 0
                and len(replay_buffer)
                >= batch_size
            ):
                update_start = (
                    time.perf_counter()
                )
                for _ in range(
                    pending_updates
                ):
                    latest_update_metrics = (
                        trainer.update(
                            replay_buffer,
                            batch_size=(
                                batch_size
                            ),
                        )
                    )
                update_seconds += (
                    time.perf_counter()
                    - update_start
                )
                pending_updates = 0

            if overlap:
                wait_start = (
                    time.perf_counter()
                )
                next_step_result = (
                    vector_env.step_wait()
                )
                env_wait_seconds += (
                    time.perf_counter()
                    - wait_start
                )

            env_collection_seconds += (
                time.perf_counter()
                - env_collection_start
            )

            (
                next_observations,
                rewards,
                terminated,
                truncated,
                next_infos,
            ) = next_step_result
            next_states = np.asarray(
                next_infos["state"],
                dtype=np.float32,
            ).copy()

            previous_global_step = (
                global_step
            )
            replay_len_before = len(
                replay_buffer
            )

            for index in range(
                num_envs
            ):
                next_env_observations = (
                    _vector_observation_at(
                        next_observations,
                        index,
                        num_envs,
                    )
                )
                env_info = (
                    _vector_info_at(
                        next_infos,
                        index,
                        num_envs,
                    )
                )

                add_environment_transition_to_replay(
                    replay_buffer,
                    observations_by_env[
                        index
                    ],
                    states[index],
                    actions_by_env[index],
                    float(
                        rewards[index]
                    ),
                    next_env_observations,
                    next_states[index],
                    bool(
                        terminated[index]
                    ),
                    bool(
                        truncated[index]
                    ),
                    env.agent_ids,
                )
                _update_vector_episode_tracker(
                    trackers[index],
                    env_info,
                    rewards[index],
                )

            global_step += num_envs
            pending_updates += (
                _count_vector_updates_due(
                    previous_global_step=(
                        previous_global_step
                    ),
                    num_envs=num_envs,
                    learning_starts=(
                        learning_starts
                    ),
                    replay_len_before=(
                        replay_len_before
                    ),
                    replay_capacity=(
                        replay_buffer.capacity
                    ),
                    batch_size=batch_size,
                    updates_per_step=(
                        updates_per_step
                    ),
                )
            )

            if (
                not overlap
                and pending_updates > 0
            ):
                update_start = (
                    time.perf_counter()
                )
                for _ in range(
                    pending_updates
                ):
                    latest_update_metrics = (
                        trainer.update(
                            replay_buffer,
                            batch_size=(
                                batch_size
                            ),
                        )
                    )
                update_seconds += (
                    time.perf_counter()
                    - update_start
                )
                pending_updates = 0

            if (
                global_step
                % log_interval
                == 0
                or global_step
                >= total_steps
            ):
                logger.log_metrics(
                    "system",
                    _vector_system_metrics(
                        trainer=trainer,
                        global_step=(
                            global_step
                        ),
                        wall_start=(
                            wall_start
                        ),
                        evaluation_seconds=(
                            evaluation_seconds
                        ),
                        env_collection_seconds=(
                            env_collection_seconds
                        ),
                        env_wait_seconds=(
                            env_wait_seconds
                        ),
                        policy_inference_seconds=(
                            policy_inference_seconds
                        ),
                        update_seconds=(
                            update_seconds
                        ),
                        num_envs=num_envs,
                        overlap=overlap,
                        initial_global_step=(
                            initial_global_step
                        ),
                        prior_metrics=(
                            prior_system_metrics
                        ),
                    ),
                    step=global_step,
                    episode=len(
                        completed_episodes
                    ),
                )

            done_mask = np.logical_or(
                terminated,
                truncated,
            )

            for index in np.flatnonzero(
                done_mask
            ):
                index = int(index)
                env_info = (
                    _vector_info_at(
                        next_infos,
                        index,
                        num_envs,
                    )
                )
                episode_metrics = (
                    _finalize_vector_episode_tracker(
                        trackers[index],
                        env_info.get(
                            "success",
                            False,
                        ),
                    )
                )
                episode_id = int(
                    trackers[index][
                        "episode_id"
                    ]
                )

                logger.log_metrics(
                    "episode",
                    episode_metrics,
                    step=global_step,
                    episode=episode_id,
                )
                completed_episodes.append(
                    {
                        "episode": (
                            episode_id
                        ),
                        **episode_metrics,
                        "terminated": bool(
                            terminated[index]
                        ),
                        "truncated": bool(
                            truncated[index]
                        ),
                    }
                )

            observations = (
                next_observations
            )
            states = next_states

            if np.any(done_mask):
                reset_seeds = [
                    None
                ] * num_envs

                for index in np.flatnonzero(
                    done_mask
                ):
                    index = int(index)
                    reset_seeds[index] = (
                        seed
                        + next_episode_id
                    )
                    trackers[index] = (
                        _new_vector_episode_tracker(
                            next_episode_id
                        )
                    )
                    next_episode_id += 1

                observations, reset_infos = (
                    vector_env.reset(
                        seed=reset_seeds,
                        options={
                            "reset_mask": (
                                done_mask
                                .astype(
                                    np.bool_
                                )
                                .copy()
                            )
                        },
                    )
                )
                states = next_states.copy()
                reset_states = np.asarray(
                    reset_infos["state"],
                    dtype=np.float32,
                )
                reset_state_mask = np.asarray(
                    reset_infos.get(
                        "_state",
                        done_mask,
                    ),
                    dtype=np.bool_,
                )
                for index in np.flatnonzero(
                    reset_state_mask
                ):
                    states[int(index)] = (
                        reset_states[
                            int(index)
                        ]
                    )

            should_evaluate = bool(
                global_step
                - last_eval_step
                >= eval_interval
                or global_step
                >= total_steps
            )
            should_checkpoint = bool(
                global_step
                - last_checkpoint_step
                >= checkpoint_interval
                and global_step
                < total_steps
            )

            if (
                overlap
                and pending_updates > 0
                and (
                    should_evaluate
                    or should_checkpoint
                )
            ):
                update_start = (
                    time.perf_counter()
                )
                for _ in range(
                    pending_updates
                ):
                    latest_update_metrics = (
                        trainer.update(
                            replay_buffer,
                            batch_size=(
                                batch_size
                            ),
                        )
                    )
                update_seconds += (
                    time.perf_counter()
                    - update_start
                )
                pending_updates = 0

            if (
                latest_update_metrics
                is not None
                and (
                    global_step % log_interval == 0
                    or global_step >= total_steps
                )
            ):
                logger.log_metrics(
                    "train",
                    filter_training_metrics_for_wandb(
                        normalized,
                        latest_update_metrics,
                    ),
                    step=global_step,
                    episode=len(
                        completed_episodes
                    ),
                )

            if should_evaluate:
                evaluation_start = (
                    time.perf_counter()
                )
                evaluation_results = (
                    evaluate_fn(
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
                evaluation_seconds += (
                    time.perf_counter()
                    - evaluation_start
                )

                logger.log_metrics(
                    "evaluation",
                    latest_evaluation,
                    step=global_step,
                    episode=len(
                        completed_episodes
                    ),
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
                        "algorithm": (
                            normalized
                        ),
                        "global_step": (
                            global_step
                        ),
                        "episode_index": len(
                            completed_episodes
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
                        "resume_safe": False,
                        "vectorized": True,
                        "num_envs": (
                            num_envs
                        ),
                        "vector_backend": (
                            vector_backend
                        ),
                    }
                    best_path = save_fn(
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
                    logger.log_model_artifact(
                        best_path,
                        aliases=["best"],
                        metadata={
                            "global_step": (
                                global_step
                            ),
                            "episode_index": len(
                                completed_episodes
                            ),
                            "num_envs": (
                                num_envs
                            ),
                        },
                    )

            if should_checkpoint:
                last_checkpoint_step = (
                    global_step
                )
                system_snapshot = (
                    _vector_system_metrics(
                        trainer=trainer,
                        global_step=(
                            global_step
                        ),
                        wall_start=(
                            wall_start
                        ),
                        evaluation_seconds=(
                            evaluation_seconds
                        ),
                        env_collection_seconds=(
                            env_collection_seconds
                        ),
                        env_wait_seconds=(
                            env_wait_seconds
                        ),
                        policy_inference_seconds=(
                            policy_inference_seconds
                        ),
                        update_seconds=(
                            update_seconds
                        ),
                        num_envs=num_envs,
                        overlap=overlap,
                        initial_global_step=(
                            initial_global_step
                        ),
                        prior_metrics=(
                            prior_system_metrics
                        ),
                    )
                )
                checkpoint_state = (
                    _make_vector_checkpoint_state(
                        algorithm=normalized,
                        seed=seed,
                        global_step=(
                            global_step
                        ),
                        best_evaluation=(
                            best_evaluation
                        ),
                        latest_evaluation=(
                            latest_evaluation
                        ),
                        last_eval_step=(
                            last_eval_step
                        ),
                        last_checkpoint_step=(
                            last_checkpoint_step
                        ),
                        rng=rng,
                        num_envs=num_envs,
                        vector_backend=(
                            vector_backend
                        ),
                        vector_env=(
                            vector_env
                        ),
                        trackers=trackers,
                        completed_episodes=(
                            completed_episodes
                        ),
                        next_episode_id=(
                            next_episode_id
                        ),
                        system_metrics=(
                            system_snapshot
                        ),
                        wandb_run_id=(
                            logger.wandb_run_id
                        ),
                        resume_safe=(
                            include_replay
                        ),
                    )
                )
                save_fn(
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

        if (
            overlap
            and pending_updates > 0
        ):
            update_start = (
                time.perf_counter()
            )
            for _ in range(
                pending_updates
            ):
                latest_update_metrics = (
                    trainer.update(
                        replay_buffer,
                        batch_size=(
                            batch_size
                        ),
                    )
                )
            update_seconds += (
                time.perf_counter()
                - update_start
            )
            pending_updates = 0

        system_metrics = (
            _vector_system_metrics(
                trainer=trainer,
                global_step=global_step,
                wall_start=wall_start,
                evaluation_seconds=(
                    evaluation_seconds
                ),
                env_collection_seconds=(
                    env_collection_seconds
                ),
                env_wait_seconds=(
                    env_wait_seconds
                ),
                policy_inference_seconds=(
                    policy_inference_seconds
                ),
                update_seconds=(
                    update_seconds
                ),
                num_envs=num_envs,
                overlap=overlap,
                initial_global_step=(
                    initial_global_step
                ),
                prior_metrics=(
                    prior_system_metrics
                ),
            )
        )
        last_checkpoint_step = (
            global_step
        )
        exact_state = (
            _make_vector_checkpoint_state(
                algorithm=normalized,
                seed=seed,
                global_step=global_step,
                best_evaluation=(
                    best_evaluation
                ),
                latest_evaluation=(
                    latest_evaluation
                ),
                last_eval_step=(
                    last_eval_step
                ),
                last_checkpoint_step=(
                    last_checkpoint_step
                ),
                rng=rng,
                num_envs=num_envs,
                vector_backend=(
                    vector_backend
                ),
                vector_env=vector_env,
                trackers=trackers,
                completed_episodes=(
                    completed_episodes
                ),
                next_episode_id=(
                    next_episode_id
                ),
                system_metrics=(
                    system_metrics
                ),
                wandb_run_id=(
                    logger.wandb_run_id
                ),
                resume_safe=(
                    include_replay
                ),
            )
        )
        latest_checkpoint_path = (
            save_fn(
                checkpoint_dir
                / "latest.pt",
                trainer,
                replay_buffer=(
                    replay_buffer
                ),
                training_state=(
                    exact_state
                ),
                include_replay=(
                    include_replay
                ),
            )
        )

        final_state = {
            "seed": seed,
            "algorithm": normalized,
            "global_step": (
                global_step
            ),
            "episode_index": len(
                completed_episodes
            ),
            "best_evaluation": (
                copy.deepcopy(
                    best_evaluation
                )
            ),
            "latest_evaluation": (
                copy.deepcopy(
                    latest_evaluation
                )
            ),
            "last_eval_step": (
                last_eval_step
            ),
            "last_checkpoint_step": (
                last_checkpoint_step
            ),
            "resume_safe": False,
            "vectorized": True,
            "num_envs": num_envs,
            "vector_backend": (
                vector_backend
            ),
        }
        final_path = save_fn(
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
        logger.log_model_artifact(
            final_path,
            aliases=["final"],
            metadata={
                "global_step": (
                    global_step
                ),
                "episode_index": len(
                    completed_episodes
                ),
                "num_envs": num_envs,
            },
        )
        logger.update_summary(
            {
                "global_step": (
                    global_step
                ),
                "episodes_completed": len(
                    completed_episodes
                ),
                "trainer_updates": int(
                    trainer.update_count
                ),
                "num_envs": num_envs,
                **system_metrics,
            },
            prefix="training_final",
        )
        logger.log_metrics(
            "system",
            system_metrics,
            step=global_step,
            episode=len(
                completed_episodes
            ),
        )

        return {
            "trainer": trainer,
            "replay_buffer": (
                replay_buffer
            ),
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
                exact_state
            ),
            "system_metrics": (
                system_metrics
            ),
            "logger_status": (
                logger.status()
            ),
            "run_dir": run_dir,
            "checkpoint_dir": (
                checkpoint_dir
            ),
            "latest_checkpoint_path": (
                latest_checkpoint_path
            ),
        }
    finally:
        vector_env.close()
        if eval_env is not None:
            eval_env.close()
        if logger is not None:
            logger.close()


# --- frozen notebook cell 216 ---
def make_experiment_run_name(
    algorithm,
    seed,
    run_suffix=None,
):
    algorithm = str(
        algorithm
    ).strip().upper()

    if algorithm not in {
        "MASAC",
        "MATD3",
    }:
        raise ValueError(
            "algorithm must be MASAC or MATD3"
        )

    seed = int(seed)

    if run_suffix is None:
        # Uses OS randomness, not Python/NumPy/Torch RNG,
        # so display-name uniqueness does not change training determinism.
        run_suffix = secrets.token_hex(
            4
        )

    run_suffix = str(
        run_suffix
    ).strip()

    if not run_suffix:
        raise ValueError(
            "run_suffix must not be empty"
        )

    return (
        f"{algorithm}-"
        f"seed{seed}-"
        f"{run_suffix}"
    )


# --- frozen notebook cell 217 ---
def run_training_experiment(
    algorithm,
    total_steps,
    seed=44,
    backend_name="simple",
    run_name=None,
    run_dir=None,
    resume_checkpoint=None,
    device=None,
    enable_csv=None,
    enable_tensorboard=None,
    enable_wandb=None,
):
    normalized = str(
        algorithm
    ).strip().lower()

    if normalized not in {
        "masac",
        "matd3",
    }:
        raise ValueError(
            "algorithm must be "
            "'masac' or 'matd3'"
        )

    seed = int(seed)

    if (
        resume_checkpoint is not None
        and run_dir is None
    ):
        run_dir = (
            Path(resume_checkpoint)
            .resolve()
            .parent
            .parent
        )

    if run_name is None:
        if (
            resume_checkpoint is not None
            and run_dir is not None
        ):
            run_name = Path(
                run_dir
            ).name
        else:
            run_name = (
                make_experiment_run_name(
                    normalized,
                    seed,
                )
            )

    env = UAVSearchEnv(
        backend_name=backend_name
    )

    try:
        if _vector_training_is_enabled(
            env,
            device,
        ):
            return train_vectorized_experiment(
                normalized,
                env,
                total_steps=total_steps,
                seed=seed,
                run_dir=run_dir,
                run_name=run_name,
                resume_checkpoint=(
                    resume_checkpoint
                ),
                device=device,
                enable_csv=enable_csv,
                enable_tensorboard=enable_tensorboard,
                enable_wandb=enable_wandb,
            )

        if normalized == "masac":
            return train_masac_experiment(
                env,
                total_steps=total_steps,
                seed=seed,
                run_dir=run_dir,
                run_name=run_name,
                resume_checkpoint=(
                    resume_checkpoint
                ),
                device=device,
                enable_csv=enable_csv,
                enable_tensorboard=(
                    enable_tensorboard
                ),
                enable_wandb=(
                    enable_wandb
                ),
            )

        return train_matd3_experiment(
            env,
            total_steps=total_steps,
            seed=seed,
            run_dir=run_dir,
            run_name=run_name,
            resume_checkpoint=(
                resume_checkpoint
            ),
            device=device,
            enable_csv=enable_csv,
            enable_tensorboard=(
                enable_tensorboard
            ),
            enable_wandb=(
                enable_wandb
            ),
        )
    finally:
        env.close()


# --- frozen notebook cell 218 ---
DEFAULT_WANDB_COMPARISON_METRICS = (
    "evaluation/success_rate",
    "evaluation/delivery_rate",
    "evaluation/confirmation_rate",
    "evaluation/return",
    "evaluation/length",
    "evaluation/false_confirmations",
    "evaluation/expired_reports",
    "evaluation/dropped_reports",
    "evaluation/obstacle_block_rate",
    "evaluation/peer_safety_block_rate",
    "evaluation/boundary_clip_rate",
    "evaluation/total_energy_j",
    "evaluation/communication_energy_j",
    "evaluation/information_gain_bits",
    "evaluation/network_pdr_percent",
    "evaluation/network_e2e_delay_ms",
    "evaluation/network_throughput_kbps",
    "evaluation/network_average_hops",
    "evaluation/network_payload_delivery_ratio",
)


# --- frozen notebook cell 219 ---
def _select_latest_wandb_training_runs(
    seed,
    algorithms=("masac", "matd3"),
    entity=None,
    project=None,
):
    wandb = importlib.import_module(
        "wandb"
    )

    if entity is None:
        entity = CONFIG.get(
            "training_wandb_entity"
        )

    if project is None:
        project = CONFIG[
            "training_wandb_project"
        ]

    api = wandb.Api(
        timeout=30
    )
    all_runs = list(
        api.runs(
            f"{entity}/{project}",
            order="-created_at",
            per_page=200,
        )
    )

    selected = {}

    for algorithm in algorithms:
        normalized = str(
            algorithm
        ).strip().lower()

        for run in all_runs:
            if run.job_type != "training":
                continue

            config = dict(
                run.config
            )

            if str(
                config.get(
                    "monitor/algorithm",
                    "",
                )
            ).lower() != normalized:
                continue

            if int(
                config.get(
                    "monitor/seed",
                    -1,
                )
            ) != int(seed):
                continue

            if run.state != "finished":
                continue

            selected[
                normalized
            ] = run
            break

    missing = [
        str(algorithm)
        for algorithm in algorithms
        if str(
            algorithm
        ).strip().lower()
        not in selected
    ]

    if missing:
        raise RuntimeError(
            "no finished W&B training run "
            "found for: "
            + ", ".join(
                missing
            )
        )

    return selected


# --- frozen notebook cell 220 ---
def log_wandb_seed_comparison_plots(
    seed=44,
    algorithms=("masac", "matd3"),
    metrics=None,
    entity=None,
    project=None,
    mode=None,
):
    wandb = importlib.import_module(
        "wandb"
    )

    seed = int(seed)

    if entity is None:
        entity = CONFIG.get(
            "training_wandb_entity"
        )

    if project is None:
        project = CONFIG[
            "training_wandb_project"
        ]

    if mode is None:
        mode = CONFIG[
            "training_wandb_mode"
        ]

    if metrics is None:
        metrics = (
            DEFAULT_WANDB_COMPARISON_METRICS
        )

    selected = (
        _select_latest_wandb_training_runs(
            seed=seed,
            algorithms=algorithms,
            entity=entity,
            project=project,
        )
    )

    analysis_name = (
        make_experiment_run_name(
            "MASAC",
            seed,
            run_suffix=(
                "comparison-"
                + secrets.token_hex(
                    3
                )
            ),
        )
        .replace(
            "MASAC",
            "COMPARE",
            1,
        )
    )

    with wandb.init(
        entity=entity,
        project=str(project),
        name=analysis_name,
        group=f"comparison-seed{seed}",
        job_type="analysis",
        tags=[
            "comparison",
            f"seed-{seed}",
            *[
                str(
                    algorithm
                ).strip().lower()
                for algorithm
                in algorithms
            ],
        ],
        config={
            "seed": seed,
            "selected_runs": {
                algorithm: {
                    "id": run.id,
                    "name": run.name,
                    "url": run.url,
                }
                for algorithm, run
                in selected.items()
            },
            **git_experiment_metadata(),
        },
        mode=str(mode),
        reinit="finish_previous",
        settings=wandb.Settings(
            x_disable_stats=bool(
                CONFIG.get(
                    "training_wandb_disable_system_stats",
                    True,
                )
            )
        ),
    ) as analysis_run:
        logged_metrics = []

        for metric in metrics:
            rows = []

            for (
                algorithm,
                source_run,
            ) in selected.items():
                history = (
                    source_run.history(
                        keys=[
                            "global_step",
                            metric,
                        ],
                        pandas=False,
                    )
                )

                for point in history:
                    step = point.get(
                        "global_step"
                    )
                    value = point.get(
                        metric
                    )

                    if (
                        step is None
                        or value is None
                    ):
                        continue

                    if not np.isfinite(
                        float(value)
                    ):
                        continue

                    rows.append(
                        [
                            float(step),
                            float(value),
                            algorithm.upper(),
                        ]
                    )

            if not rows:
                continue

            table = wandb.Table(
                data=rows,
                columns=[
                    "global_step",
                    "value",
                    "algorithm",
                ],
            )

            key = (
                metric.replace(
                    "/",
                    "_",
                )
            )

            chart = wandb.plot.line(
                table=table,
                x="global_step",
                y="value",
                stroke="algorithm",
                title=(
                    f"{metric} | seed {seed}"
                ),
                split_table=True,
            )

            analysis_run.log(
                {
                    f"comparison/{key}": (
                        chart
                    )
                }
            )
            logged_metrics.append(
                metric
            )

        return {
            "run_id": (
                analysis_run.id
            ),
            "run_url": (
                analysis_run.url
            ),
            "run_name": (
                analysis_run.name
            ),
            "seed": seed,
            "source_runs": {
                algorithm: {
                    "id": run.id,
                    "name": run.name,
                    "url": run.url,
                }
                for algorithm, run
                in selected.items()
            },
            "logged_metrics": (
                logged_metrics
            ),
        }


# Export public and internal helpers required by downstream production layers.
__all__ = [name for name in globals() if not name.startswith("__")]
