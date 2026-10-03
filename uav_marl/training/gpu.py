"""Production source migrated from frozen fix_test_gpu.ipynb.

Do not import the notebook at runtime. The notebook is retained only as a
reference/regression artifact. Production changes belong in Python modules.
Migrated from notebook cells 226..287.
"""

from ..envs.gpu import *  # noqa: F401,F403

# --- frozen notebook cell 227 ---
def load_project_namespace(repo: Path | None = None):
    """Return an explicit embedded project namespace without reading files."""
    _ = repo
    ns = {
        "gym": gym,
        "np": np,
        "threading": threading,
        "UAVSearchEnv": UAVSearchEnv,
        "HybridMASAC": HybridMASAC,
        "HybridMATD3": HybridMATD3,
        "HybridReplayBuffer": HybridReplayBuffer,
        "Report": Report,
        "UAV": UAV,
        "Obstacle": Obstacle,
        "HybridAction": HybridAction,
        "UavNetSimBackend": UavNetSimBackend,
        "project_motion_action_tensor": project_motion_action_tensor,
        "radial_squash_motion_action": radial_squash_motion_action,
        "straight_through_categorical_sample": straight_through_categorical_sample,
        "straight_through_masked_argmax": straight_through_masked_argmax,
        "train_masac": train_masac,
        "train_matd3": train_matd3,
        "set_seed": set_seed,
        "uavnetsim_available": uavnetsim_available,
        "build_transmission_intent": build_transmission_intent,
        "sample_valid_random_actions": sample_valid_random_actions,
        "add_environment_transition_to_replay": add_environment_transition_to_replay,
        "_select_vector_policy_actions": _select_vector_policy_actions,
        "_vector_observation_at": _vector_observation_at,
        "_stack_vector_actions": _stack_vector_actions,
        "_count_vector_updates_due": _count_vector_updates_due,
        "running_on_kaggle": running_on_kaggle,
        "configure_kaggle_wandb": configure_kaggle_wandb,
    }
    ns["_TORCH_SENSING_TLS"] = _get_torch_sensing_tls()
    return ns


# --- frozen notebook cell 229 ---
def _trainer(
    repo: Path,
    algorithm: str,
    device: str,
    seed: int,
):
    """Build the existing MASAC/MATD3 learner without constructing a CPU env.

    Only lightweight Gymnasium space metadata is created on the host. No CPU
    mission reset/step/sensing/network simulation is executed by the full-GPU
    training path.
    """
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    _ = repo
    gym_module = __import__("gymnasium")
    np_module = __import__("numpy")
    ns = {
        "gym": gym_module,
        "np": np_module,
        "UAVSearchEnv": UAVSearchEnv,
        "HybridMASAC": HybridMASAC,
        "HybridMATD3": HybridMATD3,
        "HybridReplayBuffer": HybridReplayBuffer,
        "UAV": UAV,
        "Obstacle": Obstacle,
        "Report": Report,
        "UavNetSimBackend": UavNetSimBackend,
        "HybridAction": HybridAction,
        "build_transmission_intent": build_transmission_intent,
        "project_motion_action_tensor": project_motion_action_tensor,
        "radial_squash_motion_action": radial_squash_motion_action,
        "straight_through_categorical_sample": straight_through_categorical_sample,
        "straight_through_masked_argmax": straight_through_masked_argmax,
    }
    gym = ns["gym"]
    np = ns["np"]
    env_cls = ns["UAVSearchEnv"]
    dummy = object.__new__(env_cls)

    U = int(CONFIG["num_uavs"])
    T = int(CONFIG["num_targets"])
    O = int(CONFIG["num_obstacles"])
    D = U + 1
    nearest = int(CONFIG["observation_nearest_obstacles"])
    patch = int(CONFIG["belief_patch_cells"])
    coarse = int(CONFIG["belief_coarse_cells"])
    critic = int(CONFIG["critic_belief_grid_cells"])
    agent_ids = tuple(f"uav_{index}" for index in range(U))
    dummy.agent_ids = agent_ids

    agent_space = gym.spaces.Dict(
        {
            "self_state": gym.spaces.Box(-1.0, 1.0, shape=(9,), dtype=np.float32),
            "gcs_relative": gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
            "neighbors": gym.spaces.Box(
                -1.0, 1.0, shape=(U - 1, 5), dtype=np.float32
            ),
            "obstacles": gym.spaces.Box(
                -1.0, 1.0, shape=(nearest, 5), dtype=np.float32
            ),
            "belief_patch": gym.spaces.Box(
                0.0, 1.0, shape=(patch, patch), dtype=np.float32
            ),
            "belief_coarse": gym.spaces.Box(
                0.0, 1.0, shape=(coarse, coarse), dtype=np.float32
            ),
            "buffer": gym.spaces.Box(0.0, 1.0, shape=(5,), dtype=np.float32),
            "destination_mask": gym.spaces.Box(
                0.0, 1.0, shape=(D,), dtype=np.float32
            ),
        }
    )
    dummy.observation_space = gym.spaces.Dict(
        {agent_id: agent_space for agent_id in agent_ids}
    )
    dummy.action_space = gym.spaces.Dict(
        {
            agent_id: gym.spaces.Dict(
                {"destination": gym.spaces.Discrete(D)}
            )
            for agent_id in agent_ids
        }
    )

    observation_dim = (
        9 + 3 + 5 * (U - 1) + 5 * nearest
        + patch * patch + coarse * coarse + 5 + D
    )
    buffer_slots = int(
        float(CONFIG["buffer_bytes"]) // float(CONFIG["report_bytes"])
    )
    pending_slots = int(
        float(CONFIG["pending_buffer_bytes"]) // float(CONFIG["report_bytes"])
    )
    per_uav_state = (
        observation_dim
        + 4 * buffer_slots
        + 4 * pending_slots
        + 2
        + critic * critic
    )
    state_dim = U * per_uav_state + 4 * T + 4 * O + 1
    dummy.state_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(state_dim,),
        dtype=np.float32,
    )

    if algorithm == "masac":
        trainer = ns["HybridMASAC"](
            dummy,
            device=device,
        )
    else:
        trainer = ns["HybridMATD3"](
            dummy,
            device=device,
            seed=int(seed),
        )
    trainer._project_namespace = ns
    return trainer


# --- frozen notebook cell 230 ---
def make_project_gpu_replay_buffer(
    trainer,
    *,
    capacity,
    num_agents,
    observation_dim,
    state_dim,
    continuous_dim,
    discrete_dim,
    device,
    seed,
    strict_cuda=True,
):
    project_ns = getattr(
        trainer,
        "_project_namespace",
        None,
    )
    if not isinstance(project_ns, dict):
        raise RuntimeError(
            "trainer is missing project namespace"
        )
    base = project_ns[
        "HybridReplayBuffer"
    ]

    class ProjectGpuReplayBuffer(
        GpuReplayBuffer,
        base,
    ):
        pass

    return ProjectGpuReplayBuffer(
        capacity=capacity,
        num_agents=num_agents,
        observation_dim=observation_dim,
        state_dim=state_dim,
        continuous_dim=continuous_dim,
        discrete_dim=discrete_dim,
        device=device,
        seed=seed,
        strict_cuda=strict_cuda,
    )


# --- frozen notebook cell 231 ---
def _matd3_actions(
    trainer,
    obs,
    mask,
    transition_steps,
    generator,
):
    with torch.no_grad():
        sampled = trainer._actor_joint_actions(
            obs,
            mask,
            straight_through=False,
            target_actor=False,
        )
        continuous = sampled["continuous"]
        destination = sampled[
            "destination_indices"
        ]
        noise = (
            torch.randn(
                continuous.shape,
                device=continuous.device,
                generator=generator,
            )
            * float(
                CONFIG[
                    "matd3_exploration_noise"
                ]
            )
        )
        continuous = (
            continuous + noise
        ).clamp(-1.0, 1.0)
        motion = continuous[..., :3]
        norm = torch.linalg.vector_norm(
            motion,
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0)
        continuous = torch.cat(
            (
                motion / norm,
                continuous[..., 3:4],
            ),
            dim=-1,
        )

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
        decay = max(
            1.0,
            float(
                CONFIG[
                    "matd3_discrete_epsilon_decay_steps"
                ]
            ),
        )
        fraction = (
            transition_steps.float()
            / decay
        ).clamp(0.0, 1.0)
        epsilon = (
            start
            + fraction * (end - start)
        )
        explore = (
            torch.rand(
                destination.shape,
                device=destination.device,
                generator=generator,
            )
            < epsilon[:, None]
        )
        random_logits = torch.rand(
            mask.shape,
            device=mask.device,
            generator=generator,
        ).masked_fill(
            mask <= 0.5,
            -1.0,
        )
        random_destination = (
            random_logits.argmax(-1)
        )
        destination = torch.where(
            explore,
            random_destination,
            destination,
        )
    return continuous, destination


# --- frozen notebook cell 232 ---
def _random_exploration_actions(
    env,
    mask,
):
    shape = (
        env.num_envs,
        env.num_uavs,
        3,
    )
    direction = torch.randn(
        shape,
        device=env.device,
        generator=env.generator,
    )
    direction = direction / (
        torch.linalg.vector_norm(
            direction,
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-12)
    )
    radius = torch.pow(
        torch.rand(
            env.num_envs,
            env.num_uavs,
            1,
            device=env.device,
            generator=env.generator,
        ),
        1.0 / 3.0,
    )
    motion = direction * radius
    power = (
        torch.rand(
            env.num_envs,
            env.num_uavs,
            1,
            device=env.device,
            generator=env.generator,
        )
        * 2.0
        - 1.0
    )
    continuous = torch.cat(
        (motion, power),
        dim=-1,
    )
    logits = torch.rand(
        mask.shape,
        device=env.device,
        generator=env.generator,
    ).masked_fill(
        mask <= 0.5,
        -1.0,
    )
    destination = logits.argmax(-1)
    return continuous, destination


# --- frozen notebook cell 234 ---
def benchmark_env(num_envs: int, steps: int, seed: int, device: str, strict_cuda: bool = True):
    env = FullGpuUAVBatchEnv(num_envs=num_envs, device=device, seed=seed, strict_cuda=strict_cuda)
    n_cuda_tensors = (
        env.assert_all_state_on_cuda()
        if env.device.type == "cuda"
        else sum(isinstance(v, torch.Tensor) for v in env.__dict__.values())
    )
    _obs, _state, mask = env.reset()
    if env.device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    transitions = 0
    for _ in range(int(steps)):
        continuous, destination = (
            _random_exploration_actions(
                env,
                mask,
            )
        )
        _obs, _state, mask, _reward, _done, _ = env.step(continuous, destination)
        transitions += num_envs
    if env.device.type == "cuda":
        torch.cuda.synchronize(env.device)
    wall = time.perf_counter() - start
    return {
        "kind": "full_gpu_env",
        "num_envs": num_envs,
        "vector_steps": int(steps),
        "transitions": transitions,
        "wall_seconds": wall,
        "transitions_per_second": transitions / wall,
        "cuda_state_tensors": n_cuda_tensors,
        "gpu_name": torch.cuda.get_device_name(env.device) if env.device.type == "cuda" else "CPU_DEBUG",
        "peak_memory_mb": torch.cuda.max_memory_allocated(env.device) / 2**20 if env.device.type == "cuda" else 0.0,
    }


# --- frozen notebook cell 236 ---
def _make_hybrid_vector_env(
    project_ns,
    num_envs,
):
    gym = project_ns["gym"]
    env_fns = [
        lambda: project_ns[
            "UAVSearchEnv"
        ](backend_name="simple")
        for _ in range(int(num_envs))
    ]
    return gym.vector.AsyncVectorEnv(
        env_fns,
        context=str(
            CONFIG.get(
                "training_vector_context",
                "fork",
            )
        ),
        shared_memory=bool(
            CONFIG.get(
                "training_vector_shared_memory",
                True,
            )
        ),
        autoreset_mode=(
            gym.vector.AutoresetMode.DISABLED
        ),
    )


# --- frozen notebook cell 237 ---
def benchmark_hybrid_env(
    repo: Path,
    num_envs: int,
    steps: int,
    seed: int,
):
    """CPU vectorized reference env throughput (no learner update)."""
    project_ns = load_project_namespace(
        repo
    )
    np = project_ns["np"]
    vector_env = _make_hybrid_vector_env(
        project_ns,
        num_envs,
    )
    template = project_ns[
        "UAVSearchEnv"
    ](backend_name="simple")
    rng = np.random.default_rng(
        int(seed) + 991
    )
    try:
        observations, _ = vector_env.reset(
            seed=[
                int(seed) + index
                for index in range(num_envs)
            ]
        )
        start = time.perf_counter()
        transitions = 0
        next_episode_seed = (
            int(seed) + 100_000
        )
        for _ in range(int(steps)):
            observations_by_env = [
                project_ns[
                    "_vector_observation_at"
                ](
                    observations,
                    index,
                    num_envs,
                )
                for index in range(
                    num_envs
                )
            ]
            actions_by_env = [
                project_ns[
                    "sample_valid_random_actions"
                ](
                    observations_by_env[index],
                    template.agent_ids,
                    rng,
                )
                for index in range(
                    num_envs
                )
            ]
            vector_actions = project_ns[
                "_stack_vector_actions"
            ](
                actions_by_env,
                template.agent_ids,
            )
            (
                next_observations,
                _rewards,
                terminated,
                truncated,
                _infos,
            ) = vector_env.step(
                vector_actions
            )
            transitions += num_envs
            done = np.logical_or(
                terminated,
                truncated,
            )
            observations = next_observations
            if np.any(done):
                seeds = [None] * num_envs
                for index in np.flatnonzero(
                    done
                ):
                    seeds[int(index)] = (
                        next_episode_seed
                    )
                    next_episode_seed += 1
                observations, _ = (
                    vector_env.reset(
                        seed=seeds,
                        options={
                            "reset_mask": (
                                done.astype(
                                    np.bool_
                                )
                            )
                        },
                    )
                )
        wall = time.perf_counter() - start
        return {
            "kind": "hybrid_cpu_env",
            "num_envs": int(num_envs),
            "vector_steps": int(steps),
            "transitions": int(
                transitions
            ),
            "wall_seconds": wall,
            "transitions_per_second": (
                transitions / wall
            ),
        }
    finally:
        vector_env.close()
        template.close()


# --- frozen notebook cell 238 ---
def benchmark_hybrid_train(
    repo: Path,
    algorithm: str,
    num_envs: int,
    total_transitions: int,
    seed: int,
    device: str,
):
    """Best current CPU-env + CUDA-learner path, including overlap."""
    algorithm = str(
        algorithm
    ).strip().lower()
    if algorithm not in {
        "masac",
        "matd3",
    }:
        raise ValueError(
            "algorithm must be masac or matd3"
        )
    if (
        int(total_transitions)
        % int(num_envs)
        != 0
    ):
        raise ValueError(
            "total_transitions must be "
            "divisible by num_envs"
        )

    project_ns = load_project_namespace(
        repo
    )
    np = project_ns["np"]
    template = project_ns[
        "UAVSearchEnv"
    ](backend_name="simple")
    trainer = _trainer(
        repo,
        algorithm,
        device,
        seed,
    )
    replay_capacity = min(
        int(
            CONFIG[
                f"{algorithm}_replay_capacity"
            ]
        ),
        max(
            4096,
            int(total_transitions),
        ),
    )
    replay = trainer._project_namespace[
        "HybridReplayBuffer"
    ](
        capacity=replay_capacity,
        num_agents=trainer.num_agents,
        observation_dim=(
            trainer.observation_dim
        ),
        state_dim=trainer.state_dim,
        continuous_dim=(
            trainer.continuous_dim
        ),
        discrete_dim=(
            trainer.discrete_dim
        ),
        seed=int(seed) + 123,
    )
    vector_env = _make_hybrid_vector_env(
        project_ns,
        num_envs,
    )

    batch_size = int(
        CONFIG[
            f"{algorithm}_batch_size"
        ]
    )
    learning_starts = int(
        CONFIG[
            f"{algorithm}_learning_starts"
        ]
    )
    updates_per_step = int(
        CONFIG[
            f"{algorithm}_updates_per_step"
        ]
    )
    rng = np.random.default_rng(
        int(seed)
    )
    overlap = bool(
        CONFIG.get(
            "training_overlap_env_and_updates",
            True,
        )
    )
    next_episode_seed = (
        int(seed) + 100_000
    )

    try:
        observations, infos = (
            vector_env.reset(
                seed=[
                    int(seed) + index
                    for index in range(
                        num_envs
                    )
                ]
            )
        )
        states = np.asarray(
            infos["state"],
            dtype=np.float32,
        ).copy()
        global_step = 0
        pending_updates = 0
        update_count = 0
        latest_metrics = None
        env_seconds = 0.0
        wait_seconds = 0.0
        update_seconds = 0.0
        policy_seconds = 0.0

        if (
            torch.device(device).type
            == "cuda"
        ):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        wall_start = time.perf_counter()
        while global_step < int(
            total_transitions
        ):
            observations_by_env = [
                project_ns[
                    "_vector_observation_at"
                ](
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

            policy_start = time.perf_counter()
            if all(
                step <= learning_starts
                for step in exploration_steps
            ):
                actions_by_env = [
                    project_ns[
                        "sample_valid_random_actions"
                    ](
                        observations_by_env[
                            index
                        ],
                        template.agent_ids,
                        rng,
                    )
                    for index in range(
                        num_envs
                    )
                ]
            elif all(
                step > learning_starts
                for step in exploration_steps
            ):
                actions_by_env = project_ns[
                    "_select_vector_policy_actions"
                ](
                    trainer,
                    observations_by_env,
                    algorithm,
                    exploration_steps,
                )
            else:
                actions_by_env = []
                for index in range(
                    num_envs
                ):
                    transition_step = (
                        exploration_steps[index]
                    )
                    if (
                        transition_step
                        <= learning_starts
                    ):
                        action = project_ns[
                            "sample_valid_random_actions"
                        ](
                            observations_by_env[
                                index
                            ],
                            template.agent_ids,
                            rng,
                        )
                    elif algorithm == "masac":
                        action = (
                            trainer.select_actions(
                                observations_by_env[
                                    index
                                ],
                                deterministic=False,
                            )
                        )
                    else:
                        action = (
                            trainer.select_actions(
                                observations_by_env[
                                    index
                                ],
                                deterministic=False,
                                exploration_step=(
                                    transition_step
                                ),
                            )
                        )
                    actions_by_env.append(
                        action
                    )
            policy_seconds += (
                time.perf_counter()
                - policy_start
            )

            vector_actions = project_ns[
                "_stack_vector_actions"
            ](
                actions_by_env,
                template.agent_ids,
            )
            env_start = time.perf_counter()
            if overlap:
                vector_env.step_async(
                    vector_actions
                )
            else:
                result = vector_env.step(
                    vector_actions
                )

            if (
                pending_updates > 0
                and len(replay)
                >= batch_size
            ):
                update_start = (
                    time.perf_counter()
                )
                for _ in range(
                    pending_updates
                ):
                    latest_metrics = (
                        trainer.update(
                            replay,
                            batch_size=(
                                batch_size
                            ),
                        )
                    )
                    update_count += 1
                update_seconds += (
                    time.perf_counter()
                    - update_start
                )
                pending_updates = 0

            if overlap:
                wait_start = (
                    time.perf_counter()
                )
                result = (
                    vector_env.step_wait()
                )
                wait_seconds += (
                    time.perf_counter()
                    - wait_start
                )
            env_seconds += (
                time.perf_counter()
                - env_start
            )

            (
                next_observations,
                rewards,
                terminated,
                truncated,
                next_infos,
            ) = result
            next_states = np.asarray(
                next_infos["state"],
                dtype=np.float32,
            ).copy()

            previous_global_step = (
                global_step
            )
            replay_len_before = len(
                replay
            )
            for index in range(
                num_envs
            ):
                next_env_observations = (
                    project_ns[
                        "_vector_observation_at"
                    ](
                        next_observations,
                        index,
                        num_envs,
                    )
                )
                project_ns[
                    "add_environment_transition_to_replay"
                ](
                    replay,
                    observations_by_env[
                        index
                    ],
                    states[index],
                    actions_by_env[index],
                    float(rewards[index]),
                    next_env_observations,
                    next_states[index],
                    bool(
                        terminated[index]
                    ),
                    bool(
                        truncated[index]
                    ),
                    template.agent_ids,
                )

            global_step += num_envs
            pending_updates += (
                project_ns[
                    "_count_vector_updates_due"
                ](
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
                        replay.capacity
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
                    latest_metrics = (
                        trainer.update(
                            replay,
                            batch_size=(
                                batch_size
                            ),
                        )
                    )
                    update_count += 1
                update_seconds += (
                    time.perf_counter()
                    - update_start
                )
                pending_updates = 0

            done = np.logical_or(
                terminated,
                truncated,
            )
            observations = (
                next_observations
            )
            states = next_states
            if np.any(done):
                seeds = [None] * num_envs
                for index in np.flatnonzero(
                    done
                ):
                    seeds[int(index)] = (
                        next_episode_seed
                    )
                    next_episode_seed += 1
                (
                    observations,
                    reset_infos,
                ) = vector_env.reset(
                    seed=seeds,
                    options={
                        "reset_mask": done.astype(
                            np.bool_
                        )
                    },
                )
                reset_states = np.asarray(
                    reset_infos["state"],
                    dtype=np.float32,
                )
                reset_mask = np.asarray(
                    reset_infos.get(
                        "_state",
                        done,
                    ),
                    dtype=np.bool_,
                )
                for index in np.flatnonzero(
                    reset_mask
                ):
                    states[int(index)] = (
                        reset_states[
                            int(index)
                        ]
                    )

        if pending_updates > 0:
            update_start = (
                time.perf_counter()
            )
            for _ in range(
                pending_updates
            ):
                latest_metrics = (
                    trainer.update(
                        replay,
                        batch_size=batch_size,
                    )
                )
                update_count += 1
            update_seconds += (
                time.perf_counter()
                - update_start
            )

        if (
            torch.device(device).type
            == "cuda"
        ):
            torch.cuda.synchronize()
        wall = (
            time.perf_counter()
            - wall_start
        )

        return {
            "kind": "hybrid_cpu_env_cuda_learner",
            "algorithm": algorithm,
            "num_envs": int(
                num_envs
            ),
            "transitions": int(
                global_step
            ),
            "updates": int(
                update_count
            ),
            "wall_seconds": wall,
            "transitions_per_second": (
                global_step / wall
            ),
            "updates_per_second": (
                update_count / wall
            ),
            "env_collection_seconds": (
                env_seconds
            ),
            "env_wait_seconds": (
                wait_seconds
            ),
            "policy_inference_seconds": (
                policy_seconds
            ),
            "update_seconds": (
                update_seconds
            ),
            "overlap_enabled": bool(
                overlap
            ),
            "gpu_name": (
                torch.cuda.get_device_name()
                if (
                    torch.device(
                        device
                    ).type
                    == "cuda"
                    and torch.cuda.is_available()
                )
                else "CPU"
            ),
            "peak_memory_mb": (
                torch.cuda.max_memory_allocated()
                / 2**20
                if (
                    torch.device(
                        device
                    ).type
                    == "cuda"
                    and torch.cuda.is_available()
                )
                else 0.0
            ),
        }
    finally:
        vector_env.close()
        template.close()


# --- frozen notebook cell 240 ---
class GpuUavNetSimBridge:
    """Packet-level UavNetSim adapter for one tensorized mission environment."""

    def __init__(self, env, project_ns, *, repo_path=None, seed=44):
        if env.num_envs != 1:
            raise ValueError(
                "GpuUavNetSimBridge currently supports exactly one mission env"
            )
        self.env = env
        self.ns = project_ns
        self.repo_path = repo_path
        self.seed = int(seed)
        self.backend = None
        self.uavs = []
        self.obstacles = []
        self.report_buffers = []
        self.pending_reports = []
        self.gcs_received_target_ids = set()
        self._last_results = []
        self._last_metrics = None
        self._timing_totals = {
            "sync_gpu_to_cpu_seconds": 0.0,
            "uavnetsim_step_seconds": 0.0,
            "sync_cpu_to_gpu_seconds": 0.0,
            "calls": 0,
        }

    @property
    def np(self):
        return self.ns["np"]

    def _tensor_numpy(self, value, dtype=None):
        array = value.detach().cpu().numpy()
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array

    def _build_mission_objects(self):
        np = self.np
        UAV = self.ns["UAV"]
        Obstacle = self.ns["Obstacle"]
        positions = self._tensor_numpy(self.env.positions[0], np.float64)
        velocities = self._tensor_numpy(self.env.velocities[0], np.float64)
        batteries = self._tensor_numpy(self.env.battery[0], np.float64)
        active = self._tensor_numpy(self.env.active[0], bool)
        self.uavs = [
            UAV(
                id=index,
                position=positions[index].copy(),
                velocity=velocities[index].copy(),
                battery_j=float(batteries[index]),
                active=bool(active[index]),
            )
            for index in range(self.env.num_uavs)
        ]
        obstacle_xy = self._tensor_numpy(self.env.obstacle_xy[0], np.float64)
        obstacle_radius = self._tensor_numpy(self.env.obstacle_radius[0], np.float64)
        obstacle_height = self._tensor_numpy(self.env.obstacle_height[0], np.float64)
        self.obstacles = [
            Obstacle(
                position=obstacle_xy[index].copy(),
                radius=float(obstacle_radius[index]),
                height=float(obstacle_height[index]),
            )
            for index in range(self.env.num_obstacles)
        ]
        self.report_buffers = [[] for _ in range(self.env.num_uavs)]
        self.pending_reports = []
        self.gcs_received_target_ids = set()
        self._sync_gpu_to_cpu()

    def _sync_gpu_to_cpu(self):
        np = self.np
        Report = self.ns["Report"]
        positions = self._tensor_numpy(self.env.positions[0], np.float64)
        velocities = self._tensor_numpy(self.env.velocities[0], np.float64)
        batteries = self._tensor_numpy(self.env.battery[0], np.float64)
        active = self._tensor_numpy(self.env.active[0], bool)
        for index, uav in enumerate(self.uavs):
            uav.position = positions[index].copy()
            uav.velocity = velocities[index].copy()
            uav.battery_j = float(batteries[index])
            uav.active = bool(active[index])

        valid = self._tensor_numpy(self.env.report_valid[0], bool)
        remaining = self._tensor_numpy(self.env.report_remaining[0], np.float64)
        created = self._tensor_numpy(self.env.report_created[0], np.int64)
        source = self._tensor_numpy(self.env.report_source[0], np.int64)
        report_size = int(CONFIG["report_bytes"])
        ttl_s = float(CONFIG["report_ttl"])
        for holder in range(self.env.num_uavs):
            buffer = self.report_buffers[holder]
            buffer.clear()
            for target in range(self.env.num_targets):
                if not valid[holder, target]:
                    continue
                delivered_bytes = int(round(report_size - float(remaining[holder, target])))
                delivered_bytes = max(0, min(report_size, delivered_bytes))
                buffer.append(
                    Report(
                        target_id=int(target),
                        source_uav=int(source[holder, target]),
                        created_step=int(created[holder, target]),
                        size_bytes=report_size,
                        ttl_s=ttl_s,
                        delivered_bytes=delivered_bytes,
                    )
                )

        pvalid = self._tensor_numpy(self.env.pending_valid[0], bool)
        premaining = self._tensor_numpy(self.env.pending_remaining[0], np.float64)
        pcreated = self._tensor_numpy(self.env.pending_created[0], np.int64)
        self.pending_reports.clear()
        for holder in range(self.env.num_uavs):
            for target in range(self.env.num_targets):
                if not pvalid[holder, target]:
                    continue
                delivered_bytes = int(round(report_size - float(premaining[holder, target])))
                delivered_bytes = max(0, min(report_size, delivered_bytes))
                self.pending_reports.append(
                    Report(
                        target_id=int(target),
                        source_uav=int(holder),
                        created_step=int(pcreated[holder, target]),
                        size_bytes=report_size,
                        ttl_s=ttl_s,
                        delivered_bytes=delivered_bytes,
                    )
                )
        self.pending_reports.sort(
            key=lambda report: (
                int(report.created_step),
                int(report.source_uav),
                int(report.target_id),
            )
        )

        delivered = self._tensor_numpy(self.env.delivered[0], bool)
        self.gcs_received_target_ids.clear()
        self.gcs_received_target_ids.update(
            int(index) for index, flag in enumerate(delivered) if flag
        )

    def _sync_cpu_to_gpu(self):
        env = self.env
        device = env.device
        valid = torch.zeros_like(env.report_valid)
        remaining = torch.zeros_like(env.report_remaining)
        created = torch.full_like(
            env.report_created,
            env.max_steps + int(CONFIG["report_ttl"]) + 1,
        )
        source = torch.full_like(env.report_source, -1)
        for holder, buffer in enumerate(self.report_buffers):
            for report in buffer:
                target = int(report.target_id)
                valid[0, holder, target] = True
                remaining[0, holder, target] = float(
                    max(0, int(report.size_bytes) - int(report.delivered_bytes))
                )
                created[0, holder, target] = int(report.created_step)
                source[0, holder, target] = int(report.source_uav)

        pvalid = torch.zeros_like(env.pending_valid)
        premaining = torch.zeros_like(env.pending_remaining)
        pcreated = torch.full_like(
            env.pending_created,
            env.max_steps + int(CONFIG["report_ttl"]) + 1,
        )
        for report in self.pending_reports:
            holder = int(report.source_uav)
            target = int(report.target_id)
            if not (0 <= holder < env.num_uavs):
                continue
            if pvalid[0, holder, target]:
                continue
            pvalid[0, holder, target] = True
            premaining[0, holder, target] = float(
                max(0, int(report.size_bytes) - int(report.delivered_bytes))
            )
            pcreated[0, holder, target] = int(report.created_step)

        delivered = torch.zeros_like(env.delivered)
        for target in self.gcs_received_target_ids:
            if 0 <= int(target) < env.num_targets:
                delivered[0, int(target)] = True

        env.report_valid = valid.to(device)
        env.report_remaining = remaining.to(device)
        env.report_created = created.to(device)
        env.report_source = source.to(device)
        env.pending_valid = pvalid.to(device)
        env.pending_remaining = premaining.to(device)
        env.pending_created = pcreated.to(device)
        env.delivered = delivered.to(device)

    def reset_from_env(self):
        if self.backend is not None:
            self.backend.close()
        self._build_mission_objects()
        self.backend = self.ns["UavNetSimBackend"](
            repo_path=self.repo_path,
            seed=self.seed,
        )
        self.backend.reset(
            uavs=self.uavs,
            gcs_position=CONFIG["gcs_position"],
            obstacles=self.obstacles,
            report_buffers=self.report_buffers,
            pending_reports=self.pending_reports,
            gcs_received_target_ids=self.gcs_received_target_ids,
        )
        self._last_results = []
        self._last_metrics = self.backend.metrics()
        self._timing_totals = {
            "sync_gpu_to_cpu_seconds": 0.0,
            "uavnetsim_step_seconds": 0.0,
            "sync_cpu_to_gpu_seconds": 0.0,
            "calls": 0,
        }

    def step(
        self,
        destination_idx,
        power_action,
        energy_budget_j=None,
    ):
        if self.backend is None:
            raise RuntimeError("GpuUavNetSimBridge must be reset before step")
        timing_start = time.perf_counter()
        self._sync_gpu_to_cpu()
        after_gpu_to_cpu = time.perf_counter()
        np = self.np
        current_step = int(self.env.step_count[0].detach().cpu().item())
        destinations = self._tensor_numpy(destination_idx[0], np.int64)
        powers = self._tensor_numpy(power_action[0, :, 0], np.float64)
        if energy_budget_j is None:
            # A missing explicit communication budget means "use the current
            # battery state as the finite upper bound". UavNetSimBackend
            # intentionally rejects +/-inf so validators and direct bridge
            # callers obey the same energy contract as the full environment.
            energy_budget = self._tensor_numpy(
                self.env.battery[0],
                np.float64,
            )
        else:
            energy_budget = self._tensor_numpy(
                torch.as_tensor(
                    energy_budget_j,
                    device=self.env.device,
                    dtype=self.env.positions.dtype,
                )[0],
                np.float64,
            )
            if energy_budget.shape != (
                self.env.num_uavs,
            ):
                raise ValueError(
                    "energy_budget_j has wrong shape"
                )
            if (
                not np.all(
                    np.isfinite(
                        energy_budget
                    )
                )
                or np.any(
                    energy_budget < 0.0
                )
            ):
                raise ValueError(
                    "energy_budget_j must be finite and >= 0"
                )

        HybridAction = self.ns["HybridAction"]
        build_intent = self.ns["build_transmission_intent"]
        intents = []
        sender_for_result = []
        for sender in range(self.env.num_uavs):
            choice_index = int(destinations[sender])
            if not 0 <= choice_index < self.env.discrete_dim:
                raise ValueError("destination index outside communication choices")
            destination = int(
                self.env.choices[sender, choice_index].detach().cpu().item()
            )
            if destination == -2:
                decoded_destination = None
                tx_power_w = 0.0
            else:
                decoded_destination = destination
                tx_power_w = (
                    float(CONFIG["tx_power_min_w"])
                    + (float(max(-1.0, min(1.0, powers[sender]))) + 1.0)
                    * 0.5
                    * (
                        float(CONFIG["tx_power_max_w"])
                        - float(CONFIG["tx_power_min_w"])
                    )
                )
            hybrid_action = HybridAction(
                movement=np.zeros(3, dtype=np.float64),
                destination=decoded_destination,
                tx_power_w=float(tx_power_w),
            )
            if decoded_destination is None:
                continue
            intent = build_intent(
                sender,
                hybrid_action,
                self.report_buffers[sender],
                current_step,
                peer_transfer_states=self.backend.peer_transfer_states,
            )
            if intent is not None:
                budget = float(
                    energy_budget[sender]
                )
                full_step_energy = (
                    float(tx_power_w)
                    * float(self.env.dt)
                )
                if budget <= 0.0:
                    intent = None
                elif budget < (
                    full_step_energy - 1e-12
                ):
                    max_bytes = int(
                        np.floor(
                            float(
                                CONFIG[
                                    "uavnetsim_bit_rate_bps"
                                ]
                            )
                            * min(
                                float(self.env.dt),
                                budget
                                / max(
                                    float(tx_power_w),
                                    1e-12,
                                ),
                            )
                            / 8.0
                            + 1e-9
                        )
                    )
                    if max_bytes <= 0:
                        intent = None
                    elif max_bytes < int(
                        intent.requested_bytes
                    ):
                        intent = type(intent)(
                            sender=int(
                                intent.sender
                            ),
                            recipient=int(
                                intent.recipient
                            ),
                            target_id=int(
                                intent.target_id
                            ),
                            requested_bytes=(
                                max_bytes
                            ),
                            tx_power_w=float(
                                intent.tx_power_w
                            ),
                        )

            if intent is not None:
                intents.append(intent)
                sender_for_result.append(sender)

        before_delivered = set(self.gcs_received_target_ids)
        self.backend.sync_positions(self.uavs)
        self.backend.set_step_energy_budget_j(
            energy_budget
        )
        results = self.backend.step(
            intents,
            dt=self.env.dt,
            current_step=current_step,
        )
        after_uavnetsim = time.perf_counter()
        communication_energy = np.minimum(
            self.backend.last_step_communication_energy_by_uav(),
            energy_budget,
        )
        attempted = torch.zeros(
            1, self.env.num_uavs, device=self.env.device, dtype=torch.bool
        )
        bytes_attempted = torch.zeros(
            1, self.env.num_uavs, device=self.env.device
        )
        bytes_tx = torch.zeros(
            1, self.env.num_uavs, device=self.env.device
        )
        link_ok = torch.zeros(
            1, self.env.num_uavs, device=self.env.device, dtype=torch.bool
        )
        for sender, intent, result in zip(
            sender_for_result,
            intents,
            results,
        ):
            attempted[0, sender] = True
            bytes_attempted[0, sender] = float(
                intent.requested_bytes
            )
            bytes_tx[0, sender] = float(
                result.get("tx_bytes", 0)
            )
            link_ok[0, sender] = bool(
                int(result.get("injected_bytes", 0)) > 0
                or int(result.get("tx_bytes", 0)) > 0
                or str(result.get("status", "")) == "in_flight"
            )

        self._sync_cpu_to_gpu()
        after_cpu_to_gpu = time.perf_counter()
        self._timing_totals["sync_gpu_to_cpu_seconds"] += (
            after_gpu_to_cpu - timing_start
        )
        self._timing_totals["uavnetsim_step_seconds"] += (
            after_uavnetsim - after_gpu_to_cpu
        )
        self._timing_totals["sync_cpu_to_gpu_seconds"] += (
            after_cpu_to_gpu - after_uavnetsim
        )
        self._timing_totals["calls"] += 1
        after_delivered = set(self.gcs_received_target_ids)
        new_delivery = len(after_delivered - before_delivered)
        comm_energy = torch.as_tensor(
            communication_energy,
            device=self.env.device,
            dtype=self.env.positions.dtype,
        ).view(1, self.env.num_uavs)
        self._last_results = results
        self._last_metrics = self.backend.metrics()
        packet_payload_bytes = max(
            1,
            int(CONFIG["uavnetsim_payload_bytes"]),
        )
        packets_tx = torch.where(
            bytes_tx > 0.0,
            torch.ceil(bytes_tx / float(packet_payload_bytes)),
            torch.zeros_like(bytes_tx),
        )
        zero_per_env = torch.zeros(
            1,
            device=self.env.device,
            dtype=self.env.positions.dtype,
        )
        return {
            "new_delivery": torch.tensor(
                [new_delivery], device=self.env.device, dtype=torch.long
            ),
            "comm_energy": comm_energy,
            "attempted": attempted,
            "bytes_attempted": bytes_attempted,
            "bytes_tx": bytes_tx,
            "packets_tx": packets_tx,
            # Detailed PHY/delay KPIs for the real UavNetSim path come from
            # backend.metrics(). These tensors keep the external-network
            # adapter schema identical to FullGpuUAVBatchEnv._network().
            "delivery_latency_sum_s": zero_per_env.clone(),
            "delivery_latency_count": zero_per_env.clone(),
            "delivery_latency_max_s": zero_per_env.clone(),
            "channel_contenders": attempted.sum(-1).to(
                self.env.positions.dtype
            ),
            "link_ok": link_ok,
            "sinr_db": torch.full_like(bytes_tx, float("nan")),
            "nlos": torch.zeros_like(attempted),
            "interference_w": torch.zeros_like(bytes_tx),
        }

    def metrics(self):
        if self.backend is None:
            return {}
        self._last_metrics = self.backend.metrics()
        return dict(self._last_metrics)

    def timing_metrics(self):
        result = dict(self._timing_totals)
        calls = max(1, int(result["calls"]))
        total = (
            float(result["sync_gpu_to_cpu_seconds"])
            + float(result["uavnetsim_step_seconds"])
            + float(result["sync_cpu_to_gpu_seconds"])
        )
        result["bridge_total_seconds"] = total
        result["uavnetsim_fraction"] = (
            float(result["uavnetsim_step_seconds"]) / total
            if total > 0.0
            else 0.0
        )
        result["avg_uavnetsim_step_seconds"] = (
            float(result["uavnetsim_step_seconds"]) / calls
        )
        return result

    def close(self):
        if self.backend is not None:
            self.backend.close()
            self.backend = None


# --- frozen notebook cell 241 ---
def copy_cpu_reference_state_to_gpu(cpu_env, gpu_env):
    """Copy one CPU mission state into one tensor env for apples-to-apples runs."""
    if gpu_env.num_envs != 1:
        raise ValueError("copy_cpu_reference_state_to_gpu requires num_envs=1")
    device = gpu_env.device
    np = __import__("numpy")

    positions = np.stack([uav.position for uav in cpu_env.uavs])
    velocities = np.stack([uav.velocity for uav in cpu_env.uavs])
    battery = np.asarray([uav.battery_j for uav in cpu_env.uavs], dtype=np.float32)
    active = np.asarray([uav.active for uav in cpu_env.uavs], dtype=bool)
    targets = np.stack([target.position for target in cpu_env.targets])
    if cpu_env.obstacles:
        obstacles = np.stack(
            [obstacle.position for obstacle in cpu_env.obstacles]
        )
    else:
        obstacles = np.empty((0, 2), dtype=np.float32)
    obstacle_radius = np.asarray(
        [obstacle.radius for obstacle in cpu_env.obstacles], dtype=np.float32
    )
    obstacle_height = np.asarray(
        [obstacle.height for obstacle in cpu_env.obstacles], dtype=np.float32
    )

    gpu_env.positions = torch.as_tensor(
        positions, device=device, dtype=gpu_env.positions.dtype
    ).unsqueeze(0)
    gpu_env.velocities = torch.as_tensor(
        velocities, device=device, dtype=gpu_env.velocities.dtype
    ).unsqueeze(0)
    gpu_env.battery = torch.as_tensor(
        battery, device=device, dtype=gpu_env.battery.dtype
    ).unsqueeze(0)
    gpu_env.active = torch.as_tensor(
        active, device=device, dtype=torch.bool
    ).unsqueeze(0)
    gpu_env.target_xy = torch.as_tensor(
        targets, device=device, dtype=gpu_env.target_xy.dtype
    ).unsqueeze(0)
    gpu_env.obstacle_xy = torch.as_tensor(
        obstacles, device=device, dtype=gpu_env.obstacle_xy.dtype
    ).unsqueeze(0)
    gpu_env.obstacle_radius = torch.as_tensor(
        obstacle_radius, device=device, dtype=gpu_env.obstacle_radius.dtype
    ).unsqueeze(0)
    gpu_env.obstacle_height = torch.as_tensor(
        obstacle_height, device=device, dtype=gpu_env.obstacle_height.dtype
    ).unsqueeze(0)
    gpu_env.confirmed = torch.as_tensor(
        [[target.confirmed for target in cpu_env.targets]],
        device=device,
        dtype=torch.bool,
    )
    gpu_env.delivered.zero_()
    for target in cpu_env.gcs_received_target_ids:
        gpu_env.delivered[0, int(target)] = True
    gpu_env.belief = torch.as_tensor(
        cpu_env.belief_maps,
        device=device,
        dtype=gpu_env.belief.dtype,
    ).unsqueeze(0)
    gpu_env.step_count.fill_(int(cpu_env.current_step))
    gpu_env.report_valid.zero_()
    gpu_env.report_remaining.zero_()
    gpu_env.report_created.fill_(
        gpu_env.max_steps + int(CONFIG["report_ttl"]) + 1
    )
    gpu_env.report_source.fill_(-1)
    for holder, buffer in enumerate(cpu_env.report_buffers):
        for report in buffer:
            target = int(report.target_id)
            gpu_env.report_valid[0, holder, target] = True
            gpu_env.report_remaining[0, holder, target] = float(
                max(0, int(report.size_bytes) - int(report.delivered_bytes))
            )
            gpu_env.report_created[0, holder, target] = int(report.created_step)
            gpu_env.report_source[0, holder, target] = int(report.source_uav)
    gpu_env.pending_valid.zero_()
    gpu_env.pending_remaining.zero_()
    gpu_env.pending_created.fill_(
        gpu_env.max_steps + int(CONFIG["report_ttl"]) + 1
    )
    for report in cpu_env.pending_reports:
        holder = int(report.source_uav)
        target = int(report.target_id)
        if 0 <= holder < gpu_env.num_uavs:
            gpu_env.pending_valid[0, holder, target] = True
            gpu_env.pending_remaining[0, holder, target] = float(
                max(0, int(report.size_bytes) - int(report.delivered_bytes))
            )
            gpu_env.pending_created[0, holder, target] = int(report.created_step)
    gpu_env.peer_received.zero_()
    gpu_env.peer_source.fill_(-1)
    gpu_env.peer_created.fill_(
        gpu_env.max_steps + int(CONFIG["report_ttl"]) + 1
    )
    gpu_env.belief_potential = gpu_env._belief_potential()
    obs, mask = gpu_env._observation()
    state = gpu_env._state(obs)
    return obs, state, mask


# --- frozen notebook cell 242 ---
def benchmark_uavnetsim_full_flow(
    repo: Path,
    *,
    repo_path,
    steps=20,
    seed=44,
    device="cuda:0",
    strict_cuda=True,
    communication_loaded=True,
):
    """CPU-UavNetSim vs tensor-env+UavNetSim using one scenario and action trace."""
    project_ns = load_project_namespace(repo)
    if not project_ns["uavnetsim_available"](repo_path):
        raise FileNotFoundError("UavNetSim repository is unavailable")

    CPUEnv = project_ns["UAVSearchEnv"]
    cpu_env = CPUEnv(
        backend_name="uavnetsim",
        network_backend_kwargs={"repo_path": repo_path},
    )
    cpu_env.reset(seed=int(seed))
    if communication_loaded:
        Report = project_ns["Report"]
        cpu_env.targets[0].confirmed = True
        cpu_env.report_buffers[0].clear()
        cpu_env.report_buffers[0].append(
            Report(
                target_id=0,
                source_uav=0,
                created_step=0,
                size_bytes=int(CONFIG["report_bytes"]),
                ttl_s=float(CONFIG["report_ttl"]),
                delivered_bytes=0,
            )
        )

    gpu_env = FullGpuUAVBatchEnv(
        num_envs=1,
        device=device,
        seed=int(seed),
        strict_cuda=bool(strict_cuda),
    )
    copy_cpu_reference_state_to_gpu(cpu_env, gpu_env)
    bridge = GpuUavNetSimBridge(
        gpu_env,
        project_ns,
        repo_path=repo_path,
        seed=int(seed),
    )
    gpu_env.attach_external_network(bridge)

    rng = project_ns["np"].random.default_rng(int(seed) + 99173)
    actions = rng.uniform(
        -1.0,
        1.0,
        size=(int(steps), gpu_env.num_uavs, gpu_env.continuous_dim),
    ).astype("float32")
    destinations = rng.integers(
        0,
        gpu_env.discrete_dim,
        size=(int(steps), gpu_env.num_uavs),
        endpoint=False,
        dtype="int64",
    )
    if communication_loaded:
        actions[:, 0, 3] = 1.0
        destinations[:, 0] = gpu_env.num_uavs

    def cpu_action_at(index):
        result = {}
        for sender, agent_id in enumerate(cpu_env.agent_ids):
            result[agent_id] = {
                "motion": actions[index, sender, :3].copy(),
                "destination": int(destinations[index, sender]),
                "power": project_ns["np"].array(
                    [actions[index, sender, 3]],
                    dtype=project_ns["np"].float32,
                ),
            }
        return result

    cpu_rewards = []
    cpu_start = time.perf_counter()
    cpu_steps = 0
    for index in range(int(steps)):
        out = cpu_env.step(cpu_action_at(index))
        cpu_rewards.append(float(out[1]))
        cpu_steps += 1
        if bool(out[2] or out[3]):
            break
    cpu_wall = time.perf_counter() - cpu_start
    cpu_network = cpu_env.network_backend.metrics()

    gpu_rewards = []
    if gpu_env.device.type == "cuda":
        torch.cuda.synchronize(gpu_env.device)
    gpu_start = time.perf_counter()
    gpu_steps = 0
    for index in range(int(steps)):
        continuous = torch.as_tensor(
            actions[index:index + 1],
            device=gpu_env.device,
            dtype=torch.float32,
        )
        destination = torch.as_tensor(
            destinations[index:index + 1],
            device=gpu_env.device,
            dtype=torch.long,
        )
        out = gpu_env.step(continuous, destination)
        gpu_rewards.append(float(out[3][0].detach().cpu().item()))
        gpu_steps += 1
        if bool(out[4][0].detach().cpu().item()):
            break
    if gpu_env.device.type == "cuda":
        torch.cuda.synchronize(gpu_env.device)
    gpu_wall = time.perf_counter() - gpu_start
    gpu_network = bridge.metrics()

    np = project_ns["np"]
    cpu_positions = np.stack(
        [uav.position for uav in cpu_env.uavs]
    )
    gpu_positions = (
        gpu_env.positions[0]
        .detach()
        .cpu()
        .numpy()
    )
    cpu_belief = np.asarray(
        cpu_env.belief_maps,
        dtype=np.float32,
    )
    gpu_belief = (
        gpu_env.belief[0]
        .detach()
        .cpu()
        .numpy()
    )
    common_steps = min(
        len(cpu_rewards),
        len(gpu_rewards),
    )
    reward_abs_diff = [
        abs(
            float(cpu_rewards[index])
            - float(gpu_rewards[index])
        )
        for index in range(common_steps)
    ]
    network_compare_keys = (
        "packets_injected",
        "packets_delivered",
        "packets_failed",
        "queue_limit_events",
        "payload_bytes_injected",
        "payload_bytes_delivered",
    )
    network_core_equal = all(
        cpu_network.get(key)
        == gpu_network.get(key)
        for key in network_compare_keys
    )
    final_cpu_confirmed = {
        int(target.id)
        for target in cpu_env.targets
        if target.confirmed
    }
    final_gpu_confirmed = {
        int(index)
        for index, flag in enumerate(
            gpu_env.confirmed[0]
            .detach()
            .cpu()
            .tolist()
        )
        if flag
    }
    final_cpu_delivered = set(
        int(value)
        for value in cpu_env.gcs_received_target_ids
    )
    final_gpu_delivered = {
        int(index)
        for index, flag in enumerate(
            gpu_env.delivered[0]
            .detach()
            .cpu()
            .tolist()
        )
        if flag
    }

    result = {
        "kind": "uavnetsim_full_flow_compare",
        "seed": int(seed),
        "requested_steps": int(steps),
        "same_initial_scenario": True,
        "same_action_trace": True,
        "parity": {
            "final_position_max_abs_diff_m": float(
                np.max(
                    np.abs(
                        cpu_positions
                        - gpu_positions
                    )
                )
            ),
            "final_belief_max_abs_diff": float(
                np.max(
                    np.abs(
                        cpu_belief
                        - gpu_belief
                    )
                )
            ),
            "reward_max_abs_diff": (
                float(max(reward_abs_diff))
                if reward_abs_diff
                else 0.0
            ),
            "network_core_equal": bool(
                network_core_equal
            ),
            "confirmed_equal": (
                final_cpu_confirmed
                == final_gpu_confirmed
            ),
            "delivered_equal": (
                final_cpu_delivered
                == final_gpu_delivered
            ),
            "cpu_confirmed": sorted(
                final_cpu_confirmed
            ),
            "gpu_confirmed": sorted(
                final_gpu_confirmed
            ),
            "cpu_delivered": sorted(
                final_cpu_delivered
            ),
            "gpu_delivered": sorted(
                final_gpu_delivered
            ),
        },
        "cpu": {
            "steps": cpu_steps,
            "wall_seconds": cpu_wall,
            "steps_per_second": cpu_steps / cpu_wall if cpu_wall > 0.0 else float("inf"),
            "reward_sum": float(sum(cpu_rewards)),
            "network": cpu_network,
        },
        "gpu_mission_plus_uavnetsim": {
            "steps": gpu_steps,
            "wall_seconds": gpu_wall,
            "steps_per_second": gpu_steps / gpu_wall if gpu_wall > 0.0 else float("inf"),
            "reward_sum": float(sum(gpu_rewards)),
            "network": gpu_network,
            "network_timing": bridge.timing_metrics(),
            "device": str(gpu_env.device),
        },
    }
    cpu_env.close()
    gpu_env.close()
    return result


# --- frozen notebook cell 243 ---
def validate_uavnetsim_gpu_direct_link(
    repo: Path,
    *,
    repo_path=None,
    seed=44,
    steps=6,
    device="cpu",
    strict_cuda=False,
):
    """Controlled direct-to-GCS validation against the real SimPy backend.

    This is intentionally a single-link case: CSMA has no collision ambiguity,
    so queue cap, payload packetization, service budget, LoS/NLoS PHY contract,
    delivery timing, reward timing and report cleanup are directly comparable.
    """
    project_ns = load_project_namespace(repo)
    if not project_ns["uavnetsim_available"](repo_path):
        raise FileNotFoundError("UavNetSim repository is unavailable")
    np = project_ns["np"]
    CPUEnv = project_ns["UAVSearchEnv"]
    Report = project_ns["Report"]
    cpu_env = CPUEnv(
        backend_name="uavnetsim",
        network_backend_kwargs={"repo_path": repo_path},
    )
    cpu_env.reset(seed=int(seed))
    cpu_env.targets[0].confirmed = True
    cpu_env.report_buffers[0].clear()
    cpu_env.report_buffers[0].append(
        Report(
            target_id=0,
            source_uav=0,
            created_step=0,
            size_bytes=int(CONFIG["report_bytes"]),
            ttl_s=float(CONFIG["report_ttl"]),
            delivered_bytes=0,
        )
    )

    gpu_env = FullGpuUAVBatchEnv(
        num_envs=1,
        device=device,
        seed=int(seed),
        strict_cuda=bool(strict_cuda),
    )
    copy_cpu_reference_state_to_gpu(cpu_env, gpu_env)
    gpu_env.network_model = "uavnetsim_gpu"

    per_step = []
    try:
        for index in range(int(steps)):
            action = np.zeros(
                (gpu_env.num_uavs, gpu_env.continuous_dim),
                dtype=np.float32,
            )
            destination = np.zeros(
                gpu_env.num_uavs,
                dtype=np.int64,
            )
            action[0, 3] = 1.0
            destination[0] = gpu_env.num_uavs

            cpu_action = {}
            for sender, agent_id in enumerate(cpu_env.agent_ids):
                cpu_action[agent_id] = {
                    "motion": action[sender, :3].copy(),
                    "destination": int(destination[sender]),
                    "power": np.asarray(
                        [action[sender, 3]],
                        dtype=np.float32,
                    ),
                }
            cpu_out = cpu_env.step(cpu_action)
            gpu_out = gpu_env.step(
                torch.as_tensor(
                    action[None],
                    device=gpu_env.device,
                    dtype=torch.float32,
                ),
                torch.as_tensor(
                    destination[None],
                    device=gpu_env.device,
                    dtype=torch.long,
                ),
            )
            cpu_network = cpu_env.network_backend.metrics()
            gpu_delivered = bool(gpu_env.delivered[0, 0].item())
            gpu_remaining = float(
                gpu_env.report_remaining[0, 0, 0].item()
            )
            gpu_delivered_bytes = int(
                round(float(CONFIG["report_bytes"]) - gpu_remaining)
            )
            per_step.append(
                {
                    "step": index + 1,
                    "cpu_payload_bytes_delivered": int(
                        cpu_network["payload_bytes_delivered"]
                    ),
                    "gpu_payload_bytes_delivered": gpu_delivered_bytes,
                    "cpu_delivered": 0 in cpu_env.gcs_received_target_ids,
                    "gpu_delivered": gpu_delivered,
                    "cpu_reward": float(cpu_out[1]),
                    "gpu_reward": float(gpu_out[3][0].item()),
                }
            )

        byte_parity = all(
            row["cpu_payload_bytes_delivered"]
            == row["gpu_payload_bytes_delivered"]
            for row in per_step
        )
        delivery_parity = all(
            row["cpu_delivered"] == row["gpu_delivered"]
            for row in per_step
        )
        reward_max_abs_diff = max(
            (
                abs(row["cpu_reward"] - row["gpu_reward"])
                for row in per_step
            ),
            default=0.0,
        )
        return {
            "passed": bool(
                byte_parity
                and delivery_parity
                and reward_max_abs_diff < 1e-4
            ),
            "byte_parity": bool(byte_parity),
            "delivery_parity": bool(delivery_parity),
            "reward_max_abs_diff": float(reward_max_abs_diff),
            "per_step": per_step,
            "cpu_network_metrics": cpu_env.network_backend.metrics(),
        }
    finally:
        cpu_env.close()
        gpu_env.close()


# --- frozen notebook cell 244 ---
def validate_uavnetsim_network_parity(
    repo: Path,
    *,
    repo_path,
    seed=44,
    device="cpu",
    strict_cuda=False,
    steps=4,
):
    """Drive two independent UavNetSim backends with identical state/intents."""
    project_ns = load_project_namespace(repo)
    CPUEnv = project_ns["UAVSearchEnv"]
    cpu_env = CPUEnv(
        backend_name="uavnetsim",
        network_backend_kwargs={"repo_path": repo_path},
    )
    cpu_env.reset(seed=int(seed))
    Report = project_ns["Report"]
    cpu_env.targets[0].confirmed = True
    cpu_env.report_buffers[0].append(
        Report(
            target_id=0,
            source_uav=0,
            created_step=0,
            size_bytes=int(CONFIG["report_bytes"]),
            ttl_s=float(CONFIG["report_ttl"]),
            delivered_bytes=0,
        )
    )

    gpu_env = FullGpuUAVBatchEnv(
        1,
        device=device,
        seed=int(seed),
        strict_cuda=bool(strict_cuda),
    )
    copy_cpu_reference_state_to_gpu(cpu_env, gpu_env)
    bridge = GpuUavNetSimBridge(
        gpu_env,
        project_ns,
        repo_path=repo_path,
        seed=int(seed),
    )
    gpu_env.attach_external_network(bridge)

    destination = torch.zeros(
        1, gpu_env.num_uavs, device=gpu_env.device, dtype=torch.long
    )
    destination[0, 0] = gpu_env.num_uavs
    power = torch.zeros(
        1, gpu_env.num_uavs, 1, device=gpu_env.device
    )
    power[0, 0, 0] = 1.0

    per_step = []
    for current_step in range(1, int(steps) + 1):
        cpu_env.current_step = current_step - 1
        gpu_env.step_count.fill_(current_step)
        action = {}
        for sender, agent_id in enumerate(cpu_env.agent_ids):
            action[agent_id] = {
                "motion": project_ns["np"].zeros(
                    3, dtype=project_ns["np"].float32
                ),
                "destination": gpu_env.num_uavs if sender == 0 else 0,
                "power": project_ns["np"].array(
                    [1.0 if sender == 0 else 0.0],
                    dtype=project_ns["np"].float32,
                ),
            }
        hybrid = cpu_env._decode_actions(cpu_env._canonical_actions(action))
        intents = []
        peer_states = cpu_env.network_backend.peer_transfer_states
        for sender, haction in enumerate(hybrid):
            intent = project_ns["build_transmission_intent"](
                sender,
                haction,
                cpu_env.report_buffers[sender],
                current_step,
                peer_transfer_states=peer_states,
            )
            if intent is not None:
                intents.append(intent)
        cpu_env.network_backend.sync_positions(cpu_env.uavs)
        cpu_results = cpu_env.network_backend.step(
            intents,
            dt=float(CONFIG["dt"]),
            current_step=current_step,
        )
        gpu_result = bridge.step(destination, power)
        cpu_metrics = cpu_env.network_backend.metrics()
        gpu_metrics = bridge.metrics()
        comparable = {
            "packets_injected": (
                cpu_metrics["packets_injected"],
                gpu_metrics["packets_injected"],
            ),
            "packets_delivered": (
                cpu_metrics["packets_delivered"],
                gpu_metrics["packets_delivered"],
            ),
            "packets_failed": (
                cpu_metrics["packets_failed"],
                gpu_metrics["packets_failed"],
            ),
            "payload_bytes_delivered": (
                cpu_metrics["payload_bytes_delivered"],
                gpu_metrics["payload_bytes_delivered"],
            ),
            "queue_limit_events": (
                cpu_metrics["queue_limit_events"],
                gpu_metrics["queue_limit_events"],
            ),
        }
        equal = all(left == right for left, right in comparable.values())
        per_step.append(
            {
                "step": current_step,
                "equal": equal,
                "metrics": comparable,
                "cpu_results": cpu_results,
                "gpu_bytes_tx": gpu_result["bytes_tx"].detach().cpu().tolist(),
            }
        )
        if not equal:
            break

    passed = all(item["equal"] for item in per_step)
    cpu_env.close()
    gpu_env.close()
    return {
        "passed": passed,
        "steps_checked": len(per_step),
        "per_step": per_step,
    }


# --- frozen notebook cell 246 ---
def _rebuild_cuda_optimizers_with_fused_adam(
    trainer,
    algorithm,
):
    """Opt-in benchmark path; safe only before the first optimizer step."""
    if trainer.device.type != "cuda":
        return False
    algorithm = str(algorithm).strip().lower()
    if algorithm == "masac":
        trainer.actor_optimizer = torch.optim.Adam(
            trainer.actor.parameters(),
            lr=float(CONFIG["masac_actor_lr"]),
            fused=True,
        )
        trainer.critic_optimizer = torch.optim.Adam(
            list(trainer.critic_1.parameters())
            + list(trainer.critic_2.parameters()),
            lr=float(CONFIG["masac_critic_lr"]),
            fused=True,
        )
        trainer.alpha_continuous_optimizer = torch.optim.Adam(
            [trainer.log_alpha_continuous],
            lr=float(CONFIG["masac_alpha_lr"]),
            fused=True,
        )
        trainer.alpha_discrete_optimizer = torch.optim.Adam(
            [trainer.log_alpha_discrete],
            lr=float(CONFIG["masac_alpha_lr"]),
            fused=True,
        )
    elif algorithm == "matd3":
        trainer.actor_optimizer = torch.optim.Adam(
            trainer.actor.parameters(),
            lr=float(CONFIG["matd3_actor_lr"]),
            fused=True,
        )
        trainer.critic_optimizer = torch.optim.Adam(
            list(trainer.critic_1.parameters())
            + list(trainer.critic_2.parameters()),
            lr=float(CONFIG["matd3_critic_lr"]),
            fused=True,
        )
    else:
        raise ValueError("unsupported algorithm")
    return True


# --- frozen notebook cell 247 ---
def _compile_trainer_modules(
    trainer,
    algorithm,
    mode="default",
):
    """Compile fixed-shape actor/critic modules in-place for benchmarking."""
    if trainer.device.type != "cuda":
        return []
    names = [
        "actor",
        "critic_1",
        "critic_2",
        "target_critic_1",
        "target_critic_2",
    ]
    if str(algorithm).strip().lower() == "matd3":
        names.append("target_actor")
    compiled = []
    for name in names:
        module = getattr(trainer, name, None)
        if isinstance(module, torch.nn.Module):
            module.compile(
                mode=str(mode),
                fullgraph=False,
            )
            compiled.append(name)
    return compiled


# --- frozen notebook cell 249 ---
def ensure_uavnetsim_reference_available(
    repo_path=None,
):
    source_available = bool(
        uavnetsim_available(
            repo_path
        )
    )
    missing_runtime = []
    for import_name in (
        "simpy",
        "trimesh",
    ):
        try:
            importlib.import_module(
                import_name
            )
        except ModuleNotFoundError:
            missing_runtime.append(
                import_name
            )

    if (
        source_available
        and not missing_runtime
    ):
        return (
            resolve_uavnetsim_repo_path(
                repo_path
            )
        )

    if not running_on_kaggle():
        if (
            source_available
            and missing_runtime
        ):
            raise FileNotFoundError(
                "UavNetSim runtime dependencies are missing: "
                f"{missing_runtime}. Install requirements.txt."
            )
        raise FileNotFoundError(
            "UavNetSim is required for authoritative "
            "network evaluation. Install requirements.txt "
            "or pass uavnetsim_repo_path."
        )

    # Keep Kaggle evaluation reproducible. UavNetSim is installed with
    # --no-deps so pip cannot replace CUDA/PyTorch/NumPy. Only lightweight
    # dependencies required by the A2A adapter are bootstrapped separately.
    install_spec = (
        "git+https://github.com/"
        "Zihao-Felix-Zhou/UavNetSim.git@"
        "04daafb815eb377409b40b285574eeb62b9a8d58"
    )
    runtime_specs = {
        "simpy": "simpy==4.1.1",
        "trimesh": "trimesh==4.12.2",
    }
    from importlib.metadata import (
        PackageNotFoundError,
        version as package_version,
    )

    protected_versions = {}
    for package_name in (
        "numpy",
        "scipy",
        "torch",
    ):
        try:
            protected_versions[
                package_name
            ] = package_version(
                package_name
            )
        except PackageNotFoundError:
            protected_versions[
                package_name
            ] = None

    dependency_results = []
    for import_name in missing_runtime:
        spec = runtime_specs[
            import_name
        ]
        dependency_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                "--quiet",
                "--no-deps",
                spec,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        dependency_results.append(
            (
                import_name,
                dependency_result,
            )
        )
        importlib.invalidate_caches()
        try:
            importlib.import_module(
                import_name
            )
        except ModuleNotFoundError as exc:
            detail = (
                dependency_result.stderr.strip()
                or dependency_result.stdout.strip()
                or "unknown pip failure"
            )
            raise RuntimeError(
                f"failed to install {spec} required by "
                f"UavNetSim: {detail}"
            ) from exc

    if source_available:
        result = None
    else:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                "--quiet",
                "--no-deps",
                install_spec,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        importlib.invalidate_caches()

    changed_packages = {}
    for (
        package_name,
        before_version,
    ) in protected_versions.items():
        try:
            after_version = package_version(
                package_name
            )
        except PackageNotFoundError:
            after_version = None
        if after_version != before_version:
            changed_packages[
                package_name
            ] = {
                "before": before_version,
                "after": after_version,
            }

    if changed_packages:
        raise RuntimeError(
            "UavNetSim bootstrap unexpectedly changed "
            "protected runtime packages: "
            f"{changed_packages}"
        )

    for (
        import_name,
        dependency_result,
    ) in dependency_results:
        if dependency_result.returncode != 0:
            detail = (
                dependency_result.stderr.strip()
                or dependency_result.stdout.strip()
                or "unknown pip failure"
            )
            raise RuntimeError(
                "failed to install pinned UavNetSim runtime "
                f"dependency {import_name}: {detail}"
            )

    if (
        result is not None
        and result.returncode != 0
    ):
        detail = (
            result.stderr.strip()
            or result.stdout.strip()
            or "unknown pip failure"
        )
        raise RuntimeError(
            "failed to install pinned UavNetSim "
            f"for Kaggle evaluation: {detail}"
        )

    if not uavnetsim_available(
        repo_path
    ):
        raise RuntimeError(
            "pinned UavNetSim installation completed "
            "but the simulator source is still unavailable"
        )

    return resolve_uavnetsim_repo_path(
        repo_path
    )


# --- frozen notebook cell 250 ---
def render_evaluation_trace_2d(
    trace,
    title=None,
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    map_size = float(
        trace["map_size"]
    )
    fig, ax = plt.subplots(
        figsize=(8.5, 8.5)
    )

    coverage = np.asarray(
        trace["coverage_grid"],
        dtype=np.float64,
    )
    if coverage.size:
        ax.imshow(
            coverage,
            origin="lower",
            extent=(
                0.0,
                map_size,
                0.0,
                map_size,
            ),
            alpha=0.18,
            vmin=0.0,
            vmax=1.0,
        )

    for obstacle in trace[
        "obstacles"
    ]:
        xy = np.asarray(
            obstacle["position"],
            dtype=np.float64,
        )
        circle = Circle(
            (
                float(xy[0]),
                float(xy[1]),
            ),
            float(
                obstacle["radius"]
            ),
            fill=False,
            linewidth=1.0,
            alpha=0.7,
        )
        ax.add_patch(
            circle
        )

    for target in trace[
        "targets"
    ]:
        p = np.asarray(
            target["position"],
            dtype=np.float64,
        )
        marker = (
            "*"
            if target[
                "delivered"
            ]
            else (
                "P"
                if target[
                    "confirmed"
                ]
                else "x"
            )
        )
        ax.scatter(
            [float(p[0])],
            [float(p[1])],
            marker=marker,
            color="tab:red",
            s=35,
        )

    trajectory = trace[
        "trajectory_xyz"
    ]
    if trajectory:
        trajectory_array = (
            np.stack(
                trajectory,
                axis=0,
            )
        )
        for uav_id in range(
            trajectory_array.shape[1]
        ):
            path = trajectory_array[
                :,
                uav_id,
                :,
            ]
            ax.plot(
                path[:, 0],
                path[:, 1],
                linewidth=1.1,
                alpha=0.85,
            )

    final_positions = np.asarray(
        trace[
            "final_uav_positions"
        ],
        dtype=np.float64,
    )
    for uav_id, position in enumerate(
        final_positions
    ):
        ax.scatter(
            [float(position[0])],
            [float(position[1])],
            marker="o",
            s=45,
        )
        ax.text(
            float(position[0]),
            float(position[1]),
            (
                f" U{uav_id}"
                f" z={position[2]:.0f}m"
            ),
            fontsize=8,
        )

    gcs = np.asarray(
        trace[
            "gcs_position"
        ],
        dtype=np.float64,
    )
    ax.scatter(
        [float(gcs[0])],
        [float(gcs[1])],
        marker="s",
        s=65,
    )
    ax.text(
        float(gcs[0]),
        float(gcs[1]),
        " GCS",
        fontsize=8,
    )

    latest_edges = {}
    for edge in trace.get(
        "communication_edges",
        [],
    ):
        key = (
            int(edge["sender"]),
            int(edge["recipient"]),
        )
        latest_edges[key] = edge
    max_edges = max(
        1,
        int(
            CONFIG.get(
                "training_visualization_max_link_edges",
                64,
            )
        ),
    )
    for edge in list(
        latest_edges.values()
    )[-max_edges:]:
        sender = int(
            edge["sender"]
        )
        recipient = int(
            edge["recipient"]
        )
        if not (
            0
            <= sender
            < final_positions.shape[0]
        ):
            continue
        stored_sender = edge.get(
            "sender_position"
        )
        stored_recipient = edge.get(
            "recipient_position"
        )
        if (
            stored_sender is not None
            and stored_recipient is not None
        ):
            p0 = np.asarray(
                stored_sender,
                dtype=np.float64,
            )[:2]
            p1 = np.asarray(
                stored_recipient,
                dtype=np.float64,
            )[:2]
        else:
            p0 = final_positions[
                sender,
                :2,
            ]
            if recipient == GCS_NODE:
                p1 = gcs[:2]
            elif (
                0
                <= recipient
                < final_positions.shape[0]
            ):
                p1 = final_positions[
                    recipient,
                    :2,
                ]
            else:
                continue
        ax.plot(
            [
                float(p0[0]),
                float(p1[0]),
            ],
            [
                float(p0[1]),
                float(p1[1]),
            ],
            linestyle="--",
            linewidth=1.0,
            alpha=0.65,
        )

    ax.set_xlim(
        0.0,
        map_size,
    )
    ax.set_ylim(
        0.0,
        map_size,
    )
    ax.set_aspect(
        "equal",
        adjustable="box",
    )
    ax.set_xlabel(
        "x (m)"
    )
    ax.set_ylabel(
        "y (m)"
    )
    if title is None:
        title = (
            "Evaluation 2D | "
            f"coverage={trace['coverage_percent']:.1f}% | "
            "target search="
            f"{trace['target_search_rate_percent']:.1f}%"
        )
    ax.set_title(
        title
    )
    ax.grid(
        True,
        alpha=0.2,
    )
    fig.tight_layout()
    return fig


# --- frozen notebook cell 251 ---
def render_evaluation_video_2d(
    trace,
    output_path,
    fps=8,
):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.patches import Circle

    output_path = Path(
        output_path
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    map_size = float(
        trace["map_size"]
    )
    trajectory = [
        np.asarray(
            value,
            dtype=np.float64,
        )
        for value in trace.get(
            "trajectory_xyz",
            [],
        )
    ]
    trace_steps = [
        int(value)
        for value in trace.get(
            "trace_steps",
            [],
        )
    ]
    coverage_history = [
        np.asarray(
            value,
            dtype=bool,
        )
        for value in trace.get(
            "coverage_history",
            [],
        )
    ]
    target_status_history = list(
        trace.get(
            "target_status_history",
            [],
        )
    )
    if not trajectory:
        raise ValueError(
            "trace has no trajectory frames"
        )

    frame_count = len(
        trajectory
    )
    if len(trace_steps) != frame_count:
        trace_steps = list(
            range(
                frame_count
            )
        )

    fig, ax = plt.subplots(
        figsize=(8.0, 8.0)
    )
    writer = FFMpegWriter(
        fps=max(
            1,
            int(fps),
        ),
        metadata={
            "title": (
                "Post-train UAV evaluation"
            )
        },
    )

    gcs = np.asarray(
        trace[
            "gcs_position"
        ],
        dtype=np.float64,
    )
    edges = list(
        trace.get(
            "communication_edges",
            [],
        )
    )

    with writer.saving(
        fig,
        str(
            output_path
        ),
        dpi=120,
    ):
        for frame_index in range(
            frame_count
        ):
            ax.clear()
            current_step = int(
                trace_steps[
                    frame_index
                ]
            )

            if (
                frame_index
                < len(
                    coverage_history
                )
            ):
                coverage = (
                    coverage_history[
                        frame_index
                    ]
                )
            else:
                coverage = np.asarray(
                    trace[
                        "coverage_grid"
                    ],
                    dtype=bool,
                )
            ax.imshow(
                coverage.astype(
                    np.float32
                ),
                origin="lower",
                extent=(
                    0.0,
                    map_size,
                    0.0,
                    map_size,
                ),
                alpha=0.18,
                vmin=0.0,
                vmax=1.0,
            )

            for obstacle in trace[
                "obstacles"
            ]:
                xy = np.asarray(
                    obstacle[
                        "position"
                    ],
                    dtype=np.float64,
                )
                ax.add_patch(
                    Circle(
                        (
                            float(
                                xy[0]
                            ),
                            float(
                                xy[1]
                            ),
                        ),
                        float(
                            obstacle[
                                "radius"
                            ]
                        ),
                        fill=False,
                        linewidth=1.0,
                        alpha=0.7,
                    )
                )

            status = (
                target_status_history[
                    frame_index
                ]
                if frame_index
                < len(
                    target_status_history
                )
                else {}
            )
            confirmed = set(
                int(value)
                for value in status.get(
                    "confirmed",
                    [],
                )
            )
            delivered = set(
                int(value)
                for value in status.get(
                    "delivered",
                    [],
                )
            )
            for target in trace[
                "targets"
            ]:
                target_id = int(
                    target[
                        "id"
                    ]
                )
                p = np.asarray(
                    target[
                        "position"
                    ],
                    dtype=np.float64,
                )
                marker = (
                    "*"
                    if target_id
                    in delivered
                    else (
                        "P"
                        if target_id
                        in confirmed
                        else "x"
                    )
                )
                ax.scatter(
                    [float(p[0])],
                    [float(p[1])],
                    marker=marker,
                    color="tab:red",
                    s=30,
                )

            history = np.stack(
                trajectory[
                    : frame_index
                    + 1
                ],
                axis=0,
            )
            current_positions = (
                history[-1]
            )
            for uav_id in range(
                current_positions.shape[0]
            ):
                path = history[
                    :,
                    uav_id,
                    :,
                ]
                ax.plot(
                    path[:, 0],
                    path[:, 1],
                    linewidth=1.0,
                    alpha=0.8,
                )
                current = (
                    current_positions[
                        uav_id
                    ]
                )
                ax.scatter(
                    [float(current[0])],
                    [float(current[1])],
                    marker="o",
                    s=42,
                )
                ax.text(
                    float(
                        current[0]
                    ),
                    float(
                        current[1]
                    ),
                    (
                        f" U{uav_id}"
                        f" z={current[2]:.0f}m"
                    ),
                    fontsize=7,
                )

            ax.scatter(
                [float(gcs[0])],
                [float(gcs[1])],
                marker="s",
                s=60,
            )
            ax.text(
                float(gcs[0]),
                float(gcs[1]),
                " GCS",
                fontsize=8,
            )

            previous_step = (
                int(
                    trace_steps[
                        frame_index - 1
                    ]
                )
                if frame_index > 0
                else 0
            )
            interval_edges = [
                edge
                for edge in edges
                if (
                    previous_step
                    < int(
                        edge.get(
                            "step",
                            current_step,
                        )
                    )
                    <= current_step
                )
            ]
            max_edges = max(
                1,
                int(
                    CONFIG.get(
                        "training_visualization_max_link_edges",
                        64,
                    )
                ),
            )
            for edge in interval_edges[
                -max_edges:
            ]:
                p0_raw = edge.get(
                    "sender_position"
                )
                p1_raw = edge.get(
                    "recipient_position"
                )
                if (
                    p0_raw is None
                    or p1_raw is None
                ):
                    continue
                p0 = np.asarray(
                    p0_raw,
                    dtype=np.float64,
                )
                p1 = np.asarray(
                    p1_raw,
                    dtype=np.float64,
                )
                ax.plot(
                    [
                        float(p0[0]),
                        float(p1[0]),
                    ],
                    [
                        float(p0[1]),
                        float(p1[1]),
                    ],
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.75,
                )

            coverage_percent = float(
                100.0
                * np.count_nonzero(
                    coverage
                )
                / max(
                    1,
                    coverage.size,
                )
            )
            target_rate = float(
                100.0
                * len(
                    confirmed
                )
                / max(
                    1,
                    len(
                        trace[
                            "targets"
                        ]
                    ),
                )
            )
            ax.set_xlim(
                0.0,
                map_size,
            )
            ax.set_ylim(
                0.0,
                map_size,
            )
            ax.set_aspect(
                "equal",
                adjustable="box",
            )
            ax.set_xlabel(
                "x (m)"
            )
            ax.set_ylabel(
                "y (m)"
            )
            ax.set_title(
                (
                    "Post-train evaluation | "
                    f"step={current_step} | "
                    f"coverage={coverage_percent:.1f}% | "
                    f"target search={target_rate:.1f}%"
                )
            )
            ax.grid(
                True,
                alpha=0.2,
            )
            fig.tight_layout()
            writer.grab_frame()

    plt.close(
        fig
    )
    return output_path


# --- frozen notebook cell 252 ---
def save_full_gpu_model_checkpoint(
    path,
    trainer,
    algorithm,
    global_step,
    episodes_completed,
    latest_evaluation=None,
    checkpoint_kind="full_gpu_model_optimizer",
):
    path = Path(
        path
    )
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    normalized = str(
        algorithm
    ).strip().lower()
    if normalized == "masac":
        saved = save_masac_checkpoint(
            path,
            trainer,
            replay_buffer=None,
            training_state={
                "global_step": int(
                    global_step
                ),
                "episode_index": int(
                    episodes_completed
                ),
                "latest_evaluation": (
                    copy.deepcopy(
                        latest_evaluation
                    )
                ),
                "resume_safe": False,
                "checkpoint_kind": str(
                    checkpoint_kind
                ),
            },
            include_replay=False,
        )
    elif normalized == "matd3":
        saved = save_matd3_checkpoint(
            path,
            trainer,
            replay_buffer=None,
            training_state={
                "global_step": int(
                    global_step
                ),
                "episode_index": int(
                    episodes_completed
                ),
                "latest_evaluation": (
                    copy.deepcopy(
                        latest_evaluation
                    )
                ),
                "resume_safe": False,
                "checkpoint_kind": str(
                    checkpoint_kind
                ),
            },
            include_replay=False,
        )
    else:
        raise ValueError(
            "algorithm must be masac or matd3"
        )
    return Path(saved)


# --- frozen notebook cell 253 ---
def evaluate_full_gpu_reference(
    trainer,
    algorithm,
    network_backend,
    seed,
    episodes=None,
    uavnetsim_repo_path=None,
):
    algorithm = str(algorithm).strip().lower()
    if algorithm not in {"masac", "matd3"}:
        raise ValueError(
            "algorithm must be masac or matd3"
        )
    if episodes is None:
        episodes = int(
            CONFIG["training_eval_episodes"]
        )
    episodes = int(episodes)
    if episodes < 1:
        raise ValueError(
            "episodes must be >= 1"
        )

    normalized_backend = str(
        network_backend
    ).strip().lower()
    if normalized_backend in {
        "uavnetsim",
        "uavnetsim_gpu",
    }:
        eval_backend = "uavnetsim"
        resolved_uavnetsim_path = (
            ensure_uavnetsim_reference_available(
                uavnetsim_repo_path
            )
        )
        backend_kwargs = {
            "repo_path": (
                resolved_uavnetsim_path
            )
        }
    else:
        # packet_approx has no event-level CPU counterpart. Evaluate its
        # learned policy in the deterministic Simple reference backend and
        # report the backend explicitly instead of inventing packet KPIs.
        eval_backend = "simple"
        backend_kwargs = {}

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else []
    )
    trainer_rng_state = None
    if hasattr(trainer, "rng"):
        trainer_rng_state = copy.deepcopy(
            trainer.rng.bit_generator.state
        )
    actor_training = bool(
        trainer.actor.training
    )

    env = None
    try:
        trainer.actor.eval()
        env = UAVSearchEnv(
            backend_name=eval_backend,
            network_backend_kwargs=(
                backend_kwargs
            ),
        )
        evaluation_seed = (
            int(seed)
            + int(
                CONFIG[
                    "training_eval_seed_offset"
                ]
            )
        )
        if algorithm == "masac":
            rows = evaluate_masac(
                env,
                trainer,
                episodes=episodes,
                seed=evaluation_seed,
            )
        else:
            rows = evaluate_matd3(
                env,
                trainer,
                episodes=episodes,
                seed=evaluation_seed,
            )
        summary = summarize_evaluation_results(
            rows
        )
        return {
            "backend": eval_backend,
            "summary": summary,
            "episodes": rows,
            "seed": evaluation_seed,
        }
    finally:
        if env is not None:
            env.close()
        trainer.actor.train(
            actor_training
        )
        random.setstate(
            python_state
        )
        np.random.set_state(
            numpy_state
        )
        torch.set_rng_state(
            torch_state
        )
        if (
            torch.cuda.is_available()
            and cuda_states
        ):
            torch.cuda.set_rng_state_all(
                cuda_states
            )
        if (
            trainer_rng_state is not None
            and hasattr(trainer, "rng")
        ):
            trainer.rng.bit_generator.state = (
                trainer_rng_state
            )


def resolve_vector_gradient_steps(
    configured_gradient_steps,
    *,
    num_envs,
    batch_size,
    train_freq=1,
):
    """Scale off-policy updates for vector collection without pretending one seed is many.

    Stable off-policy training needs optimizer work to scale with the amount of
    data collected by vectorized environments. When auto scaling is enabled,
    choose enough minibatch updates so the number of replay samples consumed
    per optimizer burst is at least `training_min_replay_sample_ratio` times
    the number of newly collected transitions. `-1` keeps exact one-gradient-
    step-per-transition semantics.
    """
    configured = int(configured_gradient_steps)
    num_envs = int(num_envs)
    batch_size = int(batch_size)
    train_freq = int(train_freq)
    if configured == 0 or configured < -1:
        raise ValueError("gradient_steps must be -1 or a positive integer")
    if num_envs < 1 or batch_size < 1 or train_freq < 1:
        raise ValueError("num_envs, batch_size and train_freq must be >= 1")
    if configured == -1:
        return -1
    if not bool(CONFIG.get("training_auto_scale_gradient_steps", False)):
        return configured

    min_ratio = float(
        CONFIG.get("training_min_replay_sample_ratio", 1.0)
    )
    if not math.isfinite(min_ratio) or min_ratio <= 0.0:
        raise ValueError("training_min_replay_sample_ratio must be > 0")
    collected_per_burst = num_envs * train_freq
    auto_min = int(
        math.ceil(
            min_ratio
            * collected_per_burst
            / batch_size
        )
    )
    return max(configured, max(1, auto_min))


# --- frozen notebook cell 254 ---
def train_full_gpu(
    repo: Path,
    algorithm: str,
    num_envs: int,
    total_transitions: int,
    seed: int,
    device: str,
    network_backend: str = "simple",
    uavnetsim_repo_path=None,
    strict_cuda: bool = True,
    compile_mode=None,
    fused_adam: bool = False,
    update_to_data_ratio=None,
    enable_wandb: bool = False,
    enable_evaluation=None,
    target_episodes=None,
):
    algorithm = str(algorithm).strip().lower()
    if algorithm not in {"masac", "matd3"}:
        raise ValueError(
            "algorithm must be masac or matd3"
        )
    if int(total_transitions) < 1:
        raise ValueError(
            "total_transitions must be >= 1"
        )
    target_episodes = (
        None
        if target_episodes is None
        else int(target_episodes)
    )
    if target_episodes is not None and target_episodes < 1:
        raise ValueError("target_episodes must be >= 1")
    network_backend = str(network_backend).strip().lower()
    if network_backend not in {
        "simple",
        "packet_approx",
        "uavnetsim_gpu",
        "uavnetsim",
    }:
        raise ValueError(
            "network_backend must be simple, packet_approx, uavnetsim_gpu, or uavnetsim"
        )
    if network_backend == "uavnetsim" and int(num_envs) != 1:
        raise ValueError(
            "UavNetSim packet-level training currently requires num_envs=1"
        )

    torch.use_deterministic_algorithms(True)
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    env = FullGpuUAVBatchEnv(
        num_envs=num_envs,
        device=device,
        seed=seed,
        strict_cuda=bool(strict_cuda),
    )
    if network_backend in {
        "simple",
        "packet_approx",
        "uavnetsim_gpu",
    }:
        env.network_model = network_backend
    if env.device.type == "cuda":
        env.assert_all_state_on_cuda()
    trainer = _trainer(
        repo,
        algorithm,
        device,
        seed,
    )
    wandb_run = _init_gpu_wandb_run(
        algorithm,
        seed,
        enabled=enable_wandb,
        network_backend=network_backend,
        num_envs=num_envs,
        gpu_count=(
            1
            if env.device.type == "cuda"
            else 0
        ),
    )
    wandb_last_log_step = -1
    if wandb_run is not None:
        wandb_run.config.update(
            {
                "target_episodes": target_episodes,
                "max_steps": int(CONFIG["max_steps"]),
            },
            allow_val_change=True,
        )
    if enable_evaluation is None:
        enable_evaluation = bool(
            enable_wandb
            and CONFIG.get(
                "training_inline_evaluation",
                False,
            )
        )
    enable_evaluation = bool(
        enable_evaluation
        and CONFIG.get(
            "training_inline_evaluation",
            False,
        )
    )
    eval_interval = max(
        1,
        int(
            CONFIG[
                "training_eval_interval_steps"
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
    last_eval_step = 0
    latest_evaluation = None
    evaluation_backend = None
    fused_adam_enabled = False
    if bool(fused_adam):
        fused_adam_enabled = (
            _rebuild_cuda_optimizers_with_fused_adam(
                trainer,
                algorithm,
            )
        )
    compiled_modules = []
    if compile_mode is not None:
        compiled_modules = _compile_trainer_modules(
            trainer,
            algorithm,
            mode=compile_mode,
        )
    if network_backend == "uavnetsim":
        bridge = GpuUavNetSimBridge(
            env,
            trainer._project_namespace,
            repo_path=uavnetsim_repo_path,
            seed=seed,
        )
        env.attach_external_network(bridge)
    else:
        bridge = None
    if (
        env.observation_dim
        != trainer.observation_dim
        or env.state_dim
        != trainer.state_dim
        or env.discrete_dim
        != trainer.discrete_dim
    ):
        raise RuntimeError(
            "full-GPU environment dimensions do not match "
            "the CPU-reference learner contract"
        )

    capacity = min(
        max(
            int(
                CONFIG[
                    f"{algorithm}_replay_capacity"
                ]
            ),
            4096,
        ),
        max(
            int(total_transitions),
            4096,
        ),
    )
    replay = make_project_gpu_replay_buffer(
        trainer,
        capacity=capacity,
        num_agents=env.num_uavs,
        observation_dim=env.observation_dim,
        state_dim=env.state_dim,
        continuous_dim=env.continuous_dim,
        discrete_dim=env.discrete_dim,
        device=device,
        seed=seed + 123,
        strict_cuda=bool(strict_cuda),
    )
    batch_size = int(
        CONFIG[f"{algorithm}_batch_size"]
    )
    learning_starts = int(
        CONFIG[
            f"{algorithm}_learning_starts"
        ]
    )
    if update_to_data_ratio is not None:
        raise ValueError(
            "update_to_data_ratio is deprecated; configure "
            "algorithm.train_freq and algorithm.gradient_steps instead"
        )
    train_freq = int(
        CONFIG.get(f"{algorithm}_train_freq", 1)
    )
    configured_gradient_steps = int(
        CONFIG.get(
            f"{algorithm}_gradient_steps",
            CONFIG.get(f"{algorithm}_updates_per_step", 1),
        )
    )
    if train_freq < 1:
        raise ValueError("train_freq must be >= 1")
    gradient_steps = resolve_vector_gradient_steps(
        configured_gradient_steps,
        num_envs=num_envs,
        batch_size=batch_size,
        train_freq=train_freq,
    )

    obs, state, mask = env.reset()
    global_step = 0
    vector_step_count = 0
    last_train_transition = int(learning_starts)
    update_count = 0
    latest_metrics = None
    completed_episodes = torch.zeros(
        num_envs,
        device=env.device,
        dtype=torch.long,
    )
    accepted_episodes = 0
    checkpoint_paths = []
    checkpoint_interval_episodes = max(
        1,
        int(
            CONFIG.get(
                "training_checkpoint_interval_episodes",
                5_000,
            )
        ),
    )
    last_checkpoint_episode = 0
    transition_offsets = torch.arange(
        num_envs,
        device=env.device,
        dtype=torch.long,
    )
    total_transitions_int = int(
        total_transitions
    )

    if env.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(
            env.device
        )
        torch.cuda.synchronize(env.device)
    start = time.perf_counter()

    while (
        global_step < total_transitions_int
        and (
            target_episodes is None
            or accepted_episodes < target_episodes
        )
    ):
        transition_steps = (
            transition_offsets
            + global_step
            + 1
        )
        warmup = (
            transition_steps
            <= learning_starts
        )

        batch_first_step = global_step + 1
        batch_last_step = global_step + num_envs
        all_warmup = batch_last_step <= learning_starts
        all_policy = batch_first_step > learning_starts

        if all_warmup:
            continuous, destination = (
                _random_exploration_actions(
                    env,
                    mask,
                )
            )
        elif all_policy:
            if algorithm == "masac":
                with torch.no_grad():
                    sampled = (
                        trainer._sample_joint_policy(
                            obs,
                            mask,
                            deterministic=False,
                        )
                    )
                    continuous = sampled[
                        "continuous"
                    ]
                    destination = sampled[
                        "destination_indices"
                    ]
            else:
                continuous, destination = (
                    _matd3_actions(
                        trainer,
                        obs,
                        mask,
                        transition_steps,
                        env.generator,
                    )
                )
        else:
            # Only one vector batch can straddle learning_starts. In that
            # boundary batch both paths are needed to preserve per-transition
            # semantics exactly.
            random_continuous, random_destination = (
                _random_exploration_actions(
                    env,
                    mask,
                )
            )
            if algorithm == "masac":
                with torch.no_grad():
                    sampled = (
                        trainer._sample_joint_policy(
                            obs,
                            mask,
                            deterministic=False,
                        )
                    )
                    policy_continuous = sampled[
                        "continuous"
                    ]
                    policy_destination = sampled[
                        "destination_indices"
                    ]
            else:
                (
                    policy_continuous,
                    policy_destination,
                ) = _matd3_actions(
                    trainer,
                    obs,
                    mask,
                    transition_steps,
                    env.generator,
                )
            continuous = torch.where(
                warmup[:, None, None],
                random_continuous,
                policy_continuous,
            )
            destination = torch.where(
                warmup[:, None],
                random_destination,
                policy_destination,
            )

        (
            next_obs,
            next_state,
            next_mask,
            reward,
            done,
            _,
        ) = env.step(
            continuous,
            destination,
        )
        take = min(
            num_envs,
            int(total_transitions)
            - global_step,
        )
        replay.add_batch(
            obs[:take],
            state[:take],
            continuous[:take],
            destination[:take],
            mask[:take],
            reward[:take],
            next_obs[:take],
            next_state[:take],
            next_mask[:take],
            done[:take],
        )
        global_step += take
        vector_step_count += 1

        should_train = (
            len(replay) >= batch_size
            and global_step > learning_starts
            and vector_step_count % train_freq == 0
        )
        if should_train:
            if gradient_steps == -1:
                updates_this_step = max(
                    0,
                    global_step
                    - max(
                        last_train_transition,
                        learning_starts,
                    ),
                )
            else:
                updates_this_step = gradient_steps
            for local_update in range(
                updates_this_step
            ):
                collected = trainer.update(
                    replay,
                    batch_size=batch_size,
                    collect_metrics=(
                        local_update
                        == updates_this_step - 1
                    ),
                )
                if collected is not None:
                    latest_metrics = collected
                update_count += 1
            last_train_transition = global_step

        should_evaluate = (
            enable_evaluation
            and (
                global_step
                - last_eval_step
                >= eval_interval
                or global_step
                >= total_transitions_int
            )
        )
        if should_evaluate:
            evaluation = (
                evaluate_full_gpu_reference(
                    trainer=trainer,
                    algorithm=algorithm,
                    network_backend=(
                        network_backend
                    ),
                    seed=seed,
                    episodes=eval_episodes,
                    uavnetsim_repo_path=(
                        uavnetsim_repo_path
                    ),
                )
            )
            latest_evaluation = (
                evaluation["summary"]
            )
            evaluation_backend = (
                evaluation["backend"]
            )
            last_eval_step = global_step

            if wandb_run is not None:
                eval_payload = {
                    "global_step": int(
                        global_step
                    ),
                    **{
                        f"evaluation/{key}": float(
                            value
                        )
                        for key, value
                        in latest_evaluation.items()
                        if isinstance(
                            value,
                            (bool, int, float),
                        )
                    },
                }
                wandb_run.log(
                    eval_payload
                )
                for key, value in (
                    latest_evaluation.items()
                ):
                    if isinstance(
                        value,
                        (bool, int, float),
                    ):
                        wandb_run.summary[
                            f"evaluation_latest/{key}"
                        ] = float(value)

        if (
            wandb_run is not None
            and (
                global_step - wandb_last_log_step
                >= max(1, int(CONFIG.get("training_log_interval_steps", 100)))
                or global_step >= total_transitions_int
            )
        ):
            elapsed_for_log = max(
                1e-12, time.perf_counter() - start
            )
            payload = {
                "global_step": int(global_step),
                "train/transitions_per_second": (
                    global_step / elapsed_for_log
                ),
                "train/updates_per_second": (
                    update_count / elapsed_for_log
                ),
            }
            if latest_metrics is not None:
                for key, value in latest_metrics.items():
                    if isinstance(value, torch.Tensor):
                        if value.numel() != 1:
                            continue
                        value = value.detach().cpu().item()
                    if isinstance(value, (bool, int, float)):
                        payload[f"train/{key}"] = float(value)
            wandb_run.log(payload)
            wandb_last_log_step = global_step

        # A scalar host check only orchestrates rare episode reset. State
        # generation/reset and all environment math remain CUDA tensor work.
        target_reached = False
        if bool(done.any().item()):
            completed_episodes += done.long()
            finished_now = int(
                done[:take].long().sum().item()
            )
            if target_episodes is None:
                accepted_episodes += finished_now
            else:
                remaining = max(
                    0,
                    target_episodes - accepted_episodes,
                )
                accepted_episodes += min(
                    finished_now,
                    remaining,
                )
            target_reached = (
                target_episodes is not None
                and accepted_episodes >= target_episodes
            )
            if wandb_run is not None:
                wandb_run.summary[
                    "episodes_completed"
                ] = int(accepted_episodes)

            should_checkpoint = (
                accepted_episodes > 0
                and (
                    accepted_episodes
                    - last_checkpoint_episode
                    >= checkpoint_interval_episodes
                    or target_reached
                )
            )
            if should_checkpoint:
                checkpoint_path = (
                    Path(repo)
                    / "checkpoints_full_gpu"
                    / (
                        f"{algorithm}_episode_"
                        f"{accepted_episodes:08d}.pt"
                    )
                )
                saved_path = save_full_gpu_model_checkpoint(
                    checkpoint_path,
                    trainer,
                    algorithm,
                    global_step,
                    accepted_episodes,
                    latest_evaluation=latest_evaluation,
                    checkpoint_kind="full_gpu_model_optimizer",
                )
                checkpoint_paths.append(
                    str(saved_path)
                )
                last_checkpoint_episode = (
                    accepted_episodes
                )

            if (
                not target_reached
                and global_step < int(total_transitions)
            ):
                obs, state, mask = env.reset(
                    mask=done
                )
            else:
                obs, state, mask = (
                    next_obs,
                    next_state,
                    next_mask,
                )
        else:
            obs, state, mask = (
                next_obs,
                next_state,
                next_mask,
            )

    if env.device.type == "cuda":
        torch.cuda.synchronize(env.device)
    wall = time.perf_counter() - start

    def metric_value(value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(
                    value.detach().cpu().item()
                )
            return None
        if isinstance(
            value,
            (int, float, bool),
        ):
            return value
        return None

    metrics = None
    if latest_metrics is not None:
        metrics = {
            key: metric_value(value)
            for key, value
            in latest_metrics.items()
            if metric_value(value)
            is not None
        }

    network_metrics = (
        bridge.metrics()
        if bridge is not None
        else None
    )
    result = {
        "kind": (
            "gpu_mission_uavnetsim_train"
            if network_backend == "uavnetsim"
            else (
                "full_gpu_uavnetsim_surrogate_train"
                if network_backend == "uavnetsim_gpu"
                else (
                    "full_gpu_packet_train"
                    if network_backend == "packet_approx"
                    else "full_gpu_train"
                )
            )
        ),
        "network_backend": network_backend,
        "algorithm": algorithm,
        "compile_mode": compile_mode,
        "compiled_modules": compiled_modules,
        "fused_adam": bool(
            fused_adam_enabled
        ),
        "num_envs": num_envs,
        "transitions": global_step,
        "updates": update_count,
        "train_freq": int(train_freq),
        "gradient_steps": int(gradient_steps),
        "episodes_completed": int(
            accepted_episodes
        ),
        "wall_seconds": wall,
        "transitions_per_second": (
            global_step / wall
        ),
        "updates_per_second": (
            update_count / wall
        ),
        "gpu_name": (
            torch.cuda.get_device_name(
                env.device
            )
            if env.device.type == "cuda"
            else None
        ),
        "peak_memory_mb": (
            torch.cuda.max_memory_allocated(
                env.device
            )
            / 2**20
            if env.device.type == "cuda"
            else None
        ),
        "latest_update_metrics": metrics,
        "model_fingerprint_sha256": (
            _trainer_parameter_fingerprint(
                trainer
            )
        ),
        "all_env_state_cuda": (
            env.device.type == "cuda"
        ),
        "network_metrics": network_metrics,
        "latest_evaluation": (
            latest_evaluation
        ),
        "evaluation_backend": (
            evaluation_backend
        ),
    }
    final_checkpoint_path = save_full_gpu_model_checkpoint(
        Path(repo)
        / "checkpoints_full_gpu"
        / (
            f"{algorithm}_final_step_{int(global_step):012d}.pt"
        ),
        trainer,
        algorithm,
        global_step,
        result["episodes_completed"],
        latest_evaluation=latest_evaluation,
        checkpoint_kind="full_gpu_model_optimizer",
    )
    result["final_checkpoint_path"] = str(final_checkpoint_path)
    checkpoint_paths.append(
        str(final_checkpoint_path)
    )
    result["checkpoint_paths"] = list(
        dict.fromkeys(checkpoint_paths)
    )
    result["wandb_run_id"] = (
        wandb_run.id if wandb_run is not None else None
    )
    result["wandb_run_name"] = (
        wandb_run.name if wandb_run is not None else None
    )
    result["wandb_run_url"] = (
        wandb_run.url if wandb_run is not None else None
    )
    if wandb_run is not None:
        if bool(
            CONFIG.get(
                "training_publish_final_checkpoint_wandb_artifact",
                True,
            )
        ):
            wandb_module = importlib.import_module(
                "wandb"
            )
            artifact = wandb_module.Artifact(
                name=(
                    f"{algorithm}-seed{seed}-"
                    f"{wandb_run.id}-checkpoint"
                ),
                type="model",
                description=(
                    "Final checkpoint for CPU "
                    "evaluation/visualization."
                ),
                metadata={
                    "algorithm": algorithm,
                    "seed": int(seed),
                    "episodes_completed": int(
                        result["episodes_completed"]
                    ),
                    "global_step": int(global_step),
                    "network_backend": str(
                        network_backend
                    ),
                    "source_run_id": str(
                        wandb_run.id
                    ),
                    "checkpoint_kind": (
                        "full_gpu_model_optimizer"
                    ),
                },
            )
            artifact.add_file(
                str(final_checkpoint_path),
                name="checkpoint.pt",
            )
            wandb_run.log_artifact(
                artifact,
                aliases=[
                    str(
                        CONFIG.get(
                            "training_final_checkpoint_artifact_alias",
                            "final",
                        )
                    ),
                    "latest",
                    (
                        "episode-"
                        f"{result['episodes_completed']}"
                    ),
                ],
            )
            artifact_ref = (
                f"{artifact.name}:"
                f"{CONFIG.get('training_final_checkpoint_artifact_alias', 'final')}"
            )
            wandb_run.summary[
                "final/checkpoint_artifact"
            ] = artifact_ref
            result["final_checkpoint_artifact"] = artifact_ref
        wandb_run.summary["final/transitions_per_second"] = result[
            "transitions_per_second"
        ]
        wandb_run.summary["final/updates_per_second"] = result[
            "updates_per_second"
        ]
        wandb_run.summary["final/gpu_count"] = (
            1
            if env.device.type == "cuda"
            else 0
        )
        wandb_run.finish()
    env.close()
    return result


# --- frozen notebook cell 256 ---
class _MASACDDPActorForward(torch.nn.Module):
    def __init__(self, actor, project_ns):
        super().__init__()
        self.actor = actor
        self._radial_squash = project_ns["radial_squash_motion_action"]
        self._categorical_sample = project_ns["straight_through_categorical_sample"]

    def forward(self, observations, destination_mask):
        mean, log_std, logits = self.actor(observations)
        std = torch.exp(log_std)
        distribution = torch.distributions.Normal(mean, std)
        pre_tanh = distribution.rsample()
        motion_action, motion_log_det = self._radial_squash(pre_tanh[..., :3])
        power_action = torch.tanh(pre_tanh[..., 3:4])
        continuous_action = torch.cat((motion_action, power_action), dim=-1)
        power_log_det = torch.log(
            1.0 - power_action.pow(2) + 1e-6
        ).sum(dim=-1, keepdim=True)
        log_probability = (
            distribution.log_prob(pre_tanh).sum(dim=-1, keepdim=True)
            - motion_log_det
            - power_log_det
        )
        discrete = self._categorical_sample(
            logits,
            destination_mask,
            temperature=CONFIG["masac_gumbel_temperature"],
            deterministic=False,
        )
        return (
            continuous_action,
            log_probability,
            discrete["action"],
            discrete["indices"],
            discrete["log_probability"],
            discrete["entropy"],
        )


# --- frozen notebook cell 257 ---
class _MATD3DDPActorForward(torch.nn.Module):
    def __init__(self, actor, project_ns):
        super().__init__()
        self.actor = actor
        self._masked_argmax = project_ns["straight_through_masked_argmax"]

    def forward(self, observations, destination_mask):
        continuous, logits = self.actor(observations)
        discrete = self._masked_argmax(logits, destination_mask)
        return (
            continuous,
            discrete["action"],
            discrete["indices"],
            discrete["probabilities"],
        )


# --- frozen notebook cell 259 ---
def _gpu_wandb_api_key_from_kaggle():
    if os.environ.get("WANDB_API_KEY"):
        return os.environ["WANDB_API_KEY"]
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        for secret_name in CONFIG.get("kaggle_wandb_secret_names", ()):
            try:
                value = client.get_secret(str(secret_name))
            except Exception:
                continue
            if value:
                os.environ["WANDB_API_KEY"] = str(value)
                return str(value)
    except Exception:
        pass
    return None


# --- frozen notebook cell 260 ---
def _init_gpu_wandb_run(
    algorithm,
    seed,
    enabled=None,
    network_backend=None,
    num_envs=None,
    gpu_count=None,
):
    if enabled is None:
        enabled = bool(CONFIG.get("training_enable_wandb", True))
    if not enabled:
        return None
    mode = str(
        CONFIG.get(
            "training_wandb_mode",
            "online",
        )
    ).strip().lower()
    if mode != "online":
        raise RuntimeError(
            "W&B full-flow logging must use mode='online'."
        )
    kaggle_account = resolve_kaggle_account()
    execution_platform = str(
        CONFIG.get(
            "_runtime_provider",
            "kaggle" if running_on_kaggle() else "local",
        )
    ).strip().lower()
    if (
        mode == "online"
        and not os.environ.get(
            "WANDB_API_KEY"
        )
    ):
        resolved_key = (
            _gpu_wandb_api_key_from_kaggle()
        )
        if (
            running_on_kaggle()
            and not resolved_key
        ):
            if bool(
                CONFIG.get(
                    "training_wandb_required",
                    True,
                )
            ):
                raise RuntimeError(
                    "W&B online mode requires a "
                    "WANDB_API_KEY Kaggle Secret"
                )
            return None
    wandb = importlib.import_module("wandb")
    import secrets
    run_id = secrets.token_hex(4)
    name_template = str(
        CONFIG.get(
            "training_wandb_name_format",
            "{algorithm}-seed{seed}-{id}",
        )
    )
    run_name = name_template.format(
        algorithm=str(algorithm).upper(),
        seed=int(seed),
        id=run_id,
    )
    try:
        run = wandb.init(
            entity=CONFIG.get("training_wandb_entity"),
            project=str(CONFIG["training_wandb_project"]),
            name=run_name,
            id=run_id,
            mode=mode,
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
    except Exception:
        if bool(CONFIG.get("training_wandb_required", True)):
            raise
        return None

    run.define_metric(
        "global_step"
    )
    run.define_metric(
        "episodes_completed"
    )
    for pattern in (
        "train/*",
        "evaluation/*",
    ):
        run.define_metric(
            pattern,
            step_metric="global_step",
            summary="none",
        )
    run.define_metric(
        "episode/*",
        step_metric="episodes_completed",
        summary="none",
    )
    run.define_metric(
        "paper/*",
        step_metric="episodes_completed",
        summary="none",
    )
    run.define_metric(
        "visualization/*",
        step_metric="episodes_completed",
        summary="none",
    )
    run.summary[
        "episodes_completed"
    ] = 0
    run.summary["kaggle_account"] = str(kaggle_account)
    run.summary["execution_platform"] = str(execution_platform)

    run.config.update(
        {
            "algorithm": str(
                algorithm
            ).lower(),
            "seed": int(seed),
            "kaggle_account": str(kaggle_account),
            "execution_platform": execution_platform,
            "experiment": CONFIG.get("_experiment_name"),
            "resolved_config_sha256": CONFIG.get(
                "_resolved_config_sha256"
            ),
            "resolved_config": CONFIG.get(
                "_resolved_run_config"
            ),
            "network_backend": str(
                (
                    network_backend
                    if network_backend is not None
                    else CONFIG.get(
                        "full_gpu_network_model",
                        "simple",
                    )
                )
            ),
            **(
                {
                    "num_envs": int(
                        num_envs
                    )
                }
                if num_envs is not None
                else {}
            ),
            **(
                {
                    "gpu_count": int(
                        gpu_count
                    )
                }
                if gpu_count is not None
                else {}
            ),
            "num_uavs": int(
                CONFIG["num_uavs"]
            ),
            "num_targets": int(
                CONFIG["num_targets"]
            ),
            "num_obstacles": int(
                CONFIG["num_obstacles"]
            ),
            "metric_schema_version": "paper-kpi-v3",
            "metric_groups": (
                "train,episode,paper,evaluation,visualization"
            ),
        },
        allow_val_change=True,
    )
    return run


# --- frozen notebook cell 261 ---
def _ddp_actor_sample_masac(trainer, actor_ddp, observations, masks):
    batch_size = int(observations.shape[0])
    flat_obs = observations.reshape(
        batch_size * trainer.num_agents,
        trainer.observation_dim,
    )
    flat_mask = masks.reshape(
        batch_size * trainer.num_agents,
        trainer.discrete_dim,
    )
    (
        continuous,
        continuous_log_probability,
        destination_one_hot,
        destination_indices,
        destination_log_probability,
        destination_entropy,
    ) = actor_ddp(flat_obs, flat_mask)
    return {
        "continuous": continuous.reshape(
            batch_size, trainer.num_agents, trainer.continuous_dim
        ),
        "continuous_log_probability": continuous_log_probability.reshape(
            batch_size, trainer.num_agents, 1
        ).sum(dim=1),
        "destination_one_hot": destination_one_hot.reshape(
            batch_size, trainer.num_agents, trainer.discrete_dim
        ),
        "destination_indices": destination_indices.reshape(
            batch_size, trainer.num_agents
        ),
        "destination_log_probability": destination_log_probability.reshape(
            batch_size, trainer.num_agents, 1
        ).sum(dim=1),
        "destination_entropy": destination_entropy.reshape(
            batch_size, trainer.num_agents, 1
        ).sum(dim=1),
    }


# --- frozen notebook cell 262 ---
def _ddp_actor_actions_matd3(
    trainer, actor_ddp, observations, masks
):
    batch_size = int(observations.shape[0])
    flat_obs = observations.reshape(
        batch_size * trainer.num_agents,
        trainer.observation_dim,
    )
    flat_mask = masks.reshape(
        batch_size * trainer.num_agents,
        trainer.discrete_dim,
    )
    continuous, one_hot, indices, probabilities = actor_ddp(
        flat_obs, flat_mask
    )
    return {
        "continuous": continuous.reshape(
            batch_size, trainer.num_agents, trainer.continuous_dim
        ),
        "destination_one_hot": one_hot.reshape(
            batch_size, trainer.num_agents, trainer.discrete_dim
        ),
        "destination_indices": indices.reshape(
            batch_size, trainer.num_agents
        ),
        "destination_probabilities": probabilities.reshape(
            batch_size, trainer.num_agents, trainer.discrete_dim
        ),
    }


# --- frozen notebook cell 263 ---
def _allreduce_parameter_grad(parameter):
    import torch.distributed as dist
    if parameter.grad is None:
        return
    dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
    parameter.grad.div_(dist.get_world_size())


# --- frozen notebook cell 264 ---
def _ddp_update_masac(
    trainer,
    replay,
    batch_size,
    actor_ddp,
    critic_1_ddp,
    critic_2_ddp,
    collect_metrics=True,
):
    batch = replay.sample(batch_size, trainer.device)
    destination_one_hot = torch.nn.functional.one_hot(
        batch["destination_indices"],
        num_classes=trainer.discrete_dim,
    ).to(torch.float32)
    replay_joint_action = trainer._joint_action_tensor(
        batch["continuous_actions"],
        destination_one_hot,
    )

    with torch.no_grad(), trainer._autocast():
        next_policy = trainer._sample_joint_policy(
            batch["next_observations"],
            batch["next_destination_masks"],
            deterministic=False,
        )
        next_joint_action = trainer._joint_action_tensor(
            next_policy["continuous"],
            next_policy["destination_one_hot"],
        )
        target_q = torch.minimum(
            trainer.target_critic_1(
                batch["next_states"], next_joint_action
            ),
            trainer.target_critic_2(
                batch["next_states"], next_joint_action
            ),
        )
        entropy_adjusted_target = (
            target_q
            - trainer.alpha_continuous.detach()
            * next_policy["continuous_log_probability"]
            - trainer.alpha_discrete.detach()
            * next_policy["destination_log_probability"]
        )
        critic_target = (
            batch["rewards"]
            + trainer.gamma
            * (1.0 - batch["terminated"])
            * entropy_adjusted_target
        )

    with trainer._autocast():
        critic_1_value = critic_1_ddp(
            batch["states"], replay_joint_action
        )
        critic_2_value = critic_2_ddp(
            batch["states"], replay_joint_action
        )
        critic_loss = (
            torch.nn.functional.mse_loss(
                critic_1_value, critic_target
            )
            + torch.nn.functional.mse_loss(
                critic_2_value, critic_target
            )
        )

    trainer.critic_optimizer.zero_grad(set_to_none=True)
    if trainer.amp_enabled:
        trainer.grad_scaler.scale(critic_loss).backward()
        trainer.grad_scaler.unscale_(trainer.critic_optimizer)
    else:
        critic_loss.backward()
    critic_grad_norm = torch.nn.utils.clip_grad_norm_(
        list(trainer.critic_1.parameters())
        + list(trainer.critic_2.parameters()),
        trainer.gradient_clip_norm,
    )
    if trainer.amp_enabled:
        trainer.grad_scaler.step(trainer.critic_optimizer)
    else:
        trainer.critic_optimizer.step()

    for critic in (trainer.critic_1, trainer.critic_2):
        for parameter in critic.parameters():
            parameter.requires_grad_(False)

    with trainer._autocast():
        policy = _ddp_actor_sample_masac(
            trainer,
            actor_ddp,
            batch["observations"],
            batch["destination_masks"],
        )
        policy_joint_action = trainer._joint_action_tensor(
            policy["continuous"],
            policy["destination_one_hot"],
        )
        policy_q = torch.minimum(
            trainer.critic_1(
                batch["states"], policy_joint_action
            ),
            trainer.critic_2(
                batch["states"], policy_joint_action
            ),
        )
        actor_loss = (
            trainer.alpha_continuous.detach()
            * policy["continuous_log_probability"]
            - trainer.alpha_discrete.detach()
            * policy["destination_entropy"]
            - policy_q
        ).mean()

    trainer.actor_optimizer.zero_grad(set_to_none=True)
    if trainer.amp_enabled:
        trainer.grad_scaler.scale(actor_loss).backward()
        trainer.grad_scaler.unscale_(trainer.actor_optimizer)
    else:
        actor_loss.backward()
    actor_grad_norm = torch.nn.utils.clip_grad_norm_(
        trainer.actor.parameters(),
        trainer.gradient_clip_norm,
    )
    if trainer.amp_enabled:
        trainer.grad_scaler.step(trainer.actor_optimizer)
        trainer.grad_scaler.update()
    else:
        trainer.actor_optimizer.step()

    for critic in (trainer.critic_1, trainer.critic_2):
        for parameter in critic.parameters():
            parameter.requires_grad_(True)

    continuous_entropy = -policy[
        "continuous_log_probability"
    ].detach()
    continuous_target = torch.full_like(
        continuous_entropy,
        float(trainer.continuous_target_entropy),
    )
    valid_counts = batch["destination_masks"].sum(
        dim=-1
    ).clamp(min=1.0)
    discrete_target = (
        trainer.discrete_target_entropy_ratio
        * torch.log(valid_counts).sum(
            dim=1, keepdim=True
        )
    )
    discrete_entropy = policy["destination_entropy"].detach()
    alpha_continuous_loss = (
        trainer.log_alpha_continuous
        * (continuous_entropy - continuous_target)
    ).mean()
    alpha_discrete_loss = (
        trainer.log_alpha_discrete
        * (discrete_entropy - discrete_target)
    ).mean()

    trainer.alpha_continuous_optimizer.zero_grad(set_to_none=True)
    alpha_continuous_loss.backward()
    _allreduce_parameter_grad(trainer.log_alpha_continuous)
    trainer.alpha_continuous_optimizer.step()

    trainer.alpha_discrete_optimizer.zero_grad(set_to_none=True)
    alpha_discrete_loss.backward()
    _allreduce_parameter_grad(trainer.log_alpha_discrete)
    trainer.alpha_discrete_optimizer.step()

    trainer._soft_update_targets()
    trainer.update_count += 1

    if not collect_metrics:
        return None
    return {
        "critic_loss": float(critic_loss.detach().cpu()),
        "actor_loss": float(actor_loss.detach().cpu()),
        "alpha_continuous_loss": float(
            alpha_continuous_loss.detach().cpu()
        ),
        "alpha_discrete_loss": float(
            alpha_discrete_loss.detach().cpu()
        ),
        "alpha_continuous": float(
            trainer.alpha_continuous.detach().cpu()
        ),
        "alpha_discrete": float(
            trainer.alpha_discrete.detach().cpu()
        ),
        "continuous_entropy": float(
            continuous_entropy.mean().cpu()
        ),
        "discrete_entropy": float(
            discrete_entropy.mean().cpu()
        ),
        "target_q_mean": float(
            critic_target.mean().detach().cpu()
        ),
        "critic_grad_norm": float(
            torch.as_tensor(critic_grad_norm).detach().cpu()
        ),
        "actor_grad_norm": float(
            torch.as_tensor(actor_grad_norm).detach().cpu()
        ),
        "update_count": int(trainer.update_count),
    }


# --- frozen notebook cell 265 ---
def _ddp_update_matd3(
    trainer,
    replay,
    batch_size,
    actor_ddp,
    critic_1_ddp,
    critic_2_ddp,
    collect_metrics=True,
):
    batch = replay.sample(batch_size, trainer.device)
    replay_destination_one_hot = torch.nn.functional.one_hot(
        batch["destination_indices"],
        num_classes=trainer.discrete_dim,
    ).to(torch.float32)
    replay_joint_action = trainer._joint_action_tensor(
        batch["continuous_actions"],
        replay_destination_one_hot,
    )

    with torch.no_grad(), trainer._autocast():
        target_policy = trainer._actor_joint_actions(
            batch["next_observations"],
            batch["next_destination_masks"],
            straight_through=False,
            target_actor=True,
        )
        noise = (
            torch.randn_like(target_policy["continuous"])
            * trainer.target_policy_noise
        ).clamp(
            -trainer.target_noise_clip,
            trainer.target_noise_clip,
        )
        target_continuous = (
            target_policy["continuous"] + noise
        ).clamp(-1.0, 1.0)
        target_continuous = torch.cat(
            (
                trainer._project_namespace[
                    "project_motion_action_tensor"
                ](target_continuous[..., :3]),
                target_continuous[..., 3:4],
            ),
            dim=-1,
        )
        target_joint_action = trainer._joint_action_tensor(
            target_continuous,
            target_policy["destination_one_hot"],
        )
        target_q = torch.minimum(
            trainer.target_critic_1(
                batch["next_states"], target_joint_action
            ),
            trainer.target_critic_2(
                batch["next_states"], target_joint_action
            ),
        )
        critic_target = (
            batch["rewards"]
            + trainer.gamma
            * (1.0 - batch["terminated"])
            * target_q
        )

    with trainer._autocast():
        critic_1_value = critic_1_ddp(
            batch["states"], replay_joint_action
        )
        critic_2_value = critic_2_ddp(
            batch["states"], replay_joint_action
        )
        critic_loss = (
            torch.nn.functional.mse_loss(
                critic_1_value, critic_target
            )
            + torch.nn.functional.mse_loss(
                critic_2_value, critic_target
            )
        )

    trainer.critic_optimizer.zero_grad(set_to_none=True)
    if trainer.amp_enabled:
        trainer.grad_scaler.scale(critic_loss).backward()
        trainer.grad_scaler.unscale_(trainer.critic_optimizer)
    else:
        critic_loss.backward()
    critic_grad_norm = torch.nn.utils.clip_grad_norm_(
        list(trainer.critic_1.parameters())
        + list(trainer.critic_2.parameters()),
        trainer.gradient_clip_norm,
    )
    if trainer.amp_enabled:
        trainer.grad_scaler.step(trainer.critic_optimizer)
    else:
        trainer.critic_optimizer.step()

    trainer.update_count += 1
    actor_updated = bool(
        trainer.update_count % trainer.policy_delay == 0
    )
    actor_loss = None
    actor_grad_norm = None

    if actor_updated:
        for critic in (trainer.critic_1, trainer.critic_2):
            for parameter in critic.parameters():
                parameter.requires_grad_(False)

        with trainer._autocast():
            policy = _ddp_actor_actions_matd3(
                trainer,
                actor_ddp,
                batch["observations"],
                batch["destination_masks"],
            )
            policy_joint_action = trainer._joint_action_tensor(
                policy["continuous"],
                policy["destination_one_hot"],
            )
            actor_loss = -trainer.critic_1(
                batch["states"],
                policy_joint_action,
            ).mean()

        trainer.actor_optimizer.zero_grad(set_to_none=True)
        if trainer.amp_enabled:
            trainer.grad_scaler.scale(actor_loss).backward()
            trainer.grad_scaler.unscale_(trainer.actor_optimizer)
        else:
            actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            trainer.actor.parameters(),
            trainer.gradient_clip_norm,
        )
        if trainer.amp_enabled:
            trainer.grad_scaler.step(trainer.actor_optimizer)
        else:
            trainer.actor_optimizer.step()

        for critic in (trainer.critic_1, trainer.critic_2):
            for parameter in critic.parameters():
                parameter.requires_grad_(True)

        trainer._soft_update(
            trainer.target_actor, trainer.actor
        )
        trainer._soft_update(
            trainer.target_critic_1, trainer.critic_1
        )
        trainer._soft_update(
            trainer.target_critic_2, trainer.critic_2
        )

    if trainer.amp_enabled:
        trainer.grad_scaler.update()
    if not collect_metrics:
        return None
    metrics = {
        "critic_loss": float(critic_loss.detach().cpu()),
        "critic_1_q_mean": float(
            critic_1_value.detach().mean().cpu()
        ),
        "critic_2_q_mean": float(
            critic_2_value.detach().mean().cpu()
        ),
        "target_q_mean": float(
            critic_target.detach().mean().cpu()
        ),
        "critic_grad_norm": float(
            torch.as_tensor(critic_grad_norm).detach().cpu()
        ),
        "actor_updated": bool(actor_updated),
        "update_count": int(trainer.update_count),
    }
    if actor_loss is not None:
        metrics["actor_loss"] = float(
            actor_loss.detach().cpu()
        )
        metrics["actor_grad_norm"] = float(
            torch.as_tensor(actor_grad_norm).detach().cpu()
        )
    return metrics


# --- frozen notebook cell 266 ---
def _episode_metric_key(name, chunk_size):
    """Use raw names for one episode; make bin averaging explicit otherwise."""
    name = str(name)
    return (
        f"episode/{name}"
        if int(chunk_size) == 1
        else f"episode/{name}_bin_mean"
    )


def _finite_episode_values(chunk, key):
    values = []
    for item in chunk:
        value = item.get(key)
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            values.append(value)
    return values


def _add_episode_value(payload, chunk, output_name, source_key):
    source_values = _finite_episode_values(chunk, source_key)
    if not source_values:
        return
    key = _episode_metric_key(output_name, len(chunk))
    if len(chunk) == 1:
        payload[key] = float(source_values[0])
    else:
        payload[key] = float(np.mean(source_values))


def add_episode_diagnostic_metrics(payload, chunk):
    """Add mission/safety/energy/network KPIs without seed-like mean/std names."""
    if not chunk:
        return payload

    direct = {
        "coverage_percent": "coverage_percent",
        "target_encounter_rate_percent": "target_encounter_rate_percent",
        "information_gain_bits": "information_gain_bits",
        "blocked_motion_rate": "blocked_motion_rate",
        "peer_safety_block_rate": "peer_safety_block_rate",
        "obstacle_block_rate": "obstacle_block_rate",
        "boundary_clip_rate": "boundary_clip_rate",
        "false_confirmations": "false_confirmations",
        "reports_created": "reports_created",
        "expired_reports": "expired_reports",
        "dropped_reports": "dropped_reports",
        "report_expiry_rate_percent": "report_expiry_rate_percent",
        "report_drop_rate_percent": "report_drop_rate_percent",
        "report_delivery_given_confirmation_rate_percent": (
            "report_delivery_given_confirmation_rate_percent"
        ),
        "communication_energy_j": "communication_energy_j",
        # This metric is intentionally a mean across UAVs inside one episode,
        # not a mean across seeds or episodes.
        "battery_remaining_mean_percent": "battery_remaining_mean_percent",
        "depleted_uav_count": "depleted_uav_count",
        "distance_total_m": "distance_total_m",
        "network_tx_attempts": "network_tx_attempts",
        "network_phy_success_percent": "network_phy_success_percent",
        "network_tx_payload_success_ratio": "network_tx_payload_success_ratio",
        "network_throughput_kbps": "network_throughput_kbps",
        "network_nlos_attempt_rate_percent": "network_nlos_attempt_rate_percent",
        "gcs_in_range_uav_fraction": "gcs_in_range_uav_fraction",
        "report_delivery_latency_s": "report_delivery_latency_s",
        "report_delivery_latency_max_s": "report_delivery_latency_max_s",
        "report_delivery_latency_sample_count": (
            "report_delivery_latency_sample_count"
        ),
        "energy_per_confirmed_target_j": "energy_per_confirmed_target_j",
        "energy_per_delivered_target_j": "energy_per_delivered_target_j",
        "time_to_first_confirm_s": "time_to_first_confirm_s",
        "time_to_all_confirm_s": "time_to_all_confirm_s",
        "time_to_first_delivery_s": "time_to_first_delivery_s",
        "first_confirmation_observed": "first_confirmation_observed",
        "all_targets_confirmed": "all_targets_confirmed",
        "first_delivery_observed": "first_delivery_observed",
    }
    for output_name, source_key in direct.items():
        _add_episode_value(
            payload,
            chunk,
            output_name,
            source_key,
        )

    return payload


def build_episode_curve_payload(
    chunk,
    episodes_completed,
    episode_component_keys,
):
    """Build raw single-episode W&B metrics; bins are explicitly named if used."""
    if not chunk:
        raise ValueError("chunk must not be empty")

    payload = {
        "episodes_completed": int(episodes_completed),
    }
    base = {
        "return": "return",
        "length": "length",
        "end_step": "end_step",
        "end_reason_code": "end_reason_code",
        "end_by_success": "end_by_success",
        "end_by_all_uavs_inactive": "end_by_all_uavs_inactive",
        "end_by_horizon": "end_by_horizon",
        "confirmed_targets": "confirmed_targets",
        "delivered_targets": "delivered_targets",
        "total_energy_j": "episode_energy_j",
        "target_search_rate_percent": "target_search_rate_percent",
        "target_delivery_rate_percent": "target_delivery_rate_percent",
        "success": "success",
    }
    for output_name, source_key in base.items():
        _add_episode_value(
            payload,
            chunk,
            output_name,
            source_key,
        )

    for component_key in episode_component_keys:
        _add_episode_value(
            payload,
            chunk,
            component_key,
            component_key,
        )

    add_episode_diagnostic_metrics(
        payload,
        chunk,
    )
    return payload


# --- frozen notebook cell 267 ---
def build_paper_curve_payload(
    chunk,
    episodes_completed,
    ema_state,
    ema_beta,
):
    """Build a paper-readable trend while preserving raw episode/* history."""
    if not chunk:
        raise ValueError("chunk must not be empty")
    beta = float(ema_beta)
    if not 0.0 <= beta < 1.0:
        raise ValueError("ema_beta must be in [0, 1)")

    candidate_sources = {
        "paper/return": "return",
        "paper/search_rate_percent": "target_search_rate_percent",
        "paper/delivery_rate_percent": "target_delivery_rate_percent",
        "paper/success_rate": "success",
        "paper/coverage_percent": "coverage_percent",
        "paper/target_encounter_rate_percent": (
            "target_encounter_rate_percent"
        ),
        "paper/end_step": "end_step",
        "paper/horizon_rate": "end_by_horizon",
        "paper/total_energy_j": "episode_energy_j",
        "paper/blocked_motion_rate": "blocked_motion_rate",
        "paper/communication_energy_j": "communication_energy_j",
        "paper/network_phy_success_percent": (
            "network_phy_success_percent"
        ),
        "paper/report_delivery_latency_s": (
            "report_delivery_latency_s"
        ),
    }
    sources = {
        output_key: source_key
        for output_key, source_key in candidate_sources.items()
        if any(
            item.get(source_key) is not None
            for item in chunk
        )
    }
    payload = {
        "episodes_completed": int(episodes_completed),
    }
    for output_key, source_key in sources.items():
        source_values = [
            float(item[source_key])
            for item in chunk
            if item.get(source_key) is not None
            and math.isfinite(
                float(item[source_key])
            )
        ]
        if not source_values:
            continue
        window_mean = float(
            np.mean(source_values)
        )
        previous = ema_state.get(output_key)
        smoothed = (
            window_mean
            if previous is None
            else beta * float(previous)
            + (1.0 - beta) * window_mean
        )
        ema_state[output_key] = float(smoothed)
        payload[output_key] = float(smoothed)
    return payload


# --- frozen notebook cell 269 ---
def paper_ema_smooth(values, beta=None):
    """Return an exponential moving average without altering raw values."""
    import numpy as np

    if beta is None:
        beta = float(
            CONFIG.get(
                "paper_plot_ema_beta",
                0.95,
            )
        )
    beta = float(beta)
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1)")

    values = np.asarray(
        values,
        dtype=np.float64,
    )
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if values.size == 0:
        return values.copy()

    output = np.empty_like(values)
    output[0] = values[0]
    for index in range(1, values.size):
        output[index] = (
            beta * output[index - 1]
            + (1.0 - beta) * values[index]
        )
    return output


# --- frozen notebook cell 270 ---
def fetch_wandb_episode_history(
    run_path,
    metric_keys,
    x_key="episodes_completed",
):
    """Fetch raw W&B history needed by publication-style plots."""
    import pandas as pd
    import wandb

    metric_keys = list(dict.fromkeys(metric_keys))
    keys = [x_key, *metric_keys]
    run = wandb.Api().run(str(run_path))

    rows = list(
        run.scan_history(
            keys=keys,
        )
    )
    if not rows:
        return pd.DataFrame(
            columns=keys
        )

    frame = pd.DataFrame(rows)
    available = [
        key
        for key in keys
        if key in frame.columns
    ]
    frame = frame[available].copy()

    if x_key not in frame.columns:
        raise KeyError(
            f"{x_key!r} is absent from W&B history for {run_path}"
        )

    frame = frame.dropna(
        subset=[x_key]
    )
    frame = frame.sort_values(
        x_key
    )
    frame = frame.drop_duplicates(
        subset=[x_key],
        keep="last",
    )
    return frame.reset_index(
        drop=True
    )


# --- frozen notebook cell 271 ---
def plot_paper_learning_curve(
    run_groups,
    y_key,
    ylabel,
    x_key="episodes_completed",
    xlabel="Episode",
    beta=None,
    ax=None,
    y_limits=None,
    output_path=None,
    dpi=300,
):
    """Plot faint raw RL traces plus a bold trend; add SEM only across >=2 runs."""
    import math
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if beta is None:
        beta = float(
            CONFIG.get(
                "paper_plot_ema_beta",
                0.95,
            )
        )
    raw_alpha = float(
        CONFIG.get(
            "paper_plot_raw_alpha",
            0.12,
        )
    )
    raw_linewidth = float(
        CONFIG.get(
            "paper_plot_raw_linewidth",
            0.9,
        )
    )
    main_linewidth = float(
        CONFIG.get(
            "paper_plot_main_linewidth",
            2.6,
        )
    )
    band_alpha = float(
        CONFIG.get(
            "paper_plot_band_alpha",
            0.16,
        )
    )

    if ax is None:
        _, ax = plt.subplots(
            figsize=(6.4, 4.4)
        )

    colors = plt.rcParams[
        "axes.prop_cycle"
    ].by_key().get(
        "color",
        [f"C{i}" for i in range(10)],
    )

    for group_index, (label, frames) in enumerate(
        run_groups.items()
    ):
        if isinstance(frames, pd.DataFrame):
            frames = [frames]
        frames = list(frames)
        if not frames:
            continue

        color = colors[
            group_index % len(colors)
        ]
        clean_runs = []
        for frame in frames:
            if (
                x_key not in frame.columns
                or y_key not in frame.columns
            ):
                continue
            data = (
                frame[[x_key, y_key]]
                .dropna()
                .sort_values(x_key)
                .drop_duplicates(
                    subset=[x_key],
                    keep="last",
                )
            )
            if data.empty:
                continue
            x = data[x_key].to_numpy(
                dtype=np.float64
            )
            y = data[y_key].to_numpy(
                dtype=np.float64
            )
            finite = (
                np.isfinite(x)
                & np.isfinite(y)
            )
            x = x[finite]
            y = y[finite]
            if x.size == 0:
                continue

            ax.plot(
                x,
                y,
                color=color,
                alpha=raw_alpha,
                linewidth=raw_linewidth,
                zorder=1,
            )
            clean_runs.append((x, y))

        if not clean_runs:
            continue

        if len(clean_runs) == 1:
            x_main, y_raw = clean_runs[0]
            y_main = paper_ema_smooth(
                y_raw,
                beta=beta,
            )
        else:
            x_min = max(
                run_x[0]
                for run_x, _ in clean_runs
            )
            x_max = min(
                run_x[-1]
                for run_x, _ in clean_runs
            )
            if x_max <= x_min:
                raise ValueError(
                    f"runs for {label!r} have no overlapping x range"
                )
            grid_size = max(
                64,
                min(
                    1200,
                    max(
                        len(run_x)
                        for run_x, _ in clean_runs
                    ),
                ),
            )
            x_main = np.linspace(
                x_min,
                x_max,
                grid_size,
            )
            interpolated = np.stack(
                [
                    np.interp(
                        x_main,
                        run_x,
                        run_y,
                    )
                    for run_x, run_y in clean_runs
                ],
                axis=0,
            )
            mean_raw = np.mean(
                interpolated,
                axis=0,
            )
            y_main = paper_ema_smooth(
                mean_raw,
                beta=beta,
            )

            sem = np.std(
                interpolated,
                axis=0,
                ddof=1,
            ) / math.sqrt(
                interpolated.shape[0]
            )
            lower = paper_ema_smooth(
                mean_raw - sem,
                beta=beta,
            )
            upper = paper_ema_smooth(
                mean_raw + sem,
                beta=beta,
            )
            ax.fill_between(
                x_main,
                lower,
                upper,
                color=color,
                alpha=band_alpha,
                linewidth=0.0,
                zorder=2,
            )

        ax.plot(
            x_main,
            y_main,
            color=color,
            linewidth=main_linewidth,
            label=str(label),
            zorder=3,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if y_limits is not None:
        ax.set_ylim(*y_limits)

    ax.grid(
        True,
        alpha=0.20,
        linewidth=0.7,
    )
    ax.legend(
        frameon=True
    )
    ax.ticklabel_format(
        style="sci",
        axis="x",
        scilimits=(4, 4),
        useMathText=True,
    )

    if output_path is not None:
        output_path = Path(
            output_path
        )
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        ax.figure.savefig(
            output_path,
            dpi=int(dpi),
            bbox_inches="tight",
        )

    ax.figure.tight_layout()
    return ax


# --- frozen notebook cell 272 ---
def plot_wandb_paper_metric(
    run_specs,
    metric_key,
    ylabel,
    x_key="episodes_completed",
    beta=None,
    ax=None,
    y_limits=None,
    output_path=None,
    dpi=300,
):
    """Fetch W&B run groups and render a publication-style learning curve."""
    run_groups = {}
    for label, paths in run_specs.items():
        if isinstance(paths, str):
            paths = [paths]
        run_groups[label] = [
            fetch_wandb_episode_history(
                run_path=path,
                metric_keys=[
                    metric_key
                ],
                x_key=x_key,
            )
            for path in paths
        ]

    return plot_paper_learning_curve(
        run_groups=run_groups,
        y_key=metric_key,
        ylabel=ylabel,
        x_key=x_key,
        beta=beta,
        ax=ax,
        y_limits=y_limits,
        output_path=output_path,
        dpi=dpi,
    )


# --- frozen notebook cell 273 ---
def _train_full_gpu_ddp_worker(
    repo,
    algorithm,
    num_envs,
    total_transitions,
    seed,
    network_backend,
    result_path,
    enable_wandb=True,
    target_episodes=None,
):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from datetime import timedelta

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(
            seconds=max(
                3600,
                int(
                    CONFIG.get(
                        "full_gpu_ddp_timeout_s",
                        3600,
                    )
                ),
            )
        ),
    )

    algorithm = str(algorithm).lower()
    if algorithm not in {"masac", "matd3"}:
        raise ValueError("algorithm must be masac or matd3")
    target_episodes = (
        None
        if target_episodes is None
        else int(target_episodes)
    )
    if (
        target_episodes is not None
        and target_episodes < 1
    ):
        raise ValueError(
            "target_episodes must be >= 1"
        )
    max_total_transitions = int(
        total_transitions
    )
    if max_total_transitions < 1:
        raise ValueError(
            "total_transitions must be >= 1"
        )
    if world_size not in {1, 2}:
        raise ValueError(
            "current distributed GPU implementation supports world_size 1 or 2"
        )

    torch.use_deterministic_algorithms(True)
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    shard_counts = _split_env_counts(num_envs, world_size)
    local_envs = int(shard_counts[rank])
    env = FullGpuUAVBatchEnv(
        num_envs=local_envs,
        device=str(device),
        seed=int(seed) + rank * 1_000_003,
        strict_cuda=True,
    )
    env.network_model = str(network_backend)
    env.assert_all_state_on_cuda()

    trainer = _trainer(
        Path(repo), algorithm, str(device), int(seed)
    )
    project_ns = trainer._project_namespace

    if algorithm == "masac":
        actor_forward = _MASACDDPActorForward(
            trainer.actor, project_ns
        ).to(device)
    else:
        actor_forward = _MATD3DDPActorForward(
            trainer.actor, project_ns
        ).to(device)

    actor_ddp = DDP(
        actor_forward,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )
    critic_1_ddp = DDP(
        trainer.critic_1,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )
    critic_2_ddp = DDP(
        trainer.critic_2,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )

    # All source networks are synchronized by DDP construction. Mirror the
    # synchronized online critics/actor into the local targets before training.
    if algorithm == "masac":
        trainer.target_critic_1.load_state_dict(
            trainer.critic_1.state_dict()
        )
        trainer.target_critic_2.load_state_dict(
            trainer.critic_2.state_dict()
        )
    else:
        trainer.target_actor.load_state_dict(
            trainer.actor.state_dict()
        )
        trainer.target_critic_1.load_state_dict(
            trainer.critic_1.state_dict()
        )
        trainer.target_critic_2.load_state_dict(
            trainer.critic_2.state_dict()
        )

    # Rank-specific data RNG after synchronized model initialization.
    random.seed(int(seed) + rank * 97_409)
    torch.cuda.manual_seed(int(seed) + rank * 97_409)

    global_capacity = min(
        max(int(CONFIG[f"{algorithm}_replay_capacity"]), 4096),
        max(int(total_transitions), 4096),
    )
    local_capacity = max(
        4096,
        int(math.ceil(global_capacity / world_size)),
    )
    replay = make_project_gpu_replay_buffer(
        trainer,
        capacity=local_capacity,
        num_agents=env.num_uavs,
        observation_dim=env.observation_dim,
        state_dim=env.state_dim,
        continuous_dim=env.continuous_dim,
        discrete_dim=env.discrete_dim,
        device=str(device),
        seed=int(seed) + 123 + rank * 10_007,
        strict_cuda=True,
    )

    global_batch_size = int(
        CONFIG[f"{algorithm}_batch_size"]
    )
    local_batch_size = max(
        1, int(math.ceil(global_batch_size / world_size))
    )
    learning_starts = int(
        CONFIG[f"{algorithm}_learning_starts"]
    )
    train_freq = int(
        CONFIG.get(f"{algorithm}_train_freq", 1)
    )
    configured_gradient_steps = int(
        CONFIG.get(
            f"{algorithm}_gradient_steps",
            CONFIG.get(f"{algorithm}_updates_per_step", 1),
        )
    )
    if train_freq < 1:
        raise ValueError("train_freq must be >= 1")
    gradient_steps = resolve_vector_gradient_steps(
        configured_gradient_steps,
        num_envs=num_envs,
        batch_size=global_batch_size,
        train_freq=train_freq,
    )

    wandb_run = (
        _init_gpu_wandb_run(
            algorithm,
            seed,
            enabled=enable_wandb,
            network_backend=network_backend,
            num_envs=num_envs,
            gpu_count=world_size,
        )
        if rank == 0
        else None
    )
    if wandb_run is not None:
        wandb_run.config.update(
            {
                "target_episodes": (
                    int(target_episodes)
                    if target_episodes
                    is not None
                    else None
                ),
                "max_steps": int(
                    CONFIG["max_steps"]
                ),
                "reward_curve_bin_episodes": int(
                    CONFIG.get(
                        "training_reward_curve_bin_episodes",
                        1,
                    )
                ),
                "paper_curve_bin_episodes": int(
                    CONFIG.get(
                        "training_paper_curve_bin_episodes",
                        128,
                    )
                ),
                "paper_curve_ema_beta": float(
                    CONFIG.get(
                        "training_paper_curve_ema_beta",
                        0.90,
                    )
                ),
                "train_freq": int(train_freq),
                "configured_gradient_steps": int(configured_gradient_steps),
                "effective_gradient_steps": int(gradient_steps),
                "gradient_steps": int(gradient_steps),
                "global_batch_size": int(global_batch_size),
                "replay_capacity": int(global_capacity),
                "replay_samples_per_new_transition": (
                    float(global_batch_size)
                    if int(gradient_steps) == -1
                    else float(gradient_steps) * float(global_batch_size)
                    / max(1.0, float(num_envs * train_freq))
                ),
                "optimizer_updates_per_1000_new_transitions": (
                    1000.0
                    if int(gradient_steps) == -1
                    else 1000.0 * float(gradient_steps)
                    / max(1.0, float(num_envs * train_freq))
                ),
                "replay_turnover_vector_steps": float(global_capacity)
                / max(1.0, float(num_envs)),
            },
            allow_val_change=True,
        )

    obs, state, mask = env.reset()
    global_step = 0
    vector_step_count = 0
    last_train_transition = int(learning_starts)
    update_count = 0
    latest_metrics = None
    completed_episodes = 0
    completed_episodes_local = 0
    episode_returns = torch.zeros(
        local_envs,
        device=device,
        dtype=torch.float32,
    )
    episode_lengths = torch.zeros(
        local_envs,
        device=device,
        dtype=torch.long,
    )
    episode_component_keys = (
        "reward_search",
        "reward_communication",
        "reward_safety",
        "reward_energy",
        "reward_mission",
    )
    episode_components = {
        key: torch.zeros(
            local_envs,
            device=device,
            dtype=torch.float32,
        )
        for key in episode_component_keys
    }
    episode_energy_j = torch.zeros(
        local_envs,
        device=device,
        dtype=torch.float32,
    )
    episode_diagnostic_keys = (
        "blocked",
        "blocked_by_peer",
        "blocked_by_obstacle",
        "boundary",
        "distance_m",
        "false_confirmations",
        "reports_created",
        "expired_reports",
        "dropped_reports",
        "comm_energy_j",
        "network_bytes_attempted",
        "network_bytes_tx",
        "network_tx_attempts",
        "network_link_successes",
        "network_nlos_attempts",
        "delivery_latency_sum_s",
        "delivery_latency_count",
        "gcs_in_range_uav_fraction",
    )
    episode_diagnostics = {
        key: torch.zeros(
            local_envs,
            device=device,
            dtype=torch.float32,
        )
        for key in episode_diagnostic_keys
    }
    episode_delivery_latency_max_s = torch.zeros(
        local_envs,
        device=device,
        dtype=torch.float32,
    )
    episode_first_confirm_step = torch.full(
        (local_envs,),
        -1,
        device=device,
        dtype=torch.long,
    )
    episode_all_confirm_step = torch.full(
        (local_envs,),
        -1,
        device=device,
        dtype=torch.long,
    )
    episode_first_delivery_step = torch.full(
        (local_envs,),
        -1,
        device=device,
        dtype=torch.long,
    )
    episode_bin_buffer = []
    episodes_logged = 0
    reward_curve_bin = max(
        1,
        int(
            CONFIG.get(
                "training_reward_curve_bin_episodes",
                1,
            )
        ),
    )
    paper_bin_buffer = []
    paper_episodes_logged = 0
    paper_curve_bin = max(
        1,
        int(
            CONFIG.get(
                "training_paper_curve_bin_episodes",
                128,
            )
        ),
    )
    paper_ema_beta = float(
        CONFIG.get(
            "training_paper_curve_ema_beta",
            0.90,
        )
    )
    if not 0.0 <= paper_ema_beta < 1.0:
        raise ValueError(
            "training_paper_curve_ema_beta must be in [0, 1)"
        )
    paper_ema_state = {}
    checkpoint_interval_episodes = max(
        1,
        int(
            CONFIG.get(
                "training_checkpoint_interval_episodes",
                5_000,
            )
        ),
    )
    eval_interval_episodes = max(
        1,
        int(
            CONFIG.get(
                "training_eval_interval_episodes",
                5_000,
            )
        ),
    )
    visualization_interval_episodes = max(
        1,
        int(
            CONFIG.get(
                "training_visualization_interval_episodes",
                10_000,
            )
        ),
    )
    last_checkpoint_episode = 0
    last_eval_episode = 0
    last_visualization_episode = 0
    checkpoint_paths = []
    shard_offsets = []
    running_offset = 0
    for count in shard_counts:
        shard_offsets.append(running_offset)
        running_offset += int(count)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    log_interval = max(
        1, int(CONFIG.get("training_log_interval_steps", 100))
    )
    last_log_step = -1
    eval_interval = max(
        1,
        int(
            CONFIG[
                "training_eval_interval_steps"
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
    last_eval_step = 0
    latest_evaluation = None
    evaluation_backend = None
    enable_evaluation = bool(
        enable_wandb
        and CONFIG.get(
            "training_inline_evaluation",
            False,
        )
    )
    enable_visualization = bool(
        enable_evaluation
        and CONFIG.get(
            "training_inline_visualization",
            False,
        )
        and CONFIG.get(
            "training_visualize_2d",
            True,
        )
    )

    try:
        while (
            global_step
            < max_total_transitions
            and (
                target_episodes is None
                or completed_episodes
                < target_episodes
            )
        ):
            local_offset = int(shard_offsets[rank])
            transition_steps = (
                torch.arange(
                    local_envs,
                    device=device,
                    dtype=torch.long,
                )
                + global_step
                + local_offset
                + 1
            )
            batch_first = global_step + 1
            batch_last = global_step + int(num_envs)
            all_warmup = batch_last <= learning_starts
            all_policy = batch_first > learning_starts

            if all_warmup:
                continuous, destination = (
                    _random_exploration_actions(env, mask)
                )
            elif all_policy:
                if algorithm == "masac":
                    with torch.no_grad():
                        sampled = trainer._sample_joint_policy(
                            obs,
                            mask,
                            deterministic=False,
                        )
                        continuous = sampled["continuous"]
                        destination = sampled[
                            "destination_indices"
                        ]
                else:
                    continuous, destination = _matd3_actions(
                        trainer,
                        obs,
                        mask,
                        transition_steps,
                        env.generator,
                    )
            else:
                random_continuous, random_destination = (
                    _random_exploration_actions(env, mask)
                )
                if algorithm == "masac":
                    with torch.no_grad():
                        sampled = trainer._sample_joint_policy(
                            obs,
                            mask,
                            deterministic=False,
                        )
                        policy_continuous = sampled["continuous"]
                        policy_destination = sampled[
                            "destination_indices"
                        ]
                else:
                    policy_continuous, policy_destination = (
                        _matd3_actions(
                            trainer,
                            obs,
                            mask,
                            transition_steps,
                            env.generator,
                        )
                    )
                warmup = transition_steps <= learning_starts
                continuous = torch.where(
                    warmup[:, None, None],
                    random_continuous,
                    policy_continuous,
                )
                destination = torch.where(
                    warmup[:, None],
                    random_destination,
                    policy_destination,
                )

            (
                next_obs,
                next_state,
                next_mask,
                reward,
                done,
                step_metrics,
            ) = env.step(continuous, destination)

            episode_returns += reward
            episode_lengths += 1
            for component_key in (
                episode_component_keys
            ):
                episode_components[
                    component_key
                ] += step_metrics[
                    component_key
                ]
            episode_energy_j += (
                step_metrics[
                    "energy_j"
                ].float()
            )
            for diagnostic_key in (
                episode_diagnostic_keys
            ):
                episode_diagnostics[
                    diagnostic_key
                ] += step_metrics[
                    diagnostic_key
                ].float()
            episode_delivery_latency_max_s = torch.maximum(
                episode_delivery_latency_max_s,
                step_metrics[
                    "delivery_latency_max_s"
                ].float(),
            )
            current_episode_step = episode_lengths.clone()
            first_confirm_mask = (
                (episode_first_confirm_step < 0)
                & (
                    step_metrics[
                        "new_confirmed"
                    ] > 0
                )
            )
            episode_first_confirm_step = torch.where(
                first_confirm_mask,
                current_episode_step,
                episode_first_confirm_step,
            )
            all_confirm_mask = (
                (episode_all_confirm_step < 0)
                & (
                    step_metrics[
                        "confirmed_count"
                    ]
                    >= int(
                        CONFIG[
                            "num_targets"
                        ]
                    )
                )
            )
            episode_all_confirm_step = torch.where(
                all_confirm_mask,
                current_episode_step,
                episode_all_confirm_step,
            )
            first_delivery_mask = (
                (episode_first_delivery_step < 0)
                & (
                    step_metrics[
                        "new_delivered"
                    ] > 0
                )
            )
            episode_first_delivery_step = torch.where(
                first_delivery_mask,
                current_episode_step,
                episode_first_delivery_step,
            )

            take_total = min(
                int(num_envs),
                int(total_transitions) - global_step,
            )
            local_take = max(
                0,
                min(
                    local_envs,
                    take_total - local_offset,
                ),
            )
            if local_take > 0:
                replay.add_batch(
                    obs[:local_take],
                    state[:local_take],
                    continuous[:local_take],
                    destination[:local_take],
                    mask[:local_take],
                    reward[:local_take],
                    next_obs[:local_take],
                    next_state[:local_take],
                    next_mask[:local_take],
                    done[:local_take],
                )

            global_step += take_total
            vector_step_count += 1

            local_ready = torch.tensor(
                [int(len(replay) >= local_batch_size)],
                device=device,
                dtype=torch.int32,
            )
            dist.all_reduce(
                local_ready, op=dist.ReduceOp.MIN
            )
            should_train = (
                int(local_ready.item()) == 1
                and global_step > learning_starts
                and vector_step_count % train_freq == 0
            )
            if should_train:
                if gradient_steps == -1:
                    updates_this_step = max(
                        0,
                        global_step
                        - max(
                            last_train_transition,
                            learning_starts,
                        ),
                    )
                else:
                    updates_this_step = gradient_steps
                for local_update in range(
                    updates_this_step
                ):
                    collect = (
                        local_update
                        == updates_this_step - 1
                    )
                    if algorithm == "masac":
                        metrics = _ddp_update_masac(
                            trainer,
                            replay,
                            local_batch_size,
                            actor_ddp,
                            critic_1_ddp,
                            critic_2_ddp,
                            collect_metrics=collect,
                        )
                    else:
                        metrics = _ddp_update_matd3(
                            trainer,
                            replay,
                            local_batch_size,
                            actor_ddp,
                            critic_1_ddp,
                            critic_2_ddp,
                            collect_metrics=collect,
                        )
                    if metrics is not None:
                        latest_metrics = metrics
                    update_count += 1
                last_train_transition = global_step

            local_done_count = int(
                done.long().sum().item()
            )
            completed_episodes_local += (
                local_done_count
            )
            global_done_tensor = torch.tensor(
                [local_done_count],
                device=device,
                dtype=torch.long,
            )
            dist.all_reduce(
                global_done_tensor,
                op=dist.ReduceOp.SUM,
            )
            global_done_count = int(
                global_done_tensor.item()
            )
            accepted_done_count = (
                global_done_count
                if target_episodes is None
                else min(
                    global_done_count,
                    max(
                        0,
                        int(target_episodes)
                        - completed_episodes,
                    ),
                )
            )

            if global_done_count > 0:
                done_indices = torch.nonzero(
                    done,
                    as_tuple=False,
                ).flatten()
                local_records = []
                if done_indices.numel() > 0:
                    confirmed = step_metrics[
                        "confirmed_count"
                    ][done_indices].float()
                    delivered = step_metrics[
                        "delivered_count"
                    ][done_indices].float()
                    success_values = step_metrics[
                        "success"
                    ][done_indices].float()
                    end_success_values = step_metrics[
                        "end_success"
                    ][done_indices].float()
                    end_inactive_values = step_metrics[
                        "end_all_uavs_inactive"
                    ][done_indices].float()
                    end_horizon_values = step_metrics[
                        "end_horizon"
                    ][done_indices].float()
                    end_reason_values = step_metrics[
                        "end_reason_code"
                    ][done_indices].float()
                    energy_values = episode_energy_j[
                        done_indices
                    ]
                    coverage_values = (
                        env.coverage_seen.sum(-1)
                        .float()
                        / float(
                            env.grid_n
                            * env.grid_n
                        )
                        * 100.0
                    )[done_indices]
                    information_gain_values = (
                        step_metrics[
                            "information_gain_bits"
                        ][done_indices]
                        .float()
                    )
                    target_encounter_values = (
                        env.target_seen.sum(-1)
                        .float()
                        / float(
                            env.num_targets
                        )
                        * 100.0
                    )[done_indices]
                    battery_remaining_values = (
                        env.battery.mean(-1)
                        / float(
                            CONFIG[
                                "battery_j"
                            ]
                        )
                        * 100.0
                    )[done_indices]
                    depleted_values = (
                        (
                            env.battery
                            <= 1e-12
                        )
                        .sum(-1)
                        .float()
                    )[done_indices]
                    diagnostic_cpu = {
                        key: episode_diagnostics[
                            key
                        ][done_indices]
                        .detach()
                        .cpu()
                        .numpy()
                        for key in episode_diagnostic_keys
                    }
                    latency_max_cpu = (
                        episode_delivery_latency_max_s[
                            done_indices
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    first_confirm_cpu = (
                        episode_first_confirm_step[
                            done_indices
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    all_confirm_cpu = (
                        episode_all_confirm_step[
                            done_indices
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    first_delivery_cpu = (
                        episode_first_delivery_step[
                            done_indices
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    coverage_cpu = (
                        coverage_values
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    information_gain_cpu = (
                        information_gain_values
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    target_encounter_cpu = (
                        target_encounter_values
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    battery_remaining_cpu = (
                        battery_remaining_values
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    depleted_cpu = (
                        depleted_values
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    returns_cpu = episode_returns[
                        done_indices
                    ].detach().cpu().numpy()
                    lengths_cpu = episode_lengths[
                        done_indices
                    ].detach().cpu().numpy()
                    confirmed_cpu = (
                        confirmed.detach().cpu().numpy()
                    )
                    delivered_cpu = (
                        delivered.detach().cpu().numpy()
                    )
                    success_cpu = (
                        success_values.detach().cpu().numpy()
                    )
                    end_success_cpu = (
                        end_success_values
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    end_inactive_cpu = (
                        end_inactive_values
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    end_horizon_cpu = (
                        end_horizon_values
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    end_reason_cpu = (
                        end_reason_values
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    energy_cpu = (
                        energy_values
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    component_cpu = {
                        key: episode_components[
                            key
                        ][done_indices]
                        .detach()
                        .cpu()
                        .numpy()
                        for key in episode_component_keys
                    }
                    for item_index in range(
                        len(returns_cpu)
                    ):
                        record = {
                            "return": float(
                                returns_cpu[
                                    item_index
                                ]
                            ),
                            "length": float(
                                lengths_cpu[
                                    item_index
                                ]
                            ),
                            "target_search_rate_percent": (
                                100.0
                                * float(
                                    confirmed_cpu[
                                        item_index
                                    ]
                                )
                                / max(
                                    1,
                                    int(
                                        CONFIG[
                                            "num_targets"
                                        ]
                                    ),
                                )
                            ),
                            "target_delivery_rate_percent": (
                                100.0
                                * float(
                                    delivered_cpu[
                                        item_index
                                    ]
                                )
                                / max(
                                    1,
                                    int(
                                        CONFIG[
                                            "num_targets"
                                        ]
                                    ),
                                )
                            ),
                            "success": float(
                                success_cpu[
                                    item_index
                                ]
                            ),
                            "confirmed_targets": float(
                                confirmed_cpu[
                                    item_index
                                ]
                            ),
                            "delivered_targets": float(
                                delivered_cpu[
                                    item_index
                                ]
                            ),
                            "episode_energy_j": float(
                                energy_cpu[
                                    item_index
                                ]
                            ),
                            "coverage_percent": float(
                                coverage_cpu[
                                    item_index
                                ]
                            ),
                            "target_encounter_rate_percent": float(
                                target_encounter_cpu[
                                    item_index
                                ]
                            ),
                            "information_gain_bits": float(
                                information_gain_cpu[
                                    item_index
                                ]
                            ),
                            "battery_remaining_mean_percent": float(
                                battery_remaining_cpu[
                                    item_index
                                ]
                            ),
                            "depleted_uav_count": float(
                                depleted_cpu[
                                    item_index
                                ]
                            ),
                            "end_step": float(
                                lengths_cpu[
                                    item_index
                                ]
                            ),
                            "end_reason_code": float(
                                end_reason_cpu[
                                    item_index
                                ]
                            ),
                            "end_by_success": float(
                                end_success_cpu[
                                    item_index
                                ]
                            ),
                            "end_by_all_uavs_inactive": float(
                                end_inactive_cpu[
                                    item_index
                                ]
                            ),
                            "end_by_horizon": float(
                                end_horizon_cpu[
                                    item_index
                                ]
                            ),
                        }
                        for key in (
                            episode_component_keys
                        ):
                            record[key] = float(
                                component_cpu[
                                    key
                                ][item_index]
                            )

                        length_steps = max(
                            1.0,
                            float(
                                lengths_cpu[
                                    item_index
                                ]
                            ),
                        )
                        possible_agent_steps = (
                            length_steps
                            * float(
                                env.num_uavs
                            )
                        )
                        diag = {
                            key: float(
                                diagnostic_cpu[
                                    key
                                ][item_index]
                            )
                            for key in episode_diagnostic_keys
                        }
                        record.update(
                            {
                                "blocked_motion_rate": (
                                    diag[
                                        "blocked"
                                    ]
                                    / possible_agent_steps
                                ),
                                "peer_safety_block_rate": (
                                    diag[
                                        "blocked_by_peer"
                                    ]
                                    / possible_agent_steps
                                ),
                                "obstacle_block_rate": (
                                    diag[
                                        "blocked_by_obstacle"
                                    ]
                                    / possible_agent_steps
                                ),
                                "boundary_clip_rate": (
                                    diag[
                                        "boundary"
                                    ]
                                    / possible_agent_steps
                                ),
                                "distance_total_m": diag[
                                    "distance_m"
                                ],
                                "false_confirmations": diag[
                                    "false_confirmations"
                                ],
                                "reports_created": diag[
                                    "reports_created"
                                ],
                                "expired_reports": diag[
                                    "expired_reports"
                                ],
                                "dropped_reports": diag[
                                    "dropped_reports"
                                ],
                                "communication_energy_j": diag[
                                    "comm_energy_j"
                                ],
                                "network_tx_attempts": diag[
                                    "network_tx_attempts"
                                ],
                                "network_phy_success_percent": (
                                    100.0
                                    * diag[
                                        "network_link_successes"
                                    ]
                                    / max(
                                        1.0,
                                        diag[
                                            "network_tx_attempts"
                                        ],
                                    )
                                ),
                                "network_tx_payload_success_ratio": (
                                    diag[
                                        "network_bytes_tx"
                                    ]
                                    / max(
                                        1.0,
                                        diag[
                                            "network_bytes_attempted"
                                        ],
                                    )
                                ),
                                "network_throughput_kbps": (
                                    diag[
                                        "network_bytes_tx"
                                    ]
                                    * 8.0
                                    / max(
                                        1e-12,
                                        length_steps
                                        * float(
                                            CONFIG[
                                                "dt"
                                            ]
                                        ),
                                    )
                                    / 1000.0
                                ),
                                "network_nlos_attempt_rate_percent": (
                                    100.0
                                    * diag[
                                        "network_nlos_attempts"
                                    ]
                                    / max(
                                        1.0,
                                        diag[
                                            "network_tx_attempts"
                                        ],
                                    )
                                ),
                                "gcs_in_range_uav_fraction": (
                                    diag[
                                        "gcs_in_range_uav_fraction"
                                    ]
                                    / length_steps
                                ),
                                "report_delivery_latency_sample_count": (
                                    diag[
                                        "delivery_latency_count"
                                    ]
                                ),
                                "report_delivery_latency_max_s": float(
                                    latency_max_cpu[
                                        item_index
                                    ]
                                ),
                            }
                        )
                        latency_count = diag[
                            "delivery_latency_count"
                        ]
                        record[
                            "report_delivery_latency_s"
                        ] = (
                            diag[
                                "delivery_latency_sum_s"
                            ]
                            / latency_count
                            if latency_count > 0.0
                            else None
                        )
                        created_reports = diag[
                            "reports_created"
                        ]
                        record[
                            "report_expiry_rate_percent"
                        ] = (
                            100.0
                            * diag[
                                "expired_reports"
                            ]
                            / created_reports
                            if created_reports > 0.0
                            else 0.0
                        )
                        record[
                            "report_drop_rate_percent"
                        ] = (
                            100.0
                            * diag[
                                "dropped_reports"
                            ]
                            / created_reports
                            if created_reports > 0.0
                            else 0.0
                        )
                        confirmed_value = float(
                            confirmed_cpu[
                                item_index
                            ]
                        )
                        delivered_value = float(
                            delivered_cpu[
                                item_index
                            ]
                        )
                        record[
                            "report_delivery_given_confirmation_rate_percent"
                        ] = (
                            100.0
                            * delivered_value
                            / confirmed_value
                            if confirmed_value > 0.0
                            else 0.0
                        )
                        record[
                            "energy_per_confirmed_target_j"
                        ] = (
                            float(
                                energy_cpu[
                                    item_index
                                ]
                            )
                            / confirmed_value
                            if confirmed_value > 0.0
                            else None
                        )
                        record[
                            "energy_per_delivered_target_j"
                        ] = (
                            float(
                                energy_cpu[
                                    item_index
                                ]
                            )
                            / delivered_value
                            if delivered_value > 0.0
                            else None
                        )

                        first_confirm_step = int(
                            first_confirm_cpu[
                                item_index
                            ]
                        )
                        all_confirm_step = int(
                            all_confirm_cpu[
                                item_index
                            ]
                        )
                        first_delivery_step = int(
                            first_delivery_cpu[
                                item_index
                            ]
                        )
                        dt_seconds = float(
                            CONFIG[
                                "dt"
                            ]
                        )
                        record[
                            "first_confirmation_observed"
                        ] = float(
                            first_confirm_step >= 0
                        )
                        record[
                            "time_to_first_confirm_s"
                        ] = (
                            first_confirm_step
                            * dt_seconds
                            if first_confirm_step >= 0
                            else None
                        )
                        record[
                            "all_targets_confirmed"
                        ] = float(
                            all_confirm_step >= 0
                        )
                        record[
                            "time_to_all_confirm_s"
                        ] = (
                            all_confirm_step
                            * dt_seconds
                            if all_confirm_step >= 0
                            else None
                        )
                        record[
                            "first_delivery_observed"
                        ] = float(
                            first_delivery_step >= 0
                        )
                        record[
                            "time_to_first_delivery_s"
                        ] = (
                            first_delivery_step
                            * dt_seconds
                            if first_delivery_step >= 0
                            else None
                        )
                        local_records.append(
                            record
                        )

                gathered_records = (
                    [None] * world_size
                    if rank == 0
                    else None
                )
                dist.gather_object(
                    local_records,
                    object_gather_list=(
                        gathered_records
                    ),
                    dst=0,
                )
                completed_episodes += (
                    accepted_done_count
                )

                if (
                    rank == 0
                    and gathered_records
                    is not None
                ):
                    accepted_records = []
                    for records in (
                        gathered_records
                    ):
                        accepted_records.extend(
                            records
                        )
                    if (
                        target_episodes
                        is not None
                    ):
                        accepted_records = (
                            accepted_records[
                                :accepted_done_count
                            ]
                        )
                    episode_bin_buffer.extend(
                        accepted_records
                    )
                    paper_bin_buffer.extend(
                        accepted_records
                    )
                    while (
                        len(paper_bin_buffer)
                        >= paper_curve_bin
                    ):
                        paper_chunk = paper_bin_buffer[
                            :paper_curve_bin
                        ]
                        del paper_bin_buffer[
                            :paper_curve_bin
                        ]
                        paper_episodes_logged += len(
                            paper_chunk
                        )
                        paper_payload = (
                            build_paper_curve_payload(
                                paper_chunk,
                                paper_episodes_logged,
                                paper_ema_state,
                                paper_ema_beta,
                            )
                        )
                        print(
                            "WANDB_BRIDGE_JSON "
                            + importlib.import_module(
                                "json"
                            ).dumps(
                                paper_payload,
                                separators=(",", ":"),
                            ),
                            flush=True,
                        )
                        if wandb_run is not None:
                            wandb_run.log(
                                paper_payload
                            )
                    while (
                        len(
                            episode_bin_buffer
                        )
                        >= reward_curve_bin
                    ):
                        chunk = (
                            episode_bin_buffer[
                                :reward_curve_bin
                            ]
                        )
                        del episode_bin_buffer[
                            :reward_curve_bin
                        ]
                        episodes_logged += len(
                            chunk
                        )
                        payload = build_episode_curve_payload(
                            chunk,
                            episodes_logged,
                            episode_component_keys,
                        )
                        print(
                            "WANDB_BRIDGE_JSON "
                            + importlib.import_module(
                                "json"
                            ).dumps(
                                payload,
                                separators=(",", ":"),
                            ),
                            flush=True,
                        )
                        if wandb_run is not None:
                            wandb_run.log(
                                payload
                            )
                            wandb_run.summary[
                                "latest_episode_return"
                            ] = float(
                                payload[
                                    _episode_metric_key(
                                        "return", len(chunk)
                                    )
                                ]
                            )
                            wandb_run.summary[
                                "latest_episode_target_search_rate_percent"
                            ] = float(
                                payload[
                                    _episode_metric_key(
                                        "target_search_rate_percent",
                                        len(chunk),
                                    )
                                ]
                            )
                            wandb_run.summary[
                                "latest_episode_target_delivery_rate_percent"
                            ] = float(
                                payload[
                                    _episode_metric_key(
                                        "target_delivery_rate_percent",
                                        len(chunk),
                                    )
                                ]
                            )
                            wandb_run.summary[
                                "latest_episode_success_rate"
                            ] = float(
                                payload[
                                    _episode_metric_key(
                                        "success", len(chunk)
                                    )
                                ]
                            )
                            wandb_run.summary[
                                "episodes_completed"
                            ] = int(
                                episodes_logged
                            )

                done_float = done.float()
                episode_returns *= (
                    1.0 - done_float
                )
                episode_lengths *= (
                    ~done
                ).long()
                episode_energy_j *= (
                    1.0
                    - done_float
                )
                for component_key in (
                    episode_component_keys
                ):
                    episode_components[
                        component_key
                    ] *= (
                        1.0
                        - done_float
                    )
                for diagnostic_key in (
                    episode_diagnostic_keys
                ):
                    episode_diagnostics[
                        diagnostic_key
                    ] *= (
                        1.0
                        - done_float
                    )
                episode_delivery_latency_max_s *= (
                    1.0
                    - done_float
                )
                reset_step_value = torch.full_like(
                    episode_first_confirm_step,
                    -1,
                )
                episode_first_confirm_step = torch.where(
                    done,
                    reset_step_value,
                    episode_first_confirm_step,
                )
                episode_all_confirm_step = torch.where(
                    done,
                    reset_step_value,
                    episode_all_confirm_step,
                )
                episode_first_delivery_step = torch.where(
                    done,
                    reset_step_value,
                    episode_first_delivery_step,
                )

            target_reached = (
                target_episodes is not None
                and completed_episodes
                >= target_episodes
            )
            transition_limit_reached = (
                global_step
                >= max_total_transitions
            )

            if bool(done.any().item()):
                if (
                    not target_reached
                    and not transition_limit_reached
                ):
                    obs, state, mask = env.reset(
                        mask=done
                    )
                else:
                    obs, state, mask = (
                        next_obs,
                        next_state,
                        next_mask,
                    )
            else:
                obs, state, mask = (
                    next_obs,
                    next_state,
                    next_mask,
                )

            should_log_training = (
                rank == 0
                and (
                    global_step
                    - last_log_step
                    >= log_interval
                    or target_reached
                    or transition_limit_reached
                )
            )
            log_flag = torch.tensor(
                [
                    1
                    if should_log_training
                    else 0
                ],
                device=device,
                dtype=torch.int32,
            )
            dist.broadcast(
                log_flag,
                src=0,
            )
            if int(log_flag.item()) == 1:
                local_reward_sum = reward.sum()
                local_reward_sq_sum = (
                    reward
                    * reward
                ).sum()
                local_reward_count = torch.tensor(
                    [
                        reward.numel()
                    ],
                    device=device,
                    dtype=torch.float32,
                )
                local_reward_min = reward.min()
                local_reward_max = reward.max()
                stats = torch.stack(
                    (
                        local_reward_sum,
                        local_reward_sq_sum,
                        local_reward_count[
                            0
                        ],
                    )
                )
                dist.all_reduce(
                    stats,
                    op=dist.ReduceOp.SUM,
                )
                dist.all_reduce(
                    local_reward_min,
                    op=dist.ReduceOp.MIN,
                )
                dist.all_reduce(
                    local_reward_max,
                    op=dist.ReduceOp.MAX,
                )

                # Action-distribution diagnostics make policy collapse and
                # boundary-driving behavior visible without changing training.
                action_count = torch.tensor(
                    [float(max(1, continuous[..., 0].numel()))],
                    device=device,
                    dtype=torch.float32,
                )
                motion = continuous[..., :3].float()
                tx_action = continuous[..., 3].float()
                action_stats = torch.stack(
                    (
                        motion[..., 0].sum(),
                        motion[..., 1].sum(),
                        motion[..., 2].sum(),
                        motion[..., 0].square().sum(),
                        motion[..., 1].square().sum(),
                        motion[..., 2].square().sum(),
                        tx_action.sum(),
                        tx_action.square().sum(),
                        (motion.abs() >= 0.95).float().sum(),
                        (motion[..., 2].abs() >= 0.95).float().sum(),
                        (destination == 0).float().sum(),
                        (
                            (destination > 0)
                            & (destination < env.discrete_dim - 1)
                        ).float().sum(),
                        (
                            destination == env.discrete_dim - 1
                        ).float().sum(),
                        action_count[0],
                    )
                )
                dist.all_reduce(
                    action_stats,
                    op=dist.ReduceOp.SUM,
                )

                component_means = {}
                for component_key in (
                    episode_component_keys
                ):
                    value = step_metrics[
                        component_key
                    ].sum()
                    dist.all_reduce(
                        value,
                        op=dist.ReduceOp.SUM,
                    )
                    component_means[
                        component_key
                    ] = float(
                        value.item()
                        / max(
                            1.0,
                            float(
                                stats[
                                    2
                                ].item()
                            ),
                        )
                    )

                if rank == 0:
                    elapsed = max(
                        1e-12,
                        time.perf_counter()
                        - start,
                    )
                    count = max(
                        1.0,
                        float(
                            stats[
                                2
                            ].item()
                        ),
                    )
                    reward_mean = float(
                        stats[
                            0
                        ].item()
                        / count
                    )
                    reward_variance = max(
                        0.0,
                        float(
                            stats[
                                1
                            ].item()
                            / count
                            - reward_mean
                            * reward_mean
                        ),
                    )
                    action_n = max(
                        1.0,
                        float(action_stats[13].item()),
                    )
                    ax_mean = float(action_stats[0].item() / action_n)
                    ay_mean = float(action_stats[1].item() / action_n)
                    az_mean = float(action_stats[2].item() / action_n)
                    ax_std = math.sqrt(max(
                        0.0,
                        float(action_stats[3].item() / action_n)
                        - ax_mean * ax_mean,
                    ))
                    ay_std = math.sqrt(max(
                        0.0,
                        float(action_stats[4].item() / action_n)
                        - ay_mean * ay_mean,
                    ))
                    az_std = math.sqrt(max(
                        0.0,
                        float(action_stats[5].item() / action_n)
                        - az_mean * az_mean,
                    ))
                    tx_mean = float(action_stats[6].item() / action_n)
                    tx_std = math.sqrt(max(
                        0.0,
                        float(action_stats[7].item() / action_n)
                        - tx_mean * tx_mean,
                    ))
                    payload = {
                        "global_step": int(
                            global_step
                        ),
                        "episodes_completed": int(
                            completed_episodes
                        ),
                        "train/transitions_per_second": (
                            global_step
                            / elapsed
                        ),
                        "train/updates_per_second": (
                            update_count
                            / elapsed
                        ),
                        "train/optimizer_updates": int(update_count),
                        "train/vector_steps": int(vector_step_count),
                        "train/optimizer_updates_per_1000_transitions": (
                            1000.0 * float(update_count)
                            / max(
                                1.0,
                                float(global_step - learning_starts),
                            )
                        ),
                        "train/replay_samples_per_new_transition_effective": (
                            float(global_batch_size)
                            * float(update_count)
                            / max(
                                1.0,
                                float(global_step - learning_starts),
                            )
                        ),
                        "train/reward_mean": (
                            reward_mean
                        ),
                        "train/reward_std": float(
                            math.sqrt(
                                reward_variance
                            )
                        ),
                        "train/reward_min": float(
                            local_reward_min.item()
                        ),
                        "train/reward_max": float(
                            local_reward_max.item()
                        ),
                        "train/action_accel_x_mean": ax_mean,
                        "train/action_accel_y_mean": ay_mean,
                        "train/action_accel_z_mean": az_mean,
                        "train/action_accel_x_std": float(ax_std),
                        "train/action_accel_y_std": float(ay_std),
                        "train/action_accel_z_std": float(az_std),
                        "train/action_tx_power_mean": tx_mean,
                        "train/action_tx_power_std": float(tx_std),
                        "train/action_motion_saturation_rate": float(
                            action_stats[8].item() / (3.0 * action_n)
                        ),
                        "train/action_altitude_saturation_rate": float(
                            action_stats[9].item() / action_n
                        ),
                        "train/action_destination_silent_rate": float(
                            action_stats[10].item() / action_n
                        ),
                        "train/action_destination_peer_rate": float(
                            action_stats[11].item() / action_n
                        ),
                        "train/action_destination_gcs_rate": float(
                            action_stats[12].item() / action_n
                        ),
                    }
                    for (
                        component_key,
                        value,
                    ) in component_means.items():
                        payload[
                            f"train/{component_key}_mean"
                        ] = value
                    if latest_metrics:
                        for (
                            key,
                            value,
                        ) in latest_metrics.items():
                            if isinstance(
                                value,
                                (
                                    bool,
                                    int,
                                    float,
                                ),
                            ):
                                payload[
                                    f"train/{key}"
                                ] = float(
                                    value
                                )
                    print(
                        "WANDB_BRIDGE_JSON "
                        + importlib.import_module(
                            "json"
                        ).dumps(
                            payload,
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            payload
                        )
                    last_log_step = (
                        global_step
                    )

            # Periodic DDP model/optimizer checkpoint. This is intentionally
            # model-resume state rather than a bit-exact replay/env snapshot.
            should_checkpoint = (
                target_episodes is not None
                and completed_episodes > 0
                and (
                    completed_episodes
                    - last_checkpoint_episode
                    >= checkpoint_interval_episodes
                    or target_reached
                )
            )
            if should_checkpoint:
                dist.barrier(
                    device_ids=[
                        local_rank
                    ]
                )
                if rank == 0:
                    checkpoint_path = (
                        Path(repo)
                        / "checkpoints_full_gpu"
                        / (
                            f"{algorithm}_episode_"
                            f"{completed_episodes:08d}.pt"
                        )
                    )
                    saved_path = (
                        save_full_gpu_model_checkpoint(
                            checkpoint_path,
                            trainer,
                            algorithm,
                            global_step,
                            completed_episodes,
                            latest_evaluation=(
                                latest_evaluation
                            ),
                            checkpoint_kind="ddp_model_optimizer",
                        )
                    )
                    checkpoint_paths.append(
                        str(saved_path)
                    )
                    if (
                        wandb_run is not None
                        and bool(
                            CONFIG.get(
                                "training_upload_periodic_checkpoints_wandb",
                                False,
                            )
                        )
                    ):
                        wandb_run.save(
                            str(saved_path),
                            base_path=str(Path(repo)),
                            policy="now",
                        )
                dist.barrier(
                    device_ids=[
                        local_rank
                    ]
                )
                last_checkpoint_episode = (
                    completed_episodes
                )

            if (
                target_episodes is not None
            ):
                should_evaluate = (
                    enable_evaluation
                    and completed_episodes > 0
                    and (
                        completed_episodes
                        - last_eval_episode
                        >= eval_interval_episodes
                        or target_reached
                    )
                )
            else:
                should_evaluate = (
                    enable_evaluation
                    and (
                        global_step
                        - last_eval_step
                        >= eval_interval
                        or transition_limit_reached
                    )
                )
            if should_evaluate:
                dist.barrier(
                    device_ids=[local_rank]
                )
                if rank == 0:
                    evaluation = (
                        evaluate_full_gpu_reference(
                            trainer=trainer,
                            algorithm=algorithm,
                            network_backend=(
                                network_backend
                            ),
                            seed=seed,
                            episodes=eval_episodes,
                            uavnetsim_repo_path=None,
                        )
                    )
                    latest_evaluation = (
                        evaluation["summary"]
                    )
                    evaluation_backend = (
                        evaluation["backend"]
                    )
                    evaluation_payload = {
                        "global_step": int(
                            global_step
                        ),
                        "episodes_completed": int(
                            completed_episodes
                        ),
                        **{
                            f"evaluation/{key}": float(
                                value
                            )
                            for key, value
                            in latest_evaluation.items()
                            if isinstance(
                                value,
                                (
                                    bool,
                                    int,
                                    float,
                                ),
                            )
                        },
                    }
                    print(
                        "WANDB_BRIDGE_JSON "
                        + importlib.import_module(
                            "json"
                        ).dumps(
                            evaluation_payload,
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            evaluation_payload
                        )
                        for key, value in (
                            latest_evaluation.items()
                        ):
                            if isinstance(
                                value,
                                (
                                    bool,
                                    int,
                                    float,
                                ),
                            ):
                                wandb_run.summary[
                                    f"evaluation_latest/{key}"
                                ] = float(
                                    value
                                )

                        for source_key, summary_key in (
                            (
                                "return",
                                "final_return",
                            ),
                            (
                                "coverage_percent",
                                "final_coverage_percent",
                            ),
                            (
                                "target_search_rate_percent",
                                "final_target_search_rate_percent",
                            ),
                            (
                                "target_delivery_rate_percent",
                                "final_target_delivery_rate_percent",
                            ),
                            (
                                "success_rate",
                                "final_success_rate",
                            ),
                        ):
                            if source_key in latest_evaluation:
                                wandb_run.summary[
                                    summary_key
                                ] = float(
                                    latest_evaluation[
                                        source_key
                                    ]
                                )

                        should_visualize = enable_visualization and (
                            completed_episodes
                            - last_visualization_episode
                            >= visualization_interval_episodes
                            or target_reached
                            or target_episodes is None
                        )
                        if (
                            should_visualize
                            and evaluation[
                                "episodes"
                            ]
                        ):
                            trace = evaluation[
                                "episodes"
                            ][0].get(
                                "visualization_trace"
                            )
                            if trace is not None:
                                fig = render_evaluation_trace_2d(
                                    trace,
                                    title=(
                                        f"{algorithm.upper()} evaluation "
                                        f"| episodes={completed_episodes:,} "
                                        f"| step={global_step:,}"
                                    ),
                                )
                                visualization_dir = (
                                    Path(repo)
                                    / "visualizations"
                                )
                                visualization_dir.mkdir(
                                    parents=True,
                                    exist_ok=True,
                                )
                                visualization_path = (
                                    visualization_dir
                                    / (
                                        f"{algorithm}_evaluation_"
                                        f"episode_{completed_episodes:08d}_"
                                        f"step_{global_step:012d}.png"
                                    )
                                )
                                fig.savefig(
                                    visualization_path,
                                    dpi=140,
                                    bbox_inches="tight",
                                )
                                wandb_run.log(
                                    {
                                        "episodes_completed": int(
                                            completed_episodes
                                        ),
                                        "visualization/evaluation_2d": (
                                            importlib.import_module(
                                                "wandb"
                                            ).Image(
                                                str(
                                                    visualization_path
                                                )
                                            )
                                        ),
                                    }
                                )
                                wandb_run.save(
                                    str(
                                        visualization_path
                                    ),
                                    base_path=str(
                                        Path(repo)
                                    ),
                                    policy="now",
                                )

                                if (
                                    target_reached
                                    and target_episodes
                                    is not None
                                ):
                                    video_dir = (
                                        Path(repo)
                                        / "videos"
                                    )
                                    video_path = (
                                        video_dir
                                        / (
                                            f"{algorithm}_post_train_"
                                            f"evaluation_episode_"
                                            f"{completed_episodes:08d}.mp4"
                                        )
                                    )
                                    render_evaluation_video_2d(
                                        trace,
                                        video_path,
                                        fps=8,
                                    )
                                    wandb_run.log(
                                        {
                                            "episodes_completed": int(
                                                completed_episodes
                                            ),
                                            "visualization/post_train_eval_video": (
                                                importlib.import_module(
                                                    "wandb"
                                                ).Video(
                                                    str(
                                                        video_path
                                                    ),
                                                    format="mp4",
                                                )
                                            ),
                                        }
                                    )
                                    wandb_run.save(
                                        str(
                                            video_path
                                        ),
                                        base_path=str(
                                            Path(repo)
                                        ),
                                        policy="now",
                                    )

                                import matplotlib.pyplot as plt
                                plt.close(
                                    fig
                                )
                                last_visualization_episode = (
                                    completed_episodes
                                )
                dist.barrier(
                    device_ids=[local_rank]
                )
                last_eval_step = (
                    global_step
                )
                last_eval_episode = (
                    completed_episodes
                )

        # Flush the final incomplete paper window without modifying raw data.
        if (
            rank == 0
            and paper_bin_buffer
        ):
            paper_chunk = list(
                paper_bin_buffer
            )
            paper_episodes_logged += len(
                paper_chunk
            )
            paper_payload = build_paper_curve_payload(
                paper_chunk,
                paper_episodes_logged,
                paper_ema_state,
                paper_ema_beta,
            )
            print(
                "WANDB_BRIDGE_JSON "
                + importlib.import_module(
                    "json"
                ).dumps(
                    paper_payload,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    paper_payload
                )

        # Flush the tail of the raw per-episode stream.
        if (
            rank == 0
            and episode_bin_buffer
        ):
            chunk = list(
                episode_bin_buffer
            )
            episodes_logged += len(
                chunk
            )
            payload = build_episode_curve_payload(
                chunk,
                episodes_logged,
                episode_component_keys,
            )
            if wandb_run is not None:
                wandb_run.log(
                    payload
                )

        torch.cuda.synchronize(device)
        wall = time.perf_counter() - start
        dist.barrier(device_ids=[rank])

        rank_result = {
            "rank": rank,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "local_envs": local_envs,
            "local_replay_size": int(len(replay)),
            "peak_memory_mb": (
                torch.cuda.max_memory_allocated(device)
                / 2**20
            ),
            "completed_episodes": (
                completed_episodes_local
            ),
        }
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(
            rank_result,
            object_gather_list=gathered,
            dst=0,
        )

        if rank == 0:
            final_checkpoint_path = (
                Path(checkpoint_paths[-1])
                if checkpoint_paths
                else None
            )
            if final_checkpoint_path is None:
                final_checkpoint_path = (
                    Path(repo)
                    / "checkpoints_full_gpu"
                    / (
                        f"{algorithm}_episode_"
                        f"{completed_episodes:08d}.pt"
                    )
                )
                final_checkpoint_path = (
                    save_full_gpu_model_checkpoint(
                        final_checkpoint_path,
                        trainer,
                        algorithm,
                        global_step,
                        completed_episodes,
                        latest_evaluation=latest_evaluation,
                        checkpoint_kind="ddp_model_optimizer",
                    )
                )
                checkpoint_paths.append(
                    str(final_checkpoint_path)
                )

            final_checkpoint_artifact = None
            if (
                wandb_run is not None
                and bool(
                    CONFIG.get(
                        "training_publish_final_checkpoint_wandb_artifact",
                        True,
                    )
                )
            ):
                wandb_module = importlib.import_module(
                    "wandb"
                )
                artifact = wandb_module.Artifact(
                    name=(
                        f"{algorithm}-seed{seed}-{wandb_run.id}-checkpoint"
                    ),
                    type="model",
                    description=(
                        "Final train-only checkpoint for CPU "
                        "evaluation/visualization."
                    ),
                    metadata={
                        "algorithm": algorithm,
                        "seed": int(seed),
                        "episodes_completed": int(completed_episodes),
                        "global_step": int(global_step),
                        "network_backend": str(network_backend),
                        "source_run_id": str(wandb_run.id),
                        "checkpoint_kind": "ddp_model_optimizer",
                    },
                )
                artifact.add_file(
                    str(final_checkpoint_path),
                    name="checkpoint.pt",
                )
                wandb_run.log_artifact(
                    artifact,
                    aliases=[
                        str(
                            CONFIG.get(
                                "training_final_checkpoint_artifact_alias",
                                "final",
                            )
                        ),
                        "latest",
                        f"episode-{completed_episodes}",
                    ],
                )
                artifact_ref = (
                    f"{artifact.name}:"
                    f"{CONFIG.get('training_final_checkpoint_artifact_alias', 'final')}"
                )
                wandb_run.summary[
                    "final/checkpoint_artifact"
                ] = artifact_ref
                final_checkpoint_artifact = artifact_ref

            result = {
                "kind": "full_gpu_ddp",
                "multi_gpu_strategy": (
                    "ddp" if world_size > 1 else "single_gpu_distributed"
                ),
                "algorithm": algorithm,
                "network_backend": network_backend,
                "used_gpu_count": world_size,
                "num_envs": int(num_envs),
                "envs_per_device": {
                    item["device"]: item["local_envs"]
                    for item in gathered
                },
                "devices": [
                    item["device"] for item in gathered
                ],
                "gpu_names": {
                    item["device"]: item["gpu_name"]
                    for item in gathered
                },
                "peak_memory_mb_per_device": {
                    item["device"]: item[
                        "peak_memory_mb"
                    ]
                    for item in gathered
                },
                "transitions": int(global_step),
                "updates": int(update_count),
                "train_freq": int(train_freq),
                "configured_gradient_steps": int(configured_gradient_steps),
                "effective_gradient_steps": int(gradient_steps),
                "gradient_steps": int(gradient_steps),
                "replay_samples_per_new_transition": (
                    float(global_batch_size)
                    if int(gradient_steps) == -1
                    else float(gradient_steps) * float(global_batch_size)
                    / max(1.0, float(num_envs * train_freq))
                ),
                "target_episodes": (
                    int(target_episodes)
                    if target_episodes
                    is not None
                    else None
                ),
                "episodes_completed": int(
                    completed_episodes
                ),
                "episodes_generated": int(
                    sum(
                        item[
                            "completed_episodes"
                        ]
                        for item in gathered
                    )
                ),
                "checkpoint_paths": list(
                    checkpoint_paths
                ),
                "final_checkpoint_path": str(final_checkpoint_path),
                "inline_evaluation_enabled": bool(
                    enable_evaluation
                ),
                "inline_visualization_enabled": bool(
                    enable_visualization
                ),
                "wall_seconds": float(wall),
                "transitions_per_second": (
                    global_step / wall
                ),
                "updates_per_second": (
                    update_count / wall
                ),
                "latest_update_metrics": (
                    latest_metrics
                ),
                "latest_evaluation": (
                    latest_evaluation
                ),
                "evaluation_backend": (
                    evaluation_backend
                ),
                "model_fingerprint_sha256": (
                    _trainer_parameter_fingerprint(
                        trainer
                    )
                ),
                "wandb_run_id": (
                    wandb_run.id
                    if wandb_run is not None
                    else None
                ),
                "wandb_run_name": (
                    wandb_run.name
                    if wandb_run is not None
                    else None
                ),
                "wandb_run_url": (
                    wandb_run.url
                    if wandb_run is not None
                    else None
                ),
            }
            if final_checkpoint_artifact is not None:
                result["final_checkpoint_artifact"] = (
                    final_checkpoint_artifact
                )
            Path(result_path).write_text(
                json.dumps(
                    result,
                    indent=2,
                    default=str,
                )
            )
            if wandb_run is not None:
                wandb_run.summary[
                    "final/transitions_per_second"
                ] = result[
                    "transitions_per_second"
                ]
                wandb_run.summary[
                    "final/updates_per_second"
                ] = result[
                    "updates_per_second"
                ]
                wandb_run.summary[
                    "final/gpu_count"
                ] = world_size
                wandb_run.finish()
        dist.barrier(device_ids=[rank])
    finally:
        env.close()
        if dist.is_initialized():
            dist.destroy_process_group()


# --- frozen notebook cell 274 ---
def _write_ddp_launcher_script(path):
    script = r"""import argparse
import pickle
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--repo", required=True)
parser.add_argument("--config-path", required=True)
parser.add_argument("--algorithm", required=True)
parser.add_argument("--num-envs", type=int, required=True)
parser.add_argument("--total-transitions", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--network-backend", required=True)
parser.add_argument("--result-path", required=True)
parser.add_argument("--enable-wandb", type=int, default=1)
parser.add_argument("--target-episodes", type=int, default=0)
args = parser.parse_args()

sys.path.insert(0, args.repo)

# Restore the parent's fully resolved experiment config before importing any
# production module that may derive module-level constants from CONFIG.
from config import CONFIG
with open(args.config_path, "rb") as handle:
    config_snapshot = pickle.load(handle)
if not isinstance(config_snapshot, dict):
    raise TypeError("DDP config snapshot must be a dictionary")
CONFIG.clear()
CONFIG.update(config_snapshot)

from uav_marl.training.gpu import _train_full_gpu_ddp_worker

_train_full_gpu_ddp_worker(
    repo=args.repo,
    algorithm=args.algorithm,
    num_envs=args.num_envs,
    total_transitions=args.total_transitions,
    seed=args.seed,
    network_backend=args.network_backend,
    result_path=args.result_path,
    enable_wandb=bool(args.enable_wandb),
    target_episodes=(
        int(args.target_episodes)
        if int(args.target_episodes) > 0
        else None
    ),
)
"""
    Path(path).write_text(script)
    return Path(path)


# --- frozen notebook cell 275 ---
def train_full_gpu_ddp(
    repo: Path,
    algorithm: str,
    num_envs: int,
    total_transitions: int,
    seed: int,
    network_backend: str = "uavnetsim_gpu",
    enable_wandb=None,
    target_episodes=None,
    world_size=None,
    selected_devices=None,
):
    visible_count = int(torch.cuda.device_count())
    if visible_count < 1:
        raise RuntimeError(
            "distributed GPU training requires at least one visible CUDA GPU"
        )
    if world_size is None:
        world_size = min(2, visible_count)
    world_size = int(world_size)
    if world_size not in {1, 2}:
        raise ValueError("world_size must be 1 or 2")
    if world_size > visible_count:
        raise RuntimeError(
            f"world_size={world_size} exceeds visible CUDA device count "
            f"{visible_count}"
        )
    if selected_devices is None:
        selected_indices = list(range(world_size))
    else:
        selected_indices = [
            int(torch.device(device).index)
            for device in selected_devices
        ]
    if len(selected_indices) != world_size:
        raise ValueError(
            "selected_devices length must equal world_size"
        )
    if len(set(selected_indices)) != len(selected_indices):
        raise ValueError("selected CUDA devices must be distinct")
    if any(
        index < 0 or index >= visible_count
        for index in selected_indices
    ):
        raise ValueError(
            "selected CUDA device index is outside the visible range"
        )
    if enable_wandb is None:
        enable_wandb = bool(
            CONFIG.get("training_enable_wandb", True)
        )
    # Resolve/login once in the parent process so rank 0 cannot fail during
    # W&B setup while rank 1 is already waiting in NCCL collectives. The
    # WANDB_API_KEY environment variable is inherited by torchrun workers.
    if bool(enable_wandb):
        mode = str(CONFIG.get("training_wandb_mode", "online"))
        if mode == "online" and not os.environ.get("WANDB_API_KEY"):
            project_ns = load_project_namespace(Path(repo))
            if bool(project_ns.get("running_on_kaggle", lambda: False)()):
                status = project_ns["configure_kaggle_wandb"](
                    verify_login=True
                )
                if not bool(status.get("configured", False)):
                    raise RuntimeError(
                        "W&B online logging is required but no attached Kaggle "
                        "WANDB_API_KEY/wandb_key secret could be configured: "
                        + str(status.get("error"))
                    )
            else:
                raise RuntimeError(
                    "W&B online logging is required but WANDB_API_KEY is not set"
                )
    import subprocess
    import tempfile
    import gc

    # The parent process may have just finished a large GPU run. Its caching
    # allocator can otherwise keep VRAM reserved while torchrun workers start.
    gc.collect()
    for device_index in selected_indices:
        with torch.cuda.device(device_index):
            torch.cuda.synchronize(device_index)
            torch.cuda.empty_cache()

    temp_dir = Path(
        tempfile.mkdtemp(prefix="uav_ddp_")
    )
    import pickle

    config_snapshot_path = temp_dir / "resolved_config.pkl"
    with config_snapshot_path.open("wb") as handle:
        pickle.dump(dict(CONFIG), handle, protocol=pickle.HIGHEST_PROTOCOL)

    launcher = _write_ddp_launcher_script(
        temp_dir / "uav_ddp_runner.py",
    )
    result_path = temp_dir / "result.json"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={world_size}",
        str(launcher),
        "--repo",
        str(Path(repo).resolve()),
        "--config-path",
        str(config_snapshot_path),
        "--algorithm",
        str(algorithm),
        "--num-envs",
        str(int(num_envs)),
        "--total-transitions",
        str(int(total_transitions)),
        "--seed",
        str(int(seed)),
        "--network-backend",
        str(network_backend),
        "--result-path",
        str(result_path),
        "--enable-wandb",
        "1" if enable_wandb else "0",
        "--target-episodes",
        (
            str(int(target_episodes))
            if target_episodes is not None
            else "0"
        ),
    ]
    stdout_path = temp_dir / "torchrun.stdout.log"
    stderr_path = temp_dir / "torchrun.stderr.log"
    child_env = os.environ.copy()
    current_visible = child_env.get(
        "CUDA_VISIBLE_DEVICES"
    )
    if current_visible:
        visible_tokens = [
            token.strip()
            for token in current_visible.split(",")
            if token.strip()
        ]
        if len(visible_tokens) < visible_count:
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES contains fewer entries than "
                "torch.cuda.device_count() reports"
            )
        selected_tokens = [
            visible_tokens[index]
            for index in selected_indices
        ]
    else:
        selected_tokens = [
            str(index)
            for index in selected_indices
        ]
    child_env["CUDA_VISIBLE_DEVICES"] = ",".join(
        selected_tokens
    )
    child_env.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )
    # Stream torchrun output into the parent notebook while also persisting
    # it. Kaggle live logs only see the notebook parent stdout/stderr; the
    # previous subprocess.run(..., stdout=file, stderr=file) implementation
    # hid every DDP training metric until the child process finished.
    with stdout_path.open(
        "w",
        buffering=1,
    ) as stdout_file:
        process = subprocess.Popen(
            command,
            cwd=str(
                Path(repo).resolve()
            ),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=child_env,
            bufsize=1,
        )
        assert (
            process.stdout
            is not None
        )
        for line in process.stdout:
            stdout_file.write(
                line
            )
            print(
                line,
                end="",
                flush=True,
            )
        returncode = process.wait()

    stdout_text = stdout_path.read_text(
        errors="replace"
    )
    stderr_text = ""
    if returncode != 0:
        raise RuntimeError(
            "DDP training failed.\nSTDOUT:\n"
            + stdout_text[-12000:]
            + "\nSTDERR:\n"
            + stderr_text[-12000:]
        )
    if not result_path.is_file():
        raise RuntimeError(
            "DDP training completed without result.json"
        )
    result = json.loads(result_path.read_text())
    result["ddp_stdout_tail"] = stdout_text[-4000:]
    result["ddp_stderr_tail"] = stderr_text[-4000:]
    result["launcher_world_size"] = int(world_size)
    result["selected_parent_devices"] = [
        f"cuda:{index}"
        for index in selected_indices
    ]
    return result


# --- frozen notebook cell 277 ---
def resolve_full_execution_mode(execution_mode=None):
    """Resolve cpu/gpu/auto while keeping one tensor implementation."""
    mode = (
        CONFIG.get("full_gpu_execution_mode", "auto")
        if execution_mode is None
        else execution_mode
    )
    mode = str(mode).strip().lower()
    if mode not in {"cpu", "gpu", "auto"}:
        raise ValueError("execution_mode must be cpu, gpu, or auto")
    if mode == "auto":
        mode = "gpu" if torch.cuda.is_available() else "cpu"
    if mode == "gpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "execution_mode='gpu' requires an available CUDA GPU"
        )
    return mode


# --- frozen notebook cell 278 ---
def resolve_full_gpu_devices(
    max_gpus=None,
    preferred_device=None,
):
    """Return CUDA devices for automatic training, capped at two.

    preferred_device selects the single-GPU device. DDP intentionally uses
    cuda:0..cuda:N-1 among the process-visible devices, matching PyTorch's
    one-process-per-GPU convention. Use CUDA_VISIBLE_DEVICES to remap physical
    GPUs before launching a multi-GPU run.
    """
    if not torch.cuda.is_available():
        return []

    visible = int(torch.cuda.device_count())
    if max_gpus is None:
        max_gpus = int(CONFIG.get("full_gpu_max_gpus", 2))
    limit = max(1, min(int(max_gpus), 2, visible))

    if preferred_device is None:
        preferred_device = CONFIG.get(
            "full_gpu_device",
            "cuda:0",
        )
    preferred_text = str(preferred_device).strip().lower()
    if preferred_text in {"auto", "cuda"}:
        preferred_index = 0
    elif preferred_text.startswith("cuda:"):
        try:
            preferred_index = int(
                preferred_text.split(":", 1)[1]
            )
        except ValueError as exc:
            raise ValueError(
                "preferred_device must be auto, cuda, or cuda:<index>"
            ) from exc
    else:
        raise ValueError(
            "preferred_device must be auto, cuda, or cuda:<index>"
        )

    if not 0 <= preferred_index < visible:
        raise ValueError(
            f"preferred CUDA device index {preferred_index} is not visible; "
            f"visible device count is {visible}"
        )

    if limit == 1:
        return [
            torch.device(
                f"cuda:{preferred_index}"
            )
        ]

    if preferred_index != 0:
        raise ValueError(
            "multi-GPU DDP uses cuda:0..cuda:N-1 among visible devices. "
            "Set runtime.device=cuda:0 and use CUDA_VISIBLE_DEVICES to "
            "choose/remap physical GPUs."
        )
    return [
        torch.device(f"cuda:{index}")
        for index in range(limit)
    ]


# --- frozen notebook cell 279 ---
def _split_env_counts(total_envs, num_shards):
    total_envs = int(total_envs)
    num_shards = int(num_shards)
    if total_envs < 1:
        raise ValueError("total_envs must be >= 1")
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    num_shards = min(num_shards, total_envs)
    base, remainder = divmod(total_envs, num_shards)
    return [base + (1 if index < remainder else 0) for index in range(num_shards)]


# --- frozen notebook cell 281 ---
def train_full_gpu_sharded(
    repo: Path,
    algorithm: str,
    num_envs: int,
    total_transitions: int,
    seed: int,
    devices=None,
    network_backend: str = "uavnetsim_gpu",
    compile_mode=None,
    fused_adam: bool = False,
    update_to_data_ratio=None,
    enable_wandb=None,
):
    """Use two CUDA devices by sharding environments and keeping one learner.

    GPU0 owns the replay buffer and learner/backward/optimizer. Environment
    shards are split across GPU0/GPU1 and stepped concurrently. Policy
    inference is centralized on GPU0 so there remains exactly one model state.
    This avoids two independently diverging MARL learners while still moving
    the dominant batched environment/network simulation onto both GPUs.
    """
    algorithm = str(algorithm).strip().lower()
    if algorithm not in {"masac", "matd3"}:
        raise ValueError("algorithm must be masac or matd3")
    network_backend = str(network_backend).strip().lower()
    if network_backend not in {"simple", "packet_approx", "uavnetsim_gpu"}:
        raise ValueError(
            "multi-GPU sharding supports simple, packet_approx, or uavnetsim_gpu"
        )
    if devices is None:
        devices = resolve_full_gpu_devices()
    devices = [torch.device(device) for device in devices]
    if len(devices) < 2:
        raise ValueError("train_full_gpu_sharded requires at least two CUDA devices")
    devices = devices[:2]
    for device in devices:
        if device.type != "cuda":
            raise ValueError("all sharded devices must be CUDA devices")
    if len({device.index for device in devices}) != len(devices):
        raise ValueError("sharded CUDA devices must be distinct")

    torch.use_deterministic_algorithms(True)
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))

    shard_counts = _split_env_counts(num_envs, len(devices))
    primary = devices[0]
    trainer = _trainer(repo, algorithm, str(primary), seed)
    if enable_wandb is None:
        enable_wandb = bool(CONFIG.get("training_enable_wandb", True))
    wandb_run = _init_gpu_wandb_run(
        algorithm,
        seed,
        enabled=bool(enable_wandb),
        network_backend=network_backend,
        num_envs=num_envs,
        gpu_count=len(devices),
    )
    wandb_last_log_step = -1
    eval_interval = max(
        1,
        int(CONFIG["training_eval_interval_steps"]),
    )
    eval_episodes = max(
        1,
        int(CONFIG["training_eval_episodes"]),
    )
    last_eval_step = 0
    latest_evaluation = None
    evaluation_backend = None
    enable_evaluation = bool(
        enable_wandb
        and CONFIG.get(
            "training_inline_evaluation",
            False,
        )
    )
    fused_adam_enabled = False
    if bool(fused_adam):
        fused_adam_enabled = _rebuild_cuda_optimizers_with_fused_adam(
            trainer, algorithm
        )
    compiled_modules = []
    if compile_mode is not None:
        compiled_modules = _compile_trainer_modules(
            trainer, algorithm, mode=compile_mode
        )

    envs = []
    shard_states = []
    offsets = []
    running_offset = 0
    for rank, (device, count) in enumerate(zip(devices, shard_counts)):
        env_seed = int(seed) + rank * 1_000_003
        env = FullGpuUAVBatchEnv(
            num_envs=int(count),
            device=str(device),
            seed=env_seed,
            strict_cuda=True,
        )
        env.network_model = network_backend
        env.assert_all_state_on_cuda()
        if (
            env.observation_dim != trainer.observation_dim
            or env.state_dim != trainer.state_dim
            or env.discrete_dim != trainer.discrete_dim
        ):
            raise RuntimeError("multi-GPU environment dimensions do not match learner")
        obs, state, mask = env.reset()
        envs.append(env)
        shard_states.append([obs, state, mask])
        offsets.append(running_offset)
        running_offset += int(count)

    capacity = min(
        max(int(CONFIG[f"{algorithm}_replay_capacity"]), 4096),
        max(int(total_transitions), 4096),
    )
    reference_env = envs[0]
    replay = make_project_gpu_replay_buffer(
        trainer,
        capacity=capacity,
        num_agents=reference_env.num_uavs,
        observation_dim=reference_env.observation_dim,
        state_dim=reference_env.state_dim,
        continuous_dim=reference_env.continuous_dim,
        discrete_dim=reference_env.discrete_dim,
        device=str(primary),
        seed=int(seed) + 123,
        strict_cuda=True,
    )
    batch_size = int(CONFIG[f"{algorithm}_batch_size"])
    learning_starts = int(CONFIG[f"{algorithm}_learning_starts"])
    if update_to_data_ratio is not None:
        raise ValueError(
            "update_to_data_ratio is deprecated; configure "
            "algorithm.train_freq and algorithm.gradient_steps instead"
        )
    train_freq = int(CONFIG.get(f"{algorithm}_train_freq", 1))
    configured_gradient_steps = int(
        CONFIG.get(
            f"{algorithm}_gradient_steps",
            CONFIG.get(f"{algorithm}_updates_per_step", 1),
        )
    )
    if train_freq < 1:
        raise ValueError("train_freq must be >= 1")
    gradient_steps = resolve_vector_gradient_steps(
        configured_gradient_steps,
        num_envs=num_envs,
        batch_size=batch_size,
        train_freq=train_freq,
    )

    policy_generator = torch.Generator(device=primary)
    policy_generator.manual_seed(int(seed) + 77_777)
    completed_episodes = 0
    global_step = 0
    vector_step_count = 0
    last_train_transition = int(learning_starts)
    update_count = 0
    latest_metrics = None
    total_transitions_int = int(total_transitions)
    per_device_env_seconds = [0.0 for _ in devices]
    transfer_seconds = 0.0
    policy_seconds = 0.0
    update_seconds = 0.0

    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    import concurrent.futures

    def move_to_primary(tensor):
        nonlocal transfer_seconds
        if tensor.device == primary:
            return tensor
        started = time.perf_counter()
        result = tensor.to(primary, non_blocking=True)
        transfer_seconds += time.perf_counter() - started
        return result

    def policy_actions_for_all(transition_steps_all):
        nonlocal policy_seconds, transfer_seconds
        started = time.perf_counter()
        obs_parts = []
        mask_parts = []
        for obs, _state, mask in shard_states:
            transfer_start = time.perf_counter()
            obs_parts.append(obs if obs.device == primary else obs.to(primary, non_blocking=True))
            mask_parts.append(mask if mask.device == primary else mask.to(primary, non_blocking=True))
            transfer_seconds += time.perf_counter() - transfer_start
        obs_all = torch.cat(obs_parts, dim=0)
        mask_all = torch.cat(mask_parts, dim=0)
        if algorithm == "masac":
            with torch.no_grad():
                sampled = trainer._sample_joint_policy(
                    obs_all, mask_all, deterministic=False
                )
                continuous_all = sampled["continuous"]
                destination_all = sampled["destination_indices"]
        else:
            continuous_all, destination_all = _matd3_actions(
                trainer,
                obs_all,
                mask_all,
                transition_steps_all,
                policy_generator,
            )
        continuous_parts = []
        destination_parts = []
        start_index = 0
        for device, count in zip(devices, shard_counts):
            stop_index = start_index + int(count)
            c = continuous_all[start_index:stop_index]
            d = destination_all[start_index:stop_index]
            transfer_start = time.perf_counter()
            if device != primary:
                c = c.to(device, non_blocking=True)
                d = d.to(device, non_blocking=True)
            transfer_seconds += time.perf_counter() - transfer_start
            continuous_parts.append(c)
            destination_parts.append(d)
            start_index = stop_index
        policy_seconds += time.perf_counter() - started
        return continuous_parts, destination_parts

    wall_start = time.perf_counter()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(envs))
    try:
        while global_step < total_transitions_int:
            transition_steps_all = (
                torch.arange(int(num_envs), device=primary, dtype=torch.long)
                + global_step
                + 1
            )
            batch_first_step = global_step + 1
            batch_last_step = global_step + int(num_envs)
            all_warmup = batch_last_step <= learning_starts
            all_policy = batch_first_step > learning_starts

            if all_warmup:
                action_parts = [
                    _random_exploration_actions(env, shard_states[index][2])
                    for index, env in enumerate(envs)
                ]
                continuous_parts = [item[0] for item in action_parts]
                destination_parts = [item[1] for item in action_parts]
            elif all_policy:
                continuous_parts, destination_parts = policy_actions_for_all(
                    transition_steps_all
                )
            else:
                policy_continuous, policy_destination = policy_actions_for_all(
                    transition_steps_all
                )
                continuous_parts = []
                destination_parts = []
                for index, env in enumerate(envs):
                    random_continuous, random_destination = _random_exploration_actions(
                        env, shard_states[index][2]
                    )
                    local_start = offsets[index]
                    local_stop = local_start + shard_counts[index]
                    warmup_local = (
                        transition_steps_all[local_start:local_stop]
                        <= learning_starts
                    ).to(env.device)
                    continuous_parts.append(
                        torch.where(
                            warmup_local[:, None, None],
                            random_continuous,
                            policy_continuous[index],
                        )
                    )
                    destination_parts.append(
                        torch.where(
                            warmup_local[:, None],
                            random_destination,
                            policy_destination[index],
                        )
                    )

            def step_one(index):
                env = envs[index]
                with torch.cuda.device(env.device):
                    started = time.perf_counter()
                    result = env.step(
                        continuous_parts[index], destination_parts[index]
                    )
                    elapsed = time.perf_counter() - started
                return index, result, elapsed

            futures = [executor.submit(step_one, index) for index in range(len(envs))]
            step_results = [None] * len(envs)
            for future in futures:
                index, result, elapsed = future.result()
                step_results[index] = result
                per_device_env_seconds[index] += elapsed

            take_total = min(
                int(num_envs), total_transitions_int - global_step
            )
            remaining_take = take_total
            for index, env in enumerate(envs):
                (
                    next_obs,
                    next_state,
                    next_mask,
                    reward,
                    done,
                    _metrics,
                ) = step_results[index]
                obs, state, mask = shard_states[index]
                take = min(int(shard_counts[index]), remaining_take)
                if take > 0:
                    replay.add_batch(
                        move_to_primary(obs[:take]),
                        move_to_primary(state[:take]),
                        move_to_primary(continuous_parts[index][:take]),
                        move_to_primary(destination_parts[index][:take]),
                        move_to_primary(mask[:take]),
                        move_to_primary(reward[:take]),
                        move_to_primary(next_obs[:take]),
                        move_to_primary(next_state[:take]),
                        move_to_primary(next_mask[:take]),
                        move_to_primary(done[:take]),
                    )
                    remaining_take -= take
                shard_states[index] = [next_obs, next_state, next_mask]

            global_step += take_total
            vector_step_count += 1
            should_train = (
                len(replay) >= batch_size
                and global_step > learning_starts
                and vector_step_count % train_freq == 0
            )
            if should_train:
                if gradient_steps == -1:
                    updates_this_step = max(
                        0,
                        global_step
                        - max(
                            last_train_transition,
                            learning_starts,
                        ),
                    )
                else:
                    updates_this_step = gradient_steps
                update_start = time.perf_counter()
                for local_update in range(updates_this_step):
                    collected = trainer.update(
                        replay,
                        batch_size=batch_size,
                        collect_metrics=(local_update == updates_this_step - 1),
                    )
                    if collected is not None:
                        latest_metrics = collected
                    update_count += 1
                update_seconds += time.perf_counter() - update_start
                last_train_transition = global_step

            should_evaluate = (
                enable_evaluation
                and (
                    global_step - last_eval_step >= eval_interval
                    or global_step >= total_transitions_int
                )
            )
            if should_evaluate:
                evaluation = evaluate_full_gpu_reference(
                    trainer=trainer,
                    algorithm=algorithm,
                    network_backend=network_backend,
                    seed=seed,
                    episodes=eval_episodes,
                    uavnetsim_repo_path=None,
                )
                latest_evaluation = evaluation["summary"]
                evaluation_backend = evaluation["backend"]
                last_eval_step = global_step
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "global_step": int(global_step),
                            **{
                                f"evaluation/{key}": float(value)
                                for key, value in latest_evaluation.items()
                                if isinstance(value, (bool, int, float))
                            },
                        }
                    )
                    for key, value in latest_evaluation.items():
                        if isinstance(value, (bool, int, float)):
                            wandb_run.summary[
                                f"evaluation_latest/{key}"
                            ] = float(value)

            if (
                wandb_run is not None
                and (
                    global_step - wandb_last_log_step
                    >= max(1, int(CONFIG.get("training_log_interval_steps", 100)))
                    or global_step >= total_transitions_int
                )
            ):
                elapsed_for_log = max(
                    1e-12, time.perf_counter() - wall_start
                )
                payload = {
                    "global_step": int(global_step),
                    "train/transitions_per_second": global_step / elapsed_for_log,
                    "train/updates_per_second": update_count / elapsed_for_log,
                }
                if latest_metrics:
                    for key, value in latest_metrics.items():
                        if isinstance(value, torch.Tensor):
                            if value.numel() != 1:
                                continue
                            value = value.detach().cpu().item()
                        if isinstance(value, (bool, int, float)):
                            payload[f"train/{key}"] = float(value)
                wandb_run.log(payload)
                wandb_last_log_step = global_step

            for index, env in enumerate(envs):
                done = step_results[index][4]
                if bool(done.any().item()):
                    completed_episodes += int(done.long().sum().item())
                    if global_step < total_transitions_int:
                        obs, state, mask = env.reset(mask=done)
                        shard_states[index] = [obs, state, mask]

        for device in devices:
            torch.cuda.synchronize(device)
        wall = time.perf_counter() - wall_start
    finally:
        executor.shutdown(wait=True)

    def scalarize(metrics):
        if metrics is None:
            return None
        result = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    result[key] = float(value.detach().cpu().item())
            elif isinstance(value, (int, float, bool)):
                result[key] = value
        return result

    peak_memory = {
        str(device): torch.cuda.max_memory_allocated(device) / 2**20
        for device in devices
    }
    gpu_names = {
        str(device): torch.cuda.get_device_name(device)
        for device in devices
    }
    result = {
        "kind": "full_gpu_auto_multi",
        "multi_gpu_strategy": "env_shard_single_learner",
        "algorithm": algorithm,
        "network_backend": network_backend,
        "num_envs": int(num_envs),
        "envs_per_device": {
            str(device): int(count)
            for device, count in zip(devices, shard_counts)
        },
        "used_gpu_count": len(devices),
        "devices": [str(device) for device in devices],
        "learner_device": str(primary),
        "gpu_names": gpu_names,
        "transitions": int(global_step),
        "updates": int(update_count),
        "train_freq": int(train_freq),
        "gradient_steps": int(gradient_steps),
        "episodes_completed": int(completed_episodes),
        "wall_seconds": float(wall),
        "transitions_per_second": global_step / wall,
        "updates_per_second": update_count / wall,
        "policy_seconds": float(policy_seconds),
        "update_seconds": float(update_seconds),
        "env_seconds_per_device": {
            str(device): float(seconds)
            for device, seconds in zip(devices, per_device_env_seconds)
        },
        "transfer_enqueue_seconds": float(transfer_seconds),
        "peak_memory_mb_per_device": peak_memory,
        "compile_mode": compile_mode,
        "compiled_modules": compiled_modules,
        "fused_adam": bool(fused_adam_enabled),
        "latest_update_metrics": scalarize(latest_metrics),
        "latest_evaluation": latest_evaluation,
        "evaluation_backend": evaluation_backend,
        "model_fingerprint_sha256": _trainer_parameter_fingerprint(trainer),
        "wandb_run_id": (
            wandb_run.id if wandb_run is not None else None
        ),
        "wandb_run_name": (
            wandb_run.name if wandb_run is not None else None
        ),
        "wandb_run_url": (
            wandb_run.url if wandb_run is not None else None
        ),
    }
    final_checkpoint_path = save_full_gpu_model_checkpoint(
        Path(repo)
        / "checkpoints_full_gpu"
        / (
            f"{algorithm}_final_step_{int(global_step):012d}.pt"
        ),
        trainer,
        algorithm,
        global_step,
        result["episodes_completed"],
        latest_evaluation=latest_evaluation,
    )
    result["final_checkpoint_path"] = str(final_checkpoint_path)
    result["checkpoint_paths"] = [str(final_checkpoint_path)]
    if wandb_run is not None:
        wandb_run.summary["final/transitions_per_second"] = result[
            "transitions_per_second"
        ]
        wandb_run.summary["final/updates_per_second"] = result[
            "updates_per_second"
        ]
        wandb_run.summary["final/gpu_count"] = len(devices)
        wandb_run.finish()
    for env in envs:
        env.close()
    return result


# --- frozen notebook cell 282 ---
def train_full_gpu_auto(
    repo: Path,
    algorithm: str,
    num_envs: int,
    total_transitions: int,
    seed: int,
    network_backend: str = "uavnetsim_gpu",
    max_gpus=None,
    auto_multi_gpu=None,
    compile_mode=None,
    fused_adam: bool = False,
    update_to_data_ratio=None,
    enable_wandb=None,
    execution_mode=None,
    target_episodes=None,
    device=None,
):
    """Run the same tensor training path in cpu/gpu/auto mode.

    cpu: force CPU for logic/regression testing.
    gpu: require CUDA and auto-select one GPU or 2-GPU DDP.
    auto: use GPU when CUDA is available, otherwise CPU.
    """
    if update_to_data_ratio is not None:
        raise ValueError(
            "update_to_data_ratio is deprecated; configure "
            "algorithm.train_freq and algorithm.gradient_steps instead"
        )
    if auto_multi_gpu is None:
        auto_multi_gpu = bool(
            CONFIG.get("full_gpu_auto_multi_gpu", True)
        )
    resolved_mode = resolve_full_execution_mode(
        execution_mode
    )
    if enable_wandb is False:
        raise RuntimeError(
            "train_full_gpu_auto is the production full-flow entry point and "
            "requires W&B online; enable_wandb=False is not allowed here."
        )
    enable_wandb = True

    if (
        bool(enable_wandb)
        and bool(
            CONFIG.get(
                "training_wandb_require_online_gpu",
                True,
            )
        )
        and str(
            CONFIG.get(
                "training_wandb_mode",
                "online",
            )
        ).strip().lower() != "online"
    ):
        raise RuntimeError(
            "Full-flow W&B is required-online, but training_wandb_mode "
            "is not 'online'."
        )

    # Resolve the account before expensive training starts. On Kaggle this is
    # also the value recorded in the W&B run-config column kaggle_account.
    resolve_kaggle_account()

    def finish_full_flow(training_result):
        postprocess_mode = str(
            CONFIG.get(
                "training_postprocess_mode",
                "separate_cpu",
            )
        ).strip().lower()
        if postprocess_mode == "same_session":
            from ..evaluation.postprocess import finalize_training_visualization

            return finalize_training_visualization(
                training_result,
                repo=repo,
                algorithm=algorithm,
                network_backend=network_backend,
                seed=seed,
            )
        if postprocess_mode != "separate_cpu":
            raise ValueError(
                "training_postprocess_mode must be separate_cpu or same_session"
            )
        result = dict(training_result)
        result["visualization_complete"] = False
        result["visualization_pending"] = True
        result["postprocess_mode"] = "separate_cpu"
        return result

    if resolved_mode == "cpu":
        result = train_full_gpu(
            repo=repo,
            algorithm=algorithm,
            num_envs=int(num_envs),
            total_transitions=int(total_transitions),
            seed=int(seed),
            device="cpu",
            network_backend=network_backend,
            strict_cuda=False,
            compile_mode=compile_mode,
            fused_adam=False,
            update_to_data_ratio=update_to_data_ratio,
            enable_wandb=bool(enable_wandb),
            target_episodes=target_episodes,
        )
        result["execution_mode"] = "cpu"
        result["used_gpu_count"] = 0
        result["devices"] = ["cpu"]
        result["multi_gpu_strategy"] = "cpu"
        result["learner_device"] = "cpu"
        return finish_full_flow(result)

    requested_gpu_limit = (
        max_gpus
        if bool(auto_multi_gpu)
        else 1
    )
    devices = resolve_full_gpu_devices(
        max_gpus=requested_gpu_limit,
        preferred_device=device,
    )
    if not devices:
        raise RuntimeError(
            "GPU mode resolved but no CUDA device is available"
        )

    strategy = str(
        CONFIG.get("full_gpu_multi_gpu_strategy", "ddp")
    ).strip().lower()

    if (
        bool(auto_multi_gpu)
        and len(devices) >= 2
        and int(num_envs) >= 2
    ):
        if strategy == "ddp":
            if compile_mode is not None:
                raise ValueError(
                    "compile_mode is not enabled in the DDP path yet; "
                    "use compile_mode=None for deterministic DDP training"
                )
            if fused_adam:
                raise ValueError(
                    "fused_adam is not enabled in the DDP path yet; "
                    "use fused_adam=False until separately benchmarked"
                )
            result = train_full_gpu_ddp(
                repo=repo,
                algorithm=algorithm,
                num_envs=int(num_envs),
                total_transitions=int(total_transitions),
                seed=int(seed),
                network_backend=network_backend,
                enable_wandb=enable_wandb,
                target_episodes=target_episodes,
                world_size=2,
                selected_devices=devices[:2],
            )
            result["execution_mode"] = "gpu"
            return finish_full_flow(result)
        if strategy == "env_shard_single_learner":
            if target_episodes is not None:
                raise ValueError(
                    "target_episodes currently requires "
                    "the DDP strategy"
                )
            result = train_full_gpu_sharded(
                repo=repo,
                algorithm=algorithm,
                num_envs=int(num_envs),
                total_transitions=int(total_transitions),
                seed=int(seed),
                devices=devices[:2],
                network_backend=network_backend,
                compile_mode=compile_mode,
                fused_adam=bool(fused_adam),
                update_to_data_ratio=update_to_data_ratio,
                enable_wandb=bool(enable_wandb),
            )
            result["execution_mode"] = "gpu"
            return finish_full_flow(result)
        raise ValueError(
            "full_gpu_multi_gpu_strategy must be ddp or "
            "env_shard_single_learner"
        )

    if target_episodes is not None:
        result = train_full_gpu_ddp(
            repo=repo,
            algorithm=algorithm,
            num_envs=int(num_envs),
            total_transitions=int(total_transitions),
            seed=int(seed),
            network_backend=network_backend,
            enable_wandb=enable_wandb,
            target_episodes=target_episodes,
            world_size=1,
            selected_devices=[devices[0]],
        )
        result["execution_mode"] = "gpu"
        return finish_full_flow(result)

    result = train_full_gpu(
        repo=repo,
        algorithm=algorithm,
        num_envs=int(num_envs),
        total_transitions=int(total_transitions),
        seed=int(seed),
        device=str(devices[0]),
        network_backend=network_backend,
        strict_cuda=True,
        compile_mode=compile_mode,
        fused_adam=bool(fused_adam),
        update_to_data_ratio=update_to_data_ratio,
        enable_wandb=bool(enable_wandb),
        target_episodes=target_episodes,
    )
    result["execution_mode"] = "gpu"
    result["used_gpu_count"] = 1
    result["devices"] = [str(devices[0])]
    result["multi_gpu_strategy"] = "single_gpu"
    result["learner_device"] = str(devices[0])
    return finish_full_flow(result)


# --- frozen notebook cell 284 ---
def benchmark_full_gpu_training_optimizations(
    repo: Path,
    *,
    algorithms=("masac", "matd3"),
    env_counts=(256, 512, 1024),
    total_transitions=8192,
    seed=44,
    device="cuda:0",
    network_backend="uavnetsim_gpu",
    test_compile=True,
    test_fused_adam=True,
    test_amp=True,
    update_to_data_ratios=(1.0,),
):
    """Short end-to-end optimization sweep without changing production defaults.

    Timing-only fields are expected to vary. Deterministic metrics/fingerprints
    are returned so any speed optimization can be rejected if it changes the
    numerical training path unexpectedly.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for optimization benchmarking")

    previous_amp = bool(CONFIG.get("training_amp_enabled", False))
    cases = [
        {
            "name": "baseline",
            "compile_mode": None,
            "fused_adam": False,
            "amp": False,
        }
    ]
    if test_fused_adam:
        cases.append(
            {
                "name": "fused_adam",
                "compile_mode": None,
                "fused_adam": True,
                "amp": False,
            }
        )
    if test_compile:
        cases.append(
            {
                "name": "compile_default_fused",
                "compile_mode": "default",
                "fused_adam": bool(test_fused_adam),
                "amp": False,
            }
        )
    if test_amp:
        cases.append(
            {
                "name": "amp_fused",
                "compile_mode": None,
                "fused_adam": bool(test_fused_adam),
                "amp": True,
            }
        )
    if test_amp and test_compile:
        cases.append(
            {
                "name": "compile_amp_fused",
                "compile_mode": "default",
                "fused_adam": bool(test_fused_adam),
                "amp": True,
            }
        )

    results = []
    try:
        for algorithm in algorithms:
            for env_count in env_counts:
                for utd in update_to_data_ratios:
                    for case in cases:
                        CONFIG["training_amp_enabled"] = bool(
                            case["amp"]
                        )
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        try:
                            item = train_full_gpu(
                                repo=repo,
                                algorithm=algorithm,
                                num_envs=int(env_count),
                                total_transitions=int(total_transitions),
                                seed=int(seed),
                                device=device,
                                network_backend=network_backend,
                                strict_cuda=True,
                                compile_mode=case["compile_mode"],
                                fused_adam=bool(case["fused_adam"]),
                                update_to_data_ratio=float(utd),
                            )
                            item["optimization_case"] = case["name"]
                            item["amp_enabled"] = bool(case["amp"])
                            results.append(item)
                        except torch.cuda.OutOfMemoryError as exc:
                            torch.cuda.empty_cache()
                            results.append(
                                {
                                    "algorithm": algorithm,
                                    "num_envs": int(env_count),
                                    "optimization_case": case["name"],
                                    "amp_enabled": bool(case["amp"]),
                                    "update_to_data_ratio": float(utd),
                                    "error": "CUDA OOM",
                                    "detail": str(exc),
                                }
                            )
        return results
    finally:
        CONFIG["training_amp_enabled"] = previous_amp


# --- frozen notebook cell 285 ---
def profile_full_gpu_step(
    repo: Path,
    *,
    algorithm="masac",
    num_envs=256,
    seed=44,
    device="cuda:0",
    network_backend="uavnetsim_gpu",
    warmup_steps=3,
    active_steps=5,
    output_trace=None,
):
    """Profile actor inference + environment/network step on CUDA.

    This intentionally excludes optimizer updates; use the end-to-end sweep for
    learner throughput. The profiler table makes environment/network hotspots
    visible before attempting more invasive rewrites.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for profiling")
    algorithm = str(algorithm).strip().lower()
    env = FullGpuUAVBatchEnv(
        num_envs=int(num_envs),
        device=device,
        seed=int(seed),
        strict_cuda=True,
    )
    env.network_model = str(network_backend)
    trainer = _trainer(repo, algorithm, device, seed)
    obs, state, mask = env.reset()

    def one_step():
        nonlocal obs, state, mask
        if algorithm == "masac":
            with torch.no_grad():
                sample = trainer._sample_joint_policy(
                    obs,
                    mask,
                    deterministic=False,
                )
                continuous = sample["continuous"]
                destination = sample["destination_indices"]
        elif algorithm == "matd3":
            transition_steps = torch.ones(
                int(num_envs),
                dtype=torch.long,
                device=env.device,
            )
            continuous, destination = _matd3_actions(
                trainer,
                obs,
                mask,
                transition_steps,
                env.generator,
            )
        else:
            raise ValueError("algorithm must be masac or matd3")
        obs, state, mask, _, done, _ = env.step(
            continuous,
            destination,
        )
        if bool(done.any().item()):
            obs, state, mask = env.reset(mask=done)

    for _ in range(int(warmup_steps)):
        one_step()
    torch.cuda.synchronize(env.device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(int(active_steps)):
            one_step()
    torch.cuda.synchronize(env.device)
    table = prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=30,
    )
    if output_trace is not None:
        prof.export_chrome_trace(str(output_trace))
    env.close()
    return table


# --- frozen notebook cell 286 ---
def benchmark_gpu_network_models(
    *,
    network_models=("packet_approx", "uavnetsim_gpu"),
    env_counts=(256, 512, 1024),
    steps=40,
    warmup_steps=4,
    seed=44,
    device="cuda:0",
):
    """Measure the cost of richer network fidelity on the same GPU env."""
    previous = CONFIG.get("full_gpu_network_model", "simple")
    results = []
    try:
        for model in network_models:
            CONFIG["full_gpu_network_model"] = str(model)
            for count in env_counts:
                torch.cuda.empty_cache()
                if int(warmup_steps) > 0:
                    benchmark_env(
                        int(count),
                        int(warmup_steps),
                        int(seed),
                        device,
                        True,
                    )
                torch.cuda.empty_cache()
                item = benchmark_env(
                    int(count),
                    int(steps),
                    int(seed),
                    device,
                    True,
                )
                item["network_model"] = str(model)
                results.append(item)
        return results
    finally:
        CONFIG["full_gpu_network_model"] = previous


# --- frozen notebook cell 287 ---
def benchmark_simple_scaling(
    repo: Path,
    *,
    cpu_env_counts=(1, 2, 4, 8, 16),
    gpu_env_counts=(1, 2, 4, 8, 16, 32, 64, 128, 256),
    vector_steps=40,
    warmup_steps=4,
    seed=44,
    device="cuda:0",
):
    results = {"cpu": [], "gpu": []}

    for count in cpu_env_counts:
        if int(warmup_steps) > 0:
            benchmark_hybrid_env(
                repo,
                num_envs=int(count),
                steps=int(warmup_steps),
                seed=int(seed),
            )
        item = benchmark_hybrid_env(
            repo,
            num_envs=int(count),
            steps=int(vector_steps),
            seed=int(seed),
        )
        results["cpu"].append(item)

    for count in gpu_env_counts:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(torch.device(device))
        if int(warmup_steps) > 0:
            benchmark_env(
                num_envs=int(count),
                steps=int(warmup_steps),
                seed=int(seed),
                device=device,
                strict_cuda=True,
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(torch.device(device))
        item = benchmark_env(
            num_envs=int(count),
            steps=int(vector_steps),
            seed=int(seed),
            device=device,
            strict_cuda=True,
        )
        results["gpu"].append(item)

    cpu_by_count = {
        int(item["num_envs"]): item
        for item in results["cpu"]
    }
    gpu_by_count = {
        int(item["num_envs"]): item
        for item in results["gpu"]
    }
    same_count = []
    for count in sorted(set(cpu_by_count) & set(gpu_by_count)):
        cpu_rate = float(cpu_by_count[count]["transitions_per_second"])
        gpu_rate = float(gpu_by_count[count]["transitions_per_second"])
        same_count.append(
            {
                "num_envs": int(count),
                "cpu_transitions_per_second": cpu_rate,
                "gpu_transitions_per_second": gpu_rate,
                "gpu_over_cpu_speedup": (
                    gpu_rate / cpu_rate
                    if cpu_rate > 0.0
                    else float("inf")
                ),
            }
        )

    best_cpu = max(
        results["cpu"],
        key=lambda item: item["transitions_per_second"],
    )
    best_gpu = max(
        results["gpu"],
        key=lambda item: item["transitions_per_second"],
    )
    results["same_count_comparison"] = same_count
    results["best_cpu"] = best_cpu
    results["best_gpu"] = best_gpu
    results["best_gpu_over_best_cpu"] = (
        float(best_gpu["transitions_per_second"])
        / float(best_cpu["transitions_per_second"])
    )
    results["same_count_crossover_envs"] = next(
        (
            item["num_envs"]
            for item in same_count
            if item["gpu_over_cpu_speedup"] > 1.0
        ),
        None,
    )
    return results


# --- migrated support helper from frozen notebook cell 293 ---
def _trainer_parameter_fingerprint(trainer):
    """SHA256 over all nn.Module parameters/buffers, independent of device."""
    import hashlib

    digest = hashlib.sha256()
    module_names = sorted(
        name
        for name, value in trainer.__dict__.items()
        if isinstance(value, torch.nn.Module)
    )
    for module_name in module_names:
        module = getattr(trainer, module_name)
        digest.update(module_name.encode("utf-8"))
        for key, tensor in sorted(
            module.state_dict().items()
        ):
            digest.update(key.encode("utf-8"))
            value = (
                tensor.detach()
                .cpu()
                .contiguous()
            )
            digest.update(
                str(value.dtype).encode("utf-8")
            )
            digest.update(
                str(tuple(value.shape)).encode(
                    "utf-8"
                )
            )
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


# Export public and internal helpers required by downstream production layers.
__all__ = [name for name in globals() if not name.startswith("__")]
