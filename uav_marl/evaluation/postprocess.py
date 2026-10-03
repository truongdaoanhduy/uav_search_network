"""Production source migrated from frozen fix_test_gpu.ipynb.

Do not import the notebook at runtime. The notebook is retained only as a
reference/regression artifact. Production changes belong in Python modules.
Migrated from notebook cells 288..298.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from ..training.gpu import *  # noqa: F401,F403

# --- frozen notebook cell 289 ---
def _uavnetsim_cpu_worker(payload):
    repo_text, seed, steps, communication_loaded = payload
    repo = Path(repo_text)
    project_ns = load_project_namespace(repo)
    np = project_ns["np"]
    CPUEnv = project_ns["UAVSearchEnv"]
    env = CPUEnv(
        backend_name="uavnetsim",
        network_backend_kwargs={"repo_path": None},
    )
    try:
        env.reset(seed=int(seed))
        if communication_loaded:
            Report = project_ns["Report"]
            env.targets[0].confirmed = True
            env.report_buffers[0].clear()
            env.report_buffers[0].append(
                Report(
                    target_id=0,
                    source_uav=0,
                    created_step=0,
                    size_bytes=int(CONFIG["report_bytes"]),
                    ttl_s=float(CONFIG["report_ttl"]),
                    delivered_bytes=0,
                )
            )
        rng = np.random.default_rng(int(seed) + 99173)
        start = time.perf_counter()
        completed = 0
        for _ in range(int(steps)):
            action = {}
            for sender, agent_id in enumerate(env.agent_ids):
                continuous = rng.uniform(
                    -1.0, 1.0, size=4
                ).astype(np.float32)
                discrete_dim = int(
                    env.action_space[
                        agent_id
                    ]["destination"].n
                )
                destination = int(
                    rng.integers(0, discrete_dim)
                )
                if communication_loaded and sender == 0:
                    continuous[3] = 1.0
                    destination = discrete_dim - 1
                action[agent_id] = {
                    "motion": continuous[:3],
                    "destination": destination,
                    "power": np.asarray(
                        [continuous[3]],
                        dtype=np.float32,
                    ),
                }
            out = env.step(action)
            completed += 1
            if bool(out[2] or out[3]):
                break
        wall = time.perf_counter() - start
        return {
            "steps": int(completed),
            "wall_seconds": float(wall),
            "network": env.network_backend.metrics(),
        }
    finally:
        env.close()


# --- frozen notebook cell 290 ---
def benchmark_uavnetsim_cpu_worker_scaling(
    repo: Path,
    *,
    env_counts=(1, 2, 4, 8),
    steps=4,
    seed=44,
    communication_loaded=True,
):
    import multiprocessing as mp

    results = []
    context = mp.get_context("fork")

    def worker_entry(payload, queue):
        try:
            queue.put(
                ("ok", _uavnetsim_cpu_worker(payload))
            )
        except BaseException as exc:
            import traceback
            queue.put(
                (
                    "error",
                    (
                        type(exc).__name__,
                        str(exc),
                        traceback.format_exc(),
                    ),
                )
            )

    for count in env_counts:
        payloads = [
            (
                str(repo),
                int(seed) + index,
                int(steps),
                bool(communication_loaded),
            )
            for index in range(int(count))
        ]
        queue = context.Queue()
        processes = [
            context.Process(
                target=worker_entry,
                args=(payload, queue),
            )
            for payload in payloads
        ]
        start = time.perf_counter()
        for process in processes:
            process.start()
        messages = [
            queue.get()
            for _ in processes
        ]
        for process in processes:
            process.join()
        wall = time.perf_counter() - start
        errors = [
            payload
            for status, payload in messages
            if status != "ok"
        ]
        if errors:
            raise RuntimeError(
                "UavNetSim worker failed: "
                + repr(errors[0])
            )
        worker_results = [
            payload
            for status, payload in messages
            if status == "ok"
        ]
        transitions = sum(
            int(item["steps"])
            for item in worker_results
        )
        results.append(
            {
                "num_envs": int(count),
                "requested_steps_per_env": int(steps),
                "transitions": int(transitions),
                "wall_seconds": float(wall),
                "transitions_per_second": (
                    transitions / wall
                    if wall > 0.0
                    else float("inf")
                ),
                "worker_wall_seconds": [
                    float(item["wall_seconds"])
                    for item in worker_results
                ],
                "network_metrics": [
                    item["network"]
                    for item in worker_results
                ],
            }
        )
    return results


# --- frozen notebook cell 292 ---
def _scalarize_metrics(metrics):
    if metrics is None:
        return None
    result = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                result[str(key)] = float(
                    value.detach().cpu().item()
                )
        elif isinstance(value, (int, float, bool)):
            result[str(key)] = value
    return result


# --- frozen notebook cell 293 ---
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


# --- frozen notebook cell 294 ---
def benchmark_uavnetsim_training_compare(
    repo: Path,
    *,
    repo_path,
    algorithm="masac",
    total_steps=8,
    seed=44,
    device="cuda:0",
    benchmark_learning_starts=4,
    benchmark_batch_size=4,
    benchmark_updates_per_step=1,
):
    """End-to-end CPU-mission vs GPU-mission training with UavNetSim on both.

    The benchmark overrides learning-start/batch settings only inside this
    function so a short run actually exercises replay sampling and gradients.
    Production CONFIG values are restored before returning.
    """
    algorithm = str(algorithm).strip().lower()
    if algorithm not in {"masac", "matd3"}:
        raise ValueError(
            "algorithm must be masac or matd3"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "training comparison requires CUDA"
        )
    total_steps = int(total_steps)
    if total_steps < 1:
        raise ValueError(
            "total_steps must be >= 1"
        )

    project_ns = load_project_namespace(repo)
    if not project_ns["uavnetsim_available"](
        repo_path
    ):
        raise FileNotFoundError(
            "UavNetSim repository is unavailable"
        )

    prefix = algorithm
    keys = {
        "learning": (
            f"{prefix}_learning_starts"
        ),
        "batch": f"{prefix}_batch_size",
        "updates": (
            f"{prefix}_updates_per_step"
        ),
    }
    previous = {
        key: CONFIG[name]
        for key, name in keys.items()
    }
    CONFIG[keys["learning"]] = int(
        benchmark_learning_starts
    )
    CONFIG[keys["batch"]] = int(
        benchmark_batch_size
    )
    CONFIG[keys["updates"]] = int(
        benchmark_updates_per_step
    )

    cpu_env = None
    try:
        CPUEnv = project_ns["UAVSearchEnv"]
        cpu_env = CPUEnv(
            backend_name="uavnetsim",
            network_backend_kwargs={
                "repo_path": repo_path,
            },
        )
        project_ns["set_seed"](int(seed))
        if algorithm == "masac":
            cpu_trainer = project_ns[
                "HybridMASAC"
            ](
                cpu_env,
                device=device,
            )
            train_fn = project_ns[
                "train_masac"
            ]
        else:
            cpu_trainer = project_ns[
                "HybridMATD3"
            ](
                cpu_env,
                device=device,
                seed=int(seed),
            )
            train_fn = project_ns[
                "train_matd3"
            ]

        torch.cuda.synchronize()
        cpu_start = time.perf_counter()
        cpu_result = train_fn(
            cpu_env,
            total_steps=total_steps,
            seed=int(seed),
            trainer=cpu_trainer,
        )
        torch.cuda.synchronize()
        cpu_wall = (
            time.perf_counter()
            - cpu_start
        )
        cpu_network = (
            cpu_env.network_backend.metrics()
        )

        torch.cuda.synchronize()
        gpu_result = train_full_gpu(
            repo=repo,
            algorithm=algorithm,
            num_envs=1,
            total_transitions=total_steps,
            seed=int(seed),
            device=device,
            network_backend="uavnetsim",
            uavnetsim_repo_path=repo_path,
        )

        return {
            "kind": (
                "uavnetsim_training_compare"
            ),
            "algorithm": algorithm,
            "seed": int(seed),
            "total_steps": total_steps,
            "benchmark_overrides": {
                keys["learning"]: int(
                    benchmark_learning_starts
                ),
                keys["batch"]: int(
                    benchmark_batch_size
                ),
                keys["updates"]: int(
                    benchmark_updates_per_step
                ),
            },
            "cpu_mission_gpu_learner": {
                "wall_seconds": cpu_wall,
                "steps_per_second": (
                    total_steps / cpu_wall
                ),
                "completed_episodes": len(
                    cpu_result[
                        "completed_episodes"
                    ]
                ),
                "completed_episode_metrics": (
                    cpu_result[
                        "completed_episodes"
                    ]
                ),
                "latest_update_metrics": (
                    _scalarize_metrics(
                        cpu_result[
                            "latest_update_metrics"
                        ]
                    )
                ),
                "model_fingerprint_sha256": (
                    _trainer_parameter_fingerprint(
                        cpu_result["trainer"]
                    )
                ),
                "network_metrics": (
                    cpu_network
                ),
            },
            "gpu_mission_uavnetsim": (
                gpu_result
            ),
        }
    finally:
        if cpu_env is not None:
            cpu_env.close()
        for key, name in keys.items():
            CONFIG[name] = previous[key]


# --- frozen notebook cell 296 ---
def configure_kaggle_huggingface():
    """Load Hugging Face token/repo id from env or Kaggle Secrets."""
    result = {
        "configured": False,
        "token_secret_name": None,
        "repo_secret_name": None,
        "repo_id": os.environ.get("HF_REPO_ID"),
        "error": None,
    }
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if running_on_kaggle():
        try:
            from kaggle_secrets import UserSecretsClient
            client = UserSecretsClient()
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            return result
        if not token:
            for name in CONFIG.get(
                "kaggle_hf_token_secret_names",
                ("HF_TOKEN",),
            ):
                try:
                    candidate = client.get_secret(str(name))
                except Exception:
                    continue
                if candidate:
                    token = candidate
                    result["token_secret_name"] = str(name)
                    break
        if not result["repo_id"]:
            for name in CONFIG.get(
                "kaggle_hf_repo_secret_names",
                ("HF_REPO_ID",),
            ):
                try:
                    candidate = client.get_secret(str(name))
                except Exception:
                    continue
                if candidate:
                    result["repo_id"] = str(candidate)
                    result["repo_secret_name"] = str(name)
                    break
    if not token:
        result["error"] = "HF token not found"
        return result
    if not result["repo_id"]:
        result["error"] = "HF_REPO_ID not found"
        return result
    os.environ["HF_TOKEN"] = str(token)
    os.environ["HF_REPO_ID"] = str(result["repo_id"])
    try:
        from huggingface_hub import HfApi
        HfApi(token=token).whoami()
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    result["configured"] = True
    return result


# --- frozen notebook cell 297 ---
def postprocess_checkpoint_cpu(
    checkpoint_path=None,
    artifact_ref=None,
    algorithm="masac",
    network_backend="uavnetsim_gpu",
    seed=None,
    episodes=None,
    output_dir="cpu_postprocess",
    source_run_id=None,
    log_wandb=False,
    upload_huggingface=None,
    render_video=True,
):
    """Evaluate + render a trained checkpoint on CPU only.

    A CPU W&B post-process run can consume the training model artifact, mirror
    the checkpoint to Hugging Face, and log deterministic evaluation media.
    """
    algorithm = str(algorithm).strip().lower()
    if algorithm not in {"masac", "matd3"}:
        raise ValueError("algorithm must be masac or matd3")
    seed = int(CONFIG["seed"] if seed is None else seed)
    episodes = int(
        CONFIG.get("postprocess_cpu_eval_episodes", 1)
        if episodes is None
        else episodes
    )
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    wandb_module = None
    if log_wandb:
        if (
            running_on_kaggle()
            and not os.environ.get("WANDB_API_KEY")
        ):
            wandb_status = configure_kaggle_wandb(
                verify_login=True
            )
            if not bool(
                wandb_status.get("configured", False)
            ):
                raise RuntimeError(
                    "CPU post-process requires W&B online access: "
                    + str(wandb_status.get("error"))
                )
        wandb_module = importlib.import_module("wandb")
        wandb_run = wandb_module.init(
            entity=CONFIG["training_wandb_entity"],
            project=CONFIG["training_wandb_project"],
            job_type="cpu_visualization",
            name=f"{algorithm.upper()}-seed{seed}-cpu-viz",
            group=(
                str(source_run_id)
                if source_run_id
                else f"{algorithm}-seed{seed}"
            ),
            mode="online",
            config={
                "algorithm": algorithm,
                "seed": seed,
                "kaggle_account": resolve_kaggle_account(),
                "source_run_id": source_run_id,
                "artifact_ref": artifact_ref,
                "execution_device": "cpu",
                "network_backend_train": str(network_backend),
            },
            settings=wandb_module.Settings(
                x_disable_stats=True,
            ),
        )

    try:
        if artifact_ref is not None:
            if wandb_run is not None:
                artifact = wandb_run.use_artifact(str(artifact_ref))
            else:
                artifact = importlib.import_module("wandb").Api().artifact(
                    str(artifact_ref)
                )
            artifact_dir = Path(artifact.download(root=str(output_dir / "artifact")))
            artifact.verify(str(artifact_dir))
            candidate = artifact_dir / "checkpoint.pt"
            if not candidate.is_file():
                candidates = sorted(artifact_dir.rglob("*.pt"))
                if not candidates:
                    raise FileNotFoundError("artifact contains no .pt checkpoint")
                candidate = candidates[-1]
            checkpoint_path = candidate
        if checkpoint_path is None:
            raise ValueError("checkpoint_path or artifact_ref is required")
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(str(checkpoint_path))

        normalized_backend = str(network_backend).strip().lower()
        if normalized_backend in {"uavnetsim", "uavnetsim_gpu"}:
            eval_backend = "uavnetsim"
            backend_kwargs = {
                "repo_path": ensure_uavnetsim_reference_available(None)
            }
        else:
            eval_backend = "simple"
            backend_kwargs = {}

        env = UAVSearchEnv(
            backend_name=eval_backend,
            network_backend_kwargs=backend_kwargs,
        )
        try:
            if algorithm == "masac":
                loaded = load_masac_checkpoint(
                    checkpoint_path,
                    env,
                    device="cpu",
                    load_replay=False,
                    restore_torch_rng=False,
                )
                trainer = loaded["trainer"]
                rows = evaluate_masac(
                    env,
                    trainer,
                    episodes=episodes,
                    seed=seed + int(CONFIG["training_eval_seed_offset"]),
                )
            else:
                loaded = load_matd3_checkpoint(
                    checkpoint_path,
                    env,
                    device="cpu",
                    load_replay=False,
                    restore_torch_rng=False,
                )
                trainer = loaded["trainer"]
                rows = evaluate_matd3(
                    env,
                    trainer,
                    episodes=episodes,
                    seed=seed + int(CONFIG["training_eval_seed_offset"]),
                )
            actor_devices = {
                str(parameter.device)
                for parameter in trainer.actor.parameters()
            }
            if actor_devices != {"cpu"}:
                raise RuntimeError(
                    f"CPU post-process loaded actor on {actor_devices}"
                )
        finally:
            env.close()

        summary = summarize_evaluation_results(rows)
        trace = rows[0].get("visualization_trace")
        if trace is None:
            raise RuntimeError("evaluation did not produce visualization_trace")
        fig = render_evaluation_trace_2d(
            trace,
            title=f"{algorithm.upper()} CPU evaluation | seed={seed}",
        )
        image_path = output_dir / f"{algorithm}_cpu_evaluation.png"
        fig.savefig(image_path, dpi=140, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)

        video_path = None
        if render_video:
            video_path = output_dir / f"{algorithm}_cpu_evaluation.mp4"
            render_evaluation_video_2d(
                trace,
                video_path,
                fps=int(CONFIG.get("postprocess_cpu_video_fps", 8)),
            )

        import hashlib
        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        training_state = loaded.get("training_state", {})
        manifest = {
            "algorithm": algorithm,
            "seed": seed,
            "source_run_id": source_run_id,
            "source_artifact": artifact_ref,
            "checkpoint_sha256": digest.hexdigest(),
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "training_global_step": training_state.get("global_step"),
            "training_episode_index": training_state.get("episode_index"),
            "evaluation_backend": eval_backend,
            "evaluation_summary": {
                key: value
                for key, value in summary.items()
                if isinstance(value, (bool, int, float, str))
            },
        }
        manifest_path = output_dir / "checkpoint_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, default=str)
        )

        hf_status = {"uploaded": False, "reason": "disabled"}
        if upload_huggingface is None:
            upload_huggingface = bool(
                CONFIG.get("postprocess_cpu_upload_huggingface", True)
            )
        if upload_huggingface:
            hf_cfg = configure_kaggle_huggingface()
            if hf_cfg.get("configured"):
                from huggingface_hub import HfApi
                api = HfApi(token=os.environ["HF_TOKEN"])
                repo_id = str(hf_cfg["repo_id"])
                api.create_repo(
                    repo_id=repo_id,
                    repo_type="model",
                    private=bool(CONFIG.get("postprocess_hf_private_repo", True)),
                    exist_ok=True,
                )
                episode_index = training_state.get("episode_index", "unknown")
                prefix = f"checkpoints/{algorithm}/seed-{seed}/episode-{episode_index}"
                api.upload_file(
                    path_or_fileobj=str(checkpoint_path),
                    path_in_repo=f"{prefix}/checkpoint.pt",
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Upload {algorithm} seed {seed} checkpoint",
                )
                api.upload_file(
                    path_or_fileobj=str(manifest_path),
                    path_in_repo=f"{prefix}/checkpoint_manifest.json",
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Add {algorithm} seed {seed} checkpoint manifest",
                )
                hf_status = {
                    "uploaded": True,
                    "repo_id": repo_id,
                    "path_prefix": prefix,
                }
            else:
                hf_status = {
                    "uploaded": False,
                    "reason": hf_cfg.get("error"),
                }

        if wandb_run is not None:
            payload = {
                f"evaluation_cpu/{key}": float(value)
                for key, value in summary.items()
                if isinstance(value, (bool, int, float))
            }
            payload["visualization/evaluation_2d"] = wandb_module.Image(
                str(image_path)
            )
            if video_path is not None:
                payload["visualization/post_train_eval_video"] = wandb_module.Video(
                    str(video_path),
                    format="mp4",
                )
            wandb_run.log(payload)
            wandb_run.summary["cpu_postprocess/checkpoint_sha256"] = digest.hexdigest()
            wandb_run.summary["cpu_postprocess/huggingface_uploaded"] = bool(
                hf_status.get("uploaded", False)
            )

        return {
            "device": "cpu",
            "evaluation_backend": str(eval_backend),
            "checkpoint_path": str(checkpoint_path),
            "image_path": str(image_path),
            "video_path": None if video_path is None else str(video_path),
            "summary": summary,
            "manifest_path": str(manifest_path),
            "huggingface": hf_status,
            "wandb_run_url": None if wandb_run is None else wandb_run.url,
        }
    finally:
        if wandb_run is not None:
            wandb_run.finish()


# --- frozen notebook cell 298 ---
def finalize_training_visualization(
    training_result,
    repo,
    algorithm,
    network_backend,
    seed,
):
    """Mandatory full-flow CPU visualization after training completes."""
    if not bool(
        CONFIG.get(
            "training_auto_postprocess_required",
            True,
        )
    ) or not bool(
        CONFIG.get(
            "training_auto_postprocess_visualization",
            True,
        )
    ):
        raise RuntimeError(
            "Full-flow visualization is mandatory and cannot be disabled."
        )

    result = dict(training_result)
    checkpoint_path = result.get("final_checkpoint_path")
    if not checkpoint_path:
        paths = list(result.get("checkpoint_paths") or [])
        if paths:
            checkpoint_path = paths[-1]
    if not checkpoint_path:
        raise RuntimeError(
            "Training completed without a final checkpoint; visualization cannot run."
        )

    run_token = str(
        result.get("wandb_run_id")
        or "no-wandb-id"
    )
    output_root = (
        Path(repo)
        / str(
            CONFIG.get(
                "training_auto_postprocess_output_dir",
                "post_train_visualization",
            )
        )
        / (
            f"{str(algorithm).lower()}-seed{int(seed)}-{run_token}"
        )
    )

    visualization = postprocess_checkpoint_cpu(
        checkpoint_path=checkpoint_path,
        artifact_ref=None,
        algorithm=algorithm,
        network_backend=network_backend,
        seed=int(seed),
        episodes=int(CONFIG.get("postprocess_cpu_eval_episodes", 1)),
        output_dir=output_root,
        source_run_id=result.get("wandb_run_id"),
        log_wandb=bool(
            CONFIG.get(
                "training_auto_postprocess_log_wandb_media",
                False,
            )
        ),
        upload_huggingface=bool(
            CONFIG.get(
                "training_auto_postprocess_upload_huggingface",
                False,
            )
        ),
        render_video=bool(
            CONFIG.get(
                "training_auto_postprocess_render_video",
                True,
            )
        ),
    )

    image_path = Path(visualization["image_path"])
    video_value = visualization.get("video_path")
    video_path = Path(video_value) if video_value else None
    if not image_path.is_file() or image_path.stat().st_size <= 0:
        raise RuntimeError(
            f"Post-train visualization PNG was not created: {image_path}"
        )
    if bool(CONFIG.get("training_auto_postprocess_render_video", True)):
        if (
            video_path is None
            or not video_path.is_file()
            or video_path.stat().st_size <= 0
        ):
            raise RuntimeError(
                f"Post-train visualization MP4 was not created: {video_path}"
            )

    result["visualization_complete"] = True
    result["visualization_image_path"] = str(image_path)
    result["visualization_video_path"] = (
        None if video_path is None else str(video_path)
    )
    result["post_train_visualization"] = visualization
    print(
        "POST_TRAIN_VISUALIZATION_COMPLETE "
        f"image={result['visualization_image_path']} "
        f"video={result['visualization_video_path']}",
        flush=True,
    )
    return result


# Export public and internal helpers required by downstream production layers.
__all__ = [name for name in globals() if not name.startswith("__")]
