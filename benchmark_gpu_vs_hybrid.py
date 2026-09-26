"""Fair benchmark: current hybrid reference vs GPU-native torch environment."""

import argparse
import json
import runpy
import subprocess
import tempfile
import time
from pathlib import Path

import torch

from test_gpu import run_gpu_training


def load_hybrid_namespace():
    tmp = Path(tempfile.gettempdir()) / "uav_hybrid_benchmark_export.py"
    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "script",
            "test.ipynb",
            "--stdout",
        ],
        check=True,
        stdout=tmp.open("w"),
    )
    return runpy.run_path(str(tmp))


def benchmark_hybrid(
    ns,
    algorithm,
    transitions,
    seed,
    learning_starts,
    batch_size,
    train,
):
    cfg = ns["CONFIG"]
    old = dict(cfg)
    cfg.update(
        {
            "training_vector_backend": "async_cpu",
            "training_cuda_num_envs": 4,
            "training_torch_sensing_enabled": False,
            "training_overlap_env_and_updates": True,
            "training_eval_interval_steps": 10**9,
            "training_checkpoint_interval_steps": 10**9,
            "training_enable_csv": False,
            "training_enable_tensorboard": False,
            "training_enable_wandb": False,
            "masac_learning_starts": learning_starts if train else 10**9,
            "matd3_learning_starts": learning_starts if train else 10**9,
            "masac_batch_size": batch_size,
            "matd3_batch_size": batch_size,
            "masac_replay_capacity": max(4096, transitions + 32),
            "matd3_replay_capacity": max(4096, transitions + 32),
        }
    )
    try:
        run_dir = Path(tempfile.mkdtemp(prefix=f"hybrid-{algorithm}-"))
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = ns["run_training_experiment"](
            algorithm,
            total_steps=transitions,
            seed=seed,
            backend_name="simple",
            device="cuda:0",
            run_dir=run_dir,
            run_name=f"hybrid-{algorithm}",
            enable_csv=False,
            enable_tensorboard=False,
            enable_wandb=False,
        )
        torch.cuda.synchronize()
        outer_wall = time.perf_counter() - start
        metrics = result.get("system_metrics", {})
        return {
            "implementation": "hybrid_async_cpu_cuda_learner",
            "algorithm": algorithm,
            "num_envs": 4,
            "transitions": transitions,
            "updates": int(result["trainer"].update_count),
            "core_transitions_per_second": metrics.get(
                "session_transitions_per_second",
                metrics.get("transitions_per_second"),
            ),
            "outer_wall_seconds": outer_wall,
            "outer_transitions_per_second": transitions / outer_wall,
            "gpu_peak_memory_allocated_mb": (
                torch.cuda.max_memory_allocated() / 1024**2
            ),
        }
    finally:
        cfg.clear()
        cfg.update(old)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transitions", type=int, default=512)
    parser.add_argument("--learning-starts", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument(
        "--gpu-envs",
        type=int,
        nargs="+",
        default=[4, 16, 32],
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This comparison requires a CUDA GPU")

    ns = load_hybrid_namespace()
    results = []
    for algorithm in ("masac", "matd3"):
        results.append(
            benchmark_hybrid(
                ns,
                algorithm,
                args.transitions,
                args.seed,
                args.learning_starts,
                args.batch_size,
                True,
            )
        )
        for num_envs in args.gpu_envs:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            result = run_gpu_training(
                algorithm,
                args.transitions,
                num_envs,
                "cuda:0",
                seed=args.seed,
                learning_starts=args.learning_starts,
                batch_size=args.batch_size,
                replay_capacity=max(4096, args.transitions + 32),
                updates_per_transition=1,
                train=True,
            )
            results.append(result)

    # Environment/policy collection only: no learner updates.
    for algorithm in ("masac", "matd3"):
        results.append(
            benchmark_hybrid(
                ns,
                algorithm,
                800,
                args.seed,
                10**9,
                args.batch_size,
                False,
            )
        )
        for num_envs in args.gpu_envs:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            results.append(
                run_gpu_training(
                    algorithm,
                    800,
                    num_envs,
                    "cuda:0",
                    seed=args.seed,
                    learning_starts=10**9,
                    batch_size=args.batch_size,
                    replay_capacity=1024,
                    train=False,
                )
            )

    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "seed": args.seed,
        "training_transitions": args.transitions,
        "learning_starts": args.learning_starts,
        "batch_size": args.batch_size,
        "results": results,
    }
    Path("gpu_vs_hybrid_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )
    print("GPU_VS_HYBRID_RESULT", json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
