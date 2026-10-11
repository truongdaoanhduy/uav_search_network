"""Bounded, two-GPU runtime check; passing does not establish learning quality."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _check_replay_roundtrip(device):
    import torch
    from uav_marl.envs.gpu import GpuReplayBuffer
    replay = GpuReplayBuffer(8, 1, 1, 1, 1, 2, "cpu", strict_cuda=False)
    ids = torch.arange(8, device=device, dtype=torch.float32) + 0.1234567
    obs = ids.view(8, 1, 1)
    state = ids.view(8, 1)
    mask = torch.ones(8, 1, 2, device=device)
    replay.add_batch(
        obs, state + 10, obs + 20, (ids.long() % 2).view(8, 1),
        mask, ids + 30, obs + 100, state + 110, mask,
        torch.zeros(8, device=device),
    )
    batch = replay.sample(8, device)
    assert all(value.dtype == (torch.long if name == "destination_indices" else torch.float32)
               for name, value in batch.items())
    sampled_ids = batch["observations"].flatten()
    assert bool(torch.isin(sampled_ids, ids).all())
    assert torch.equal(batch["states"].flatten(), sampled_ids + 10)
    assert torch.equal(batch["continuous_actions"].flatten(), sampled_ids + 20)
    assert torch.equal(batch["destination_indices"].flatten(), sampled_ids.long() % 2)
    assert torch.equal(batch["rewards"].flatten(), sampled_ids + 30)
    assert torch.equal(batch["next_observations"].flatten(), sampled_ids + 100)
    assert torch.equal(batch["next_states"].flatten(), sampled_ids + 110)
    assert all(value.device == torch.device(device) for value in batch.values())
    del replay, batch


def main():
    from hydra import compose, initialize_config_dir
    from uav_marl.configuration import apply_to_legacy_config, validate_config
    from config import CONFIG

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="config")
    parser.add_argument("--vector-steps", type=int, default=104)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name=args.config_name)
    validate_config(cfg)
    assert cfg.algorithm.name == "masac" and cfg.runtime.max_gpus == 2
    transitions = int(cfg.runtime.num_envs) * args.vector_steps
    if transitions <= max(cfg.algorithm.learning_starts, cfg.algorithm.replay_capacity):
        raise ValueError("Smoke must allocate the full replay and reach optimizer updates.")
    apply_to_legacy_config(CONFIG, cfg)
    # Runtime test only: no paper run, inline evaluation, video, or online artifacts.
    CONFIG["training_enable_wandb"] = False
    CONFIG["training_inline_evaluation"] = False
    CONFIG["training_inline_visualization"] = False

    # Keep the production import order: apply resolved configuration first.
    import torch
    from uav_marl.training.gpu import train_full_gpu_ddp

    if torch.cuda.device_count() < 2:
        raise RuntimeError("This smoke requires two visible CUDA GPUs.")
    for rank in range(2):
        _check_replay_roundtrip(f"cuda:{rank}")

    # Host observations; /proc/meminfo need not reflect a container RAM limit.
    import threading
    import resource
    def host_memory():
        fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        return {key: int(fields[key].split()[0]) / 1024 for key in ("MemTotal", "MemAvailable")}
    host_before = host_memory()
    lowest_available = [host_before["MemAvailable"]]
    stop = threading.Event()
    def monitor_host_memory():
        while not stop.wait(0.5):
            lowest_available[0] = min(lowest_available[0], host_memory()["MemAvailable"])
    monitor = threading.Thread(target=monitor_host_memory, daemon=True)
    monitor.start()
    try:
        result = train_full_gpu_ddp(
            repo=ROOT, algorithm="masac", num_envs=int(cfg.runtime.num_envs),
            total_transitions=transitions, seed=int(cfg.seed),
            network_backend=str(cfg.runtime.network_backend),
            enable_wandb=False, target_episodes=None, world_size=2,
        )
    finally:
        stop.set()
        monitor.join()
    assert result["used_gpu_count"] == 2
    assert result["transitions"] >= transitions
    assert result["updates"] > 0
    assert result["checkpoint_paths"]
    assert all(Path(path).is_file() and Path(path).stat().st_size > 0
               for path in result["checkpoint_paths"])
    metrics = result["latest_update_metrics"]
    assert metrics and all(math.isfinite(float(value)) for value in metrics.values())
    result["validation_scope"] = "GPU allocation and finite optimizer updates; no convergence claim"
    result["cross_device_replay_roundtrip_passed"] = True
    result["replay_storage_device"] = str(cfg.runtime.offpolicy.replay_storage_device)
    result["ram_measurement_scope"] = "Host /proc/meminfo and largest individual child RSS; not container headroom"
    result["host_total_mb"] = host_before["MemTotal"]
    result["host_min_available_mb"] = lowest_available[0]
    result["largest_child_peak_rss_mb"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    result["configured_replay_capacity"] = int(cfg.algorithm.replay_capacity)
    result["configured_learning_starts"] = int(cfg.algorithm.learning_starts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
