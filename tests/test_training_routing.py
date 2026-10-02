from pathlib import Path

import nbformat
import torch

ROOT = Path(__file__).resolve().parents[1]


def _train_auto_source():
    nb = nbformat.read(ROOT / "fix_test_gpu.ipynb", as_version=4)
    hits = [
        cell.source
        for cell in nb.cells
        if cell.cell_type == "code"
        and cell.source.lstrip().startswith("def train_full_gpu_auto(")
    ]
    assert len(hits) == 1
    return hits[0]


def _namespace(devices):
    calls = []

    def train_full_gpu_ddp(**kwargs):
        calls.append(("distributed", kwargs))
        return {
            "wandb_run_id": "fake",
            "final_checkpoint_path": "/tmp/fake.pt",
        }

    def train_full_gpu(**kwargs):
        calls.append(("direct", kwargs))
        return {
            "wandb_run_id": "fake",
            "final_checkpoint_path": "/tmp/fake.pt",
        }

    def train_full_gpu_sharded(**kwargs):
        calls.append(("sharded", kwargs))
        return {
            "wandb_run_id": "fake",
            "final_checkpoint_path": "/tmp/fake.pt",
        }

    def finalize_training_visualization(result, **kwargs):
        result = dict(result)
        result["visualization_complete"] = True
        return result

    ns = {
        "Path": Path,
        "CONFIG": {
            "full_gpu_auto_multi_gpu": True,
            "training_wandb_require_online_gpu": True,
            "training_wandb_mode": "online",
            "full_gpu_multi_gpu_strategy": "ddp",
            "masac_train_freq": 1,
            "masac_gradient_steps": 1,
            "masac_updates_per_step": 1,
            "matd3_train_freq": 1,
            "matd3_gradient_steps": 1,
            "matd3_updates_per_step": 1,
        },
        "resolve_full_execution_mode": lambda mode: "gpu",
        "resolve_kaggle_account": lambda: "None",
        "resolve_full_gpu_devices": lambda **kwargs: devices,
        "train_full_gpu_ddp": train_full_gpu_ddp,
        "train_full_gpu": train_full_gpu,
        "train_full_gpu_sharded": train_full_gpu_sharded,
        "finalize_training_visualization": finalize_training_visualization,
    }
    exec(_train_auto_source(), ns)  # noqa: S102 - isolated test namespace.
    return ns, calls


def test_one_gpu_episode_budget_uses_full_metrics_distributed_worker():
    ns, calls = _namespace([torch.device("cuda:0")])
    out = ns["train_full_gpu_auto"](
        repo=ROOT,
        algorithm="masac",
        num_envs=2048,
        total_transitions=51_200_000,
        seed=44,
        network_backend="uavnetsim_gpu",
        max_gpus=1,
        auto_multi_gpu=False,
        enable_wandb=True,
        execution_mode="gpu",
        target_episodes=50_000,
        device="cuda:0",
    )
    assert out["visualization_complete"] is True
    assert len(calls) == 1
    kind, kwargs = calls[0]
    assert kind == "distributed"
    assert kwargs["world_size"] == 1
    assert kwargs["selected_devices"] == [torch.device("cuda:0")]
    assert kwargs["target_episodes"] == 50_000


def test_two_gpu_episode_budget_uses_world_size_two():
    ns, calls = _namespace(
        [torch.device("cuda:0"), torch.device("cuda:1")]
    )
    out = ns["train_full_gpu_auto"](
        repo=ROOT,
        algorithm="masac",
        num_envs=4096,
        total_transitions=53_248_000,
        seed=44,
        network_backend="uavnetsim_gpu",
        max_gpus=2,
        auto_multi_gpu=True,
        enable_wandb=True,
        execution_mode="gpu",
        target_episodes=50_000,
        device="cuda:0",
    )
    assert out["visualization_complete"] is True
    assert len(calls) == 1
    kind, kwargs = calls[0]
    assert kind == "distributed"
    assert kwargs["world_size"] == 2
    assert kwargs["selected_devices"] == [
        torch.device("cuda:0"),
        torch.device("cuda:1"),
    ]
