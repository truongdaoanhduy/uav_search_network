from pathlib import Path

import torch

import uav_marl.training.gpu as training

ROOT = Path(__file__).resolve().parents[1]


def _patch_runtime(monkeypatch, devices):
    calls = []

    def fake_ddp(**kwargs):
        calls.append(("distributed", kwargs))
        return {
            "wandb_run_id": "fake",
            "final_checkpoint_path": "/tmp/fake.pt",
        }

    def fake_direct(**kwargs):
        calls.append(("direct", kwargs))
        return {
            "wandb_run_id": "fake",
            "final_checkpoint_path": "/tmp/fake.pt",
        }

    def fake_sharded(**kwargs):
        calls.append(("sharded", kwargs))
        return {
            "wandb_run_id": "fake",
            "final_checkpoint_path": "/tmp/fake.pt",
        }

    def fake_finalize(result, **kwargs):
        result = dict(result)
        result["visualization_complete"] = True
        return result

    monkeypatch.setattr(training, "resolve_full_execution_mode", lambda mode: "gpu")
    monkeypatch.setattr(training, "resolve_kaggle_account", lambda: "None")
    monkeypatch.setattr(
        training,
        "resolve_full_gpu_devices",
        lambda **kwargs: devices,
    )
    monkeypatch.setattr(training, "train_full_gpu_ddp", fake_ddp)
    monkeypatch.setattr(training, "train_full_gpu", fake_direct)
    monkeypatch.setattr(training, "train_full_gpu_sharded", fake_sharded)

    # same_session imports this function lazily from evaluation.postprocess.
    from uav_marl.evaluation import postprocess

    monkeypatch.setattr(
        postprocess,
        "finalize_training_visualization",
        fake_finalize,
    )

    for key, value in {
        "full_gpu_auto_multi_gpu": True,
        "training_wandb_require_online_gpu": True,
        "training_wandb_mode": "online",
        "training_postprocess_mode": "separate_cpu",
        "full_gpu_multi_gpu_strategy": "ddp",
        "masac_train_freq": 1,
        "masac_gradient_steps": 1,
        "masac_updates_per_step": 1,
        "matd3_train_freq": 1,
        "matd3_gradient_steps": 1,
        "matd3_updates_per_step": 1,
    }.items():
        monkeypatch.setitem(training.CONFIG, key, value)

    return calls


def test_one_gpu_episode_budget_uses_full_metrics_distributed_worker(monkeypatch):
    calls = _patch_runtime(monkeypatch, [torch.device("cuda:0")])
    out = training.train_full_gpu_auto(
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
    assert out["visualization_complete"] is False
    assert out["visualization_pending"] is True
    assert out["postprocess_mode"] == "separate_cpu"
    assert len(calls) == 1
    kind, kwargs = calls[0]
    assert kind == "distributed"
    assert kwargs["world_size"] == 1
    assert kwargs["selected_devices"] == [torch.device("cuda:0")]
    assert kwargs["target_episodes"] == 50_000


def test_two_gpu_episode_budget_uses_world_size_two(monkeypatch):
    calls = _patch_runtime(
        monkeypatch,
        [torch.device("cuda:0"), torch.device("cuda:1")],
    )
    out = training.train_full_gpu_auto(
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
    assert out["visualization_complete"] is False
    assert out["visualization_pending"] is True
    assert out["postprocess_mode"] == "separate_cpu"
    assert len(calls) == 1
    kind, kwargs = calls[0]
    assert kind == "distributed"
    assert kwargs["world_size"] == 2
    assert kwargs["selected_devices"] == [
        torch.device("cuda:0"),
        torch.device("cuda:1"),
    ]


def test_same_session_mode_uses_lazy_python_evaluation(monkeypatch):
    calls = _patch_runtime(monkeypatch, [torch.device("cuda:0")])
    monkeypatch.setitem(
        training.CONFIG,
        "training_postprocess_mode",
        "same_session",
    )
    out = training.train_full_gpu_auto(
        repo=ROOT,
        algorithm="masac",
        num_envs=32,
        total_transitions=128,
        seed=44,
        network_backend="uavnetsim_gpu",
        max_gpus=1,
        auto_multi_gpu=False,
        enable_wandb=True,
        execution_mode="gpu",
        target_episodes=4,
        device="cuda:0",
    )
    assert out["visualization_complete"] is True
    assert len(calls) == 1
