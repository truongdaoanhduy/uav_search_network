from pathlib import Path

from uav_marl.training.gpu import CONFIG, train_full_gpu


ROOT = Path(__file__).resolve().parents[1]


def test_train_full_gpu_cpu_initializes_environment_before_action_selection(monkeypatch):
    monkeypatch.setitem(CONFIG, "masac_learning_starts", 2000)
    monkeypatch.setitem(CONFIG, "masac_batch_size", 4)
    monkeypatch.setitem(CONFIG, "masac_replay_capacity", 64)
    monkeypatch.setitem(CONFIG, "training_inline_evaluation", False)
    monkeypatch.setitem(CONFIG, "training_checkpoint_interval_episodes", 1000)

    result = train_full_gpu(
        repo=ROOT,
        algorithm="masac",
        num_envs=1,
        total_transitions=1,
        seed=44,
        device="cpu",
        network_backend="simple",
        strict_cuda=False,
        enable_wandb=False,
        target_episodes=None,
    )

    assert result["transitions"] == 1
