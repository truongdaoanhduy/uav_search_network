from pathlib import Path

import pytest

from uav_marl.algorithms.masac import evaluate_masac, load_masac_checkpoint
from uav_marl.algorithms.matd3 import evaluate_matd3, load_matd3_checkpoint
from uav_marl.envs.uav_search import UAVSearchEnv
from uav_marl.training.gpu import CONFIG, train_full_gpu


ROOT = Path(__file__).resolve().parents[1]


def test_train_full_gpu_cpu_initializes_environment_before_action_selection(monkeypatch, tmp_path):
    monkeypatch.setitem(CONFIG, "masac_learning_starts", 2000)
    monkeypatch.setitem(CONFIG, "masac_batch_size", 4)
    monkeypatch.setitem(CONFIG, "masac_replay_capacity", 64)
    monkeypatch.setitem(CONFIG, "training_inline_evaluation", False)
    monkeypatch.setitem(CONFIG, "training_checkpoint_interval_episodes", 1000)

    result = train_full_gpu(
        repo=tmp_path,
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


@pytest.mark.parametrize("algorithm", ["masac", "matd3"])
def test_train_full_gpu_cpu_runs_optimizer_and_checkpoint_for_both_algorithms(
    monkeypatch, tmp_path, algorithm
):
    monkeypatch.setitem(CONFIG, f"{algorithm}_hidden_dims", (32, 32))
    monkeypatch.setitem(CONFIG, f"{algorithm}_learning_starts", 1)
    monkeypatch.setitem(CONFIG, f"{algorithm}_batch_size", 4)
    monkeypatch.setitem(CONFIG, f"{algorithm}_replay_capacity", 64)
    monkeypatch.setitem(CONFIG, f"{algorithm}_train_freq", 1)
    monkeypatch.setitem(CONFIG, f"{algorithm}_gradient_steps", 1)
    monkeypatch.setitem(CONFIG, "training_auto_scale_gradient_steps", False)
    monkeypatch.setitem(CONFIG, "training_inline_evaluation", False)
    monkeypatch.setitem(CONFIG, "training_checkpoint_interval_episodes", 1000)
    monkeypatch.setitem(CONFIG, "max_steps", 5)

    result = train_full_gpu(
        repo=tmp_path,
        algorithm=algorithm,
        num_envs=1,
        total_transitions=8,
        seed=44,
        device="cpu",
        network_backend="simple",
        strict_cuda=False,
        enable_wandb=False,
        enable_evaluation=False,
        target_episodes=None,
    )

    assert result["algorithm"] == algorithm
    assert result["transitions"] == 8
    assert result["updates"] > 0
    assert result["latest_update_metrics"] is not None
    checkpoint = Path(result["final_checkpoint_path"])
    assert checkpoint.is_file()
    assert checkpoint.stat().st_size > 0

    env = UAVSearchEnv(backend_name="simple")
    try:
        if algorithm == "masac":
            loaded = load_masac_checkpoint(
                checkpoint,
                env,
                device="cpu",
                load_replay=False,
                restore_torch_rng=False,
            )
            rows = evaluate_masac(env, loaded["trainer"], episodes=1, seed=10044)
        else:
            loaded = load_matd3_checkpoint(
                checkpoint,
                env,
                device="cpu",
                load_replay=False,
                restore_torch_rng=False,
            )
            rows = evaluate_matd3(env, loaded["trainer"], episodes=1, seed=10044)
        assert len(rows) == 1
        assert rows[0]["length"] == 5
        assert rows[0]["terminated"] is True
        assert rows[0]["truncated"] is False
    finally:
        env.close()


def test_periodic_single_device_checkpoint_is_uploaded_before_finish(monkeypatch, tmp_path):
    import uav_marl.training.gpu as training

    class RunConfig(dict):
        def update(self, *args, **kwargs):
            kwargs.pop("allow_val_change", None)
            return super().update(*args, **kwargs)

    class Run:
        id = "checkpoint-test"
        name = "checkpoint-test"
        url = "https://example.invalid/checkpoint-test"
        config = RunConfig()
        summary = {}
        uploaded = []
        finished = False

        def log(self, payload):
            pass

        def save(self, path, *, base_path, policy):
            assert not self.finished
            assert Path(path).is_file()
            assert Path(path).is_relative_to(base_path)
            assert policy == "now"
            self.uploaded.append(path)

        def finish(self):
            self.finished = True

    run = Run()
    monkeypatch.setattr(training, "_init_gpu_wandb_run", lambda *args, **kwargs: run)
    for key, value in {
        "masac_hidden_dims": (16, 16),
        "masac_learning_starts": 2000,
        "masac_batch_size": 4,
        "masac_replay_capacity": 64,
        "training_inline_evaluation": False,
        "training_checkpoint_interval_episodes": 1,
        "training_upload_periodic_checkpoints_wandb": True,
        "training_publish_final_checkpoint_wandb_artifact": False,
        "max_steps": 2,
    }.items():
        monkeypatch.setitem(CONFIG, key, value)

    result = train_full_gpu(
        repo=tmp_path, algorithm="masac", num_envs=1,
        total_transitions=2, seed=44, device="cpu",
        network_backend="simple", strict_cuda=False,
        enable_wandb=True, enable_evaluation=False, target_episodes=1,
    )
    assert result["episodes_completed"] == 1
    assert len(run.uploaded) == 1
    assert "episode_00000001" in Path(run.uploaded[0]).name
