from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.kaggle_pipeline import _push_kernel, build_kernels
from uav_marl.handoff import (
    CHECKPOINT_FILENAME,
    HANDOFF_FILENAME,
    create_training_handoff,
    load_training_handoff,
    run_cpu_postprocess_from_handoff,
)
from uav_marl.training.gpu import _train_full_gpu_ddp_worker

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = str((ROOT / "configs").resolve())


def _cfg():
    with initialize_config_dir(
        version_base="1.3",
        config_dir=CONFIG_DIR,
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "algorithm=masac",
                "runtime=local_debug",
                "experiment=smoke",
            ],
        )
    return OmegaConf.to_container(cfg, resolve=True)


def test_training_handoff_copies_and_verifies_checkpoint(tmp_path):
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"model-checkpoint-bytes")

    result = create_training_handoff(
        {
            "final_checkpoint_path": str(checkpoint),
            "wandb_run_id": "abc123",
            "episodes_completed": 10,
            "global_step": 42,
            "resolved_config_sha256": "cfg-hash",
        },
        _cfg(),
        repo=tmp_path,
    )
    assert result["visualization_complete"] is False
    assert result["visualization_pending"] is True
    assert result["postprocess_mode"] == "separate_cpu"

    handoff_path = Path(result["handoff_path"])
    copied_checkpoint = Path(result["handoff_checkpoint_path"])
    assert handoff_path.name == HANDOFF_FILENAME
    assert copied_checkpoint.name == CHECKPOINT_FILENAME
    assert copied_checkpoint.read_bytes() == checkpoint.read_bytes()

    payload, loaded_checkpoint = load_training_handoff(handoff_path)
    assert payload["algorithm"] == "masac"
    assert payload["seed"] == 44
    assert loaded_checkpoint == copied_checkpoint


def test_training_handoff_detects_checkpoint_tampering(tmp_path):
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"good")

    result = create_training_handoff(
        {"final_checkpoint_path": str(checkpoint)},
        _cfg(),
        repo=tmp_path,
    )
    copied_checkpoint = Path(result["handoff_checkpoint_path"])
    copied_checkpoint.write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="size mismatch|SHA256 mismatch"):
        load_training_handoff(result["handoff_path"])


def test_kaggle_pipeline_builds_gpu_then_cpu_kernel(tmp_path):
    gpu_dir, cpu_dir, gpu_ref, cpu_ref = build_kernels(
        tmp_path,
        username="demo-user",
        commit="a" * 40,
        algorithm="masac",
        runtime="kaggle_2xt4",
        experiment="smoke",
        seed=44,
        gpu_kernel_slug="uav-marl-gpu-train",
        cpu_kernel_slug="uav-marl-cpu-visualize",
        machine_shape="NvidiaTeslaT4",
        overrides=["experiment.total_episodes=10"],
    )

    gpu_meta = json.loads((gpu_dir / "kernel-metadata.json").read_text())
    cpu_meta = json.loads((cpu_dir / "kernel-metadata.json").read_text())

    assert gpu_ref == "demo-user/uav-marl-gpu-train"
    assert cpu_ref == "demo-user/uav-marl-cpu-visualize"
    assert gpu_meta["enable_gpu"] is True
    assert gpu_meta["machine_shape"] == "NvidiaTeslaT4"
    assert gpu_meta["kernel_sources"] == []

    assert cpu_meta["enable_gpu"] is False
    assert cpu_meta["machine_shape"] == ""
    assert cpu_meta["kernel_sources"] == [gpu_ref]

    gpu_script = next(gpu_dir.glob("*.py")).read_text()
    cpu_script = next(cpu_dir.glob("*.py")).read_text()
    compile(gpu_script, "<gpu-kernel>", "exec")
    compile(cpu_script, "<cpu-kernel>", "exec")

    assert "GPU_TRAINING_HANDOFF_READY" in gpu_script
    assert "CPU_VISUALIZATION_NEXT_STAGE.json" in gpu_script
    assert "https://www.kaggle.com/code/demo-user/uav-marl-cpu-visualize" in gpu_script
    assert "torch.cuda.is_available()" in cpu_script
    assert "--handoff" in cpu_script
    assert "/kaggle/input" in cpu_script


def test_default_experiment_uses_separate_cpu_session():
    cfg = _cfg()
    assert cfg["experiment"]["visualization"]["session"] == "separate_cpu"


def test_generated_kernels_keep_repo_checkout_out_of_kaggle_working(tmp_path):
    gpu_dir, cpu_dir, _, _ = build_kernels(
        tmp_path,
        username="demo-user",
        commit="b" * 40,
        algorithm="masac",
        runtime="kaggle_2xt4",
        experiment="smoke",
        seed=44,
        gpu_kernel_slug="gpu-clean-output",
        cpu_kernel_slug="cpu-clean-output",
        machine_shape="NvidiaTeslaT4",
        overrides=[],
    )
    gpu_script = next(gpu_dir.glob("*.py")).read_text()
    cpu_script = next(cpu_dir.glob("*.py")).read_text()
    assert 'Path("/tmp/uav_search_network")' in gpu_script
    assert 'Path("/tmp/uav_search_network")' in cpu_script
    assert 'Path("/kaggle/working/uav_search_network")' not in gpu_script
    assert 'Path("/kaggle/working/uav_search_network")' not in cpu_script


def test_cpu_completion_manifest_keeps_training_provenance(tmp_path, monkeypatch):
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"checkpoint")

    training = create_training_handoff(
        {
            "final_checkpoint_path": str(checkpoint),
            "wandb_run_id": "run123",
            "wandb_run_url": "https://example.invalid/run123",
        },
        _cfg(),
        repo=tmp_path,
    )

    image = tmp_path / "fake.png"
    video = tmp_path / "fake.mp4"
    manifest = tmp_path / "fake_manifest.json"

    def fake_postprocess_checkpoint_cpu(**kwargs):
        image.write_bytes(b"png")
        video.write_bytes(b"mp4")
        manifest.write_text("{}")
        return {
            "image_path": str(image),
            "video_path": str(video),
            "manifest_path": str(manifest),
            "wandb_run_url": kwargs.get("source_run_id"),
        }

    from uav_marl import evaluation

    monkeypatch.setattr(
        evaluation,
        "postprocess_checkpoint_cpu",
        fake_postprocess_checkpoint_cpu,
    )

    output_dir = tmp_path / "postprocess"
    result = run_cpu_postprocess_from_handoff(
        training["handoff_path"],
        repo=tmp_path,
        output_dir=output_dir,
        log_wandb=False,
        upload_huggingface=False,
    )
    completion = json.loads(
        (output_dir / "cpu_visualization_complete.json").read_text()
    )

    assert result["algorithm"] == "masac"
    assert completion["algorithm"] == "masac"
    assert completion["seed"] == 44
    assert completion["network_backend"] == "simple"
    assert completion["experiment"] == "smoke"
    assert completion["source_provider"] == "local"
    assert completion["source_wandb_run_id"] == "run123"
    assert "source_git_commit" in completion

def test_ddp_final_artifact_is_attached_after_result_initialization():
    source = inspect.getsource(_train_full_gpu_ddp_worker)
    result_init = source.index('result = {')
    artifact_attach = source.index('result["final_checkpoint_artifact"]')
    assert result_init < artifact_attach


def test_kaggle_cpu_kernel_can_log_wandb_with_private_credential_dataset(tmp_path):
    credential_ref = "demo-user/private-wandb-credential"
    _gpu_dir, cpu_dir, gpu_ref, cpu_ref = build_kernels(
        tmp_path,
        username="demo-user",
        commit="c" * 40,
        algorithm="masac",
        runtime="kaggle_2xt4",
        experiment="smoke",
        seed=44,
        gpu_kernel_slug="existing-gpu-run",
        cpu_kernel_slug="cpu-authoritative-eval",
        machine_shape="NvidiaTeslaT4",
        overrides=[],
        cpu_log_wandb=True,
        cpu_dataset_sources=[credential_ref],
    )

    cpu_meta = json.loads((cpu_dir / "kernel-metadata.json").read_text())
    cpu_script = next(cpu_dir.glob("*.py")).read_text()

    assert gpu_ref == "demo-user/existing-gpu-run"
    assert cpu_ref == "demo-user/cpu-authoritative-eval"
    assert cpu_meta["enable_gpu"] is False
    assert cpu_meta["kernel_sources"] == [gpu_ref]
    assert cpu_meta["dataset_sources"] == [credential_ref]
    assert "wandb_api_key.txt" in cpu_script
    assert "wandb_secret.json" in cpu_script
    assert 'credential_payload["api_key"]' in cpu_script
    assert 'command.append("--log-wandb")' in cpu_script
    assert 'os.environ["WANDB_API_KEY"] = api_key' in cpu_script
    compile(cpu_script, "<cpu-authoritative-eval>", "exec")


def test_kaggle_cpu_wandb_requires_one_credential_dataset(tmp_path):
    with pytest.raises(ValueError, match="credential dataset"):
        build_kernels(
            tmp_path,
            username="demo-user",
            commit="d" * 40,
            algorithm="masac",
            runtime="kaggle_2xt4",
            experiment="smoke",
            seed=44,
            gpu_kernel_slug="existing-gpu-run",
            cpu_kernel_slug="cpu-authoritative-eval",
            machine_shape="NvidiaTeslaT4",
            overrides=[],
            cpu_log_wandb=True,
            cpu_dataset_sources=[],
        )


def test_cpu_postprocess_uses_current_wandb_stats_setting():
    source = (ROOT / "uav_marl" / "evaluation" / "postprocess.py").read_text()
    assert "x_disable_stats=True" in source
    assert "_disable_stats=True" not in source.replace("x_disable_stats=True", "")


def test_cpu_completion_distinguishes_training_and_evaluation_backends():
    source = (ROOT / "uav_marl" / "handoff.py").read_text()
    assert '"training_network_backend"' in source
    assert '"evaluation_network_backend"' in source
    postprocess_source = (ROOT / "uav_marl" / "evaluation" / "postprocess.py").read_text()
    assert '"evaluation_backend": str(eval_backend)' in postprocess_source


def test_kaggle_gpu_kernel_can_read_private_wandb_credential_dataset(tmp_path):
    credential_ref = "demo-user/private-wandb-credential"
    gpu_dir, _cpu_dir, gpu_ref, _cpu_ref = build_kernels(
        tmp_path,
        username="demo-user",
        commit="e" * 40,
        algorithm="masac",
        runtime="kaggle_2xt4",
        experiment="smoke",
        seed=44,
        gpu_kernel_slug="gpu-wandb-dataset",
        cpu_kernel_slug="cpu-unused",
        machine_shape="NvidiaTeslaT4",
        overrides=[],
        gpu_dataset_sources=[credential_ref],
    )
    gpu_meta = json.loads((gpu_dir / "kernel-metadata.json").read_text())
    gpu_script = next(gpu_dir.glob("*.py")).read_text()
    assert gpu_ref == "demo-user/gpu-wandb-dataset"
    assert gpu_meta["enable_gpu"] is True
    assert gpu_meta["dataset_sources"] == [credential_ref]
    assert "wandb_api_key.txt" in gpu_script
    assert "wandb_secret.json" in gpu_script
    assert 'credential_payload["api_key"]' in gpu_script
    assert 'os.environ["WANDB_API_KEY"] = api_key' in gpu_script
    compile(gpu_script, "<gpu-wandb-dataset>", "exec")


def test_pipeline_supports_cloud_kaggle_token_without_local_key_file():
    source = (ROOT / "scripts" / "kaggle_pipeline.py").read_text()
    assert 'os.environ.get("KAGGLE_API_TOKEN", "").strip()' in source
    assert '"cloud_controller"' in source
    assert 'env["KAGGLE_API_TOKEN"] = cloud_token' in source


def test_github_workflow_summary_links_gpu_and_cpu_kernels():
    source = (ROOT / ".github" / "workflows" / "kaggle_gpu_cpu_pipeline.yml").read_text()
    assert "GITHUB_STEP_SUMMARY" in source
    assert "GPU Kaggle kernel" in source
    assert "CPU visualization kernel" in source
    assert "https://www.kaggle.com/code/${KAGGLE_USERNAME}/${GPU_SLUG}" in source
    assert "https://www.kaggle.com/code/${KAGGLE_USERNAME}/${CPU_SLUG}" in source


def test_kaggle_push_session_timeout_is_separate_from_controller_wait(tmp_path, monkeypatch):
    calls = []

    class Result:
        stdout = "pushed"
        stderr = ""
        returncode = 0

    def fake_run(args, **kwargs):
        calls.append((list(args), dict(kwargs)))
        return Result()

    monkeypatch.setattr("scripts.kaggle_pipeline._run", fake_run)
    _push_kernel(
        tmp_path,
        env={},
        accelerator="NvidiaTeslaT4",
        session_timeout_seconds=43_200,
    )

    command, kwargs = calls[0]
    assert command[0:3] == ["kaggle", "kernels", "push"]
    assert command[command.index("--timeout") + 1] == "43200"
    assert kwargs["timeout"] < 43_200


def test_full_gpu_checkpoint_directory_persists_on_kaggle(tmp_path, monkeypatch):
    import uav_marl.training.gpu as training

    monkeypatch.setattr(training, "running_on_kaggle", lambda: True)
    assert training._full_gpu_checkpoint_dir(tmp_path) == Path(
        "/kaggle/working/checkpoints_full_gpu"
    )


def test_full_gpu_checkpoint_directory_stays_local_off_kaggle(tmp_path, monkeypatch):
    import uav_marl.training.gpu as training

    monkeypatch.setattr(training, "running_on_kaggle", lambda: False)
    assert training._full_gpu_checkpoint_dir(tmp_path) == tmp_path / "checkpoints_full_gpu"


def test_reconciler_only_selects_pipeline_gpu_refs_for_requested_user():
    from scripts.reconcile_kaggle_pipeline import _cpu_ref_for_gpu, _gpu_refs

    refs = {
        "haibro1234/uav-masac-gha-123-1-gpu",
        "haibro1234/uav-matd3-gha-456-2-gpu",
        "haibro1234/unrelated-gpu",
        "someone/uav-masac-gha-789-1-gpu",
    }
    assert _gpu_refs(refs, "haibro1234") == [
        "haibro1234/uav-masac-gha-123-1-gpu",
        "haibro1234/uav-matd3-gha-456-2-gpu",
    ]
    assert _cpu_ref_for_gpu("haibro1234/uav-masac-gha-123-1-gpu") == (
        "haibro1234/uav-masac-gha-123-1-cpu"
    )


def test_reconciler_does_not_duplicate_active_gpu_stage(tmp_path, monkeypatch):
    from scripts import reconcile_kaggle_pipeline as reconcile

    monkeypatch.setattr(reconcile, "_kernel_status", lambda ref, env: "running")
    monkeypatch.setattr(
        reconcile,
        "_kernel_source_commit",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not pull active GPU")),
    )
    result = reconcile.reconcile_one(
        "haibro1234/uav-masac-gha-123-1-gpu",
        refs={"haibro1234/uav-masac-gha-123-1-gpu"},
        env={},
        temp_root=tmp_path,
        cpu_credential_dataset=None,
        cpu_log_wandb=False,
        cpu_session_timeout_seconds=10_800,
    )
    assert result == "gpu_active"


def test_training_handoff_preserves_architecture_metadata(tmp_path):
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"checkpoint")
    cfg = _cfg()
    training = create_training_handoff(
        {
            "final_checkpoint_path": str(checkpoint),
            "wandb_run_id": "run-architecture",
            "wandb_run_name": "MASAC-LeakyReLU-Kaiming-NoLN-H512x256-seed44-GPU-run-architecture",
        },
        cfg,
        repo=tmp_path,
    )
    payload, _ = load_training_handoff(training["handoff_path"])
    assert payload["architecture"]["activation"] == "leaky_relu"
    assert payload["architecture"]["weight_init"] == "kaiming"
    assert payload["architecture"]["layer_norm"] is False
    assert payload["source_wandb_run_name"].startswith("MASAC-LeakyReLU-Kaiming-NoLN")


def test_huggingface_checkpoint_prefix_separates_architecture_variants():
    from uav_marl.evaluation.postprocess import huggingface_checkpoint_prefix

    assert huggingface_checkpoint_prefix(
        algorithm="masac",
        architecture="prelu_kaiming_ln",
        seed=44,
        episode_index=20000,
    ) == "checkpoints/masac/prelu_kaiming_ln/seed-44/episode-20000"
