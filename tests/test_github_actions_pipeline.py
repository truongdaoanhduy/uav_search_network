from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "kaggle_gpu_cpu_pipeline.yml"


def test_cloud_pipeline_workflow_has_gpu_then_cpu_contract():
    source = WORKFLOW.read_text()
    data = yaml.safe_load(source)

    workflow_root = data.get("on", data.get(True))
    assert "workflow_dispatch" in workflow_root
    assert "gpu_train_then_cpu_visualize" in data["jobs"]
    assert "secrets.KAGGLE_API_TOKEN" in source
    assert "--machine-shape NvidiaTeslaT4" in source
    assert "--experiment paper_20k" in source
    assert "runtime.num_envs=${{ inputs.envs }}" in source
    assert "runtime.max_gpus=2" in source
    assert "runtime.auto_multi_gpu=true" in source
    assert "runtime.cuda_graph_policy_actions=${CUDA_GRAPH}" in source
    assert "--cpu-log-wandb" in source
    assert "--cpu-credential-dataset" in source
    assert "actions/checkout@v7" in source
    assert "actions/setup-python@v7" in source
    assert "actions/upload-artifact@v7" in source


def test_cloud_pipeline_defaults_to_requested_20k_3000step_run():
    source = WORKFLOW.read_text()
    assert 'default: "20000"' in source
    assert 'default: "4096"' in source
    assert 'default: "3000"' in source
    assert "default: true" in source


def test_cloud_pipeline_dispatches_selected_algorithm():
    source = WORKFLOW.read_text()
    data = yaml.safe_load(source)
    workflow_root = data.get("on", data.get(True))
    inputs = workflow_root["workflow_dispatch"]["inputs"]

    assert inputs["algorithm"]["type"] == "choice"
    assert inputs["algorithm"]["default"] == "masac"
    assert inputs["algorithm"]["options"] == ["masac", "matd3"]
    assert 'ALGORITHM="${{ inputs.algorithm }}"' in source
    assert '--algorithm "${ALGORITHM}"' in source
    assert 'GPU_SLUG="uav-${ALGORITHM}-gha-${GITHUB_RUN_ID}-1-gpu"' in source
    assert 'CPU_SLUG="uav-${ALGORITHM}-gha-${GITHUB_RUN_ID}-1-cpu"' in source


def test_cloud_pipeline_launch_is_detached_from_long_kaggle_runtime():
    source = WORKFLOW.read_text()
    data = yaml.safe_load(source)
    job = data["jobs"]["gpu_train_then_cpu_visualize"]

    assert int(job["timeout-minutes"]) <= 30
    assert "--launch-only" in source
    assert "--gpu-session-timeout-seconds 43200" in source
    assert "--gpu-timeout-seconds 14400" not in source


def test_cloud_pipeline_keeps_periodic_checkpoint_defaults_for_recovery():
    source = WORKFLOW.read_text()
    assert "experiment.checkpoint.interval_episodes=${{ inputs.episodes }}" not in source
    assert "experiment.checkpoint.interval_steps=100000000" not in source


def test_cloud_pipeline_has_scheduled_reconciler():
    reconcile = ROOT / ".github" / "workflows" / "kaggle_gpu_cpu_reconcile.yml"
    assert reconcile.is_file()
    source = reconcile.read_text()
    data = yaml.safe_load(source)
    workflow_root = data.get("on", data.get(True))
    assert "schedule" in workflow_root
    assert "workflow_dispatch" in workflow_root
    assert "scripts/reconcile_kaggle_pipeline.py" in source
    assert "secrets.KAGGLE_API_TOKEN" in source


def test_cloud_workflows_match_the_current_github_kaggle_credential_owner():
    pipeline_source = WORKFLOW.read_text()
    reconcile = ROOT / ".github" / "workflows" / "kaggle_gpu_cpu_reconcile.yml"
    reconcile_source = reconcile.read_text()

    for source in (pipeline_source, reconcile_source):
        assert "KAGGLE_USERNAME: haibro1234" in source
        assert "haibro1234/uav-wandb-credential-masac50k-d59860c" in source
    assert "KAGGLE_ACCOUNT_NAME: account_02" in pipeline_source
