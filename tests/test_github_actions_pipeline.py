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
    assert "actions/upload-artifact@v4" in source


def test_cloud_pipeline_defaults_to_requested_20k_3000step_run():
    source = WORKFLOW.read_text()
    assert 'default: "20000"' in source
    assert 'default: "4096"' in source
    assert 'default: "3000"' in source
    assert "default: true" in source
