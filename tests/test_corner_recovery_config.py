"""Validate the actual launch profiles, not only a special recovery override."""
from pathlib import Path
import warnings

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from uav_marl.configuration import validate_config


def _compose(name="config", *overrides):
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
        return compose(config_name=name, overrides=list(overrides))


@pytest.mark.parametrize("runtime", ["kaggle_2xt4", "vast_1gpu", "vast_2gpu", "local_debug"])
@pytest.mark.parametrize("algorithm", ["masac", "matd3"])
def test_default_launch_profiles_pass_temporal_guard(runtime, algorithm):
    cfg = _compose("config", f"runtime={runtime}", f"algorithm={algorithm}")
    validate_config(cfg)


def test_original_run_budget_is_still_rejected():
    cfg = _compose("config", "algorithm.replay_capacity=50000", "algorithm.learning_starts=2000")
    with pytest.raises(ValueError, match="temporal coverage"):
        validate_config(cfg)


def test_recovery_profile_has_temporal_coverage():
    old, new = _compose(), _compose("corner_recovery")
    assert old.task == new.task
    assert old.reward == new.reward
    assert old.architecture == new.architecture
    assert new.algorithm.name == "masac"
    assert new.runtime.num_envs == 3072
    horizon = min(new.task.scenario.max_steps, 1.0 / (1.0 - new.algorithm.gamma))
    assert new.algorithm.replay_capacity / new.runtime.num_envs >= horizon
    assert new.algorithm.learning_starts / new.runtime.num_envs >= horizon
    assert new.experiment.total_episodes == 20000
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_config(new)
    assert not any("temporal coverage" in str(w.message) for w in caught)


def test_runtime_budget_preserves_explicit_algorithm_overrides():
    cfg = _compose("config", "algorithm.replay_capacity=600000", "algorithm.learning_starts=500000")
    validate_config(cfg)
    assert cfg.algorithm.replay_capacity == 600000
    assert cfg.algorithm.learning_starts == 500000


def test_cpu_profile_keeps_small_replay_default():
    cfg = _compose("config", "runtime=local_debug")
    assert cfg.algorithm.replay_capacity == 50000
    assert cfg.algorithm.learning_starts == 2000


@pytest.mark.parametrize("override", ["algorithm.gamma=0.999", "runtime.num_envs=8192"])
def test_guard_rechecks_explicit_overrides(override):
    with pytest.raises(ValueError, match="temporal coverage"):
        validate_config(_compose("config", override))


@pytest.mark.parametrize("algorithm", ["masac", "matd3"])
def test_generated_kaggle_command_resolves_and_validates(algorithm):
    # Parse the command emitted by the production generator without executing it.
    import ast
    from scripts.kaggle_pipeline import _gpu_script

    source = _gpu_script(
        commit="d13fd9e", algorithm=algorithm, runtime="kaggle_2xt4",
        experiment="paper_20k", seed=44, cpu_ref="owner/cpu",
        overrides=["runtime.num_envs=4096", "runtime.max_gpus=2",
                   "runtime.auto_multi_gpu=true", "runtime.cuda_graph_policy_actions=false"],
    )
    command_node = next(
        node.value for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "command" for target in node.targets)
    )
    # First two entries are Python/Path expressions, the rest are literal Hydra args.
    command = [ast.literal_eval(node) for node in command_node.left.elts[2:]]
    command += ast.literal_eval(command_node.right)
    cfg = _compose(command[1], *command[2:])
    plain = validate_config(cfg)
    assert plain["runtime"]["num_envs"] == 4096
    assert OmegaConf.to_container(cfg, resolve=True)["algorithm"]["replay_capacity"] >= 409600


@pytest.mark.parametrize("device", ["cp", "cuda:99"])
def test_invalid_replay_storage_device_fails_before_launch(device):
    with pytest.raises(ValueError, match="replay_storage_device"):
        validate_config(_compose("config", f"runtime.offpolicy.replay_storage_device={device}"))
