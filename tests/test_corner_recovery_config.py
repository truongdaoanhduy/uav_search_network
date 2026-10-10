from pathlib import Path
import warnings
import pytest
from hydra import compose, initialize_config_dir
from uav_marl.configuration import validate_config


def _compose(name):
    with initialize_config_dir(version_base='1.3', config_dir=str(Path(__file__).resolve().parents[1] / 'configs')):
        return compose(config_name=name)


def test_old_4096_env_production_config_is_rejected_for_temporal_coverage():
    with pytest.raises(ValueError, match='temporal coverage'):
        validate_config(_compose('config'))


def test_recovery_profile_has_temporal_coverage_without_sacrificing_gpu_parallelism():
    old, new = _compose('config'), _compose('corner_recovery')
    assert old.task == new.task
    assert old.reward == new.reward
    assert old.architecture == new.architecture
    assert new.algorithm.name == 'masac'
    assert new.runtime.num_envs == 3072
    effective_horizon = min(
        new.task.scenario.max_steps,
        1.0 / (1.0 - new.algorithm.gamma),
    )
    assert new.algorithm.replay_capacity / new.runtime.num_envs >= effective_horizon
    assert new.algorithm.learning_starts / new.runtime.num_envs >= effective_horizon
    assert new.experiment.total_episodes == 20000
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        validate_config(new)
    assert not any('temporal coverage' in str(w.message) for w in caught)


def test_vast_gpu_profile_is_also_rejected_when_temporal_coverage_is_too_short():
    cfg = _compose('config')
    cfg.runtime.provider = 'vast'
    cfg.runtime.max_gpus = 1
    cfg.runtime.auto_multi_gpu = False
    cfg.runtime.num_envs = 2048
    with pytest.raises(ValueError, match='temporal coverage'):
        validate_config(cfg)
