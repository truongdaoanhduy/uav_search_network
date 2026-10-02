"""Single experiment runner shared by Kaggle, Vast and local debug."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from .configuration import (
    apply_to_legacy_config,
    derive_transition_budget,
    describe_run,
    validate_config,
)
from .handoff import create_training_handoff
from .notebook_engine import load_notebook_engine


def _cfg_value(cfg: dict[str, Any], path: str) -> Any:
    current: Any = cfg
    for part in path.split("."):
        current = current[part]
    return current


def run_experiment(
    cfg: DictConfig,
    *,
    repo: str | Path | None = None,
) -> dict[str, Any]:
    plain = validate_config(cfg)
    repo_path = Path(repo or Path(__file__).resolve().parents[1]).resolve()

    from config import CONFIG as LEGACY_CONFIG

    apply_to_legacy_config(LEGACY_CONFIG, plain)

    summary = describe_run(plain)
    if bool(plain.get("dry_run", False)):
        return {
            "dry_run": True,
            **summary,
            "resolved_config": plain,
        }

    namespace = load_notebook_engine(repo_path)
    train_fn = namespace["train_full_gpu_auto"]

    total_episodes = int(
        _cfg_value(plain, "experiment.total_episodes")
    )
    num_envs = int(_cfg_value(plain, "runtime.num_envs"))
    max_steps = int(_cfg_value(plain, "task.scenario.max_steps"))
    total_transitions = derive_transition_budget(
        total_episodes=total_episodes,
        num_envs=num_envs,
        max_steps=max_steps,
    )

    result = train_fn(
        repo=repo_path,
        algorithm=str(_cfg_value(plain, "algorithm.name")),
        num_envs=num_envs,
        total_transitions=total_transitions,
        seed=int(_cfg_value(plain, "seed")),
        network_backend=str(
            _cfg_value(plain, "runtime.network_backend")
        ),
        max_gpus=int(_cfg_value(plain, "runtime.max_gpus")),
        auto_multi_gpu=bool(
            _cfg_value(plain, "runtime.auto_multi_gpu")
        ),
        compile_mode=_cfg_value(plain, "runtime.compile_mode"),
        fused_adam=bool(_cfg_value(plain, "runtime.fused_adam")),
        enable_wandb=True,
        execution_mode=str(
            _cfg_value(plain, "runtime.execution_mode")
        ),
        target_episodes=total_episodes,
        device=str(_cfg_value(plain, "runtime.device")),
    )
    result["experiment"] = str(
        _cfg_value(plain, "experiment.name")
    )
    result["runtime_provider"] = str(
        _cfg_value(plain, "runtime.provider")
    )
    result["resolved_config_sha256"] = str(
        LEGACY_CONFIG["_resolved_config_sha256"]
    )
    result["derived_transition_ceiling"] = int(total_transitions)

    visualization_session = str(
        _cfg_value(
            plain,
            "experiment.visualization.session",
        )
    ).strip().lower()
    if visualization_session == "separate_cpu":
        result = create_training_handoff(
            result,
            plain,
            repo=repo_path,
        )
    elif visualization_session != "same_session":
        raise ValueError(
            "experiment.visualization.session must be separate_cpu or same_session"
        )
    return result
