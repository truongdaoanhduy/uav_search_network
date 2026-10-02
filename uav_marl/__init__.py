"""Composable UAV MARL research package.

Production training/evaluation lives in Python modules. The historical
fix_test_gpu.ipynb notebook is retained only for regression/reference tests.
"""

from .configuration import (
    apply_to_legacy_config,
    derive_transition_budget,
    resolved_config_sha256,
    validate_config,
)


def run_experiment(*args, **kwargs):
    """Lazy import to keep configuration-only imports lightweight."""
    from .runner import run_experiment as _run_experiment

    return _run_experiment(*args, **kwargs)


__all__ = [
    "apply_to_legacy_config",
    "derive_transition_budget",
    "resolved_config_sha256",
    "run_experiment",
    "validate_config",
]
