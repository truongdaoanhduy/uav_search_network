"""MARL algorithm registry and lazy learner exports."""

from .registry import (
    AlgorithmDefinition,
    available_algorithms,
    get_algorithm,
    register_algorithm,
)


def __getattr__(name):
    if name in {"HybridMASAC", "HybridMASACActor"}:
        from .masac import HybridMASAC, HybridMASACActor

        return {
            "HybridMASAC": HybridMASAC,
            "HybridMASACActor": HybridMASACActor,
        }[name]
    if name in {"HybridMATD3", "HybridMATD3Actor"}:
        from .matd3 import HybridMATD3, HybridMATD3Actor

        return {
            "HybridMATD3": HybridMATD3,
            "HybridMATD3Actor": HybridMATD3Actor,
        }[name]
    raise AttributeError(name)


__all__ = [
    "AlgorithmDefinition",
    "HybridMASAC",
    "HybridMASACActor",
    "HybridMATD3",
    "HybridMATD3Actor",
    "available_algorithms",
    "get_algorithm",
    "register_algorithm",
]
