"""Algorithm registry metadata for the composable experiment runner."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlgorithmDefinition:
    name: str
    legacy_prefix: str
    off_policy: bool
    supports_ddp: bool = True


_REGISTRY: dict[str, AlgorithmDefinition] = {}


def register_algorithm(definition: AlgorithmDefinition) -> None:
    name = definition.name.strip().lower()
    if not name:
        raise ValueError("algorithm name must not be empty")
    if name in _REGISTRY:
        raise ValueError(f"algorithm already registered: {name}")
    _REGISTRY[name] = definition


def get_algorithm(name: str) -> AlgorithmDefinition:
    normalized = str(name).strip().lower()
    try:
        return _REGISTRY[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"unknown algorithm {normalized!r}; available: {available}"
        ) from exc


def available_algorithms() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


register_algorithm(
    AlgorithmDefinition(
        name="masac",
        legacy_prefix="masac",
        off_policy=True,
    )
)
register_algorithm(
    AlgorithmDefinition(
        name="matd3",
        legacy_prefix="matd3",
        off_policy=True,
    )
)
