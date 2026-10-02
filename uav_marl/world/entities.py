"""Production source migrated from frozen fix_test_gpu.ipynb.

Do not import the notebook at runtime. The notebook is retained only as a
reference/regression artifact. Production changes belong in Python modules.
Migrated from notebook cells 7..10.
"""

from ..common import *  # noqa: F401,F403

# --- frozen notebook cell 7 ---
@dataclass
class UAV:
    id: int
    position: np.ndarray
    velocity: np.ndarray
    battery_j: float

    active: bool = True


# --- frozen notebook cell 8 ---
@dataclass
class Target:
    id: int
    position: np.ndarray
    confirmed: bool = False


# --- frozen notebook cell 9 ---
@dataclass
class Obstacle:
    position: np.ndarray
    radius: float
    height: float

    def __post_init__(self):
        self.position = np.asarray(
            self.position,
            dtype=np.float64,
        )
        self.radius = float(self.radius)
        self.height = float(self.height)

        if self.position.shape != (2,):
            raise ValueError(
                "Obstacle.position must be the XY center with shape (2,)"
            )

        if not np.all(np.isfinite(self.position)):
            raise ValueError(
                "Obstacle.position must contain only finite values"
            )

        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError(
                "Obstacle.radius must be finite and > 0"
            )

        if not np.isfinite(self.height) or self.height <= 0.0:
            raise ValueError(
                "Obstacle.height must be finite and > 0"
            )


# --- frozen notebook cell 10 ---
@dataclass
class Report:
    target_id: int
    source_uav: int
    created_step: int
    size_bytes: int
    ttl_s: float
    delivered_bytes: int = 0

    def __post_init__(self):
        for name, value in (
            ("target_id", self.target_id),
            ("source_uav", self.source_uav),
            ("created_step", self.created_step),
            ("size_bytes", self.size_bytes),
            ("delivered_bytes", self.delivered_bytes),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
            ):
                raise TypeError(
                    f"{name} must be an integer"
                )

        self.target_id = int(self.target_id)
        self.source_uav = int(self.source_uav)
        self.created_step = int(self.created_step)
        self.size_bytes = int(self.size_bytes)
        self.delivered_bytes = int(self.delivered_bytes)
        self.ttl_s = float(self.ttl_s)

        if self.target_id < 0:
            raise ValueError(
                "target_id must be >= 0"
            )

        if self.source_uav < 0:
            raise ValueError(
                "source_uav must be >= 0"
            )

        if self.created_step < 0:
            raise ValueError(
                "created_step must be >= 0"
            )

        if self.size_bytes <= 0:
            raise ValueError(
                "size_bytes must be > 0"
            )

        if (
            not np.isfinite(self.ttl_s)
            or self.ttl_s <= 0.0
        ):
            raise ValueError(
                "ttl_s must be finite and > 0"
            )

        if not (
            0
            <= self.delivered_bytes
            <= self.size_bytes
        ):
            raise ValueError(
                "delivered_bytes must be in "
                "[0, size_bytes]"
            )


# Export public and internal helpers required by downstream production layers.
__all__ = [name for name in globals() if not name.startswith("__")]
